"""Build the preprocessing cache: per-series fixed-spacing uint8 stacks (EXECUTION_PLAN Phase 2.1).

Per user decision 2026-09-06 (quota question still with admins): build ONLY the single
uint8/384px/0.3mm variant — no second 512px crop variant.

Per series:
  1. Decode every slice (pydicom + pylibjpeg), apply RescaleSlope/Intercept.
  2. Order slices by ImagePositionPatient projected onto the slice normal (cross product of the
     two ImageOrientationPatient direction cosines) — NOT InstanceNumber alone (EXECUTION_PLAN
     2.2). Also compare against the InstanceNumber-implied order and flag if they disagree, and
     flag irregular inter-slice spacing (possible missing slices / interleaved acquisition).
  3. Resample each 2D slice from its native PixelSpacing to isotropic 0.3mm/px.
  4. Center-crop-or-pad to 384x384.
  5. Per-series intensity normalize: clip to the series' own p1-p99.5 percentile range, scale to
     uint8 [0, 255].
  6. Stack into one [n_slices, 384, 384] uint8 array, save as data/cache/<study_uid>/<series_uid>.npy.

Never crashes on one bad series — errors are logged into the manifest and the run continues (a
crash here would waste the whole array-job slot for nothing).

Sharded for SLURM array jobs: --shard S --num-shards N processes studies where index % N == S.
Resumable: skips a series if its .npy output already exists.

Usage (see slurm/cache_build.sbatch):
    python scripts/build_cache.py --raw data/raw --out data/cache \
        --manifest-out artifacts/cache_manifests --shard 0 --num-shards 20 \
        --spacing 0.3 --size 384 --dtype uint8
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

try:
    from pydicom.pixels import apply_modality_lut
except ImportError:  # older pydicom layout
    from pydicom.pixel_data_handlers.util import apply_modality_lut


@dataclass
class SeriesResult:
    study_uid: str
    series_uid: str
    ok: bool = False
    error: str = ""
    n_slices: int = 0
    native_spacing: str = ""
    order_mismatch: bool = False
    irregular_spacing: bool = False
    out_path: str = ""
    skipped_existing: bool = False


def slice_normal(iop: list[float]) -> np.ndarray:
    row = np.array(iop[0:3], dtype=np.float64)
    col = np.array(iop[3:6], dtype=np.float64)
    return np.cross(row, col)


def load_and_order_series(series_dir: Path) -> tuple[list[np.ndarray], dict]:
    dcm_files = sorted(series_dir.glob("*.dcm"))
    datasets = []
    for f in dcm_files:
        ds = pydicom.dcmread(str(f), force=True)
        datasets.append(ds)
    if not datasets:
        raise ValueError("no .dcm files in series dir")

    iop = datasets[0].get("ImageOrientationPatient")
    instance_numbers = [int(ds.get("InstanceNumber", i)) for i, ds in enumerate(datasets)]
    if iop is not None:
        normal = slice_normal([float(x) for x in iop])
        projections = [
            float(np.dot(np.array([float(x) for x in ds.get("ImagePositionPatient", [0, 0, i])]), normal))
            for i, ds in enumerate(datasets)
        ]
    else:
        projections = list(range(len(datasets)))  # no orientation info — fall back to file order

    ipp_order = np.argsort(projections)
    inst_order = np.argsort(instance_numbers)
    order_mismatch = not np.array_equal(ipp_order, inst_order)

    sorted_projections = np.array(projections)[ipp_order]
    gaps = np.diff(sorted_projections)
    irregular_spacing = False
    if len(gaps) > 1:
        nonzero_gaps = np.abs(gaps[np.abs(gaps) > 1e-6])
        if len(nonzero_gaps) > 1:
            irregular_spacing = bool(nonzero_gaps.max() > 3 * nonzero_gaps.min())
        elif len(gaps) > 0 and np.any(np.abs(gaps) <= 1e-6):
            irregular_spacing = True  # duplicate slice positions

    ordered_datasets = [datasets[i] for i in ipp_order]
    pixel_spacing = ordered_datasets[0].get("PixelSpacing", [1.0, 1.0])
    meta = {
        "n_slices": len(ordered_datasets),
        "native_spacing": str([float(x) for x in pixel_spacing]),
        "order_mismatch": order_mismatch,
        "irregular_spacing": irregular_spacing,
    }
    return ordered_datasets, meta


def resample_and_crop(ds, target_spacing: float, target_size: int) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    try:
        arr = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:  # noqa: BLE001 - not all series have a valid modality LUT
        pass

    row_spacing, col_spacing = [float(x) for x in ds.get("PixelSpacing", [1.0, 1.0])]
    scale_row = row_spacing / target_spacing
    scale_col = col_spacing / target_spacing
    new_rows = max(1, int(round(arr.shape[0] * scale_row)))
    new_cols = max(1, int(round(arr.shape[1] * scale_col)))

    from scipy.ndimage import zoom
    resampled = zoom(arr, (new_rows / arr.shape[0], new_cols / arr.shape[1]), order=1)

    out = np.zeros((target_size, target_size), dtype=np.float32)
    r, c = resampled.shape
    # center crop
    r0 = max(0, (r - target_size) // 2)
    c0 = max(0, (c - target_size) // 2)
    cropped = resampled[r0:r0 + target_size, c0:c0 + target_size]
    # center pad if smaller than target
    pr0 = max(0, (target_size - cropped.shape[0]) // 2)
    pc0 = max(0, (target_size - cropped.shape[1]) // 2)
    out[pr0:pr0 + cropped.shape[0], pc0:pc0 + cropped.shape[1]] = cropped
    return out


def normalize_uint8(volume: np.ndarray) -> np.ndarray:
    p1, p995 = np.percentile(volume, [1, 99.5])
    if p995 <= p1:
        p995 = p1 + 1.0
    clipped = np.clip(volume, p1, p995)
    scaled = (clipped - p1) / (p995 - p1) * 255.0
    return scaled.astype(np.uint8)


def process_series(study_uid: str, series_uid: str, series_dir: Path, out_dir: Path,
                    target_spacing: float, target_size: int) -> SeriesResult:
    result = SeriesResult(study_uid=study_uid, series_uid=series_uid)
    out_path = out_dir / study_uid / f"{series_uid}.npy"
    if out_path.exists():
        result.ok = True
        result.skipped_existing = True
        result.out_path = str(out_path)
        return result

    try:
        ordered_datasets, meta = load_and_order_series(series_dir)
        slices = [resample_and_crop(ds, target_spacing, target_size) for ds in ordered_datasets]
        volume = np.stack(slices, axis=0)
        volume_uint8 = normalize_uint8(volume)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, volume_uint8)

        result.ok = True
        result.n_slices = meta["n_slices"]
        result.native_spacing = meta["native_spacing"]
        result.order_mismatch = meta["order_mismatch"]
        result.irregular_spacing = meta["irregular_spacing"]
        result.out_path = str(out_path)
    except Exception as exc:  # noqa: BLE001 - one bad series must not kill the shard
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="data/raw (has train_series/)")
    parser.add_argument("--out", type=Path, required=True, help="data/cache output root")
    parser.add_argument("--manifest-out", type=Path, default=Path("artifacts/cache_manifests"))
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--spacing", type=float, default=0.3)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--dtype", type=str, default="uint8", choices=["uint8"])
    args = parser.parse_args()

    train_series_dir = args.raw / "train_series"
    study_dirs = sorted(p for p in train_series_dir.iterdir() if p.is_dir())
    my_studies = [d for i, d in enumerate(study_dirs) if i % args.num_shards == args.shard]
    print(f"Shard {args.shard}/{args.num_shards}: {len(my_studies)} studies "
          f"(of {len(study_dirs)} total).", file=sys.stderr)

    results: list[SeriesResult] = []
    for i, study_dir in enumerate(my_studies):
        study_uid = study_dir.name
        series_dirs = sorted(p for p in study_dir.iterdir() if p.is_dir())
        for series_dir in series_dirs:
            series_uid = series_dir.name
            r = process_series(study_uid, series_uid, series_dir, args.out,
                                args.spacing, args.size)
            results.append(r)
        if (i + 1) % 50 == 0:
            n_ok = sum(1 for r in results if r.ok)
            print(f"  {i + 1}/{len(my_studies)} studies, {n_ok}/{len(results)} series ok",
                  file=sys.stderr)

    manifest_rows = [{
        "study_uid": r.study_uid, "series_uid": r.series_uid, "ok": r.ok, "error": r.error,
        "n_slices": r.n_slices, "native_spacing": r.native_spacing,
        "order_mismatch": r.order_mismatch, "irregular_spacing": r.irregular_spacing,
        "out_path": r.out_path, "skipped_existing": r.skipped_existing,
    } for r in results]
    manifest_df = pd.DataFrame(manifest_rows)
    args.manifest_out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_out / f"shard_{args.shard:03d}.parquet"
    manifest_df.to_parquet(manifest_path, index=False)

    n_ok = int(manifest_df["ok"].sum())
    n_err = len(manifest_df) - n_ok
    n_order_mismatch = int(manifest_df["order_mismatch"].sum())
    n_irregular = int(manifest_df["irregular_spacing"].sum())
    print(f"Shard {args.shard} done: {n_ok} ok, {n_err} errors, "
          f"{n_order_mismatch} order-mismatch, {n_irregular} irregular-spacing. "
          f"Manifest: {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
