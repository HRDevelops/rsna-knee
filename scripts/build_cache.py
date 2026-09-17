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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.preprocess import (  # noqa: E402
    load_and_order_series, normalize_uint8, resample_and_crop, slice_normal,
)


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
    parser.add_argument("--study-list", type=Path, default=None,
                         help="Optional file, one study_uid per line -- restrict to this subset "
                              "(2026-09-17, quota-cheap resolution ablation: build a cache for a "
                              "small subset instead of the full ~4400-study corpus). Default: "
                              "every study under --raw/train_series, unchanged behavior.")
    args = parser.parse_args()

    train_series_dir = args.raw / "train_series"
    study_dirs = sorted(p for p in train_series_dir.iterdir() if p.is_dir())
    if args.study_list is not None:
        wanted = {line.strip() for line in args.study_list.read_text().splitlines() if line.strip()}
        study_dirs = [d for d in study_dirs if d.name in wanted]
        print(f"--study-list restricted to {len(study_dirs)}/{len(wanted)} requested studies found "
              f"under {train_series_dir}.", file=sys.stderr)
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
