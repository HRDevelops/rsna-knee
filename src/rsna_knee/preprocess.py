"""DICOM decode/order/resample/normalize — the exact per-series pipeline used to build the
training cache (scripts/build_cache.py). Factored out so scripts/build_cache.py and the inference
path (scripts/infer.py, notebooks/submit-exp002.ipynb) share one implementation instead of two
copies that could silently drift apart (a mismatch here would degrade the LB score without ever
raising an error).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom

try:
    from pydicom.pixels import apply_modality_lut
except ImportError:  # older pydicom layout
    from pydicom.pixel_data_handlers.util import apply_modality_lut


def slice_normal(iop: list[float]) -> np.ndarray:
    row = np.array(iop[0:3], dtype=np.float64)
    col = np.array(iop[3:6], dtype=np.float64)
    return np.cross(row, col)


def load_and_order_series(series_dir: Path) -> tuple[list, dict]:
    """Reads every .dcm in series_dir, orders by ImagePositionPatient projected onto the slice
    normal (not InstanceNumber — see scripts/build_cache.py module docstring for why)."""
    dcm_files = sorted(Path(series_dir).glob("*.dcm"))
    datasets = [pydicom.dcmread(str(f), force=True) for f in dcm_files]
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
        projections = list(range(len(datasets)))

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
            irregular_spacing = True

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
    r0 = max(0, (r - target_size) // 2)
    c0 = max(0, (c - target_size) // 2)
    cropped = resampled[r0:r0 + target_size, c0:c0 + target_size]
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


def decode_series_to_uint8(series_dir: Path, target_spacing: float = 0.3,
                            target_size: int = 384) -> tuple[np.ndarray, dict]:
    """Full pipeline for one series dir -> [n_slices, target_size, target_size] uint8, plus the
    first DICOM's raw Laterality-family tag values (for side resolution)."""
    ordered_datasets, meta = load_and_order_series(series_dir)
    slices = [resample_and_crop(ds, target_spacing, target_size) for ds in ordered_datasets]
    volume = np.stack(slices, axis=0)
    volume_uint8 = normalize_uint8(volume)
    ds0 = ordered_datasets[0]
    meta["tag_laterality"] = str(ds0.get("Laterality", "")) or str(ds0.get("ImageLaterality", ""))
    return volume_uint8, meta
