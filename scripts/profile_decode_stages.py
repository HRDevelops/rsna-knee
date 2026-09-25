# Copyright 2026 Husnain Rasool
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workstream C, step 1: where does exp018's decode time actually go?

Re-runs rsna_knee.wmil.decode_study's exact logic on real studies with a timer around every stage, and
counts the work: files whose headers are read, slices pixel-decoded, slices stacked (all of which are used,
since every stacked slice falls in some 3-slice window). Stages:

  count    glob-count *.dcm per series (feeds pick_slots)
  header   pydicom.dcmread(stop_before_pixels=True) for EVERY file of each chosen series (ordering)
  order    _order_from_headers + _geometry (pure python/numpy)
  read     pydicom.dcmread(full) of each selected file (pixel data still compressed)
  pixel    ds.pixel_array (the actual JPEG-Lossless / JPEG-2000 / raw decode)
  resample apply_modality_lut + scipy.ndimage.zoom(order=1) + crop/pad  (resample_and_crop)
  norm     normalize_uint8 (percentile clip) per series
  orient   normalize_orientation per study

The same run also measures candidate EXACT-output speedups (header reads restricted to the tags the
ordering/geometry code reads, via `specific_tags`) and asserts they reproduce the reference arrays
bit-for-bit. Single-threaded for the stage breakdown; see --workers for pool throughput.

    python scripts/profile_decode_stages.py --n-studies 200
"""
from __future__ import annotations

import argparse
import glob as _glob
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pydicom.pixel_data_handlers.util import apply_modality_lut  # noqa: E402

from rsna_knee import wmil  # noqa: E402
from rsna_knee.preprocess import _order_from_headers, normalize_uint8  # noqa: E402

# Every tag the ordering (_order_from_headers) and geometry (_geometry) code reads from a header.
ORDER_TAGS = ["ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber", "PixelSpacing",
              "Rows", "Columns", "Laterality", "ImageLaterality"]


def resample(ds, T: dict) -> np.ndarray:
    """preprocess.resample_and_crop, verbatim, with the pixel / resample split timed separately."""
    t0 = time.perf_counter()
    arr = ds.pixel_array.astype(np.float32)
    T["pixel"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        arr = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:  # noqa: BLE001
        pass
    row_spacing, col_spacing = [float(x) for x in ds.get("PixelSpacing", [1.0, 1.0])]
    new_rows = max(1, int(round(arr.shape[0] * row_spacing / wmil.TARGET_SPACING)))
    new_cols = max(1, int(round(arr.shape[1] * col_spacing / wmil.TARGET_SPACING)))
    from scipy.ndimage import zoom
    resampled = zoom(arr, (new_rows / arr.shape[0], new_cols / arr.shape[1]), order=1)
    size = wmil.IMG
    out = np.zeros((size, size), dtype=np.float32)
    r, c = resampled.shape
    r0, c0 = max(0, (r - size) // 2), max(0, (c - size) // 2)
    cropped = resampled[r0:r0 + size, c0:c0 + size]
    pr0, pc0 = max(0, (size - cropped.shape[0]) // 2), max(0, (size - cropped.shape[1]) // 2)
    out[pr0:pr0 + cropped.shape[0], pc0:pc0 + cropped.shape[1]] = cropped
    T["resample"] += time.perf_counter() - t0
    return out


def series_slot(series_dir: Path, plane: str, n_take: int, T: dict, C: dict, header_tags) -> tuple:
    files = sorted(Path(series_dir).glob("*.dcm"))
    t0 = time.perf_counter()
    kw = {"specific_tags": header_tags} if header_tags else {}
    headers = [pydicom.dcmread(str(f), force=True, stop_before_pixels=True, **kw) for f in files]
    T["header"] += time.perf_counter() - t0
    C["headers_read"] += len(files)
    t0 = time.perf_counter()
    order, meta = _order_from_headers(headers)
    geom = wmil._geometry(headers, order)
    if plane == "Sagittal" and geom["normal_x"] < 0:
        order = order[::-1]
    n = len(order)
    lo, hi = int(round(wmil.SPAN[0] * (n - 1))), int(round(wmil.SPAN[1] * (n - 1)))
    take = np.clip(np.round(np.linspace(lo, hi, n_take)).astype(int), 0, n - 1)
    T["order"] += time.perf_counter() - t0
    decoded = {}
    for t in sorted(set(int(v) for v in take)):
        t0 = time.perf_counter()
        ds = pydicom.dcmread(str(files[order[t]]), force=True)
        T["read"] += time.perf_counter() - t0
        decoded[t] = resample(ds, T)
        C["pixel_decoded"] += 1
    slices = [decoded[int(t)] for t in take]
    C["stacked"] += len(slices)
    t0 = time.perf_counter()
    vol = normalize_uint8(np.stack(slices, axis=0))
    T["norm"] += time.perf_counter() - t0
    return vol, {**meta, **geom, "n_native": n}


def study(study_dir: Path, rows: list, T: dict, C: dict, header_tags=None):
    t0 = time.perf_counter()
    rr = []
    for r in rows:
        n = len(_glob.glob(str(Path(study_dir) / str(r["series_uid"]) / "*.dcm")))
        if n:
            rr.append({**r, "n_slices": n})
    T["count"] += time.perf_counter() - t0
    C["series_in_study"] += len(rr)
    chosen = wmil.pick_slots(rr)
    C["series_chosen"] += len(chosen)
    stack = np.zeros((wmil.DEPTH, wmil.IMG, wmil.IMG), dtype=np.uint8)
    present = np.zeros(wmil.NUM_SLOTS, dtype=bool)
    metas = [None] * wmil.NUM_SLOTS
    for i in range(wmil.NUM_SLOTS):
        r = chosen.get(i)
        if r is None:
            continue
        vol, meta = series_slot(Path(study_dir) / str(r["series_uid"]), wmil.SLOT_PLANE[i], wmil.SLOT_LEN[i], T, C,
                                header_tags)
        stack[wmil.SLOT_OFF[i]:wmil.SLOT_OFF[i] + wmil.SLOT_LEN[i]] = vol
        present[i] = True
        metas[i] = meta
    t0 = time.perf_counter()
    sides = wmil.resolve_slot_sides(metas, None)
    out = wmil.normalize_orientation(stack, metas, sides)
    T["orient"] += time.perf_counter() - t0
    return out, present


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-studies", type=int, default=200)
    ap.add_argument("--workers", type=int, nargs="*", default=[3, 4])
    args = ap.parse_args()

    import pandas as pd
    root = Path(os.environ["RSNA_ROOT"])
    raw = root / "data/raw"
    sdf = pd.read_csv(raw / "train_series.csv").rename(columns={
        "StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid",
        "Anatomical_Plane": "plane", "Fluid_Sensitive": "fluid"})
    uids = list(pd.read_parquet(root / "artifacts/exp018_coatnet2_wmil_v3labels/fold0/oof.parquet")["study_uid"])[:args.n_studies]
    rows = {u: sdf[sdf.study_uid == u][["series_uid", "plane", "fluid"]].to_dict("records") for u in uids}
    print(f"{len(uids)} fold-0 studies | cpus visible {os.cpu_count()} | OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}",
          flush=True)

    # ---- 1. single-threaded stage breakdown on the reference path, and the exactness reference
    T, C = defaultdict(float), defaultdict(int)
    ref = {}
    t_all = time.perf_counter()
    for u in uids:
        ref[u] = study(raw / "train_series" / u, rows[u], T, C)
    wall = time.perf_counter() - t_all
    n = len(uids)
    print(f"\n== reference decode (exp018 logic), single thread: {wall / n:.3f} s/study ==")
    for k in ("count", "header", "order", "read", "pixel", "resample", "norm", "orient"):
        print(f"  {k:<9} {T[k] / n * 1000:8.1f} ms/study  {T[k] / wall:6.1%}")
    print(f"  work/study: {C['series_in_study'] / n:.1f} series present, {C['series_chosen'] / n:.1f} chosen, "
          f"{C['headers_read'] / n:.1f} headers read, {C['pixel_decoded'] / n:.1f} slices pixel-decoded, "
          f"{C['stacked'] / n:.1f} slices stacked (all used by >=1 window)")

    # ---- 2. candidate exact speedup: header reads restricted to the tags the ordering code reads
    T2, C2 = defaultdict(float), defaultdict(int)
    t_all = time.perf_counter()
    mism = 0
    for u in uids:
        st, pr = study(raw / "train_series" / u, rows[u], T2, C2, header_tags=ORDER_TAGS)
        if not (np.array_equal(st, ref[u][0]) and np.array_equal(pr, ref[u][1])):
            mism += 1
    wall2 = time.perf_counter() - t_all
    print(f"\n== candidate: specific_tags header reads: {wall2 / n:.3f} s/study (header "
          f"{T2['header'] / n * 1000:.1f} ms vs {T['header'] / n * 1000:.1f} ms) | studies not bit-identical: "
          f"{mism}/{n} ==")

    # ---- 3. pool throughput at Kaggle-like worker counts (reference logic), wall-clock per study
    for w in args.workers:
        def one(u):
            return study(raw / "train_series" / u, rows[u], defaultdict(float), defaultdict(int))
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=w) as pool:
            list(pool.map(one, uids))
        print(f"== pool: {w} worker thread(s): {(time.perf_counter() - t0) / n:.3f} s/study wall ==", flush=True)


if __name__ == "__main__":
    main()
