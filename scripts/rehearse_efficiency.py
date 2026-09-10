"""Local/HPC rehearsal of notebooks/submit-efficiency.ipynb's exact logic (fp16, batched decode,
fold-count ladder) against local paths — CLAUDE.md §9 load-test rehearsal.

Batched decode uses ThreadPoolExecutor, not ProcessPoolExecutor (2026-09-10 finding, see
infer_core.decode_one_for_pool's docstring): a process pool hit two real, compounding problems in
a notebook-loaded-package context — (1) forking after CUDA is already initialized in the parent
deadlocks the child, and (2) even switching to a spawn context, the custom sys.modules-injection
package loader the notebooks use isn't visible to a freshly-spawned interpreter, so the worker
function fails to import there either. Threads sidestep both (shared process, no pickling) and
still parallelize the expensive part of decode, since DICOM decompression (pylibjpeg/gdcm) and
numpy/scipy resampling are C extensions that release the GIL.

Usage:
    python scripts/rehearse_efficiency.py \
        --weights-dir $RSNA_ROOT/artifacts/exp008_dinov2s_slot_distilled \
        --raw $RSNA_ROOT/artifacts/synthetic_loadtest \
        --out submission_efficiency.csv --budget-seconds 2700
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.dataset import TARGET_COLUMNS, _patch_size_for_backbone  # noqa: E402
from rsna_knee.infer_core import (  # noqa: E402
    WallClockGuard, decode_one_for_pool, nearest_patch_multiple, prior_row, run_study_inference,
)
from rsna_knee.model import SlotModel  # noqa: E402

BACKBONE = "vit_small_patch14_dinov2.lvd142m"
SLICES_PER_SERIES = 16
IMAGE_SIZE = 320


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_efficiency.csv"))
    parser.add_argument("--budget-seconds", type=float, default=45 * 60)
    parser.add_argument("--calibration-studies", type=int, default=10)
    parser.add_argument("--decode-chunk-size", type=int, default=24)
    # os.cpu_count() reports the NODE's full CPU count on this cluster's shared SLURM nodes,
    # not the --cpus-per-task allocation (found 2026-09-10: requested 16, os.cpu_count() reported
    # 192, spawning 192 worker processes that thrashed each other on 16 real cores and never
    # finished 20 studies' decode). SLURM_CPUS_PER_TASK reflects the actual allocation and should
    # be trusted first; Kaggle's os.cpu_count() (no SLURM) is expected to already be correct.
    _default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 4)
    parser.add_argument("--decode-workers", type=int, default=max(1, min(_default_workers - 1, 32)))
    parser.add_argument("--limit", type=int, default=None, help="truncate test set for a quick sanity pass")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_size = nearest_patch_multiple(IMAGE_SIZE, _patch_size_for_backbone(BACKBONE))
    autocast_dtype = torch.float16
    print(f"device={device} out_size={out_size} decode_workers={args.decode_workers}", file=sys.stderr)

    all_models = []
    for fold_dir in sorted(args.weights_dir.glob("fold*")):
        ckpt_path = fold_dir / "best.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = SlotModel(BACKBONE, num_targets=len(TARGET_COLUMNS), pretrained=False).to(device)
        model.load_state_dict(ckpt["ema"])
        model.eval()
        all_models.append(model)
    print(f"Loaded {len(all_models)}/5 folds.", file=sys.stderr)
    if not all_models:
        raise RuntimeError("No fold models loaded.")

    test_df = pd.read_csv(args.raw / "test.csv")
    if args.limit:
        test_df = test_df.iloc[:args.limit].reset_index(drop=True)
    series_df = pd.read_csv(args.raw / "test_series.csv").rename(
        columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"}
    )
    series_root = args.raw / "test_series"
    print(f"{len(test_df)} test studies, {len(series_df)} series.", file=sys.stderr)

    def decode_chunk(study_uids: list[str]) -> dict:
        tasks = [
            (uid, series_df[series_df["study_uid"] == uid].to_dict("records"), str(series_root),
             SLICES_PER_SERIES, out_size)
            for uid in study_uids
        ]
        results = {}
        with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
            futures = {pool.submit(decode_one_for_pool, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                uid, slots, tag_sides, err = fut.result()
                results[uid] = (slots, tag_sides, err)
        return results

    LADDER_FOLD_COUNTS = [5, 4, 3, 2, 1]
    guard = WallClockGuard(args.budget_seconds)
    n_calib = min(args.calibration_studies, len(test_df))
    calib_uids = list(test_df["StudyInstanceUID"].iloc[:n_calib])
    t0 = time.time()
    calib_decoded = decode_chunk(calib_uids)
    calib_rows = []
    for uid in calib_uids:
        slots, tag_sides, err = calib_decoded[uid]
        if slots and err is None:
            probs = run_study_inference(slots, tag_sides, all_models, device, SLICES_PER_SERIES, out_size, autocast_dtype)
            calib_rows.append({"StudyInstanceUID": uid, **{c: float(p) for c, p in zip(TARGET_COLUMNS, probs)}})
    calib_elapsed = time.time() - t0
    per_study_5fold = calib_elapsed / max(n_calib, 1)
    print(f"Calibration: {n_calib} studies, {calib_elapsed:.1f}s ({per_study_5fold:.2f}s/study at 5 folds)",
          file=sys.stderr)

    chosen_n_folds = LADDER_FOLD_COUNTS[-1]
    for n_folds in LADDER_FOLD_COUNTS:
        projected_per_study = per_study_5fold * 0.5 + (per_study_5fold * 0.5) * (n_folds / 5)
        projected_total = projected_per_study * len(test_df)
        fits = projected_total <= guard.budget * 0.85
        print(f"  {n_folds} folds: projected {projected_total / 60:.1f} min -> {'FITS' if fits else 'too slow'}",
              file=sys.stderr)
        if fits:
            chosen_n_folds = n_folds
            break
    active_models = all_models[:chosen_n_folds]
    print(f"Chosen fold count: {chosen_n_folds}/5", file=sys.stderr)

    rows = list(calib_rows) if chosen_n_folds == 5 else []
    done = {r["StudyInstanceUID"] for r in rows}
    remaining = [u for u in test_df["StudyInstanceUID"] if u not in done]

    n_prior_fallback = n_dual_flip = 0
    for chunk_start in range(0, len(remaining), args.decode_chunk_size):
        chunk_uids = remaining[chunk_start:chunk_start + args.decode_chunk_size]
        if guard.over(margin=0.05):
            for uid in chunk_uids + remaining[chunk_start + args.decode_chunk_size:]:
                rows.append({"StudyInstanceUID": uid, **{c: float(p) for c, p in zip(TARGET_COLUMNS, prior_row())}})
                n_prior_fallback += 1
            break
        decoded = decode_chunk(chunk_uids)
        for uid in chunk_uids:
            row = {"StudyInstanceUID": uid}
            try:
                slots, tag_sides, err = decoded[uid]
                if err is not None or not slots:
                    raise RuntimeError(err or "no series decoded successfully")
                if all(s not in ("R", "L") for s in tag_sides):
                    n_dual_flip += 1
                probs = run_study_inference(slots, tag_sides, active_models, device, SLICES_PER_SERIES, out_size, autocast_dtype)
                row.update({c: float(p) for c, p in zip(TARGET_COLUMNS, probs)})
            except Exception as e:  # noqa: BLE001
                n_prior_fallback += 1
                row.update({c: float(p) for c, p in zip(TARGET_COLUMNS, prior_row())})
            rows.append(row)
        print(f"  {len(rows)}/{len(test_df)} studies, {guard.elapsed():.0f}s elapsed "
              f"({guard.fraction_used():.1%} of budget)", file=sys.stderr)

    sub_df = pd.DataFrame(rows)[["StudyInstanceUID"] + TARGET_COLUMNS]
    sub_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(sub_df)} rows, folds={chosen_n_folds}/5, "
          f"{n_prior_fallback} prior fallbacks, {n_dual_flip} dual-flip, {guard.elapsed():.0f}s total.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
