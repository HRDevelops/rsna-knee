"""Local/HPC rehearsal of notebooks/submit-efficiency.ipynb's exact logic (fp16, batched decode,
formula-driven fold-count selection) against local paths — CLAUDE.md §9 load-test rehearsal.

Batched decode uses ThreadPoolExecutor, not ProcessPoolExecutor (2026-09-10 finding, see
infer_core.decode_one_for_pool's docstring): a process pool hit two real, compounding problems in
a notebook-loaded-package context — (1) forking after CUDA is already initialized in the parent
deadlocks the child, and (2) even switching to a spawn context, the custom sys.modules-injection
package loader the notebooks use isn't visible to a freshly-spawned interpreter, so the worker
function fails to import there either. Threads sidestep both (shared process, no pickling) and
still parallelize the expensive part of decode, since DICOM decompression (pylibjpeg/gdcm) and
numpy/scipy resampling are C extensions that release the GIL.

--fast-decode (2026-09-10) uses infer_core.decode_one_for_pool_fast (header-only ordering pass +
pixel decode of only the T slices actually used) instead of a full-series decode — see
preprocess.decode_series_to_uint8_fast for the validated approximation. Fold count is picked to
minimize the CLAUDE.md §4 efficiency formula (not "does it fit a budget") — see OOF_AUC_BY_FOLDS
below and notebooks/submit-efficiency.ipynb cell-6 for the mirrored logic.

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
    WallClockGuard, decode_one_for_pool, decode_one_for_pool_fast, nearest_patch_multiple,
    prior_row, run_study_inference,
)
from rsna_knee.model import SlotModel  # noqa: E402

BACKBONE = "vit_small_patch14_dinov2.lvd142m"
SLICES_PER_SERIES = 16
IMAGE_SIZE = 320

# Mirrors notebooks/submit-efficiency.ipynb cell-2 (2026-09-10) — see that cell for full rationale.
OOF_MACRO_AUC_5FOLD = 0.8708
FOLD_COUNT_AUC_DELTA = {1: -0.0060, 2: -0.0030, 3: -0.0014, 4: -0.0004, 5: 0.0}
OOF_AUC_BY_FOLDS = {k: OOF_MACRO_AUC_5FOLD + d for k, d in FOLD_COUNT_AUC_DELTA.items()}
ASSUMED_MAX_AUC = 0.92
EFF_DENOM = 0.5 - ASSUMED_MAX_AUC


def efficiency_score(auc: float, runtime_seconds: float) -> float:
    return auc / EFF_DENOM + runtime_seconds / 32400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_efficiency.csv"))
    # 8h true Kaggle wall-clock safety cap (2026-09-10 correction — this used to default to the
    # CLAUDE.md §8 45min *target*, conflating a performance goal with the crash-prevention budget;
    # matches submit-efficiency.ipynb cell-2 and rehearse_ensemble.py's default).
    parser.add_argument("--budget-seconds", type=float, default=8 * 3600)
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
    parser.add_argument("--fast-decode", action="store_true",
                         help="decode_one_for_pool_fast (header-only ordering + T-slice-only pixel "
                              "decode) instead of the full-series decode — see "
                              "preprocess.decode_series_to_uint8_fast for the validated approximation")
    args = parser.parse_args()
    decode_worker_fn = decode_one_for_pool_fast if args.fast_decode else decode_one_for_pool

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
            futures = {pool.submit(decode_worker_fn, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                uid, slots, tag_sides, err = fut.result()
                results[uid] = (slots, tag_sides, err)
        return results

    guard = WallClockGuard(args.budget_seconds)
    n_calib = min(args.calibration_studies, len(test_df))
    calib_uids = list(test_df["StudyInstanceUID"].iloc[:n_calib])

    decode_t0 = time.time()
    calib_decoded = decode_chunk(calib_uids)
    decode_calib_elapsed = time.time() - decode_t0

    model_t0 = time.time()
    calib_rows = []
    for uid in calib_uids:
        slots, tag_sides, err = calib_decoded[uid]
        if slots and err is None:
            probs = run_study_inference(slots, tag_sides, all_models, device, SLICES_PER_SERIES, out_size, autocast_dtype)
            calib_rows.append({"StudyInstanceUID": uid, **{c: float(p) for c, p in zip(TARGET_COLUMNS, probs)}})
    model_calib_elapsed = time.time() - model_t0

    decode_per_study = decode_calib_elapsed / max(n_calib, 1)
    model_per_study_5fold = model_calib_elapsed / max(n_calib, 1)
    print(f"Calibration: {n_calib} studies — decode {decode_calib_elapsed:.1f}s ({decode_per_study:.3f}s/study), "
          f"model-5fold {model_calib_elapsed:.1f}s ({model_per_study_5fold:.3f}s/study)", file=sys.stderr)

    print(f"Fold-count selection — efficiency formula (maxAUC assumed {ASSUMED_MAX_AUC}):", file=sys.stderr)
    eff_by_folds = {}
    for k in sorted(OOF_AUC_BY_FOLDS):
        projected_per_study = decode_per_study + model_per_study_5fold * (k / 5)
        projected_total = projected_per_study * len(test_df)
        auc = OOF_AUC_BY_FOLDS[k]
        score = efficiency_score(auc, projected_total)
        fits = projected_total <= guard.budget * 0.85
        eff_by_folds[k] = (score, projected_total, fits)
        print(f"  {k} fold(s): OOF AUC {auc:.4f}, projected {projected_total / 60:.1f} min "
              f"({'fits' if fits else 'EXCEEDS BUDGET — excluded'}), efficiency score {score:.4f}", file=sys.stderr)

    feasible = {k: v for k, v in eff_by_folds.items() if v[2]}
    chosen_n_folds = min(feasible, key=lambda k: feasible[k][0]) if feasible else min(eff_by_folds, key=lambda k: eff_by_folds[k][1])
    active_models = all_models[:chosen_n_folds]
    print(f"Chosen fold count: {chosen_n_folds}/5 (efficiency score {eff_by_folds[chosen_n_folds][0]:.4f})", file=sys.stderr)

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
