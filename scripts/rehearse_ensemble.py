"""Local/HPC rehearsal of notebooks/submit-ensemble.ipynb's exact logic (frozen recipe file,
degradation ladder, calibration-based rung selection, multi-GPU member sharding, decode/inference
overlap) against local paths instead of /kaggle/input — CLAUDE.md §9 load-test rehearsal, run
before ever touching Kaggle. Mirrors the notebook cell-for-cell rather than reusing
scripts/infer.py, since the ensemble notebook's ladder logic doesn't exist there.

2026-09-11 T4 adaptation: fp16 (not bf16 — T4/sm_75 has no native bf16 tensor cores), members
split across however many CUDA devices are visible (2 on Kaggle's T4 x2; falls back to one device
on a normal single-GPU HPC job) and run from separate threads so different-device members overlap,
and decode of study N+1 runs in a background thread pool while study N is on the GPU
(infer_core.prefetch_decode) instead of the old decode-then-infer lockstep.

Usage:
    python scripts/rehearse_ensemble.py \
        --recipe src/rsna_knee/ensemble_recipe_v1.yaml \
        --weights-root $RSNA_ROOT/artifacts \
        --raw $RSNA_ROOT/artifacts/synthetic_loadtest \
        --out submission_ensemble.csv --budget-seconds 2700
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.dataset import TARGET_COLUMNS, _patch_size_for_backbone  # noqa: E402
from rsna_knee.infer_core import (  # noqa: E402
    WallClockGuard, decode_one_for_pool, ensemble_blend, load_ensemble_recipe, nearest_patch_multiple,
    prefetch_decode, prior_row, run_study_inference,
)
from rsna_knee.model import SlotModel  # noqa: E402

AUTOCAST_DTYPE = torch.float16


def load_member_models(weights_dir: Path, backbone: str, device: str) -> list:
    models = []
    for fold_dir in sorted(weights_dir.glob("fold*")):
        ckpt_path = fold_dir / "best.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = SlotModel(backbone, num_targets=len(TARGET_COLUMNS), pretrained=False).to(device)
        model.load_state_dict(ckpt["ema"])
        model.eval()
        models.append(model)
    return models


def assign_member_devices(members: dict, n_gpus: int, fallback_device: str) -> dict[str, str]:
    if n_gpus <= 1:
        return {name: fallback_device for name in members}
    loads = [0.0] * n_gpus
    assignment = {}
    for name, spec in sorted(members.items(),
                              key=lambda kv: -(3.0 if kv[1]["backbone"].startswith("vit_base") else 1.0)):
        cost = 3.0 if spec["backbone"].startswith("vit_base") else 1.0
        gpu = min(range(n_gpus), key=lambda i: loads[i])
        assignment[name] = f"cuda:{gpu}"
        loads[gpu] += cost
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True, help="dir with test.csv/test_series.csv/test_series/")
    parser.add_argument("--out", type=Path, default=Path("submission_ensemble.csv"))
    parser.add_argument("--budget-seconds", type=float, default=8 * 3600)
    parser.add_argument("--calibration-studies", type=int, default=10)
    _default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 4)
    parser.add_argument("--decode-workers", type=int, default=max(1, min(_default_workers - 1, 32)))
    parser.add_argument("--limit", type=int, default=None, help="truncate test set for a quick sanity pass")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    loaded = load_ensemble_recipe(args.recipe)
    members, recipe = loaded["members"], loaded["recipe"]
    global_members = list(members.keys())
    out_size = nearest_patch_multiple(384, _patch_size_for_backbone(next(iter(members.values()))["backbone"]))
    member_device = assign_member_devices(members, n_gpus, device)
    print(f"Recipe v{loaded['raw']['version']}, members={global_members}, out_size={out_size}, "
          f"device={device}, n_gpus={n_gpus}, decode_workers={args.decode_workers}", file=sys.stderr)
    print(f"  member device assignment: {member_device}", file=sys.stderr)

    member_models, member_rank_refs = {}, {}
    for name, spec in members.items():
        weights_dir = args.weights_root / name
        member_models[name] = load_member_models(weights_dir, spec["backbone"], member_device[name])
        npz_path = weights_dir / "rank_reference.npz"
        member_rank_refs[name] = dict(np.load(npz_path)) if npz_path.exists() else {}
        print(f"  {name} -> {member_device[name]}: {len(member_models[name])}/5 folds loaded", file=sys.stderr)

    test_df = pd.read_csv(args.raw / "test.csv")
    if args.limit:
        test_df = test_df.iloc[:args.limit].reset_index(drop=True)
    series_df = pd.read_csv(args.raw / "test_series.csv").rename(
        columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"}
    )
    series_root = args.raw / "test_series"
    print(f"{len(test_df)} test studies, {len(series_df)} series.", file=sys.stderr)

    SLICES_PER_SERIES = 24
    LADDER = [
        ("no-TTA + 3 members", global_members),
        ("no-TTA + 2 members", [m for m in global_members if "exp003" not in m]),
        ("smallest model only", [m for m in global_members if members[m]["backbone"].startswith("vit_small")]),
    ]

    def relative_cost(mems: list[str]) -> float:
        return sum(3.0 if members[m]["backbone"].startswith("vit_base") else 1.0 for m in mems)

    member_pool = ThreadPoolExecutor(max_workers=max(1, len(global_members)))

    def run_member(name: str, slots, tag_sides) -> np.ndarray:
        models = member_models[name]
        if not models:
            return prior_row()
        return run_study_inference(slots, tag_sides, models, member_device[name], SLICES_PER_SERIES, out_size, AUTOCAST_DTYPE)

    def run_ensemble(slots, tag_sides, mems: list[str]) -> np.ndarray:
        futures = {m: member_pool.submit(run_member, m, slots, tag_sides) for m in mems}
        member_probs = {m: f.result() for m, f in futures.items()}
        active_recipe = {label: ([m for m in orig if m in mems], method) for label, (orig, method) in recipe.items()}
        return ensemble_blend(member_probs, active_recipe, member_rank_refs)

    def study_task(study_uid: str) -> tuple:
        study_series = series_df[series_df["study_uid"] == study_uid].to_dict("records")
        return (study_uid, study_series, str(series_root), SLICES_PER_SERIES, out_size)

    guard = WallClockGuard(args.budget_seconds)
    n_calib = min(args.calibration_studies, len(test_df))
    calib_tasks = [study_task(u) for u in test_df["StudyInstanceUID"].iloc[:n_calib]]
    t0 = time.time()
    calib_rows = []
    for study_uid, slots, tag_sides, err in prefetch_decode(calib_tasks, decode_one_for_pool, args.decode_workers):
        if slots and err is None:
            probs = run_ensemble(slots, tag_sides, LADDER[0][1])
            calib_rows.append({"StudyInstanceUID": study_uid, **{c: float(p) for c, p in zip(TARGET_COLUMNS, probs)}})
    calib_elapsed = time.time() - t0
    per_study_0 = calib_elapsed / max(n_calib, 1)
    cost0 = relative_cost(LADDER[0][1])
    print(f"Calibration: {n_calib} studies, {calib_elapsed:.1f}s ({per_study_0:.2f}s/study, decode-overlapped)",
          file=sys.stderr)

    chosen_idx = len(LADDER) - 1
    for idx, (label, mems) in enumerate(LADDER):
        projected = per_study_0 * (relative_cost(mems) / cost0) * len(test_df)
        fits = projected <= guard.budget * 0.85
        print(f"  rung {idx} ({label}): projected {projected / 60:.1f} min -> {'FITS' if fits else 'too slow'}",
              file=sys.stderr)
        if fits:
            chosen_idx = idx
            break
    rung_label, active_members = LADDER[chosen_idx]
    print(f"Chosen rung: {chosen_idx} ({rung_label})", file=sys.stderr)

    rows = list(calib_rows) if chosen_idx == 0 else []
    done = {r["StudyInstanceUID"] for r in rows}
    remaining_uids = [u for u in test_df["StudyInstanceUID"] if u not in done]
    remaining_tasks = [study_task(u) for u in remaining_uids]

    n_prior_fallback = n_dual_flip = 0
    processed = len(rows)
    for study_uid, slots, tag_sides, err in prefetch_decode(remaining_tasks, decode_one_for_pool, args.decode_workers):
        if guard.over(margin=0.05):
            break
        row = {"StudyInstanceUID": study_uid}
        try:
            if err is not None or not slots:
                raise RuntimeError(err or "no series decoded successfully")
            if all(s not in ("R", "L") for s in tag_sides):
                n_dual_flip += 1
            probs = run_ensemble(slots, tag_sides, active_members)
            row.update({c: float(p) for c, p in zip(TARGET_COLUMNS, probs)})
        except Exception as e:  # noqa: BLE001
            n_prior_fallback += 1
            row.update({c: float(p) for c, p in zip(TARGET_COLUMNS, prior_row())})
        rows.append(row)
        processed += 1
        if processed % 100 == 0 or processed == len(test_df):
            print(f"  {processed}/{len(test_df)} studies, {guard.elapsed():.0f}s elapsed "
                  f"({guard.fraction_used():.1%} of budget)", file=sys.stderr)

    if processed < len(test_df):
        done_now = {r["StudyInstanceUID"] for r in rows}
        skipped = [u for u in remaining_uids if u not in done_now]
        print(f"  [BUDGET] {guard.elapsed():.0f}s used, {processed}/{len(test_df)} done — priors for the rest.",
              file=sys.stderr)
        for uid in skipped:
            rows.append({"StudyInstanceUID": uid, **{c: float(p) for c, p in zip(TARGET_COLUMNS, prior_row())}})
            n_prior_fallback += 1

    sub_df = pd.DataFrame(rows)[["StudyInstanceUID"] + TARGET_COLUMNS]
    sub_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(sub_df)} rows, rung={chosen_idx} ({rung_label}), "
          f"{n_prior_fallback} prior fallbacks, {n_dual_flip} dual-flip, {guard.elapsed():.0f}s total.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
