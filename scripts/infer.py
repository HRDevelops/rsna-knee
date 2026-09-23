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

"""CLI wrapper around src/rsna_knee/infer_core.py — runs the exact inference logic the Kaggle
notebook uses, but against local/HPC paths with a GPU available. Two uses:
  1. Rehearsal: point --raw at data/raw (has the tiny 3-study public sample test_series/) to
     validate the full DICOM->submission.csv path end-to-end before ever touching Kaggle.
  2. A synthetic load test: --repeat N duplicates the sample studies under fake UIDs to approximate
     the real ~1300-study test set's wall-clock behavior (CLAUDE.md §9).

Usage:
    python scripts/infer.py --raw $RSNA_ROOT/data/raw \
        --weights-dir $RSNA_ROOT/artifacts/exp002_dinov2s_slot_v2labels \
        --backbone vit_small_patch14_dinov2.lvd142m \
        --out submission.csv --budget-seconds 1800
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.dataset import TARGET_COLUMNS, _patch_size_for_backbone  # noqa: E402
from rsna_knee.infer_core import (  # noqa: E402
    WallClockGuard, decode_study_slots, nearest_patch_multiple, prior_row, run_study_inference,
)
from rsna_knee.model import SlotModel  # noqa: E402


def load_models(weights_dir: Path, backbone: str, device: str) -> list[torch.nn.Module]:
    models = []
    for fold_dir in sorted(weights_dir.glob("fold*")):
        ckpt_path = fold_dir / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = fold_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = SlotModel(backbone, num_targets=len(TARGET_COLUMNS), pretrained=False).to(device)
        state = ckpt.get("ema", ckpt["model"])
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        print(f"Loaded {fold_dir.name} from {ckpt_path.name}", file=sys.stderr)
    if not models:
        raise RuntimeError(f"No checkpoints found under {weights_dir}")
    return models


def build_synthetic_repeat(raw_dir: Path, n: int, tmp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Duplicate the tiny sample test set n times under synthetic study UIDs (symlink series dirs,
    no data copy) to approximate the real ~1300-study test set for a wall-clock load test."""
    test_df = pd.read_csv(raw_dir / "test.csv")
    series_df = pd.read_csv(raw_dir / "test_series.csv")
    series_root = raw_dir / "test_series"
    tmp_series_root = tmp_dir / "test_series"
    tmp_series_root.mkdir(parents=True, exist_ok=True)

    new_test_rows, new_series_rows = [], []
    for rep in range(n):
        for _, row in test_df.iterrows():
            new_uid = f"{row['StudyInstanceUID']}_rep{rep}"
            new_test_rows.append({"StudyInstanceUID": new_uid})
            study_series = series_df[series_df["StudyInstanceUID"] == row["StudyInstanceUID"]]
            for _, srow in study_series.iterrows():
                new_series_rows.append({**srow.to_dict(), "StudyInstanceUID": new_uid})
            dst = tmp_series_root / new_uid
            if not dst.exists():
                dst.symlink_to((series_root / row["StudyInstanceUID"]).resolve(), target_is_directory=True)
    return pd.DataFrame(new_test_rows), pd.DataFrame(new_series_rows), tmp_series_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--slices-per-series", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--budget-seconds", type=float, default=1800)
    parser.add_argument("--repeat", type=int, default=0, help="synthetic load-test multiplier, 0=off")
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/rsna_infer_repeat"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    guard = WallClockGuard(args.budget_seconds)
    T = args.slices_per_series
    out_size = nearest_patch_multiple(args.image_size, _patch_size_for_backbone(args.backbone))

    if args.repeat > 0:
        test_df, series_df, series_root = build_synthetic_repeat(args.raw, args.repeat, args.tmp_dir)
        print(f"Synthetic load test: {len(test_df)} studies ({args.repeat}x the sample set)", file=sys.stderr)
    else:
        test_df = pd.read_csv(args.raw / "test.csv")
        series_df = pd.read_csv(args.raw / "test_series.csv")
        series_root = args.raw / "test_series"

    series_df = series_df.rename(columns={"StudyInstanceUID": "study_uid", "SeriesInstanceUID": "series_uid"})
    models = load_models(args.weights_dir, args.backbone, device)
    print(f"Ensemble: {len(models)} folds, T={T}, out_size={out_size}, device={device}", file=sys.stderr)

    rows = []
    n_prior_fallback = n_dual_flip = 0
    for i, study_uid in enumerate(test_df["StudyInstanceUID"]):
        row = {"StudyInstanceUID": study_uid}
        try:
            if guard.over(margin=0.1):
                print(f"  [BUDGET] {guard.elapsed():.0f}s used of {args.budget_seconds:.0f}s "
                      f"budget at study {i}/{len(test_df)} — falling back to priors for the rest.",
                      file=sys.stderr)
                raise TimeoutError("wall-clock budget exhausted")

            study_series = series_df[series_df["study_uid"] == study_uid].to_dict("records")
            series_dir = series_root / study_uid
            slots, tag_sides = decode_study_slots(study_uid, study_series, series_dir, T, out_size)
            if not slots:
                raise RuntimeError("no series decoded successfully for this study")
            if all(s not in ("R", "L") for s in tag_sides):
                n_dual_flip += 1

            probs = run_study_inference(slots, tag_sides, models, device, T, out_size)
            row.update({c: float(p) for c, p in zip(TARGET_COLUMNS, probs)})
        except Exception as e:  # noqa: BLE001 - CLAUDE.md §9: never crash, fall back to priors
            print(f"  [WARN] {study_uid}: {type(e).__name__}: {e} — using label priors", file=sys.stderr)
            n_prior_fallback += 1
            row.update({c: float(p) for c, p in zip(TARGET_COLUMNS, prior_row())})
        rows.append(row)
        if (i + 1) % 50 == 0 or (i + 1) == len(test_df):
            print(f"  {i + 1}/{len(test_df)} studies, {guard.elapsed():.0f}s elapsed", file=sys.stderr)

    sub_df = pd.DataFrame(rows)[["StudyInstanceUID"] + TARGET_COLUMNS]
    sub_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(sub_df)} rows, {n_prior_fallback} full-prior fallbacks, "
          f"{n_dual_flip} dual-flip (unresolved side) studies, {guard.elapsed():.0f}s total",
          file=sys.stderr)

    if args.repeat > 0:
        shutil.rmtree(args.tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
