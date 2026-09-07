"""Repack an existing full checkpoint.pt (model+optimizer+scheduler+ema) into a lean best.pt
(model+ema only) for folds trained before scripts/train.py started writing best.pt itself
(exp002's run predates that change). No retraining, no eval — same EMA weights, just strips the
resume-only state before these get uploaded as a Kaggle Dataset (CLAUDE.md §9).

Usage: python scripts/repack_best.py --exp exp002_dinov2s_slot_v2labels
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rsna_root = Path(os.environ.get("RSNA_ROOT", "."))
    exp_dir = (args.out or rsna_root / "artifacts") / args.exp

    for fold_dir in sorted(exp_dir.glob("fold*")):
        ckpt_path = fold_dir / "checkpoint.pt"
        best_path = fold_dir / "best.pt"
        if not ckpt_path.exists() or best_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        summary_path = fold_dir / "summary.csv"
        val_macro = gold_macro = float("nan")
        if summary_path.exists():
            row = pd.read_csv(summary_path).iloc[0]
            val_macro, gold_macro = row["val_macro_auc"], row["gold58_macro_auc"]
        torch.save(
            {"model": ckpt["model"], "ema": ckpt["ema"], "epoch": ckpt["epoch"],
             "val_macro_auc": val_macro, "gold58_macro_auc": gold_macro},
            best_path,
        )
        before = ckpt_path.stat().st_size / 1e6
        after = best_path.stat().st_size / 1e6
        print(f"{fold_dir.name}: {before:.0f}MB -> {after:.0f}MB ({best_path})")


if __name__ == "__main__":
    main()
