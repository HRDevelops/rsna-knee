"""Builds the distillation teacher target for exp008 (efficiency track) — blends the chosen
ensemble members ({exp006, exp003, exp005}, per-label prob-mean/rank-mean per
artifacts/ensemble_recipe.csv, 2026-09-09) over ALL 4,407 studies: OOF predictions for the ~4,349
non-gold studies (out-of-fold, unbiased), fold-ensembled predictions for the 58 gold studies
(never in any training fold, so averaging the 5 independently-trained folds' models is legitimate,
same pattern as build_labels_v3_candidate.py). Confidence weight per label = inter-model agreement
across the 3 ensemble members (tighter agreement -> higher weight), same spirit as the existing
label-blend scripts.

Usage:
    python scripts/build_ensemble_teacher.py \
        --members exp006_dinov2b_slot_v3labels exp003_dinov2b_slot_v2labels exp005_dinov2s_slot_v3labels \
        --recipe artifacts/ensemble_recipe.csv \
        --out artifacts/labels/ensemble_teacher_v1.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MIN_CONFIDENCE = 0.1


def load_full_oof(exp_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/oof.parquet"))]
    assert len(parts) == 5, f"expected 5 fold oof.parquet under {exp_dir}"
    return pd.concat(parts, ignore_index=True).set_index("study_uid")


def load_gold_fold_ensemble(exp_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/gold58_predictions.parquet"))]
    assert len(parts) == 5, f"expected 5 fold gold58_predictions.parquet under {exp_dir}"
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby("study_uid")[TARGETS].mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/labels/ensemble_teacher_v1.parquet"))
    args = parser.parse_args()

    rsna_root = Path(os.environ.get("RSNA_ROOT", "."))
    root = rsna_root / "artifacts"
    recipe_df = pd.read_csv(args.recipe).set_index("label")

    all_df = {}
    for name in args.members:
        exp_dir = root / name
        oof = load_full_oof(exp_dir)
        gold = load_gold_fold_ensemble(exp_dir)
        all_df[name] = pd.concat([oof[TARGETS], gold[TARGETS]])
    common_studies = None
    for df in all_df.values():
        common_studies = df.index if common_studies is None else common_studies.union(df.index)
    print(f"{len(common_studies)} studies covered ({len(args.members)} members)")

    result_rows = {"study_uid": common_studies}
    for target in TARGETS:
        method = recipe_df.loc[target, "method"]
        matrix = pd.DataFrame({name: all_df[name].loc[common_studies, target] for name in args.members},
                               index=common_studies)
        if method == "rank-mean":
            blended = matrix.rank(pct=True).mean(axis=1)
        else:
            blended = matrix.mean(axis=1)
        agreement = 1.0 - (matrix.max(axis=1) - matrix.min(axis=1))  # tighter spread -> higher agreement
        confidence = agreement.clip(lower=MIN_CONFIDENCE, upper=1.0)

        result_rows[f"{target}__probability"] = blended.to_numpy()
        result_rows[f"{target}__confidence_weight"] = confidence.to_numpy()
        result_rows[f"{target}__method"] = method

    out_df = pd.DataFrame(result_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {len(out_df)} studies")
    for target in TARGETS:
        print(f"  {target}: method={recipe_df.loc[target, 'method']} "
              f"mean_prob={out_df[f'{target}__probability'].mean():.3f} "
              f"mean_confidence={out_df[f'{target}__confidence_weight'].mean():.3f}")


if __name__ == "__main__":
    main()
