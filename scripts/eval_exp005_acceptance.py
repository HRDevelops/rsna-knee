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

"""exp005 (noisy-student round 1) pre-registered acceptance test — docs/EXPERIMENTS.md 2026-09-08,
written BEFORE exp005 was launched. exp005 must beat exp002 on BOTH:
  1. OOF macro-AUC vs the ORIGINAL labels_v2_final report labels (exp005's own training-time OOF
     metric is against labels_v3_candidate instead — NOT this criterion — so exp005's OOF
     predictions are re-scored here against labels_v2_final specifically).
  2. gold-58 macro-AUC (already computed correctly in each fold's summary.csv regardless of which
     labels file trained the model, since gold truth always comes from data/raw/train.csv).
If it wins only on gold-58, that's a blend-overfitting signal (labels_v3_candidate's weights were
themselves LOO-fit on gold-58) — flag, don't accept.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def load_summaries(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(exp_dir.glob("fold*")):
        csv_path = fold_dir / "summary.csv"
        if csv_path.exists():
            rows.append(pd.read_csv(csv_path).iloc[0])
    return pd.DataFrame(rows)


def load_full_oof(exp_dir: Path) -> pd.DataFrame | None:
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/oof.parquet"))]
    return pd.concat(parts, ignore_index=True) if len(parts) == 5 else None


def macro_auc_vs_labels(oof_df: pd.DataFrame, labels_parquet: Path) -> tuple[float, dict]:
    """Re-scores OOF predictions against a SPECIFIC labels file's binarized (>=0.5) truth —
    independent of whichever labels file the model was actually trained/evaluated against
    during training."""
    truth_df = pd.read_parquet(labels_parquet)
    truth = {
        r["study_uid"]: np.array([1.0 if r[f"{c}__probability"] >= 0.5 else 0.0 for c in TARGETS])
        for r in truth_df.to_dict("records")
    }
    matched = oof_df[oof_df["study_uid"].isin(truth)].reset_index(drop=True)
    truth_matrix = np.stack([truth[uid] for uid in matched["study_uid"]])
    per_label = {}
    for i, col in enumerate(TARGETS):
        y_true = truth_matrix[:, i]
        y_score = matched[col].to_numpy()
        per_label[col] = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else float("nan")
    valid = [v for v in per_label.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan"), per_label


if __name__ == "__main__":
    rsna_root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent))
    root = rsna_root / "artifacts"
    labels_v2_final = root / "labels" / "labels_v2_final.parquet"

    exp002_dir = root / "exp002_dinov2s_slot_v2labels"
    exp005_dir = root / "exp005_dinov2s_slot_v3labels"

    exp002_summary = load_summaries(exp002_dir)
    exp005_summary = load_summaries(exp005_dir)

    if len(exp002_summary) < 5:
        print(f"exp002 incomplete ({len(exp002_summary)}/5 folds) — cannot run acceptance test.")
        raise SystemExit(1)
    if len(exp005_summary) < 5:
        print(f"exp005 incomplete ({len(exp005_summary)}/5 folds) — not ready for acceptance test yet.")
        raise SystemExit(0)

    exp002_oof = load_full_oof(exp002_dir)
    exp005_oof = load_full_oof(exp005_dir)

    # Criterion 1: OOF vs ORIGINAL labels_v2_final (exp002's native metric already IS this, since
    # it was trained+evaluated on labels_v2_final; exp005 needs re-scoring since it trained on v3).
    exp002_oof_v2final, exp002_per_label_v2final = macro_auc_vs_labels(exp002_oof, labels_v2_final)
    exp005_oof_v2final, exp005_per_label_v2final = macro_auc_vs_labels(exp005_oof, labels_v2_final)

    exp002_gold58 = exp002_summary["gold58_macro_auc"].mean()
    exp005_gold58 = exp005_summary["gold58_macro_auc"].mean()

    print("=== Criterion 1: OOF macro-AUC vs ORIGINAL labels_v2_final (independent target) ===")
    print(f"exp002: {exp002_oof_v2final:.4f}  (native metric, unchanged)")
    print(f"exp005: {exp005_oof_v2final:.4f}  (re-scored — exp005's native training metric was vs labels_v3_candidate instead)")
    win_criterion1 = exp005_oof_v2final > exp002_oof_v2final

    print("\n=== Criterion 2: gold-58 macro-AUC ===")
    print(f"exp002: {exp002_gold58:.4f}")
    print(f"exp005: {exp005_gold58:.4f}")
    win_criterion2 = exp005_gold58 > exp002_gold58

    print(f"\nCriterion 1 (OOF vs labels_v2_final): {'WIN' if win_criterion1 else 'LOSE'} "
          f"({exp005_oof_v2final - exp002_oof_v2final:+.4f})")
    print(f"Criterion 2 (gold-58 macro):          {'WIN' if win_criterion2 else 'LOSE'} "
          f"({exp005_gold58 - exp002_gold58:+.4f})")

    if win_criterion1 and win_criterion2:
        verdict = "ACCEPT — exp005 beats exp002 on both pre-registered criteria."
    elif win_criterion2 and not win_criterion1:
        verdict = ("FLAG — wins gold-58 ONLY. This is exactly the blend-overfitting failure mode "
                   "(labels_v3_candidate was itself LOO-calibrated on gold-58) — do NOT proceed "
                   "without further scrutiny.")
    else:
        verdict = "REJECT — exp005 does not beat exp002 on the pre-registered criteria."
    print(f"\nVERDICT: {verdict}")

    print("\n=== Per-label: OOF vs labels_v2_final ===")
    per_label_df = pd.DataFrame({
        "label": TARGETS,
        "exp002_OOF_v2final": [exp002_per_label_v2final[t] for t in TARGETS],
        "exp005_OOF_v2final": [exp005_per_label_v2final[t] for t in TARGETS],
    })
    per_label_df["delta"] = per_label_df["exp005_OOF_v2final"] - per_label_df["exp002_OOF_v2final"]
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 4)
    print(per_label_df.to_string(index=False))

    out_path = root / "exp005_acceptance_report.csv"
    per_label_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
