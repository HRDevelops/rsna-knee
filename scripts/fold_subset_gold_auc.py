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

"""Fold-count -> AUC sensitivity, on gold-58 (the only set where every fold's model predicts every
study, so subsets of folds are directly comparable — the same reasoning and methodology used for
exp008's FOLD_COUNT_AUC_DELTA table, docs/EXPERIMENTS.md 2026-09-10/model_cards/exp008). For each
subset size k=1..5, averages macro-AUC over ALL C(5,k) equal-weight subsets (not just the best
one), so the reported number is an unbiased expectation over "which k folds a notebook happens to
load", not a best-case cherry-pick.

    python scripts/fold_subset_gold_auc.py --experiment exp018_coatnet2_wmil_v3labels
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import TARGETS, load_gold_truth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts" / a.experiment
    gold_truth = load_gold_truth(root / "data/raw/train.csv")

    folds = []
    for k in range(5):
        p = art / f"fold{k}/gold58_predictions.parquet"
        if not p.exists():
            sys.exit(f"missing {p}")
        folds.append(pd.read_parquet(p).set_index("study_uid")[TARGETS])
    uids = sorted(set(folds[0].index) & set(gold_truth))
    y = np.stack([[gold_truth[u][i] for u in uids] for i in range(12)]).astype(int)
    stacks = [f.loc[uids, TARGETS].to_numpy().T for f in folds]  # each [12, N]

    def macro(s: np.ndarray) -> float:
        return float(np.mean([roc_auc_score(y[i], s[i]) for i in range(12) if len(set(y[i])) > 1]))

    full = macro(np.mean(stacks, axis=0))
    lines = [f"# Fold-count -> gold-58 AUC sensitivity — `{a.experiment}` (all-subsets average, n={len(uids)} gold studies)",
             "", f"Full 5-fold gold-58 macro: **{full:.4f}**", "",
             "| k folds | mean macro over all C(5,k) subsets | Δ vs 5-fold | # subsets averaged |",
             "|---|---|---|---|"]
    deltas = {}
    for k in range(1, 6):
        vals = [macro(np.mean([stacks[i] for i in combo], axis=0)) for combo in itertools.combinations(range(5), k)]
        m = float(np.mean(vals))
        deltas[k] = m - full
        lines.append(f"| {k} | {m:.4f} | {m - full:+.4f} | {len(vals)} |")
    lines += ["", "Python dict for reuse (mirrors exp008's `FOLD_COUNT_AUC_DELTA`):",
              "```python", "FOLD_COUNT_AUC_DELTA = {" + ", ".join(f"{k}: {v:+.4f}" for k, v in deltas.items()) + "}",
              "```"]
    rep = "\n".join(lines) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
