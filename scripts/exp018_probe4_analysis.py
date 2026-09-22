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

"""Probe #4 pre-submission analysis (2026-09-22, after exp018 scored 0.929 LB): per-label OOF
correlation between exp018 and the deployed 3-member ensemble (diversity check, same method as
scripts/eval_exp009_acceptance.py), and the rank-space blend at weights 0.6/0.7/0.75/0.8 (the
weight is a user decision informed by the 0.929-vs-0.893 LB gap, NOT picked by OOF argmax —
reported for reference only, see the caveat printed at the end).

    python scripts/exp018_probe4_analysis.py --weights 0.6 0.7 0.75 0.8 --out artifacts/exp018_probe4_analysis.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import TARGETS, load_full_oof, load_gold_fold_ensemble, load_gold_truth, load_truth  # noqa: E402

CANDIDATE = "exp018_coatnet2_wmil_v3labels"
MEMBERS = ["exp006_dinov2b_slot_v3labels", "exp003_dinov2b_slot_v2labels", "exp005_dinov2s_slot_v3labels"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.6, 0.7, 0.75, 0.8])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts"
    truth = load_truth(art / "labels/labels_v2_final.parquet")
    gold_truth = load_gold_truth(root / "data/raw/train.csv")
    clusters = pd.read_parquet(art / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]

    names = MEMBERS + [CANDIDATE]
    oof = {n: load_full_oof(art / n) for n in names}
    gold = {n: load_gold_fold_ensemble(art / n) for n in names}
    uids = [u for u in oof[CANDIDATE].index if u in truth and u in clusters.index
            and all(u in oof[n].index for n in MEMBERS)]
    y = np.stack([[truth[u][i] for u in uids] for i in range(12)]).astype(int)
    P = {n: oof[n].loc[uids, TARGETS].to_numpy().T for n in names}                       # [12, N]
    ens_prob = np.mean([P[m] for m in MEMBERS], axis=0)
    R = {n: np.stack([rankdata(P[n][i]) / len(uids) for i in range(12)]) for n in names}
    ens_rank = np.mean([R[m] for m in MEMBERS], axis=0)

    gu = sorted(set.intersection(*[set(gold[n].index) for n in names]) & set(gold_truth))
    gy = np.stack([[gold_truth[u][i] for u in gu] for i in range(12)]).astype(int)
    GP = {n: gold[n].loc[gu, TARGETS].to_numpy().T for n in names}
    ens_gold_prob = np.mean([GP[m] for m in MEMBERS], axis=0)
    GR = {n: np.stack([rankdata(GP[n][i]) / len(gu) for i in range(12)]) for n in names}
    ens_gold_rank = np.mean([GR[m] for m in MEMBERS], axis=0)

    def macro(s, yy, idx=None):
        idx = np.arange(yy.shape[1]) if idx is None else idx
        return float(np.mean([roc_auc_score(yy[i, idx], s[i, idx]) for i in range(12)]))

    # --- 1. per-label diversity: Pearson correlation, exp018 raw prob vs deployed-ensemble prob-mean, on OOF
    lines = ["# exp018 probe #4 pre-submission analysis (2026-09-22)", "",
             f"exp018 scored **0.929 public LB** (deployed 3-member ensemble: 0.893; +0.036). Everything below is "
             f"OOF/gold-58 (not LB) — advisory, per the caveat at the end.", "",
             "## 1. Per-label OOF correlation: exp018 vs the deployed ensemble (diversity check)", "",
             "Pearson correlation of raw probabilities on the same OOF studies (method: `scripts/eval_exp009_acceptance.py`, "
             "the same check used to reject exp009 as redundant at corr 0.9296). Lower = more genuinely different errors.", "",
             "| label | corr(exp018, ensemble) |", "|---|---|"]
    corrs = []
    for i, c in enumerate(TARGETS):
        r = float(np.corrcoef(P[CANDIDATE][i], ens_prob[i])[0, 1])
        corrs.append(r)
        lines.append(f"| {c} | {r:.4f} |")
    mean_corr = float(np.mean(corrs))
    lines += ["", f"**Mean per-label correlation: {mean_corr:.4f}** "
                  f"({'below 0.5 — propose a lower weight than 0.75, see the note below' if mean_corr < 0.5 else 'above 0.5 — the pre-declared 0.75 stands'})."]

    # --- 2. blend table at the requested weights, rank-space (empirical-CDF at deployment; population-rank OOF here)
    cid = clusters.loc[uids].to_numpy()
    groups = {c: np.where(cid == c)[0] for c in np.unique(cid)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[c] for c in rng.choice(keys, len(keys), replace=True)]) for _ in range(a.reps)]
    base_prob_macro, base_rank_macro = macro(ens_prob, y), macro(ens_rank, y)
    base_name = "prob-mean" if base_prob_macro >= base_rank_macro else "rank-mean"
    base = ens_prob if base_name == "prob-mean" else ens_rank
    base_gold = ens_gold_prob if base_name == "prob-mean" else ens_gold_rank

    lines += ["", "## 2. Blend at the requested weights (rank space; the same empirical-CDF mechanism already "
                  "deployed for the 3-member ensemble, `infer_core.ensemble_blend`)", "",
              f"Deployed-ensemble baseline used below: {base_name} {macro(base, y):.4f} OOF / {macro(base_gold, gy):.4f} gold-58. "
              f"exp018 solo: {macro(P[CANDIDATE], y):.4f} OOF / {macro(GP[CANDIDATE], gy):.4f} gold-58.", "",
              "| w(exp018) | OOF macro | Δ vs deployed ensemble | 95% CI (paired) | gold-58 macro (Δ) |",
              "|---|---|---|---|---|"]
    for w in sorted(set(a.weights) | {0.0, 1.0}):
        s = w * R[CANDIDATE] + (1 - w) * ens_rank
        gs = w * GR[CANDIDATE] + (1 - w) * ens_gold_rank
        d = macro(s, y) - macro(base, y)
        bd = np.array([macro(s, y, ix) - macro(base, y, ix) for ix in boots])
        lo, hi = np.percentile(bd, [2.5, 97.5])
        mark = " *(requested)*" if w in a.weights else " *(reference)*"
        lines.append(f"| {w:.2f}{mark} | {macro(s, y):.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{macro(gs, gy):.4f} ({macro(gs, gy) - macro(base_gold, gy):+.4f}) |")
    lines += ["", "**Caveat (why this table must NOT drive the weight choice):** exp018's own OOF (0.8816) already "
                  "undersold it by ~4 points relative to its LB score (0.929) — the same report-derived-label proxy "
                  "problem documented for every prior experiment in this project, just larger here because exp018's "
                  "image-only signal generalises to the real (image-reviewed) test labels better than a report-label "
                  "training/eval target can measure. There is no reason to expect the BLEND's OOF number to be any "
                  "less biased downward. OOF above is reported for completeness and for the diversity/correlation "
                  "read only — the actual weight decision rests on the LB gap (0.929 vs 0.893, a 3.6-point edge to "
                  "exp018) plus the correlation in §1, per the pre-registered instruction."]

    rep = "\n".join(lines) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
