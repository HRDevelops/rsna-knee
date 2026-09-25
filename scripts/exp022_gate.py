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

"""exp022 gate, exactly as pre-registered in docs/EXPERIMENTS.md (2026-09-26, before any exp022 result):

  1. mean per-label OOF Pearson correlation, exp022 vs exp018, < 0.90 (per-label values reported);
  2. rank-space blend of exp021's 10 checkpoints (prob-mean of the two seeds, then ranked) with exp022
     (ranked) at the PRE-DECLARED weight w(exp022) = 0.50 improves OOF macro vs exp021 alone by >= +0.002;
  3. gold-58 of that same w = 0.50 blend >= exp021's gold-58 (point estimate).

Other weights are printed for information only; they do not decide the gate. OOF vs labels_v2_final with the
same paired cluster bootstrap over report clusters as exp016/exp018/exp021. Ranks are per label over the
evaluated population (OOF: the 4,349 non-gold studies; gold-58: the 58), i.e. the rank-mean the deployed
empirical-CDF mechanism reproduces at test time.

    python scripts/exp022_gate.py --out artifacts/exp022_gate.md
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

S42, S43, D = "exp018_coatnet2_wmil_v3labels", "exp021_coatnet2_wmil_seed43", "exp022_dinov2b_wmil"
W_GATE = 0.50
CORR_BAR, GAIN_BAR = 0.90, 0.002


def ranks(p: np.ndarray) -> np.ndarray:  # [12, N] -> per-label ranks in (0, 1]
    return np.stack([rankdata(p[i]) / p.shape[1] for i in range(p.shape[0])])


def macro(s, y, idx=None) -> float:
    idx = np.arange(y.shape[1]) if idx is None else idx
    return float(np.mean([roc_auc_score(y[i, idx], s[i, idx]) for i in range(12) if len(set(y[i, idx])) == 2]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts"
    oof = {n: load_full_oof(art / n) for n in (S42, S43, D)}
    gold = {n: load_gold_fold_ensemble(art / n) for n in (S42, S43, D)}
    miss = [n for n in (S42, S43, D) if oof[n] is None or gold[n] is None]
    if miss:
        sys.exit(f"incomplete (need 5 folds each): {miss}")
    truth = load_truth(art / "labels/labels_v2_final.parquet")
    gtruth = load_gold_truth(root / "data/raw/train.csv")
    clusters = pd.read_parquet(art / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]

    uids = sorted(set.intersection(*[set(oof[n].index) for n in oof]) & set(truth) & set(clusters.index))
    y = np.stack([[truth[u][i] for u in uids] for i in range(12)]).astype(int)
    P = {n: oof[n].loc[uids, TARGETS].to_numpy().T for n in oof}
    ens10 = (P[S42] + P[S43]) / 2                      # exp021's 10 checkpoints (each study: fold k of both seeds)
    R10, R22 = ranks(ens10), ranks(P[D])

    gu = sorted(set.intersection(*[set(gold[n].index) for n in gold]) & set(gtruth))
    gy = np.stack([[gtruth[u][i] for u in gu] for i in range(12)]).astype(int)
    G = {n: gold[n].loc[gu, TARGETS].to_numpy().T for n in gold}
    g10 = (G[S42] + G[S43]) / 2
    GR10, GR22 = ranks(g10), ranks(G[D])

    cid = clusters.loc[uids].to_numpy()
    groups = {c: np.where(cid == c)[0] for c in np.unique(cid)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[c] for c in rng.choice(keys, len(keys), replace=True)]) for _ in range(a.reps)]

    # ---- criterion 1: diversity vs exp018
    corr = [float(np.corrcoef(P[D][i], P[S42][i])[0, 1]) for i in range(12)]
    c1 = float(np.mean(corr)) < CORR_BAR

    base = macro(R10, y)   # rank transform is monotone per label, so this equals exp021's prob-mean OOF macro
    L = [f"# exp022 gate (pre-registered 2026-09-26)", "",
         f"OOF: {len(uids)} non-gold studies vs `labels_v2_final`, {len(keys)} report clusters, {a.reps} paired "
         f"cluster-bootstrap reps. Gold-58 n={len(gu)}.", "",
         f"exp022 solo: OOF {macro(P[D], y):.4f}, gold-58 {macro(G[D], gy):.4f} | exp018 solo: OOF {macro(P[S42], y):.4f}, "
         f"gold-58 {macro(G[S42], gy):.4f} | exp021 (10 ckpt): OOF {base:.4f}, gold-58 {macro(g10, gy):.4f}", "",
         "## Criterion 1 — diversity (mean per-label OOF correlation with exp018 < 0.90)", "",
         "| " + " | ".join(t[:10] for t in TARGETS) + " | **mean** |", "|" + "---|" * 13,
         "| " + " | ".join(f"{c:.3f}" for c in corr) + f" | **{np.mean(corr):.3f}** |", "",
         f"-> {'PASS' if c1 else 'FAIL'} ({np.mean(corr):.3f} {'<' if c1 else '>='} {CORR_BAR})", "",
         "## Criteria 2-3 — rank blend with exp021 (gate row: w(exp022) = 0.50; others informational)", "",
         "| w(exp022) | OOF macro | Δ vs exp021 | 95% CI (paired) | P(Δ≤0) | gold-58 | Δ gold-58 |", "|---|---|---|---|---|---|---|"]
    gate_row = None
    for w in (0.25, 0.35, 0.50, 0.65):
        s = (1 - w) * R10 + w * R22
        d = macro(s, y) - base
        bd = np.array([macro(s, y, ix) - macro(R10, y, ix) for ix in boots])
        lo, hi = np.percentile(bd, [2.5, 97.5])
        gs = macro((1 - w) * GR10 + w * GR22, gy)
        gd = gs - macro(g10, gy)
        tag = " **(gate)**" if w == W_GATE else ""
        L.append(f"| {w:.2f}{tag} | {macro(s, y):.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {(bd <= 0).mean():.3f} | "
                 f"{gs:.4f} | {gd:+.4f} |")
        if w == W_GATE:
            gate_row = (d, gd)
    c2, c3 = gate_row[0] >= GAIN_BAR, gate_row[1] >= 0
    L += ["", f"Criterion 2 (Δ OOF >= +{GAIN_BAR} at w=0.50): {'PASS' if c2 else 'FAIL'} ({gate_row[0]:+.4f})",
          f"Criterion 3 (gold-58 Δ >= 0 at w=0.50): {'PASS' if c3 else 'FAIL'} ({gate_row[1]:+.4f})", "",
          f"**GATE: {'PASS -> stage + placeholder check + probe' if (c1 and c2 and c3) else 'FAIL -> stop, no probe'}**"]
    rep = "\n".join(L) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
