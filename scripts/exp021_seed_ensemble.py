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

"""exp021 verdict: exp018 (seed 42) vs exp021 (seed 43, otherwise identical) vs their 10-checkpoint
prob-mean -- the candidate for the one reserved Kaggle probe.

What each row means:
- OOF (vs labels_v2_final, the cross-experiment comparable metric): every study is predicted only by the
  fold models that never saw it, so the seed ensemble's OOF is a 2-model mean per study (fold k of each
  seed). Scored with the same paired cluster bootstrap over report clusters as exp016/exp018.
- Gold-58 (fold-ensembled): each seed's 5 fold models all predict all 58 gold studies, so the ensemble row
  is the mean of 10 checkpoints -- exactly what the Kaggle notebook computes at test time. n = 58, SE of
  a macro-AUC ~0.03: advisory only, and shown with its own bootstrap CI so its width is visible.

Also reported, because it is the reason exp021 exists: how far two runs of the IDENTICAL recipe land
apart (seed 42 vs seed 43), per fold and in total. That is the noise floor any single-run comparison in
this project has been judged against without being measured.

    python scripts/exp021_seed_ensemble.py --out artifacts/exp021_seed_ensemble.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import TARGETS, load_full_oof, load_gold_fold_ensemble, load_gold_truth, load_truth  # noqa: E402

A, B = "exp018_coatnet2_wmil_v3labels", "exp021_coatnet2_wmil_seed43"


def macro(s: np.ndarray, y: np.ndarray, idx=None) -> float:
    idx = np.arange(y.shape[1]) if idx is None else idx
    vals = [roc_auc_score(y[i, idx], s[i, idx]) for i in range(12) if len(set(y[i, idx])) == 2]
    return float(np.mean(vals))


def per_label(s: np.ndarray, y: np.ndarray) -> list[float]:
    return [roc_auc_score(y[i], s[i]) if len(set(y[i])) == 2 else float("nan") for i in range(12)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts"

    oof = {n: load_full_oof(art / n) for n in (A, B)}
    gold = {n: load_gold_fold_ensemble(art / n) for n in (A, B)}
    missing = [n for n in (A, B) if oof[n] is None or gold[n] is None]
    if missing:
        sys.exit(f"incomplete (need all 5 folds): {missing}")

    truth = load_truth(art / "labels/labels_v2_final.parquet")
    gold_truth = load_gold_truth(root / "data/raw/train.csv")
    clusters = pd.read_parquet(art / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]

    # ---- OOF: identical study set for all three rows
    uids = sorted(set(oof[A].index) & set(oof[B].index) & set(truth) & set(clusters.index))
    y = np.stack([[truth[u][i] for u in uids] for i in range(12)]).astype(int)
    P = {n: oof[n].loc[uids, TARGETS].to_numpy().T for n in (A, B)}
    P["ens"] = (P[A] + P[B]) / 2

    cid = clusters.loc[uids].to_numpy()
    groups = {c: np.where(cid == c)[0] for c in np.unique(cid)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[c] for c in rng.choice(keys, len(keys), replace=True)]) for _ in range(a.reps)]

    def paired(s1, s2):
        d = macro(s1, y) - macro(s2, y)
        bd = np.array([macro(s1, y, ix) - macro(s2, y, ix) for ix in boots])
        lo, hi = np.percentile(bd, [2.5, 97.5])
        return d, lo, hi, float((bd <= 0).mean())

    # ---- gold-58: 10-checkpoint mean for the ensemble row
    gu = sorted(set(gold[A].index) & set(gold[B].index) & set(gold_truth))
    gy = np.stack([[gold_truth[u][i] for u in gu] for i in range(12)]).astype(int)
    G = {n: gold[n].loc[gu, TARGETS].to_numpy().T for n in (A, B)}
    G["ens"] = (G[A] + G[B]) / 2
    grng = np.random.default_rng(1)
    gboots = [grng.integers(0, len(gu), len(gu)) for _ in range(a.reps)]

    def gold_ci(s1, s2=None):
        vals = []
        for ix in gboots:
            v1 = macro(s1, gy, ix)
            vals.append(v1 - macro(s2, gy, ix) if s2 is not None else v1)
        return np.percentile(vals, [2.5, 97.5])

    # ---- seed-to-seed noise floor, per fold (fold-val and gold-58 exactly as the gate reports them)
    fold_rows = []
    for k in range(5):
        sa = pd.read_csv(art / A / f"fold{k}/summary.csv").iloc[0]
        sb = pd.read_csv(art / B / f"fold{k}/summary.csv").iloc[0]
        fold_rows.append((k, sa["val_macro_auc"], sb["val_macro_auc"], sa["gold58_macro_auc"], sb["gold58_macro_auc"]))

    L = [f"# exp021 verdict: seed ensemble (exp018 seed 42 + exp021 seed 43, 10 checkpoints)", "",
         f"OOF: {len(uids)} non-gold studies vs `labels_v2_final`, {len(keys)} report clusters, {a.reps} paired "
         f"cluster-bootstrap reps. Gold-58: n={len(gu)}, {a.reps} study-level bootstrap reps.", "",
         "## Headline", "",
         "| model | checkpoints | OOF macro | Δ OOF vs exp018 | 95% CI (paired) | P(Δ≤0) | gold-58 macro | gold-58 95% CI | Δ gold-58 vs exp018 (95% CI) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, key, nck in (("exp018 (seed 42)", A, 5), ("exp021 (seed 43)", B, 5), ("**10-ckpt prob-mean**", "ens", 10)):
        om, gm = macro(P[key], y), macro(G[key], gy)
        glo, ghi = gold_ci(G[key])
        if key == A:
            L.append(f"| {name} | {nck} | {om:.4f} | — | — | — | {gm:.4f} | [{glo:.4f}, {ghi:.4f}] | — |")
        else:
            d, lo, hi, p = paired(P[key], P[A])
            gd = gm - macro(G[A], gy)
            gdl, gdh = gold_ci(G[key], G[A])
            L.append(f"| {name} | {nck} | {om:.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} | {gm:.4f} | "
                     f"[{glo:.4f}, {ghi:.4f}] | {gd:+.4f} [{gdl:+.4f}, {gdh:+.4f}] |")

    L += ["", "## Seed-to-seed noise floor (identical recipe, seed 42 vs 43)", "",
          "| fold | fold-val s42 | fold-val s43 | Δ | gold-58 s42 | gold-58 s43 | Δ |", "|---|---|---|---|---|---|---|"]
    dv, dg = [], []
    for k, va, vb, ga, gb in fold_rows:
        dv.append(vb - va); dg.append(gb - ga)
        L.append(f"| {k} | {va:.4f} | {vb:.4f} | {vb - va:+.4f} | {ga:.4f} | {gb:.4f} | {gb - ga:+.4f} |")
    L += ["", f"Mean |Δ| across folds: fold-val {np.mean(np.abs(dv)):.4f}, gold-58 {np.mean(np.abs(dg)):.4f}; "
              f"max |Δ|: fold-val {np.max(np.abs(dv)):.4f}, gold-58 {np.max(np.abs(dg)):.4f}. "
              f"Any single-fold, single-seed comparison smaller than these is inside run-to-run noise.", ""]

    L += ["## Per label (OOF, and gold-58 10-ckpt vs exp018)", "",
          "| label | OOF s42 | OOF s43 | OOF ens | seed corr (OOF) | gold s42 | gold ens |", "|---|---|---|---|---|---|---|"]
    oa, ob, oe = per_label(P[A], y), per_label(P[B], y), per_label(P["ens"], y)
    ga_, ge_ = per_label(G[A], gy), per_label(G["ens"], gy)
    for i, t in enumerate(TARGETS):
        r = float(np.corrcoef(P[A][i], P[B][i])[0, 1])
        L.append(f"| {t} | {oa[i]:.4f} | {ob[i]:.4f} | {oe[i]:.4f} | {r:.3f} | {ga_[i]:.4f} | {ge_[i]:.4f} |")
    L += ["", "Seed correlation near 1 = little for averaging to cancel; the ensemble gain is bounded by how much "
              "the two seeds disagree.", ""]

    rep = "\n".join(L) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
