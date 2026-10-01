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

"""Offline blend gate, exp021 + the scored slot ensemble, exactly as pre-registered in docs/EXPERIMENTS.md
(2026-09-26, commit 185eab7, before any number was computed).

CoAt arm = exp021 10-checkpoint mean. Slot arm = {exp006, exp003, exp005} through ensemble_recipe_v1.yaml as the
0.893 notebook computes it (empirical_percentile for rank-mean labels). Per-label percentiles against each arm's
OOF population, one global slot weight alpha in {0, .1, .2, .3, .4}, nested CV over 5 GroupKFold outer folds by
report_cluster_id, alpha rule = smallest alpha within 0.0005 of the best inner gain.

`--slot-arm exp005` (re-gate pre-registered 2026-10-01, commit a05ad9f) swaps ONLY the slot arm for exp005's 5-fold
mean (the model the scored ensemble notebook falls back to on T4); criteria, grid and alpha rule are unchanged.

    python scripts/blend_v5_gate.py --out artifacts/blend_v5_gate.md
    python scripts/blend_v5_gate.py --slot-arm exp005 --out artifacts/blend_v5_gate_exp005.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import TARGETS, load_full_oof, load_gold_fold_ensemble, load_gold_truth, load_truth  # noqa: E402

S42, S43 = "exp018_coatnet2_wmil_v3labels", "exp021_coatnet2_wmil_seed43"
MEMBERS = ["exp006_dinov2b_slot_v3labels", "exp003_dinov2b_slot_v2labels", "exp005_dinov2s_slot_v3labels"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]
TOL, CORR_BAR, GAIN_BAR, GOLD_BAR, IQR_BAR = 0.0005, 0.92, 0.0020, 0.9091, 0.20


def pct(x: np.ndarray, ref: np.ndarray) -> np.ndarray:  # [12, n] against [12, N] -> infer_core.empirical_percentile
    return np.stack([np.searchsorted(np.sort(ref[i]), x[i]) / ref.shape[1] for i in range(12)])


def macro(s, y, idx=None) -> float:
    idx = np.arange(y.shape[1]) if idx is None else idx
    return float(np.mean([roc_auc_score(y[i, idx], s[i, idx]) for i in range(12)]))


def slot_arm(member: dict[str, np.ndarray], ref: dict[str, np.ndarray], recipe: dict) -> np.ndarray:
    """member/ref: {name: [12, n]} / {name: [12, N_oof]}; the recipe's prob-mean or rank-mean per label."""
    out = np.zeros_like(member[MEMBERS[0]])
    for i, t in enumerate(TARGETS):
        spec = recipe[t]
        if spec["method"] == "rank-mean":
            vals = [np.searchsorted(np.sort(ref[m][i]), member[m][i]) / ref[m].shape[1] for m in spec["members"]]
        else:
            vals = [member[m][i] for m in spec["members"]]
        out[i] = np.mean(vals, axis=0)
    return out


def pick_alpha(gains: dict[float, float]) -> float:
    best = max(gains.values())
    return min(a for a in ALPHAS if gains[a] >= best - TOL)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--slot-arm", choices=["recipe", "exp005"], default="recipe")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts"
    recipe_path = root / "code/src/rsna_knee/ensemble_recipe_v1.yaml"
    if not recipe_path.exists():
        recipe_path = Path(__file__).resolve().parent.parent / "src/rsna_knee/ensemble_recipe_v1.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())["recipe"]
    names = MEMBERS + [S42, S43]
    oof = {n: load_full_oof(art / n) for n in names}
    gold = {n: load_gold_fold_ensemble(art / n) for n in names}
    miss = [n for n in names if oof[n] is None or gold[n] is None]
    if miss:
        sys.exit(f"incomplete (need 5 folds each): {miss}")
    truth = load_truth(art / "labels/labels_v2_final.parquet")
    gtruth = load_gold_truth(root / "data/raw/train.csv")
    clusters = pd.read_parquet(art / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]

    uids = sorted(set.intersection(*[set(oof[n].index) for n in names]) & set(truth) & set(clusters.index))
    y = np.stack([[truth[u][i] for u in uids] for i in range(12)]).astype(int)
    P = {n: oof[n].loc[uids, TARGETS].to_numpy().T for n in names}                      # [12, N]
    coat = (P[S42] + P[S43]) / 2
    exp005 = "exp005_dinov2s_slot_v3labels"
    slot = slot_arm(P, P, recipe) if a.slot_arm == "recipe" else P[exp005]
    gu = sorted(set.intersection(*[set(gold[n].index) for n in names]) & set(gtruth))
    gy = np.stack([[gtruth[u][i] for u in gu] for i in range(12)]).astype(int)
    G = {n: gold[n].loc[gu, TARGETS].to_numpy().T for n in names}
    gcoat = (G[S42] + G[S43]) / 2                    # both seeds' 5-fold means = the 10-checkpoint mean
    gslot = slot_arm(G, P, recipe) if a.slot_arm == "recipe" else G[exp005]   # exp005: 5-fold mean

    pc, ps = pct(coat, coat), pct(slot, slot)
    gpc, gps = pct(gcoat, coat), pct(gslot, slot)
    blend = {al: (1 - al) * pc + al * ps for al in ALPHAS}
    gblend = {al: (1 - al) * gpc + al * gps for al in ALPHAS}
    base = macro(coat, y)
    gbase = macro(gcoat, gy)

    cid = clusters.loc[uids].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(uids)), groups=cid))
    cross = np.zeros_like(pc)
    fold_rows, fold_alphas = [], []
    for k, (inner, outer) in enumerate(folds):
        b0 = macro(blend[0.0], y, inner)
        gains = {al: macro(blend[al], y, inner) - b0 for al in ALPHAS}
        ak = pick_alpha(gains)
        fold_alphas.append(ak)
        cross[:, outer] = blend[ak][:, outer]
        og = macro(blend[ak], y, outer) - macro(blend[0.0], y, outer)
        fold_rows.append(f"| {k} | {len(outer)} | " + " | ".join(f"{gains[al]:+.4f}" for al in ALPHAS)
                         + f" | **{ak:.1f}** | {og:+.4f} |")

    groups = {c: np.where(cid == c)[0] for c in np.unique(cid)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[c] for c in rng.choice(keys, len(keys), replace=True)]) for _ in range(a.reps)]

    corr = [float(spearmanr(coat[i], slot[i]).statistic) for i in range(12)]
    gain = macro(cross, y) - base
    bd = np.array([macro(cross, y, ix) - macro(coat, y, ix) for ix in boots])
    lo, hi = np.percentile(bd, [2.5, 97.5])
    frac_pos = float((bd > 0).mean())
    full_gains = {al: macro(blend[al], y) - base for al in ALPHAS}
    a_dep = pick_alpha(full_gains)
    g_dep = macro(gblend[a_dep], gy)
    iqr = float(np.percentile(fold_alphas, 75) - np.percentile(fold_alphas, 25))
    crit = [
        ("1 mean per-label Spearman < 0.92", float(np.mean(corr)) < CORR_BAR, f"{np.mean(corr):.4f}"),
        ("2 cross-fitted gain >= +0.0020", gain >= GAIN_BAR, f"{gain:+.4f}"),
        ("3 bootstrap CI lo > 0 and >= 95% reps > 0", lo > 0 and frac_pos >= 0.95, f"CI [{lo:+.4f}, {hi:+.4f}], {frac_pos:.1%} > 0"),
        ("4 gold-58 at deploy alpha >= 0.9091", g_dep >= GOLD_BAR, f"{g_dep:.4f} (alpha {a_dep:.1f})"),
        ("5 outer alpha IQR <= 0.20", iqr <= IQR_BAR, f"{iqr:.2f} (alphas {fold_alphas})"),
    ]

    title = ("exp021 + slot ensemble (pre-registered 2026-09-26, commit 185eab7)" if a.slot_arm == "recipe"
             else "exp021 + exp005 (re-gate pre-registered 2026-10-01, commit a05ad9f)")
    L = [f"# Offline blend gate: {title}", "",
         f"OOF: {len(uids)} non-gold studies vs `labels_v2_final`, {len(keys)} report clusters, {a.reps} paired "
         f"cluster-bootstrap reps. Gold-58 n={len(gu)}. Slot arm: {'recipe ' + recipe_path.name if a.slot_arm == 'recipe' else 'exp005 5-fold mean'}.", "",
         f"exp021 (CoAt arm): OOF {base:.4f}, gold-58 {gbase:.4f} | slot arm: OOF {macro(slot, y):.4f}, "
         f"gold-58 {macro(gslot, gy):.4f}", "",
         "## Criterion 1: per-label Spearman, exp021 vs slot arm (OOF)", "",
         "| " + " | ".join(t[:10] for t in TARGETS) + " | **mean** |", "|" + "---|" * 13,
         "| " + " | ".join(f"{c:.3f}" for c in corr) + f" | **{np.mean(corr):.3f}** |", "",
         "## Nested CV: inner gains per alpha (vs alpha 0, inner data only) and chosen alpha_k", "",
         "| outer fold | n outer | " + " | ".join(f"α={al:.1f}" for al in ALPHAS) + " | α_k | outer gain at α_k |",
         "|" + "---|" * (len(ALPHAS) + 4), *fold_rows, "",
         "## Alpha curve on the full OOF (information only)", "",
         "| α (slot) | OOF macro | Δ vs exp021 | 95% CI (paired) | % reps > 0 | gold-58 | Δ gold-58 |", "|---|---|---|---|---|---|---|"]
    for al in ALPHAS:
        b = np.array([macro(blend[al], y, ix) - macro(coat, y, ix) for ix in boots])
        l2, h2 = np.percentile(b, [2.5, 97.5])
        gs = macro(gblend[al], gy)
        L.append(f"| {al:.1f} | {macro(blend[al], y):.4f} | {full_gains[al]:+.4f} | [{l2:+.4f}, {h2:+.4f}] | "
                 f"{(b > 0).mean():.1%} | {gs:.4f} | {gs - gbase:+.4f} |")
    L += ["", "## Gate", "", "| criterion | result | value |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if ok else 'FAIL'} | {v} |" for n, ok, v in crit]
    passed = all(ok for _, ok, _ in crit)
    L += ["", f"**GATE: {'PASS -> build submit-blend-v5 at alpha ' + format(a_dep, '.1f') if passed else 'FAIL -> stop, no rescue'}**"]
    rep = "\n".join(L) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
