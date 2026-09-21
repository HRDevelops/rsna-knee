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

"""exp018 blend test (after all 5 folds): candidate solo vs exp006, and candidate + the DEPLOYED 3-member
ensemble {exp006, exp003, exp005} at several weights (rank-mean and prob-mean), with a paired cluster bootstrap
over report clusters (same machinery as exp016_verdict.py; macro SD ~0.0041) against the deployed ensemble.
The weight is chosen in-sample from a coarse grid, so the best-weight row is optimistic; the fixed 0.5 row is the
honest one. Gold-58 (fold-ensembled, n=58) is reported as a tripwire.

    python scripts/exp018_blend.py [--candidate exp018_coatnet2_wmil_v3labels] --out artifacts/exp018_blend.md
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

MEMBERS = ["exp006_dinov2b_slot_v3labels", "exp003_dinov2b_slot_v2labels", "exp005_dinov2s_slot_v3labels"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="exp018_coatnet2_wmil_v3labels")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    art = root / "artifacts"
    truth = load_truth(art / "labels/labels_v2_final.parquet")
    gold_truth = load_gold_truth(root / "data/raw/train.csv")
    clusters = pd.read_parquet(art / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]

    names = MEMBERS + [a.candidate]
    oof = {n: load_full_oof(art / n) for n in names}
    gold = {n: load_gold_fold_ensemble(art / n) for n in names}
    missing = [n for n in names if oof[n] is None]
    if missing:
        sys.exit(f"incomplete (need 5 folds): {missing}")
    uids = [u for u in oof[a.candidate].index if u in truth and u in clusters.index and all(u in oof[n].index for n in MEMBERS)]
    y = np.stack([[truth[u][i] for u in uids] for i in range(12)]).astype(int)
    P = {n: oof[n].loc[uids, TARGETS].to_numpy().T for n in names}                      # [12, N]
    R = {n: np.stack([rankdata(P[n][i]) / len(uids) for i in range(12)]) for n in names}
    gu = sorted(set.intersection(*[set(gold[n].index) for n in names]) & set(gold_truth))
    gy = np.stack([[gold_truth[u][i] for u in gu] for i in range(12)]).astype(int)
    GP = {n: gold[n].loc[gu, TARGETS].to_numpy().T for n in names}
    GR = {n: np.stack([rankdata(GP[n][i]) / len(gu) for i in range(12)]) for n in names}

    def blend(space: dict, w: float) -> np.ndarray:
        ens = np.mean([space[m] for m in MEMBERS], axis=0)
        return w * space[a.candidate] + (1 - w) * ens

    def macro(s, yy, idx=None):
        idx = np.arange(yy.shape[1]) if idx is None else idx
        return float(np.mean([roc_auc_score(yy[i, idx], s[i, idx]) for i in range(12)]))

    full = np.arange(len(uids))
    cid = clusters.loc[uids].to_numpy()
    groups = {c: np.where(cid == c)[0] for c in np.unique(cid)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[c] for c in rng.choice(keys, len(keys), replace=True)]) for _ in range(a.reps)]

    ens_p, ens_r = blend(P, 0.0), blend(R, 0.0)
    base_name, base_p = ("rank-mean", ens_r) if macro(ens_r, y) >= macro(ens_p, y) else ("prob-mean", ens_p)
    base_gold = blend(GR, 0.0) if base_name == "rank-mean" else blend(GP, 0.0)
    lines = [f"# exp018 blend test — candidate `{a.candidate}` (pre-registered bar: compare with the paired CI; "
             f"macro SD ≈ 0.0041)", "",
             f"{len(uids)} non-gold studies, {len(keys)} report clusters, {a.reps} paired cluster-bootstrap reps.", "",
             f"Deployed ensemble {{006,003,005}}: prob-mean {macro(ens_p, y):.4f}, rank-mean {macro(ens_r, y):.4f} "
             f"(baseline used below = better of the two: **{base_name} {macro(base_p, y):.4f}**); "
             f"gold-58 {macro(base_gold, gy):.4f}.", "",
             f"Candidate solo: OOF {macro(P[a.candidate], y):.4f} (exp006 solo {macro(P[MEMBERS[0]], y):.4f}); "
             f"gold-58 {macro(GP[a.candidate], gy):.4f} (exp006 {macro(GP[MEMBERS[0]], gy):.4f}).", "",
             "| blend | space | w(candidate) | OOF macro | Δ vs deployed | 95% CI (paired) | gold-58 (Δ) |", "|---|---|---|---|---|---|---|"]
    best = None
    for space_name, sp, gsp in (("rank-mean", R, GR), ("prob-mean", P, GP)):
        for w in a.weights:
            s = blend(sp, w)
            d = macro(s, y) - macro(base_p, y)
            bd = np.array([macro(s, y, ix) - macro(base_p, y, ix) for ix in boots])
            lo, hi = np.percentile(bd, [2.5, 97.5])
            g = macro(blend(gsp, w), gy)
            lines.append(f"| candidate+ens | {space_name} | {w:.2f} | {macro(s, y):.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                         f"{g:.4f} ({g - macro(base_gold, gy):+.4f}) |")
            if best is None or d > best[0]:
                best = (d, space_name, w)
    lines += ["", f"Best in-sample cell: {best[1]} w={best[2]:.2f} ({best[0]:+.4f}) — chosen from {2 * len(a.weights)} cells, "
                  f"so optimistic; w=0.50 rank-mean is the pre-declared headline row.", ""]
    rep = "\n".join(lines) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
