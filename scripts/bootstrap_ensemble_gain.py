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

"""Sampling-noise check for OOF macro-AUC differences (2026-09-20).

Paired CLUSTER bootstrap: resample report clusters (artifacts/report_clusters.parquet) with
replacement, gather all their studies, and recompute macro-AUC for every recipe on the SAME resample —
so each reported delta is paired and near-duplicate reports (which make studies non-independent) don't
make the interval look tighter than it is. Fixed recipes only: this measures sampling noise of a delta,
NOT the selection optimism of choosing members/methods on the same data (greedy selection and the
per-label max over prob-mean/rank-mean both pick winners in-sample, so the true out-of-sample gain of a
re-selected recipe is smaller than its point estimate).

    python scripts/bootstrap_ensemble_gain.py --reps 300
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
from build_ensemble import TARGETS, best_method_score, load_full_oof, load_truth  # noqa: E402

POOL = {
    "exp003": "exp003_dinov2b_slot_v2labels", "exp005": "exp005_dinov2s_slot_v3labels",
    "exp006": "exp006_dinov2b_slot_v3labels", "exp010": "exp010_dinov2b_slot_v3labels_t32",
    "exp012": "exp012_dinov2b_slot_v3labels_acl_prior", "exp015": "exp015_dinov2b_slot_v3labels_448",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent)) / "artifacts"

    oof = {k: load_full_oof(root / v) for k, v in POOL.items()}
    truth = load_truth(root / "labels/labels_v2_final.parquet")
    clusters = pd.read_parquet(root / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]
    uids = [u for u in oof["exp006"].index if u in truth and u in clusters.index]
    y = {t: np.array([truth[u][i] for u in uids]) for i, t in enumerate(TARGETS)}
    mats = {t: pd.DataFrame({k: oof[k].loc[uids, t] for k in POOL}, index=uids) for t in TARGETS}

    def scores(members: list, methods: dict) -> dict:
        out = {}
        for t in TARGETS:
            m = mats[t][members]
            out[t] = (m.rank(pct=True).mean(axis=1) if methods[t] == "rank-mean" else m.mean(axis=1)).to_numpy()
        return out

    def csv_recipe(path: Path):
        df = pd.read_csv(path)
        return df["members"].iloc[0].split(","), dict(zip(df["label"], df["method"]))

    def short(name):
        return next(k for k, v in POOL.items() if v == name)

    old_members, old_methods = csv_recipe(root / "ensemble_recipe_v1_accepted.csv")
    new_members, new_methods = csv_recipe(root / "ensemble_recipe_reselect_20260918.csv")
    three = ["exp012", "exp003", "exp006"]
    three_methods = {t: best_method_score(mats[t], three, y[t])[0] for t in TARGETS}

    recipes = {
        "deployed 3-member {006,003,005}": scores([short(m) for m in old_members], old_methods),
        "reselected 6-member (all)": scores([short(m) for m in new_members], new_methods),
        "3-member {012,003,006}": scores(three, three_methods),
        "solo exp006": scores(["exp006"], {t: "prob-mean" for t in TARGETS}),
        "solo exp010 (T=32)": scores(["exp010"], {t: "prob-mean" for t in TARGETS}),
        "solo exp012 (ACL prior)": scores(["exp012"], {t: "prob-mean" for t in TARGETS}),
        "solo exp015 (448px)": scores(["exp015"], {t: "prob-mean" for t in TARGETS}),
    }

    def macro(sc: dict, idx: np.ndarray) -> float:
        return float(np.mean([roc_auc_score(y[t][idx], sc[t][idx]) for t in TARGETS]))

    full_idx = np.arange(len(uids))
    point = {k: macro(v, full_idx) for k, v in recipes.items()}

    cid = clusters.loc[uids].to_numpy()
    groups = {c: np.where(cid == c)[0] for c in np.unique(cid)}
    keys = np.array(list(groups.keys()))
    rng = np.random.default_rng(args.seed)
    boots = {k: [] for k in recipes}
    for _ in range(args.reps):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        for k, v in recipes.items():
            boots[k].append(macro(v, idx))
    boots = {k: np.array(v) for k, v in boots.items()}

    print(f"{len(uids)} studies, {len(keys)} report clusters, {args.reps} cluster-bootstrap reps\n")
    print("| recipe | macro (point) | bootstrap SD |")
    print("|---|---|---|")
    for k in recipes:
        print(f"| {k} | {point[k]:.4f} | {boots[k].std():.4f} |")

    def delta(a: str, b: str) -> None:
        d = boots[a] - boots[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"| {a} vs {b} | {point[a] - point[b]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {(d > 0).mean():.2f} |")

    print("\n| paired delta | point | 95% CI (cluster bootstrap) | P(delta > 0) |")
    print("|---|---|---|---|")
    base = "deployed 3-member {006,003,005}"
    delta("reselected 6-member (all)", base)
    delta("3-member {012,003,006}", base)
    for solo in ("solo exp010 (T=32)", "solo exp012 (ACL prior)", "solo exp015 (448px)"):
        delta(solo, "solo exp006")


if __name__ == "__main__":
    main()
