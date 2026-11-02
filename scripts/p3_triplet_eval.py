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

"""Pre-registered triplet-spacing test (docs/EXPERIMENTS.md, 2026-11-02): each arm vs B5, fold 0, seed-averaged,
paired report-cluster bootstrap (1,000 reps, rng 0, same scheme as ladder_label_bootstrap.py).

Adoption rule: ACL 95% CI lower bound > 0 AND macro point change >= -0.002. If both arms qualify, larger macro wins.

    RSNA_ROOT=... python scripts/p3_triplet_eval.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train import load_soft_truth  # noqa: E402
from rsna_knee.wmil import TARGET_COLUMNS as T  # noqa: E402

A = Path(os.environ["RSNA_ROOT"]) / "artifacts"
RUNS = {"B5": ("p0b_B5_s42", "p0b_B5_s43"),
        "tri7": ("p3_tri7_s42", "p3_tri7_s43"),
        "tri10": ("p3_tri10_s42", "p3_tri10_s43")}
folds = pd.read_parquet(A / "folds.parquet")
val = set(folds[folds.fold == 0].study_uid)
truth = load_soft_truth(A / "labels/labels_v3_candidate.parquet")
cl = pd.read_parquet(A / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]


def pred(r):
    ps = []
    for e in RUNS[r]:
        d = pd.read_parquet(A / e / "fold0/oof.parquet")
        ps.append(d[d.study_uid.isin(val)].set_index("study_uid")[T])
    return (ps[0] + ps[1]) / 2


P = {r: pred(r) for r in RUNS}
u = sorted(set.intersection(*[set(p.index) for p in P.values()]) & set(truth) & set(cl.index))
Y = np.stack([truth[x] for x in u])
P = {r: p.loc[u].values for r, p in P.items()}
c = cl.loc[u].values
keys = np.unique(c)
idx = {k: np.where(c == k)[0] for k in keys}
rng = np.random.default_rng(0)
boots = [np.concatenate([idx[k] for k in rng.choice(keys, len(keys))]) for _ in range(1000)]
ACL = T.index("ACL")


def aucs(Pm, ii):
    return np.array([roc_auc_score(Y[ii, j], Pm[ii, j]) if len(set(Y[ii, j])) > 1 else np.nan for j in range(12)])


allx = np.arange(len(u))
print(f"{len(u)} fold-0 studies, {len(keys)} report clusters, seed-averaged predictions")
for r in RUNS:
    a = aucs(P[r], allx)
    print(f"{r:6s} macro {np.nanmean(a):.4f}  ACL {a[ACL]:.4f}")
qual = {}
for arm in ("tri7", "tri10"):
    full = aucs(P[arm], allx) - aucs(P["B5"], allx)
    bs = np.stack([aucs(P[arm], ii) - aucs(P["B5"], ii) for ii in boots])
    lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
    mac = np.nanmean(bs, 1)
    m = np.nanmean(full)
    ok = lo[ACL] > 0 and m >= -0.002
    if ok:
        qual[arm] = m
    print(f"\nB5 -> {arm}: macro {m:+.4f} [{np.percentile(mac, 2.5):+.4f}, {np.percentile(mac, 97.5):+.4f}]"
          f"  ->  {'QUALIFIES' if ok else 'does not qualify'}")
    for j, t in enumerate(T):
        flag = " *" if (lo[j] > 0 or hi[j] < 0) else ""
        print(f"   {t:17s} {full[j]:+.4f} [{lo[j]:+.4f}, {hi[j]:+.4f}]{flag}")
print("\nDECISION:", f"adopt {max(qual, key=qual.get)}" if qual else "no arm qualifies; B5 unchanged")
