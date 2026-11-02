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

"""ACL forensics step 1 (docs/research/NEXT_STEPS_2026-11.md s4): paired per-label cluster-bootstrap CIs of the Phase 0B
ladder steps from existing fold-0 predictions (seed-averaged over 42 / 43; report clusters; 1,000 reps). No training.

    RSNA_ROOT=... python scripts/ladder_label_bootstrap.py
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
RUNS = {"B0": ("exp018_coatnet2_wmil_v3labels", "exp021_coatnet2_wmil_seed43")}
RUNS.update({r: (f"p0b_{r}_s42", f"p0b_{r}_s43") for r in ("B1", "B2", "B3", "B4", "B5")})
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


def aucs(Pm, ii):
    return np.array([roc_auc_score(Y[ii, j], Pm[ii, j]) if len(set(Y[ii, j])) > 1 else np.nan for j in range(12)])


print(f"{len(u)} fold-0 studies, {len(keys)} report clusters, seed-averaged predictions")
for a, b in [("B0", "B1"), ("B1", "B2"), ("B2", "B3"), ("B3", "B4"), ("B4", "B5"), ("B0", "B5")]:
    full = aucs(P[b], np.arange(len(u))) - aucs(P[a], np.arange(len(u)))
    bs = np.stack([aucs(P[b], ii) - aucs(P[a], ii) for ii in boots])
    lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
    mac = np.nanmean(bs, 1)
    print(f"\n{a} -> {b}: macro {np.nanmean(full):+.4f} [{np.percentile(mac, 2.5):+.4f}, {np.percentile(mac, 97.5):+.4f}]")
    for j, t in enumerate(T):
        flag = " *" if (lo[j] > 0 or hi[j] < 0) else ""
        print(f"   {t:17s} {full[j]:+.4f} [{lo[j]:+.4f}, {hi[j]:+.4f}]{flag}")
