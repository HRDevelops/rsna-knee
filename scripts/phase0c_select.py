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

"""Phase 0C selection exactly as pre-registered (docs/EXPERIMENTS.md, 2026-10-25): default 6x6 + bagged; another grid
only if its 2-seed mean bagged fold-val macro-AUC beats 6x6 by >= +0.004; dense only if, on the chosen grid, the
paired cluster bootstrap (seed-averaged predictions, report clusters, 1,000 reps) gives delta > 0 and CI lower > 0.
Gold-58 is not read. Also prints the ACL documentation item (B0 vs B5, per seed).

    RSNA_ROOT=... python scripts/phase0c_select.py
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
means = {}
for g in (4, 6, 12):
    s = [pd.read_csv(A / f"p0c_g{g}_s{x}/fold0/summary.csv").iloc[0] for x in (42, 43)]
    bg = [r["val_macro_auc_bagged"] for r in s]
    dn = [r["val_macro_auc_dense"] for r in s]
    means[g] = np.mean(bg)
    pk = s[0].get("peak_gb_dense", float("nan"))
    print(f"g{g}: bagged {bg[0]:.4f} / {bg[1]:.4f} mean {np.mean(bg):.4f} | dense {dn[0]:.4f} / {dn[1]:.4f} "
          f"mean {np.mean(dn):.4f} | dense peak GPU {pk:.1f} GB")
g_star = 6
for g in (4, 12):
    if means[g] - means[6] >= 0.004 and means[g] - means[6] >= means[g_star] - means[6]:
        g_star = g
print(f"grid rule: deltas vs 6x6 {{4: {means[4] - means[6]:+.4f}, 12: {means[12] - means[6]:+.4f}}} -> g* = {g_star}")

truth = load_soft_truth(A / "labels/labels_v3_candidate.parquet")
cl = pd.read_parquet(A / "report_clusters.parquet").set_index("study_uid")["report_cluster_id"]


def avg(mode):
    a = [pd.read_parquet(A / f"p0c_g{g_star}_s{x}/fold0/oof_{mode}.parquet").set_index("study_uid")[T] for x in (42, 43)]
    return (a[0] + a[1]) / 2


B, D = avg("bagged"), avg("dense")
u = sorted(set(B.index) & set(D.index) & set(truth) & set(cl.index))
Y = np.stack([truth[x] for x in u])
B, D = B.loc[u].values, D.loc[u].values


def mac(P, y):
    return np.mean([roc_auc_score(y[:, j], P[:, j]) for j in range(12) if len(set(y[:, j])) > 1])


d0 = mac(D, Y) - mac(B, Y)
c = cl.loc[u].values
keys = np.unique(c)
idx = {k: np.where(c == k)[0] for k in keys}
rng = np.random.default_rng(0)
ds = []
for _ in range(1000):
    ii = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
    ds.append(mac(D[ii], Y[ii]) - mac(B[ii], Y[ii]))
lo, hi = np.percentile(ds, [2.5, 97.5])
m_star = "dense" if (d0 > 0 and lo > 0) else "bagged"
print(f"mode rule on g{g_star} (seed-averaged, {len(u)} studies, {len(keys)} clusters): dense - bagged {d0:+.4f} "
      f"[{lo:+.4f}, {hi:+.4f}] -> m* = {m_star}")
print(f"FROZEN for G: grid {g_star}x{g_star}, inference {m_star}")
for x, e0 in ((42, "exp018_coatnet2_wmil_v3labels"), (43, "exp021_coatnet2_wmil_seed43")):
    a = pd.read_csv(A / f"{e0}/fold0/summary.csv").iloc[0]
    b = pd.read_csv(A / f"p0b_B5_s{x}/fold0/summary.csv").iloc[0]
    print(f"ACL seed {x}: B0 {a['val_ACL']:.4f} -> B5 {b['val_ACL']:.4f} ({b['val_ACL'] - a['val_ACL']:+.4f})")
