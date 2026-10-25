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

"""Mechanics check of the Phase 1 donor-swap control (not a metric search): how many fold-0 val studies received a
donor, how much the donor geometry differs from the study's own (relative cross-plane layout), and how much the
predictions moved. RSNA_ROOT=... python scripts/phase1_swap_check.py"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train_tokens import donor_map  # noqa: E402
from rsna_knee.wmil_v2 import load_manifest_v2  # noqa: E402

A = Path(os.environ["RSNA_ROOT"]) / "artifacts"
man = load_manifest_v2(A / "cache_manifests_v2")
side = pd.read_csv(A / "final_side.csv").set_index("study_uid")["final_side"].to_dict()
f = pd.read_parquet(A / "folds.parquet")
val = sorted(f[f.fold == 0].study_uid)
tr = sorted(f[(f.fold != 0) & (f.fold != -1)].study_uid)
dm = donor_map(val, tr, man, side)
print(f"val studies {len(val)}, with a donor {len(dm)}")


def layout(u):
    """Centre of each plane's longest series relative to the study's sagittal centre (mm)."""
    g = man[man.study_uid == u]
    c = {}
    for p, x in g.groupby("plane"):
        r = x.sort_values("n_slices").iloc[-1]
        ipp = np.asarray(r["ipp"])
        c[p] = ipp.mean(0)
    ref = c.get("Sagittal", np.mean(list(c.values()), 0))
    return {p: v - ref for p, v in c.items()}


diffs = []
for u, v in list(dm.items())[:200]:
    a, b = layout(u), layout(v)
    diffs += [np.linalg.norm(a[p] - b[p]) for p in a if p in b and p != "Sagittal"]
print(f"relative plane-centre difference own vs donor (mm): median {np.median(diffs):.1f}, p90 {np.percentile(diffs, 90):.1f}")
for s in (42, 43):
    a = pd.read_parquet(A / f"p1_graw_s{s}/fold0/oof_dense.parquet").set_index("study_uid").sort_index()
    b = pd.read_parquet(A / f"p1_graw_s{s}/fold0/oof_dense_donor.parquet").set_index("study_uid").sort_index()
    d = (a - b).abs().values
    print(f"seed {s}: mean |dprob| {d.mean():.5f}, max {d.max():.4f}, studies changed > 1e-3: {(d.max(1) > 1e-3).mean():.3f}")
