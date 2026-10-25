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

"""Pre-launch check of the Phase 0B ladder data path (CPU). (1) Rung B1 must reproduce exp018's evaluation windows:
same number, same order, same laterality handling; pixels differ only by the resampler, so each window must correlate
> 0.97 with exp018's and sit at zero shift. (2) Every rung returns well-formed batches; prints window counts.

    RSNA_ROOT=... python scripts/check_wmil_v2.py 40
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee import wmil  # noqa: E402
from rsna_knee.wmil_v2 import WMILv2Dataset, load_manifest_v2  # noqa: E402

R = Path(os.environ["RSNA_ROOT"])
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
man2 = load_manifest_v2(R / "artifacts/cache_manifests_v2")
man1 = pd.concat([pd.read_parquet(f) for f in (R / "artifacts/cache_manifests_wmil").glob("shard_*.parquet")])
man1 = man1[man1.ok].set_index("study_uid")
side_df = pd.read_csv(R / "artifacts/final_side.csv")
final_side = side_df.set_index("study_uid")["final_side"].to_dict()
bilateral = set(side_df.loc[side_df["source"].isin(["conflict_resolved_by_report", "conflict_unresolved"]), "study_uid"])
uids = sorted(set(man2.study_uid) & set(man1.index))[:N]

ds = WMILv2Dataset(uids, R / "data/cache_v2", man2, None, final_side, bilateral, train=False, k=12,
                   series="exp018_slots", slices="exp018", plane_mix="uniform", crop="c140")
corrs, nbad = [], 0
for i, u in enumerate(uids):
    item = ds[i]
    r = man1.loc[u]
    present = np.array([bool(r[f"present_{j}"]) for j in range(wmil.NUM_SLOTS)])
    metas = [{"row_x": float(r[f"row_x_{j}"]), "cx": float(r[f"cx_{j}"]), "tag_laterality": r[f"tag_{j}"]}
             if present[j] else None for j in range(wmil.NUM_SLOTS)]
    if u in bilateral:
        sides = [wmil.normalize_tag_side(m["tag_laterality"]) if m else None for m in metas]
    else:
        fs = final_side.get(u)
        sides = wmil.resolve_slot_sides(metas, fs if isinstance(fs, str) else None)
    stack = wmil.normalize_orientation(np.load(R / f"data/cache_wmil/{u}.npy"), metas, sides)
    cs = wmil.candidate_centers(present)
    w1 = np.stack([stack[c - 1:c + 2] for c in cs]).astype(float)
    w2 = item["windows"][item["wmask"]].numpy().astype(float)
    if len(w1) != len(w2):
        nbad += 1
        print(f"{u[-8:]}: window count v1 {len(w1)} vs v2 {len(w2)}")
        continue
    # window-MIL is permutation-invariant over windows; exp018 reverses a right sagittal slot's whole slice order, so
    # its windows are the same triplets enumerated in reverse. Match each exp018 window to its best v2 window.
    A = (w1 - w1.mean(axis=(1, 2, 3), keepdims=True)).reshape(len(w1), -1)
    B = (w2 - w2.mean(axis=(1, 2, 3), keepdims=True)).reshape(len(w2), -1)
    C = (A / np.linalg.norm(A, axis=1, keepdims=True)) @ (B / np.linalg.norm(B, axis=1, keepdims=True)).T
    best = C.argmax(1)
    if len(set(best.tolist())) != len(best):
        nbad += 1
        print(f"{u[-8:]}: matching is not one-to-one")
    corrs += C.max(1).tolist()
corrs = np.array(corrs)
print(f"B1 vs exp018: {len(uids)} studies, count mismatches {nbad}, windows {len(corrs)}, corr median "
      f"{np.median(corrs):.4f}, p1 {np.percentile(corrs, 1):.4f}, min {corrs.min():.4f}, share > 0.97 "
      f"{(corrs > 0.97).mean():.3%}")

for rung, opt in {"B2": dict(series="exp018_slots", slices="native", plane_mix="exp018", crop="c140"),
                  "B3": dict(series="all", slices="native", plane_mix="exp018", crop="c140"),
                  "B4": dict(series="all", slices="native", plane_mix="balanced", crop="c140"),
                  "B5": dict(series="all", slices="native", plane_mix="balanced", crop="f160")}.items():
    tr = WMILv2Dataset(uids, R / "data/cache_v2", man2, None, final_side, bilateral, train=True, k=12, **opt)
    ev = WMILv2Dataset(uids, R / "data/cache_v2", man2, None, final_side, bilateral, train=False, k=12, **opt)
    a, b = tr[0], ev[0]
    n_eval = [int(ev[j]["wmask"].sum()) for j in range(min(10, len(uids)))]
    print(f"{rung}: train {tuple(a['windows'].shape)} eval {tuple(b['windows'].shape)} eval windows/study "
          f"median {int(np.median(n_eval))} max {max(n_eval)}")
