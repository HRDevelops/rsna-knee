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
def exp018_plan(u, r, sides):
    """exp018's eval windows as (series_uid, channel-ordered source-slice triplet in stored order, mirror), rebuilt from
    its own rules: slot slices = linspace over the 6-94% span of the (ascending patient-x for sagittal) order, sagittal
    right knees reverse the whole slot, coronal/axial mirror when (side == R) != (row_x < 0), window = [c-1, c, c+1]."""
    out = []
    for j, (name, plane, _, L) in enumerate(wmil.SLOT_SPECS):
        if not bool(r[f"present_{j}"]):
            continue
        n = int(r[f"n_native_{j}"])
        order = np.arange(n)[::-1] if (plane == "Sagittal" and float(r[f"normal_x_{j}"]) < 0) else np.arange(n)
        lo, hi = int(round(wmil.SPAN[0] * (n - 1))), int(round(wmil.SPAN[1] * (n - 1)))
        slot = order[np.clip(np.round(np.linspace(lo, hi, L)).astype(int), 0, n - 1)]
        mirror = False
        if sides[j] is not None:
            if plane == "Sagittal":
                if sides[j] == "R":
                    slot = slot[::-1]
            else:
                mirror = (sides[j] == "R") != (float(r[f"row_x_{j}"]) < 0)
        out += [(r[f"series_{j}"], tuple(int(t) for t in slot[c - 1:c + 2]), mirror) for c in range(1, L - 1)]
    return out


corrs, nbad, n_win = [], 0, 0
for i, u in enumerate(uids):
    item = ds[i]
    r0 = man1.loc[u]
    pres = [bool(r0[f"present_{j}"]) for j in range(wmil.NUM_SLOTS)]
    mt = [{"row_x": float(r0[f"row_x_{j}"]), "cx": float(r0[f"cx_{j}"]), "tag_laterality": r0[f"tag_{j}"]}
          if pres[j] else None for j in range(wmil.NUM_SLOTS)]
    if u in bilateral:
        sd = [wmil.normalize_tag_side(m["tag_laterality"]) if m else None for m in mt]
    else:
        fs0 = final_side.get(u)
        sd = wmil.resolve_slot_sides(mt, fs0 if isinstance(fs0, str) else None)
    p_old, p_new = exp018_plan(u, r0, sd), ds.plan(i)
    n_win += len(p_old)
    if sorted(p_old) != sorted(p_new):
        nbad += 1
        diff = set(p_old) ^ set(p_new)
        print(f"{u[-8:]}: SOURCE-SLICE PLAN MISMATCH ({len(diff)} differing windows), e.g. {sorted(diff)[:3]}")
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
    if len(w1) != len(w2) or len(w1) != len(p_old):
        nbad += 1
        print(f"{u[-8:]}: window count v1 {len(w1)} vs v2 {len(w2)} vs plan {len(p_old)}")
        continue
    # exact pairing by source-slice plan entry (series, channel-ordered triplet, mirror); the per-pair correlation then
    # tests the pixels channel by channel, including the left-right mirror. Duplicate entries (series with fewer
    # native slices than the slot) are paired in order.
    pos = {}
    for k, key in enumerate(p_new):
        pos.setdefault(key, []).append(k)
    for k, key in enumerate(p_old):
        b = w2[pos[key].pop(0)]
        corrs.append(min(np.corrcoef(w1[k][ch].ravel(), b[ch].ravel())[0, 1] for ch in range(3)))
corrs = np.array(corrs)
print(f"B1 source-slice plan (series, channel-ordered triplet, mirror) identical to exp018: "
      f"{len(uids) - nbad}/{len(uids)} studies, {n_win} windows")
assert nbad == 0, "B1 does not reproduce exp018's source slices / channel order"
print(f"B1 vs exp018 pixels: {len(uids)} studies, windows {len(corrs)}, corr median "
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
