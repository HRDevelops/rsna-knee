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

"""exp018 cache audit (CPU): coverage, slot presence, and — the reason this exists — whether the geometric
side rule (image-centre patient-x sign) agrees with the validated sides we already have (final_side.csv:
DICOM tag, cross-checked against report text), because ~34% of studies have no DICOM laterality tag and the
deployed pipeline never resolved them. Writes artifacts/exp018_cache_audit.md."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsna_knee.wmil import NUM_SLOTS, SLOT_SPECS, normalize_tag_side, side_from_geometry  # noqa: E402

root = Path(os.environ["RSNA_ROOT"])
man = pd.concat([pd.read_parquet(f) for f in sorted((root / "artifacts/cache_manifests_wmil").glob("shard_*.parquet"))])
n_all, ok = len(man), man[man["ok"]].copy()
lines = ["# exp018 cache audit", "", f"studies attempted {n_all}, decoded {len(ok)}, failed {n_all - len(ok)}", ""]
for e in man[~man["ok"]]["error"].head(3):
    lines.append(f"- failure example: `{e.splitlines()[0][:160]}`")

lines += ["", "## Slot presence", "", "| slot | studies with slot | share |", "|---|---|---|"]
for i, s in enumerate(SLOT_SPECS):
    n = int(ok[f"present_{i}"].sum())
    lines.append(f"| {s[0]} ({s[3]} slices) | {n} | {n / len(ok):.1%} |")
n_slots = ok[[f"present_{i}" for i in range(NUM_SLOTS)]].sum(axis=1)
lines += ["", "studies by number of slots filled: " + ", ".join(f"{k}: {int((n_slots == k).sum())}" for k in range(NUM_SLOTS + 1)), ""]

sag_nx = pd.concat([ok["normal_x_0"], ok["normal_x_1"]]).dropna()
cor_ax_rx = pd.concat([ok[f"row_x_{i}"] for i in (2, 3, 4)]).dropna()
lines += [
    "## Orientation conventions actually present", "",
    f"- sagittal normal_x < 0 (ordering reversed to ascending x): {(sag_nx < 0).mean():.1%} of {len(sag_nx)} series",
    f"- coronal/axial row_x < 0 (image-right points to patient RIGHT; extra horizontal flip): "
    f"{(cor_ax_rx < 0).mean():.2%} of {len(cor_ax_rx)} series",
    f"- series where InstanceNumber order != geometric order: "
    f"{np.mean([ok[f'order_mismatch_{i}'][ok[f'present_{i}']].mean() for i in range(NUM_SLOTS)]):.1%}", ""]

# side rule vs validated sides
fs = pd.read_csv(root / "artifacts/final_side.csv").set_index("study_uid")
cx = ok[[f"cx_{i}" for i in range(NUM_SLOTS)]].median(axis=1)
geo = cx.map(side_from_geometry)
tag = ok[[f"tag_{i}" for i in range(NUM_SLOTS)]].apply(
    lambda r: (lambda s: next(iter(s)) if len(s) == 1 else None)({normalize_tag_side(v) for v in r} - {None}), axis=1)
lines += ["## Geometric side rule (|median centre x| >= 20 mm; x<0 = right) vs validated side", "",
          "| reference set | n | geometry resolves | agreement where resolved |", "|---|---|---|---|"]
for name, ref in (("final_side source=tag", fs[fs["source"] == "tag"]["final_side"]),
                  ("final_side source=report", fs[fs["source"] == "report"]["final_side"])):
    ref = ref[ref.isin(["L", "R"])].reindex(ok.index).dropna()
    g = geo.reindex(ref.index)
    res = g.notna()
    lines.append(f"| {name} | {len(ref)} | {res.mean():.1%} | {(g[res] == ref[res]).mean():.2%} |")
none = fs[fs["source"] == "none_needs_classifier"].index.intersection(ok.index)
lines += ["", f"studies with no validated side (final_side none_needs_classifier): {len(none)}; "
              f"geometry resolves {geo.reindex(none).notna().mean():.1%} of them "
              f"(right {(geo.reindex(none) == 'R').mean():.1%} / left {(geo.reindex(none) == 'L').mean():.1%})"]
rep = "\n".join(lines) + "\n"
print(rep)
(root / "artifacts/exp018_cache_audit.md").write_text(rep, encoding="utf-8")
