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

"""Choose the gold-58 studies for the attention figure BEFORE any attention map is looked at.

Pre-declared rule (fixed here so the choice cannot be tuned to what the maps look like): for each finding in
FINDINGS, in order, take the gold-positive study with the HIGHEST exp021 (10-checkpoint) probability for that
finding that has not been taken yet. This picks the clearest, most confidently detected cases, so the figure is a
best-case illustration of where the model looks, not an estimate of how often it looks in the right place (the
slot-level attention summary over all 58 studies is the non-selected view).

    python scripts/fig_select_attention_studies.py        # writes docs/figures/_data/attention_selection.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as fc  # noqa: E402

FINDINGS = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Effusion", "Baker's", "Contusion"]


def main() -> None:
    d = fc.load_predictions()
    gold, y, uids = d["gold"]["exp021"], d["y_gold"], d["uids_gold"]
    taken: set[str] = set()
    rows = []
    for f in FINDINGS:
        i = fc.TARGETS.index(f)
        pos = [(gold.loc[u, f], u) for j, u in enumerate(uids) if y[i, j] == 1]
        p, u = max((pu for pu in pos if pu[1] not in taken))
        taken.add(u)
        j = uids.index(u)
        rows.append({"finding": f, "study_uid": u, "exp021_prob": round(float(p), 4),
                     "all_gold_positive": "; ".join(t for k, t in enumerate(fc.TARGETS) if y[k, j] == 1),
                     "n_gold_positive_candidates": len(pos)})
    out = pd.DataFrame(rows)
    out.to_csv(fc.DATA / "attention_selection.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 90)
    print(out.drop(columns="study_uid").to_string())


if __name__ == "__main__":
    main()
