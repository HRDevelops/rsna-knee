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

"""Generate LaTeX table bodies for the Dec 5 report from the same per-study predictions as the figures, so no
per-label number is typed by hand.

  docs/report/tables/per_label_auc.tex   per-label AUC, OOF and gold-58, exp018 and exp021 (macro row last)

    python scripts/report_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_common as fc  # noqa: E402


def main() -> None:
    d = fc.load_predictions()           # asserts the ledger's four headline macro-AUCs
    out = fc.ROOT / "docs/report/tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cols = {"oof": {}, "gold": {}}
    for split in ("oof", "gold"):
        y = d["y_oof" if split == "oof" else "y_gold"]
        for m in ("exp018", "exp021"):
            cols[split][m] = [roc_auc_score(y[i], d[split][m][t].to_numpy()) for i, t in enumerate(fc.TARGETS)]
    for i, t in enumerate(fc.TARGETS):
        name = t.replace("'", "'")
        rows.append(f"{name} & {cols['oof']['exp018'][i]:.3f} & {cols['oof']['exp021'][i]:.3f} & "
                    f"{cols['gold']['exp018'][i]:.3f} & {cols['gold']['exp021'][i]:.3f} \\\\")
    rows.append("\\midrule")
    # The last row deliberately has no row terminator: main.tex writes the closing row break after \input, because a
    # row break at the very end of an \input file followed by \bottomrule raises "Misplaced \noalign".
    rows.append("Macro & " + " & ".join(f"{np.mean(cols[s][m]):.4f}" for s in ("oof", "gold") for m in ("exp018", "exp021")))
    (out / "per_label_auc.tex").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(rows))
    print(f"wrote {out / 'per_label_auc.tex'}")


if __name__ == "__main__":
    main()
