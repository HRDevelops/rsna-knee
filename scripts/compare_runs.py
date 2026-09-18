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

"""Per-label comparison of finished 5-fold runs against a baseline (2026-09-18).

Same evaluation protocol as scripts/build_ensemble.py (imported, not re-implemented): OOF AUC of
each run's full 5-fold out-of-fold predictions against the ORIGINAL labels_v2_final report labels
binarized at 0.5, plus a gold-58 tripwire column (fold-ensembled predictions vs true gold labels —
n=58, noisy, never a selector). Prints per-label AUC and delta-vs-baseline for OOF and gold-58.

    python scripts/compare_runs.py --baseline exp006_dinov2b_slot_v3labels \\
        --runs exp010_dinov2b_slot_v3labels_t32 exp012_dinov2b_slot_v3labels_acl_prior \\
               exp015_dinov2b_slot_v3labels_448 --out artifacts/compare_vs_exp006.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import (  # noqa: E402
    TARGETS, load_full_oof, load_gold_fold_ensemble, load_gold_truth, load_truth, safe_auc,
)


def per_label_auc(pred_df: pd.DataFrame, truth: dict) -> dict:
    uids = [u for u in pred_df.index if u in truth]
    y = np.array([truth[u] for u in uids])
    out = {}
    for i, t in enumerate(TARGETS):
        out[t] = safe_auc(y[:, i], pred_df.loc[uids, t].to_numpy())
    out["macro"] = float(np.nanmean([out[t] for t in TARGETS]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    truth = load_truth(root / "artifacts/labels/labels_v2_final.parquet")
    gold_truth = load_gold_truth(root / "data/raw/train.csv")

    results = {}
    for name in [args.baseline] + args.runs:
        exp_dir = root / "artifacts" / name
        oof, gold = load_full_oof(exp_dir), load_gold_fold_ensemble(exp_dir)
        if oof is None:
            print(f"[skip] {name}: fewer than 5 folds with oof.parquet (run incomplete)", file=sys.stderr)
            continue
        results[name] = {"oof": per_label_auc(oof, truth),
                         "gold": per_label_auc(gold, gold_truth) if gold is not None else None}
    if args.baseline not in results:
        sys.exit(f"baseline {args.baseline} has no complete 5-fold OOF")

    cols = TARGETS + ["macro"]
    lines = [f"# Per-label comparison vs baseline `{args.baseline}`", "",
             "OOF = full 5-fold OOF vs labels_v2_final (binarized 0.5, non-gold). "
             "Gold = fold-ensembled predictions vs true gold-58 labels (n=58: tripwire only).", ""]
    for split, title in (("oof", "OOF macro/per-label AUC"), ("gold", "gold-58 AUC (tripwire, noisy)")):
        lines += [f"## {title}", "", "| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        base = results[args.baseline][split]
        for name, r in results.items():
            v = r[split]
            if v is None or base is None:
                continue
            lines.append(f"| {name} | " + " | ".join(f"{v[c]:.4f}" for c in cols) + " |")
        lines += ["", f"### delta vs {args.baseline}", "", "| run | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for name, r in results.items():
            v = r[split]
            if name == args.baseline or v is None or base is None:
                continue
            lines.append(f"| {name} | " + " | ".join(f"{v[c] - base[c]:+.4f}" for c in cols) + " |")
        lines.append("")
    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)


if __name__ == "__main__":
    main()
