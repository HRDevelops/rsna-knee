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

"""One-off analysis: exp002 full eval per Part 3 of the 2026-09-08 housekeeping instructions.

Reads summary.csv from each exp002 fold, reports OOF (vs report/soft labels) and gold-58 macro +
per-label AUC (mean +/- std across folds), compares against exp001, and cross-references the label
engine's own gold-58 LOO AUC (docs/label_blend_v2_final.md) to flag labels where the image model's
gold-58 AUC already exceeds the label engine's — Phase 4 blend candidates per CLAUDE.md §6 step 5.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Label engine's own strict-LOO gold-58 AUC per label (docs/label_blend_v2_final.md).
LABEL_ENGINE_LOO_AUC = {
    "ACL": 0.978, "MCL": 0.943, "Medial Meniscus": 0.892, "Lateral Meniscus": 0.812,
    "Medial OA": 0.865, "Lateral OA": 0.727, "PF OA": 0.788, "Effusion": 0.815,
    "Synovitis": 0.711, "Baker's": 0.862, "Contusion": 0.731, "Fracture": 0.747,
}


def load_summaries(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(exp_dir.glob("fold*")):
        csv_path = fold_dir / "summary.csv"
        if csv_path.exists():
            rows.append(pd.read_csv(csv_path).iloc[0])
    return pd.DataFrame(rows)


def report(exp_dir: Path, label: str) -> pd.DataFrame:
    df = load_summaries(exp_dir)
    print(f"\n=== {label} ({len(df)} folds) ===")
    print(f"OOF macro-AUC (vs report labels): {df['val_macro_auc'].mean():.4f} +/- {df['val_macro_auc'].std():.4f}  "
          f"[{df['val_macro_auc'].min():.4f}, {df['val_macro_auc'].max():.4f}]")
    print(f"Gold-58 macro-AUC:                {df['gold58_macro_auc'].mean():.4f} +/- {df['gold58_macro_auc'].std():.4f}  "
          f"[{df['gold58_macro_auc'].min():.4f}, {df['gold58_macro_auc'].max():.4f}]")
    return df


def per_label_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        val_mean = df[f"val_{t}"].mean()
        gold_mean = df[f"gold_{t}"].mean()
        engine_auc = LABEL_ENGINE_LOO_AUC[t]
        rows.append({
            "label": t,
            "image_OOF_AUC": val_mean,
            "image_gold58_AUC": gold_mean,
            "label_engine_LOO_AUC": engine_auc,
            "image_beats_engine_on_gold58": gold_mean > engine_auc,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rsna_root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent))
    root = rsna_root / "artifacts"
    df1 = report(root / "exp001_dinov2s_slot", "exp001 (v1 labels, pipeline validation)")
    df2 = report(root / "exp002_dinov2s_slot_v2labels", "exp002 (labels_v2_final, stability bundle)")

    print("\n=== Per-label: exp002 image model vs label engine (gold-58 LOO) ===")
    tbl = per_label_table(df2)
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 4)
    print(tbl.to_string(index=False))

    print("\n=== Phase 4 blend candidates (image gold58 AUC > label engine LOO AUC) ===")
    print(tbl[tbl["image_beats_engine_on_gold58"]].to_string(index=False))

    out_path = root / "exp002_eval_report.csv"
    tbl.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
