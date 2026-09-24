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

"""exp018 fold-0 gate (pre-registered 2026-09-21, before any result existed).

KILL RULE: fold-0 gold-58 macro-AUC of the top-3-epoch-averaged model < 0.880 -> HALT: write the report,
launch nothing. Otherwise launch folds 1-4 (through scripts/safe_sbatch.py, i.e. with the disk-projection
guard). Gold-58 was never used for selection (epochs are chosen by fold-validation AUC), and n=58 is a coarse
instrument (per-fold gold macro of the deployed exp006 models ranges ~0.87-0.89): clearing 0.880 means "not
worse than the deployed recipe on the tripwire", NOT evidence of improvement — the report also gives fold-0 OOF
against the same fold of exp006 for that reason.

    python scripts/exp018_gate.py [--no-launch]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import TARGETS, load_truth  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

EXP, BASE, GATE = "exp018_coatnet2_wmil_v3labels", "exp006_dinov2b_slot_v3labels", 0.880


def macro_vs_truth(df: pd.DataFrame, truth: dict) -> tuple[float, dict]:
    df = df[df["study_uid"].isin(truth)]
    y = np.stack([truth[u] for u in df["study_uid"]])
    per = {c: roc_auc_score(y[:, i], df[c].to_numpy()) for i, c in enumerate(TARGETS) if len(set(y[:, i])) > 1}
    return float(np.mean(list(per.values()))), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-launch", action="store_true")
    args = ap.parse_args()
    root = Path(os.environ["RSNA_ROOT"])
    art = root / "artifacts"
    f0 = art / EXP / "fold0"
    out = art / "exp018_fold0_gate.md"
    if not (f0 / "summary.csv").exists():
        out.write_text(f"# exp018 fold-0 gate\n\nNO RESULT: {f0}/summary.csv missing — the fold-0 job did not finish. "
                       f"Nothing launched.\n")
        print("fold 0 did not finish; nothing launched")
        return 1
    s = pd.read_csv(f0 / "summary.csv").iloc[0]
    gold = float(s["gold58_macro_auc"])
    v2 = load_truth(art / "labels/labels_v2_final.parquet")
    oof_new = pd.read_parquet(f0 / "oof.parquet")
    oof_base = pd.read_parquet(art / BASE / "fold0/oof.parquet")
    m_new, p_new = macro_vs_truth(oof_new, v2)
    m_base, p_base = macro_vs_truth(oof_base, v2)
    b0 = pd.read_csv(art / BASE / "fold0/summary.csv").iloc[0]
    lines = [f"# exp018 fold-0 gate (rule: gold-58 macro < {GATE} -> HALT)", "",
             f"| | exp018 (fold 0) | exp006 (fold 0) |", "|---|---|---|",
             f"| fold-0 OOF macro vs labels_v2_final (n={len(oof_new)}) | **{m_new:.4f}** | {m_base:.4f} |",
             f"| fold-0 val macro vs labels_v3 (selection metric) | {s['val_macro_auc']:.4f} | {b0['val_macro_auc']:.4f} |",
             f"| fold-0 gold-58 macro (tripwire, n=58) | **{gold:.4f}** | {b0['gold58_macro_auc']:.4f} |", "",
             "| label | exp018 OOF | exp006 OOF | Δ | exp018 gold | exp006 gold |", "|---|---|---|---|---|---|"]
    for c in TARGETS:
        lines.append(f"| {c} | {p_new[c]:.4f} | {p_base[c]:.4f} | {p_new[c] - p_base[c]:+.4f} | "
                     f"{s[f'gold_{c}']:.4f} | {b0[f'gold_{c}']:.4f} |")
    passed = gold >= GATE
    lines += ["", f"**Verdict: {'CLEARS the gate' if passed else 'FAILS the gate -> HALT, nothing launched'}** "
                  f"(gold-58 {gold:.4f} {'>=' if passed else '<'} {GATE}).",
              "Caveat: one fold, n=58 gold; a pass means 'not worse than the deployed recipe on the tripwire'. "
              "The fold-0 OOF comparison above is the informative number."]
    if passed and not args.no_launch:
        cmd = [sys.executable, "scripts/safe_sbatch.py", "--", "--array=1-4",
               "slurm/train_wmil.sbatch", "configs/exp018_coatnet2_wmil_v3labels.yaml"]
        r = subprocess.run(cmd, cwd=root / "code", capture_output=True, text=True)
        lines += ["", "Launch of folds 1-4 via safe_sbatch:", "```", (r.stdout + r.stderr).strip()[-1500:], "```"]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
