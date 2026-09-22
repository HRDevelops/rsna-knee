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

"""Efficiency-track table (CLAUDE.md §4: Eff = AUC/(0.5-maxAUC) + seconds/32400, minimized) comparing
exp008 (deployed) against exp018 at several fold counts / GPU counts.

AUC anchor: each 5-fold config's REAL public-LB score (exp008 0.885, exp018 0.929) — more trustworthy
than either model's OOF, and for exp018 specifically OOF is known to undersell it by ~4 points (see
exp018_probe4_analysis.md). Reduced-fold AUC = that anchor + a gold-58 all-subsets-averaged delta
(scripts/fold_subset_gold_auc.py; exp008's delta table was computed the same way, 2026-09-10).

Runtime: H100-measured decode/model split (this project's standard method), decode RESCALED for
Kaggle's real CPU allocation (the deployed efficiency notebook assumes ~4 CPUs -> 3 decode workers;
our H100 profiling jobs use 8 -- see --profiled-decode-workers/--kaggle-decode-workers), model time
x3.5 for T4 (docs/SUBMISSION_RUNBOOK.md), decode/model overlapped (bound by whichever is slower, not
their sum -- matches how the ensemble/efficiency notebooks already project T4 runtime). T4 x2 assumes
per-study data-parallel across GPUs (this project's inference driver design): model time roughly
halves, decode does not (decode workers are shared, not per-GPU).

    python scripts/exp018_efficiency_table.py --out artifacts/exp018_efficiency_table.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

# --- measured/logged inputs (see docstring for provenance) -----------------------------------------
EXP008_LB, EXP018_LB = 0.885, 0.929
ASSUMED_MAX_AUC = 0.95
N_TEST = 1300
EFF_DENOM = 0.5 - ASSUMED_MAX_AUC  # -0.45

# gold-58 all-subsets fold-count deltas (macro-AUC vs that model's own full 5-fold)
EXP008_FOLD_DELTA = {1: -0.0060, 2: -0.0030, 3: -0.0014, 4: -0.0004, 5: 0.0}     # docs/EXPERIMENTS.md 2026-09-10
EXP018_FOLD_DELTA = {1: -0.0067, 2: -0.0023, 3: -0.0007, 4: -0.0002, 5: 0.0}     # scripts/fold_subset_gold_auc.py, 2026-09-22

# exp008: real full-1300 H100 rehearsal (docs/EXPERIMENTS.md 2026-09-10), decode/model overlap already
# built into the notebook (infer_core.prefetch_decode) so the logged 447s IS the overlapped total for
# 4/5 folds at DECODE_WORKERS=3 (Kaggle-realistic) -- no rescaling needed, it was measured on Kaggle-like
# CPU provisioning already. T4 projection already logged: ~0.535 s/study T4 x1 (max(decode, model x3.5)).
EXP008_T4X1_S_PER_STUDY_5FOLD = 0.535   # docs/EXPERIMENTS.md 2026-09-10 (measured at 4/5 folds; ~fold-count-flat since decode dominates)
EXP008_DECODE_S_PER_STUDY = 0.508 * (1300 / 1300)  # T4-projected decode floor, same source

# exp018: H100 profiling job 26078, 300 real studies, DECODE_WORKERS=8 (the H100 profiling job's CPU
# allocation) -- NOT Kaggle's real ~3 decode workers, so decode must be rescaled by worker-count ratio
# before projecting (single-thread-equivalent CPU-seconds/study is worker-count-invariant; per-study
# WALL time is not).
EXP018_H100_DECODE_S_PER_STUDY = 0.264
EXP018_H100_MODEL_S_PER_STUDY_5FOLD = 0.359
PROFILED_DECODE_WORKERS = 8
KAGGLE_DECODE_WORKERS = 3               # matches DECODE_WORKERS in notebooks/submit-efficiency.ipynb cell-3
T4_MODEL_MULTIPLIER = 3.5


def efficiency_score(auc: float, seconds: float) -> float:
    return auc / EFF_DENOM + seconds / 32400


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    decode_kaggle = EXP018_H100_DECODE_S_PER_STUDY * (PROFILED_DECODE_WORKERS / KAGGLE_DECODE_WORKERS)
    model_h100_per_fold = EXP018_H100_MODEL_S_PER_STUDY_5FOLD / 5

    rows = []
    # exp008, as already deployed (5-fold, T4 x1)
    auc = EXP008_LB
    sec = EXP008_T4X1_S_PER_STUDY_5FOLD * N_TEST
    rows.append(("exp008 (deployed, 5-fold, T4 x1)", 5, 1, auc, sec, efficiency_score(auc, sec)))

    for k in range(1, 6):
        model_t4 = model_h100_per_fold * k * T4_MODEL_MULTIPLIER
        auc = EXP018_LB + EXP018_FOLD_DELTA[k]
        for ngpu in (1, 2):
            per_study = max(decode_kaggle, model_t4 / ngpu)
            sec = per_study * N_TEST
            rows.append((f"exp018 ({k}-fold, T4 x{ngpu})", k, ngpu, auc, sec, efficiency_score(auc, sec)))

    lines = ["# Efficiency-track table (2026-09-22)", "",
             f"maxAUC assumed **{ASSUMED_MAX_AUC}** (per this instruction; the deployed efficiency notebook's own "
             f"internal fold-selection logic still uses 0.92 — not changed here, see the recommendation below). "
             f"AUC anchored on each 5-fold config's real public LB (exp008 {EXP008_LB}, exp018 {EXP018_LB}); reduced-fold "
             f"AUC = anchor + a gold-58 all-subsets-averaged delta. Kaggle decode workers assumed = {KAGGLE_DECODE_WORKERS} "
             f"(the deployed efficiency notebook's own assumption); exp018's H100 profiling used "
             f"{PROFILED_DECODE_WORKERS} workers and is rescaled by that ratio "
             f"({EXP018_H100_DECODE_S_PER_STUDY:.3f}s -> {decode_kaggle:.3f}s/study).", "",
             "| config | folds | GPUs | AUC | projected seconds (T4) | projected minutes | Eff score (lower=better) |",
             "|---|---|---|---|---|---|---|"]
    for name, k, ngpu, auc, sec, eff in rows:
        lines.append(f"| {name} | {k} | {ngpu} | {auc:.4f} | {sec:.0f} | {sec / 60:.1f} | {eff:.4f} |")

    best = min(rows, key=lambda r: r[5])
    exp008_row = rows[0]
    lines += ["", f"**Efficiency-optimal: {best[0]}** (score {best[5]:.4f} vs exp008's {exp008_row[5]:.4f}).",
              "", "## Reading the table",
              f"- The AUC term barely moves with fold count (exp018 drops only {-EXP018_FOLD_DELTA[1]:.4f} macro going "
              f"5->1 fold, per the gold-58 subset check) while the runtime term moves a lot, so the formula strongly "
              f"favours FEWER folds here — the same shape exp008's own fold-count selection found, just more extreme "
              f"because exp018's window-MIL head already averages ~50 windows per study per fold, so one fold alone "
              f"is a fairly stable estimate.",
              f"- Even exp018's slowest row (5-fold, T4 x1, ~{rows[5][4] / 60:.0f} min) beats exp008 on the formula, "
              f"because the {EXP018_LB - EXP008_LB:+.3f} AUC gap is worth far more than the extra runtime costs at "
              f"this scale (the AUC term's sensitivity is 1/{-EFF_DENOM:.2f} ≈ {1 / -EFF_DENOM:.2f} per point of AUC, "
              f"vs 1/32400 ≈ {1/32400:.6f} per second of runtime).",
              "", "## Caveats",
              "- The T4 multiplier (3.5x) and the decode-worker rescaling are both carried over assumptions, not "
              "independently measured for CoAtNet-2 or for Kaggle's real per-competition CPU allocation — treat the "
              "seconds column as a plausible range, same caveat as every prior projection in this project.",
              "- AUC-by-fold-count uses gold-58 (n=58) deltas, which are themselves noisy; the table's ranking is "
              "robust to that noise only because the runtime term dominates the differences between exp018 rows.",
              "- Real Kaggle backend scored-run timing is opaque to us (established 2026-09-17, docs/EXPERIMENTS.md) "
              "— every 'projected seconds' figure here, for BOTH models, is a projection from H100 profiling, not a "
              "read of an actual scored run's wall clock.",
              "- If exp018 is staged as the new efficiency notebook, its own in-notebook calibration (like "
              "submit-efficiency.ipynb cell-6/7) should still pick the fold count live on Kaggle's actual hardware "
              "rather than trusting this table blindly — this table is what informs which candidate rungs to build "
              "the ladder from, and suggests raising ASSUMED_MAX_AUC to 0.95 in that notebook to match this analysis."]
    rep = "\n".join(lines) + "\n"
    print(rep)
    if a.out:
        a.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
