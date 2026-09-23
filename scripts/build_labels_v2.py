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

"""Build labels_v2.parquet: per-label Platt-calibrated (gold-58 LOO-validated) blend of the
Qwen and Llama label-engine runs, with per-study confidence weighted by inter-model agreement
(2026-09-07 strategy review step 6).

For each of the 12 targets:
  1. Fit a 1D Platt scaler (logistic regression on the model's raw probability -> gold binary
     label) separately for Qwen and Llama, using ALL 58 gold studies — this is the calibrator
     actually applied to the full 4407-study dataset.
  2. Validate that calibrator with leave-one-out cross-validation on gold-58 (refit on the other
     57 each time, predict the held-out study) — reported as a diagnostic (LOO Brier score / AUC),
     not the deployed function, since with n=58 a straight non-LOO evaluation would be optimistic.
  3. Blend: per-study probability = mean of the two calibrated probabilities. Per-study confidence
     = 1 - |calibrated_qwen - calibrated_llama| (full agreement -> weight 1, total disagreement ->
     floored at a minimum so a disagreeing label still contributes some signal, not zero).
  4. That per-study confidence is further scaled by a per-LABEL global agreement factor (derived
     from the Qwen/Llama raw-probability correlation across all 4407 studies) so labels the two
     LLMs systematically disagree on end up down-weighted overall, not just noisy per study.

Usage:
    python scripts/build_labels_v2.py \
        --qwen artifacts/labels/qwen2.5-72b-instruct-awq_v2.parquet \
        --llama artifacts/labels/llama-3.3-70b-instruct-awq.parquet \
        --gold data/raw/train.csv \
        --out artifacts/labels/labels_v2.parquet \
        --out-report docs/label_blend_v2.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
MIN_CONFIDENCE = 0.1  # floor so a fully-disagreeing label still carries some training signal
MIN_LABEL_SCALE = 0.2  # floor for the per-label global agreement scaling


def load_gold(gold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gold_csv).rename(columns={"StudyInstanceUID": "study_uid"})
    return df.loc[df[TARGET_COLUMNS[0]].notna(), ["study_uid"] + TARGET_COLUMNS].reset_index(drop=True)


def fit_platt(raw: np.ndarray, gold_binary: np.ndarray) -> LogisticRegression:
    # sklearn's default L2 regularization (C=1.0) is inappropriate for classic 1D Platt scaling —
    # confirmed empirically: with the default, MCL's fitted decision boundary required a raw
    # probability of 1.09 (impossible, max is 1.0) to cross calibrated 0.5, so NOT ONE of the 4407
    # studies ever landed positive for MCL/Lateral OA/Contusion after calibration, regardless of
    # how confident the LLM was. Platt scaling is meant to be an (effectively) unregularized 1D
    # logistic fit; C=1e6 approximates that while keeping the same well-tested sklearn solver.
    clf = LogisticRegression(C=1e6)
    clf.fit(raw.reshape(-1, 1), gold_binary)
    return clf


def loo_calibration_diagnostic(raw: np.ndarray, gold_binary: np.ndarray) -> tuple[float, float]:
    n = len(raw)
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if len(set(gold_binary[mask])) < 2:
            loo_preds[i] = raw[i]  # can't fit a 1-class logistic regression, fall back to raw
            continue
        clf = fit_platt(raw[mask], gold_binary[mask])
        loo_preds[i] = clf.predict_proba(raw[i:i + 1].reshape(-1, 1))[0, 1]
    brier = brier_score_loss(gold_binary, loo_preds)
    auc = roc_auc_score(gold_binary, loo_preds) if len(set(gold_binary)) > 1 else float("nan")
    return brier, auc


def calibrate_full(clf: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    return clf.predict_proba(raw.reshape(-1, 1))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/labels/labels_v2.parquet"))
    parser.add_argument("--out-report", type=Path, default=Path("docs/label_blend_v2.md"))
    args = parser.parse_args()

    qwen_df = pd.read_parquet(args.qwen).set_index("study_uid")
    llama_df = pd.read_parquet(args.llama).set_index("study_uid")
    gold_df = load_gold(args.gold).set_index("study_uid")

    common_studies = qwen_df.index.intersection(llama_df.index)
    print(f"{len(common_studies)} studies in both Qwen and Llama outputs "
          f"(Qwen={len(qwen_df)}, Llama={len(llama_df)})")

    result_rows = {"study_uid": common_studies}
    report_lines = ["# Label blend v2 — Qwen + Llama, Platt-calibrated on gold-58\n"]
    report_lines.append("| Target | Qwen LOO AUC | Llama LOO AUC | Qwen LOO Brier | Llama LOO Brier | "
                         "Inter-LLM correlation (all 4407) | Label agreement scale |")
    report_lines.append("|---|---|---|---|---|---|---|")

    for target in TARGET_COLUMNS:
        gold_studies = gold_df.index.intersection(qwen_df.index).intersection(llama_df.index)
        gold_binary = gold_df.loc[gold_studies, target].astype(int).to_numpy()
        qwen_raw_gold = qwen_df.loc[gold_studies, f"{target}__probability"].to_numpy()
        llama_raw_gold = llama_df.loc[gold_studies, f"{target}__probability"].to_numpy()

        qwen_clf = fit_platt(qwen_raw_gold, gold_binary)
        llama_clf = fit_platt(llama_raw_gold, gold_binary)
        qwen_loo_brier, qwen_loo_auc = loo_calibration_diagnostic(qwen_raw_gold, gold_binary)
        llama_loo_brier, llama_loo_auc = loo_calibration_diagnostic(llama_raw_gold, gold_binary)

        qwen_raw_all = qwen_df.loc[common_studies, f"{target}__probability"].to_numpy()
        llama_raw_all = llama_df.loc[common_studies, f"{target}__probability"].to_numpy()
        qwen_cal = calibrate_full(qwen_clf, qwen_raw_all)
        llama_cal = calibrate_full(llama_clf, llama_raw_all)

        blended_prob = (qwen_cal + llama_cal) / 2.0
        per_study_agreement = 1.0 - np.abs(qwen_cal - llama_cal)
        per_study_agreement = np.clip(per_study_agreement, MIN_CONFIDENCE, 1.0)

        label_corr = float(np.corrcoef(qwen_raw_all, llama_raw_all)[0, 1])
        label_scale = np.clip((label_corr + 1) / 2, MIN_LABEL_SCALE, 1.0)  # map [-1,1] -> [0,1], floored

        final_confidence = per_study_agreement * label_scale

        result_rows[f"{target}__probability"] = blended_prob
        result_rows[f"{target}__confidence_weight"] = final_confidence
        result_rows[f"{target}__qwen_calibrated"] = qwen_cal
        result_rows[f"{target}__llama_calibrated"] = llama_cal

        report_lines.append(
            f"| {target} | {qwen_loo_auc:.3f} | {llama_loo_auc:.3f} | {qwen_loo_brier:.3f} | "
            f"{llama_loo_brier:.3f} | {label_corr:.3f} | {label_scale:.3f} |"
        )

    out_df = pd.DataFrame(result_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    report_lines.append(f"\n{len(common_studies)} studies blended. Confidence weight floors: "
                         f"per-study >= {MIN_CONFIDENCE}, per-label scale >= {MIN_LABEL_SCALE} "
                         f"(a fully-disagreeing label still contributes some training signal, "
                         f"never fully zeroed).\n")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote {args.out} and {args.out_report}")


if __name__ == "__main__":
    main()
