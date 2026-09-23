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

"""Build labels_v2_final.parquet: per-label choice among {Qwen-only, Llama-only, 50/50 blend},
selected by STRICT leave-one-out AUC on gold-58 (2026-09-07 strategy review, part B step 5).

For each of the 12 targets, three candidate calibrated predictors are LOO-evaluated on gold-58:
  - qwen_only:  Platt-calibrated Qwen probability alone
  - llama_only: Platt-calibrated Llama probability alone
  - blend:      mean of the two calibrated probabilities (each model's Platt scaler refit inside
                the same LOO fold as the other, so the blend's LOO score isn't leaking info the
                single-model LOO scores didn't also have access to)
Whichever of the three has the highest LOO AUC for that label is DEPLOYED for that label across
all 4407 studies (the deployed calibrator(s) are refit on all 58 gold points, same as
build_labels_v2.py — LOO is only used to CHOOSE the strategy and report its out-of-sample quality,
never as the deployed function itself).

Confidence weight: blend-selected labels get the same per-study agreement-based weight (floored)
as build_labels_v2.py; single-model-selected labels get a flat 1.0 (we've explicitly determined
that model is the best available source for this label — full trust, still floored downstream at
0.1 by the dataset loader regardless).

Usage:
    python scripts/build_labels_v2_final.py \
        --qwen artifacts/labels/qwen2.5-72b-instruct-awq_v2.parquet \
        --llama artifacts/labels/llama-3.3-70b-instruct-awq.parquet \
        --gold data/raw/train.csv \
        --out artifacts/labels/labels_v2_final.parquet \
        --out-report docs/label_blend_v2_final.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
MIN_CONFIDENCE = 0.1
MIN_LABEL_SCALE = 0.2


def load_gold(gold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gold_csv).rename(columns={"StudyInstanceUID": "study_uid"})
    return df.loc[df[TARGET_COLUMNS[0]].notna(), ["study_uid"] + TARGET_COLUMNS].set_index("study_uid")


def fit_platt(raw: np.ndarray, gold_binary: np.ndarray) -> LogisticRegression:
    # C=1e6: effectively unregularized, standard for 1D Platt scaling. sklearn's default L2
    # (C=1.0) was confirmed to break calibration entirely for low-base-rate labels (see
    # 2026-09-07 EXPERIMENTS.md entry: MCL/Lateral OA/Contusion never crossed 0.5 for ANY of the
    # 4407 studies under the default).
    clf = LogisticRegression(C=1e6)
    clf.fit(raw.reshape(-1, 1), gold_binary)
    return clf


def strict_loo_predictions(raw_qwen: np.ndarray, raw_llama: np.ndarray,
                            gold_binary: np.ndarray) -> dict[str, np.ndarray]:
    """Returns LOO predictions (one per gold study) for all three candidate strategies, each
    fold's calibrator(s) refit on the other 57 — a genuinely out-of-sample evaluation."""
    n = len(gold_binary)
    loo_qwen = np.zeros(n)
    loo_llama = np.zeros(n)
    loo_blend = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if len(set(gold_binary[mask])) < 2:
            loo_qwen[i] = raw_qwen[i]
            loo_llama[i] = raw_llama[i]
            loo_blend[i] = (raw_qwen[i] + raw_llama[i]) / 2
            continue
        qwen_clf = fit_platt(raw_qwen[mask], gold_binary[mask])
        llama_clf = fit_platt(raw_llama[mask], gold_binary[mask])
        cq = qwen_clf.predict_proba(raw_qwen[i:i + 1].reshape(-1, 1))[0, 1]
        cl = llama_clf.predict_proba(raw_llama[i:i + 1].reshape(-1, 1))[0, 1]
        loo_qwen[i] = cq
        loo_llama[i] = cl
        loo_blend[i] = (cq + cl) / 2
    return {"qwen_only": loo_qwen, "llama_only": loo_llama, "blend": loo_blend}


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/labels/labels_v2_final.parquet"))
    parser.add_argument("--out-report", type=Path, default=Path("docs/label_blend_v2_final.md"))
    args = parser.parse_args()

    qwen_df = pd.read_parquet(args.qwen).set_index("study_uid")
    llama_df = pd.read_parquet(args.llama).set_index("study_uid")
    gold_df = load_gold(args.gold)

    common_studies = qwen_df.index.intersection(llama_df.index)
    gold_studies = gold_df.index.intersection(common_studies)
    print(f"{len(common_studies)} studies total, {len(gold_studies)} in gold-58 overlap")

    result_rows = {"study_uid": common_studies}
    table_rows = []

    for target in TARGET_COLUMNS:
        gold_binary = gold_df.loc[gold_studies, target].astype(int).to_numpy()
        raw_qwen_gold = qwen_df.loc[gold_studies, f"{target}__probability"].to_numpy()
        raw_llama_gold = llama_df.loc[gold_studies, f"{target}__probability"].to_numpy()

        loo_preds = strict_loo_predictions(raw_qwen_gold, raw_llama_gold, gold_binary)
        loo_aucs = {strategy: safe_auc(gold_binary, preds) for strategy, preds in loo_preds.items()}
        chosen = max(loo_aucs, key=lambda k: (loo_aucs[k] if not np.isnan(loo_aucs[k]) else -1))

        raw_qwen_all = qwen_df.loc[common_studies, f"{target}__probability"].to_numpy()
        raw_llama_all = llama_df.loc[common_studies, f"{target}__probability"].to_numpy()

        qwen_clf_full = fit_platt(raw_qwen_gold, gold_binary)
        llama_clf_full = fit_platt(raw_llama_gold, gold_binary)
        qwen_cal_all = qwen_clf_full.predict_proba(raw_qwen_all.reshape(-1, 1))[:, 1]
        llama_cal_all = llama_clf_full.predict_proba(raw_llama_all.reshape(-1, 1))[:, 1]

        if chosen == "qwen_only":
            final_prob = qwen_cal_all
            confidence = np.ones(len(common_studies))
            source_weights = "qwen=1.0, llama=0.0"
        elif chosen == "llama_only":
            final_prob = llama_cal_all
            confidence = np.ones(len(common_studies))
            source_weights = "qwen=0.0, llama=1.0"
        else:
            final_prob = (qwen_cal_all + llama_cal_all) / 2.0
            per_study_agreement = np.clip(1.0 - np.abs(qwen_cal_all - llama_cal_all), MIN_CONFIDENCE, 1.0)
            label_corr = float(np.corrcoef(raw_qwen_all, raw_llama_all)[0, 1])
            label_scale = np.clip((label_corr + 1) / 2, MIN_LABEL_SCALE, 1.0)
            confidence = per_study_agreement * label_scale
            source_weights = "qwen=0.5, llama=0.5 (agreement-weighted confidence)"

        result_rows[f"{target}__probability"] = final_prob
        result_rows[f"{target}__confidence_weight"] = confidence
        result_rows[f"{target}__source"] = chosen

        table_rows.append({
            "target": target, "chosen": chosen, "source_weights": source_weights,
            "qwen_only_loo_auc": loo_aucs["qwen_only"], "llama_only_loo_auc": loo_aucs["llama_only"],
            "blend_loo_auc": loo_aucs["blend"], "chosen_loo_auc": loo_aucs[chosen],
            "mean_confidence": float(np.mean(confidence)),
        })

    out_df = pd.DataFrame(result_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    table_df = pd.DataFrame(table_rows)
    macro_chosen_loo = table_df["chosen_loo_auc"].mean()

    lines = ["# Label blend v2 FINAL — per-label strategy selection by strict LOO on gold-58\n"]
    lines.append(f"Macro LOO AUC of the CHOSEN per-label strategy: **{macro_chosen_loo:.3f}**\n")
    lines.append("| Target | Chosen | Source weights | Qwen-only LOO AUC | Llama-only LOO AUC | "
                  "Blend LOO AUC | Mean confidence |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in table_rows:
        lines.append(
            f"| {r['target']} | **{r['chosen']}** | {r['source_weights']} | "
            f"{r['qwen_only_loo_auc']:.3f} | {r['llama_only_loo_auc']:.3f} | {r['blend_loo_auc']:.3f} | "
            f"{r['mean_confidence']:.3f} |"
        )
    lines.append(f"\n{len(common_studies)} studies. All LOO numbers are strict leave-one-out "
                  f"(each held-out gold study's prediction comes from calibrator(s) fit on the "
                  f"OTHER 57 only) — this is what decided which strategy is deployed per label, "
                  f"not a post-hoc rationalization.\n")

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} and {args.out_report}")
    print(table_df.to_string(index=False))


if __name__ == "__main__":
    main()
