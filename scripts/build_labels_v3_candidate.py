"""Phase 4 blend candidate: labels_v3 = per-label weighted blend of {report label engine
(labels_v2_final), exp002 image model OOF} — CLAUDE.md §6 step 5 ("Round-2 noisy student: blend
image-model OOF probs with report probs"). NOT deployed/retrained on yet — this produces a
candidate parquet + a gold-58 sanity table for review before committing to a training round
(2026-09-08 instruction).

Direction is pre-registered per label from the 2026-09-08 exp002 eval (docs/EXPERIMENTS.md):
10 labels where exp002's gold-58 AUC already beat the label engine's own LOO AUC get an
"image-heavy" search grid (w_image in [0.5, 1.0]); ACL and MCL, where the report still wins
clearly, get a "report-heavy" grid (w_image in [0.0, 0.5]). Within its pre-registered direction,
the actual weight is chosen by strict LOO AUC on gold-58 (same rigor as
build_labels_v2_final.py): both sources are Platt-calibrated separately, refit inside each held-out
fold, so the LOO score is genuinely out-of-sample — this only picks the MAGNITUDE, not whether to
override the pre-registered direction (protects against overfitting the direction to noise on
n=58).

Usage:
    python scripts/build_labels_v3_candidate.py \
        --report artifacts/labels/labels_v2_final.parquet \
        --exp002-dir artifacts/exp002_dinov2s_slot_v2labels \
        --gold data/raw/train.csv \
        --out artifacts/labels/labels_v3_candidate.parquet \
        --out-report docs/label_blend_v3_candidate.md
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

# Pre-registered direction per label (docs/EXPERIMENTS.md 2026-09-08 exp002 eval: 10/12 labels had
# image gold58 AUC > label engine LOO AUC; only ACL/MCL still favored the report).
REPORT_HEAVY_LABELS = {"ACL", "MCL"}
IMAGE_HEAVY_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REPORT_HEAVY_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def load_gold(gold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gold_csv).rename(columns={"StudyInstanceUID": "study_uid"})
    return df.loc[df[TARGET_COLUMNS[0]].notna(), ["study_uid"] + TARGET_COLUMNS].set_index("study_uid")


def load_image_oof(exp_dir: Path) -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/oof.parquet"))]
    assert len(parts) == 5, f"expected 5 fold oof.parquet under {exp_dir}, found {len(parts)}"
    return pd.concat(parts, ignore_index=True).set_index("study_uid")


def load_image_gold(exp_dir: Path) -> pd.DataFrame:
    """Fold-ensembled (mean across the 5 folds' models) gold-58 predictions — each fold's model
    never trained on gold studies, so averaging the 5 independently-trained models here is a
    legitimate ensemble, not a leak."""
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/gold58_predictions.parquet"))]
    assert len(parts) == 5, f"expected 5 fold gold58_predictions.parquet under {exp_dir}, found {len(parts)}"
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby("study_uid")[TARGET_COLUMNS].mean()


def fit_platt(raw: np.ndarray, gold_binary: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(C=1e6)  # effectively unregularized — see build_labels_v2_final.py
    clf.fit(raw.reshape(-1, 1), gold_binary)
    return clf


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else float("nan")


def strict_loo_blend_search(raw_report: np.ndarray, raw_image: np.ndarray, gold_binary: np.ndarray,
                             weight_grid: list[float]) -> tuple[float, dict[float, float]]:
    """For each candidate weight, LOO-evaluate the blend (both sources' Platt calibrators refit
    per held-out fold). Returns (best_weight, {weight: loo_auc})."""
    n = len(gold_binary)
    loo_report = np.zeros(n)
    loo_image = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if len(set(gold_binary[mask])) < 2:
            loo_report[i] = raw_report[i]
            loo_image[i] = raw_image[i]
            continue
        report_clf = fit_platt(raw_report[mask], gold_binary[mask])
        image_clf = fit_platt(raw_image[mask], gold_binary[mask])
        loo_report[i] = report_clf.predict_proba(raw_report[i:i + 1].reshape(-1, 1))[0, 1]
        loo_image[i] = image_clf.predict_proba(raw_image[i:i + 1].reshape(-1, 1))[0, 1]

    aucs = {}
    for w in weight_grid:
        blended = w * loo_image + (1 - w) * loo_report
        aucs[w] = safe_auc(gold_binary, blended)
    best_w = max(aucs, key=lambda k: (aucs[k] if not np.isnan(aucs[k]) else -1))
    return best_w, aucs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exp002-dir", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/labels/labels_v3_candidate.parquet"))
    parser.add_argument("--out-report", type=Path, default=Path("docs/label_blend_v3_candidate.md"))
    args = parser.parse_args()

    report_df = pd.read_parquet(args.report).set_index("study_uid")
    image_oof_df = load_image_oof(args.exp002_dir)
    image_gold_df = load_image_gold(args.exp002_dir)
    gold_df = load_gold(args.gold)

    # Full deployment set: report engine covers all 4407; image covers OOF (non-gold, 4349) plus
    # the fold-ensembled gold-58 predictions — union is the same 4407 studies.
    image_all_df = pd.concat([
        image_oof_df[TARGET_COLUMNS],
        image_gold_df[TARGET_COLUMNS],
    ])
    common_studies = report_df.index.intersection(image_all_df.index)
    gold_studies = gold_df.index.intersection(common_studies)
    print(f"{len(common_studies)} studies total (report ^ image coverage), "
          f"{len(gold_studies)} in gold-58 overlap")

    result_rows = {"study_uid": common_studies}
    table_rows = []

    for target in TARGET_COLUMNS:
        grid = REPORT_HEAVY_GRID if target in REPORT_HEAVY_LABELS else IMAGE_HEAVY_GRID
        direction = "report-heavy" if target in REPORT_HEAVY_LABELS else "image-heavy"

        gold_binary = gold_df.loc[gold_studies, target].astype(int).to_numpy()
        raw_report_gold = report_df.loc[gold_studies, f"{target}__probability"].to_numpy()
        raw_image_gold = image_gold_df.loc[gold_studies, target].to_numpy()

        best_w, weight_aucs = strict_loo_blend_search(raw_report_gold, raw_image_gold, gold_binary, grid)
        report_only_auc = weight_aucs.get(0.0)
        image_only_auc = weight_aucs.get(1.0)

        raw_report_all = report_df.loc[common_studies, f"{target}__probability"].to_numpy()
        raw_image_all = image_all_df.loc[common_studies, target].to_numpy()

        report_clf_full = fit_platt(raw_report_gold, gold_binary)
        image_clf_full = fit_platt(raw_image_gold, gold_binary)
        report_cal_all = report_clf_full.predict_proba(raw_report_all.reshape(-1, 1))[:, 1]
        image_cal_all = image_clf_full.predict_proba(raw_image_all.reshape(-1, 1))[:, 1]

        final_prob = best_w * image_cal_all + (1 - best_w) * report_cal_all
        per_study_agreement = np.clip(1.0 - np.abs(image_cal_all - report_cal_all), MIN_CONFIDENCE, 1.0)
        label_corr = float(np.corrcoef(raw_report_all, raw_image_all)[0, 1])
        label_scale = np.clip((label_corr + 1) / 2, MIN_LABEL_SCALE, 1.0)
        confidence = per_study_agreement * label_scale

        result_rows[f"{target}__probability"] = final_prob
        result_rows[f"{target}__confidence_weight"] = confidence
        result_rows[f"{target}__source"] = f"blend_w_image={best_w:.1f}_{direction}"

        table_rows.append({
            "target": target, "direction": direction, "chosen_w_image": best_w,
            "report_only_loo_auc": report_only_auc, "image_only_loo_auc": image_only_auc,
            "chosen_blend_loo_auc": weight_aucs[best_w], "mean_confidence": float(np.mean(confidence)),
            "grid_aucs": {round(w, 1): round(a, 4) if not np.isnan(a) else None for w, a in weight_aucs.items()},
        })

    out_df = pd.DataFrame(result_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    table_df = pd.DataFrame(table_rows)
    macro_chosen_loo = table_df["chosen_blend_loo_auc"].mean()
    macro_report_only = table_df["report_only_loo_auc"].mean()
    macro_image_only = table_df["image_only_loo_auc"].mean()

    lines = ["# Label blend v3 CANDIDATE — image (exp002 OOF) + report (labels_v2_final) blend, "
             "per-label weight chosen by strict LOO on gold-58\n"]
    lines.append("**NOT DEPLOYED — candidate for review, no retrain has happened on this yet.**\n")
    lines.append(f"Macro LOO AUC: chosen blend **{macro_chosen_loo:.3f}** vs report-only "
                 f"{macro_report_only:.3f} vs image-only {macro_image_only:.3f}\n")
    lines.append("| Target | Direction | w_image | Report-only LOO | Image-only LOO | "
                  "Chosen-blend LOO | Mean confidence |")
    lines.append("|---|---|---|---|---|---|---|")
    def fmt(v):
        return f"{v:.3f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "n/a (not in grid)"

    for r in table_rows:
        lines.append(
            f"| {r['target']} | {r['direction']} | **{r['chosen_w_image']:.1f}** | "
            f"{fmt(r['report_only_loo_auc'])} | {fmt(r['image_only_loo_auc'])} | "
            f"{r['chosen_blend_loo_auc']:.3f} | {r['mean_confidence']:.3f} |"
        )
    lines.append(f"\n{len(common_studies)} studies. Strict LOO throughout (each held-out gold "
                 f"study's blend prediction comes from calibrators fit on the OTHER 57 only). "
                 f"Weight search was constrained to the pre-registered direction per label "
                 f"(image-heavy [0.5-1.0] for the 10 labels where exp002's gold58 AUC beat the "
                 f"label engine's own LOO AUC; report-heavy [0.0-0.5] for ACL/MCL) — LOO only "
                 f"picked the magnitude within that direction, not whether to flip it.\n")
    lines.append("## Full weight-grid LOO AUC per label\n")
    lines.append("| Target | " + " | ".join(f"w={w}" for w in sorted(set(IMAGE_HEAVY_GRID + REPORT_HEAVY_GRID))) + " |")
    lines.append("|---|" + "---|" * len(set(IMAGE_HEAVY_GRID + REPORT_HEAVY_GRID)))
    for r in table_rows:
        cells = [f"{r['grid_aucs'].get(w, ''):.3f}" if r['grid_aucs'].get(w) is not None else ""
                 for w in sorted(set(IMAGE_HEAVY_GRID + REPORT_HEAVY_GRID))]
        lines.append(f"| {r['target']} | " + " | ".join(cells) + " |")

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} and {args.out_report}")
    print(table_df.drop(columns="grid_aucs").to_string(index=False))


if __name__ == "__main__":
    main()
