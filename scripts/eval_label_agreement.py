"""Score LLM-extracted labels against the 58 gold (image-reviewed) studies (EXECUTION_PLAN 1.4).

This is the tripwire for the whole label-engine workstream (CLAUDE.md §2, §7): the 58 gold
studies are a validation set for the LABELER, never training fodder. This script produces:
  1. a per-label agreement table (accuracy, F1, Cohen's kappa, ROC-AUC) for each model passed in,
     plus an ensemble-mean row if more than one model is given;
  2. an error ledger (every gold/prediction disagreement, with the model's evidence quote) to
     drive rubric iteration per CLAUDE.md §6 step 3-4.

Usage:
    python scripts/eval_label_agreement.py \
        --labels artifacts/labels/qwen2.5-72b-awq.parquet artifacts/labels/llama-3.3-70b.parquet \
        --gold data/raw/train.csv \
        --out docs/gold_agreement_v1.md \
        --out-ledger artifacts/labels/error_ledger_v1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

DECISION_THRESHOLD = 0.5  # binarization point for accuracy/F1/kappa only; AUC uses raw probs.


def load_gold(gold_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gold_csv)
    df = df.rename(columns={"StudyInstanceUID": "study_uid"})
    gold_mask = df[TARGET_COLUMNS[0]].notna()
    gold = df.loc[gold_mask, ["study_uid", "Report"] + TARGET_COLUMNS].copy()
    for col in TARGET_COLUMNS:
        gold[col] = gold[col].astype(float)
    return gold.reset_index(drop=True)


def load_model_labels(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def model_probs_for_target(model_df: pd.DataFrame, target: str) -> pd.Series:
    return model_df.set_index("study_uid")[f"{target}__probability"]


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return None


def score_one_model(gold: pd.DataFrame, model_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        probs = model_probs_for_target(model_df, target)
        merged = gold[["study_uid", target]].merge(
            probs.rename("prob"), left_on="study_uid", right_index=True, how="inner"
        )
        n_missing = len(gold) - len(merged)
        y_true = merged[target].to_numpy()
        y_prob = merged["prob"].to_numpy()
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        rows.append({
            "model": model_name,
            "target": target,
            "n_gold": len(gold),
            "n_matched": len(merged),
            "n_missing": n_missing,
            "gold_positive_rate": y_true.mean() if len(y_true) else np.nan,
            "pred_positive_rate": y_pred.mean() if len(y_pred) else np.nan,
            "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
            "f1": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
            "cohen_kappa": cohen_kappa_score(y_true, y_pred) if len(y_true) > 0 and len(np.unique(y_true)) > 1 else np.nan,
            "roc_auc": safe_auc(y_true, y_prob),
        })
    return pd.DataFrame(rows)


def build_ensemble_probs(gold: pd.DataFrame, model_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Study x target mean probability across all given models (simple ensemble)."""
    frames = []
    for target in TARGET_COLUMNS:
        per_model = []
        for name, df in model_dfs.items():
            per_model.append(model_probs_for_target(df, target).rename(name))
        combined = pd.concat(per_model, axis=1)
        frames.append(combined.mean(axis=1).rename(f"{target}__probability"))
    ensemble = pd.concat(frames, axis=1)
    ensemble.index.name = "study_uid"
    return ensemble.reset_index()


def build_error_ledger(gold: pd.DataFrame, model_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        prob_col = f"{target}__probability"
        present_col = f"{target}__present"
        evidence_col = f"{target}__evidence_quote"
        severity_col = f"{target}__severity_phrase"
        merged = gold[["study_uid", "Report", target]].merge(
            model_df[["study_uid", prob_col, present_col, evidence_col, severity_col]],
            on="study_uid", how="inner",
        )
        pred = (merged[prob_col] >= DECISION_THRESHOLD).astype(int)
        disagree = merged[pred != merged[target].astype(int)]
        for _, row in disagree.iterrows():
            rows.append({
                "model": model_name,
                "target": target,
                "study_uid": row["study_uid"],
                "gold_label": int(row[target]),
                "predicted_present": row[present_col],
                "predicted_probability": row[prob_col],
                "severity_phrase": row[severity_col],
                "evidence_quote": row[evidence_col],
                "report_excerpt": str(row["Report"])[:400],
            })
    return pd.DataFrame(rows)


def write_markdown_report(all_scores: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Gold-58 label agreement\n"]
    for model_name, group in all_scores.groupby("model"):
        macro_auc = group["roc_auc"].mean(skipna=True)
        macro_kappa = group["cohen_kappa"].mean(skipna=True)
        lines.append(f"## {model_name}\n")
        lines.append(f"Macro ROC-AUC vs gold-58: **{macro_auc:.3f}** "
                      f"(macro Cohen's kappa: {macro_kappa:.3f})\n")
        lines.append("| Target | n | Gold pos rate | Pred pos rate | Accuracy | F1 | Kappa | ROC-AUC |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in group.sort_values("roc_auc", na_position="first").iterrows():
            auc_str = f"{r['roc_auc']:.3f}" if pd.notna(r["roc_auc"]) else "n/a"
            kappa_str = f"{r['cohen_kappa']:.3f}" if pd.notna(r["cohen_kappa"]) else "n/a"
            lines.append(
                f"| {r['target']} | {r['n_matched']}/{r['n_gold']} | "
                f"{r['gold_positive_rate']:.2f} | {r['pred_positive_rate']:.2f} | "
                f"{r['accuracy']:.3f} | {r['f1']:.3f} | {kappa_str} | {auc_str} |"
            )
        lines.append("")
        below_threshold = group[group["roc_auc"] < 0.80]
        if len(below_threshold):
            flagged = ", ".join(below_threshold["target"].tolist())
            lines.append(f"**Below the 0.80 gold-agreement exit criterion (CLAUDE.md §6 step 1.5): "
                          f"{flagged}.** Expected candidates: Synovitis, Contusion (reports "
                          f"under-specify these) — anything else here needs a rubric fix.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, nargs="+", required=True,
                         help="One or more label_reports.py output parquets (one per model)")
    parser.add_argument("--gold", type=Path, required=True, help="train.csv (has the 58 gold rows)")
    parser.add_argument("--out", type=Path, default=Path("docs/gold_agreement.md"))
    parser.add_argument("--out-ledger", type=Path, default=Path("artifacts/labels/error_ledger.csv"))
    args = parser.parse_args()

    gold = load_gold(args.gold)
    print(f"{len(gold)} gold-58 studies loaded.")

    model_dfs = {p.stem: load_model_labels(p) for p in args.labels}

    all_scores = []
    all_ledgers = []
    for name, df in model_dfs.items():
        scores = score_one_model(gold, df, name)
        all_scores.append(scores)
        all_ledgers.append(build_error_ledger(gold, df, name))

    if len(model_dfs) > 1:
        ensemble_df = build_ensemble_probs(gold, model_dfs)
        # ensemble has no present/evidence columns; error ledger skipped for it, scoring still works
        ensemble_df_full = ensemble_df.copy()
        for target in TARGET_COLUMNS:
            ensemble_df_full[f"{target}__present"] = np.where(
                ensemble_df_full[f"{target}__probability"] >= DECISION_THRESHOLD, "yes", "no"
            )
            ensemble_df_full[f"{target}__evidence_quote"] = ""
            ensemble_df_full[f"{target}__severity_phrase"] = ""
        all_scores.append(score_one_model(gold, ensemble_df_full, "ensemble_mean"))
        all_ledgers.append(build_error_ledger(gold, ensemble_df_full, "ensemble_mean"))

    scores_df = pd.concat(all_scores, ignore_index=True)
    ledger_df = pd.concat(all_ledgers, ignore_index=True)

    write_markdown_report(scores_df, args.out)
    args.out_ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger_df.to_csv(args.out_ledger, index=False)

    print(f"Wrote agreement report to {args.out}")
    print(f"Wrote {len(ledger_df)} disagreement rows to {args.out_ledger}")


if __name__ == "__main__":
    main()
