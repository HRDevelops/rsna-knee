"""exp009 (ConvNeXt-V2-Base diversity round) pre-registered acceptance test —
configs/exp009_convnextv2b_slot_v3labels.yaml, 2026-09-10. Bar: OOF macro-AUC vs labels_v2_final
>= 0.855 AND mean per-label OOF-prediction correlation with exp006 < 0.90. Greedy ensemble
re-selection only runs if both hold.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

OOF_AUC_BAR = 0.855
CORR_BAR = 0.90


def load_summaries(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(exp_dir.glob("fold*")):
        csv_path = fold_dir / "summary.csv"
        if csv_path.exists():
            rows.append(pd.read_csv(csv_path).iloc[0])
    return pd.DataFrame(rows)


def load_full_oof(exp_dir: Path) -> pd.DataFrame | None:
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/oof.parquet"))]
    return pd.concat(parts, ignore_index=True) if len(parts) == 5 else None


def macro_auc_vs_labels(oof_df: pd.DataFrame, labels_parquet: Path) -> tuple[float, dict]:
    truth_df = pd.read_parquet(labels_parquet)
    truth = {
        r["study_uid"]: np.array([1.0 if r[f"{c}__probability"] >= 0.5 else 0.0 for c in TARGETS])
        for r in truth_df.to_dict("records")
    }
    matched = oof_df[oof_df["study_uid"].isin(truth)].reset_index(drop=True)
    truth_matrix = np.stack([truth[uid] for uid in matched["study_uid"]])
    per_label = {}
    for i, col in enumerate(TARGETS):
        y_true = truth_matrix[:, i]
        y_score = matched[col].to_numpy()
        per_label[col] = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else float("nan")
    valid = [v for v in per_label.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan"), per_label


def per_label_correlation(oof_a: pd.DataFrame, oof_b: pd.DataFrame) -> tuple[float, dict]:
    merged = oof_a.merge(oof_b, on="study_uid", suffixes=("_a", "_b"))
    per_label = {}
    for c in TARGETS:
        ca, cb = f"{c}_a", f"{c}_b"
        if ca in merged.columns and cb in merged.columns:
            per_label[c] = float(np.corrcoef(merged[ca], merged[cb])[0, 1])
    valid = list(per_label.values())
    return float(np.mean(valid)) if valid else float("nan"), per_label


if __name__ == "__main__":
    rsna_root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent))
    root = rsna_root / "artifacts"
    labels_v2_final = root / "labels" / "labels_v2_final.parquet"

    exp009_dir = root / "exp009_convnextv2b_slot_v3labels"
    exp006_dir = root / "exp006_dinov2b_slot_v3labels"

    exp009_summary = load_summaries(exp009_dir)
    if len(exp009_summary) < 5:
        print(f"exp009 incomplete ({len(exp009_summary)}/5 folds) — not ready for acceptance test yet.")
        raise SystemExit(0)

    exp009_oof = load_full_oof(exp009_dir)
    exp006_oof = load_full_oof(exp006_dir)

    exp009_oof_v2final, exp009_per_label_v2final = macro_auc_vs_labels(exp009_oof, labels_v2_final)
    exp009_gold58 = exp009_summary["gold58_macro_auc"].mean()
    exp009_val_native = exp009_summary["val_macro_auc"].mean()  # native metric, vs labels_v3_candidate

    corr_mean, corr_per_label = per_label_correlation(exp009_oof, exp006_oof)

    print("=== exp009 vs. acceptance bar ===")
    print(f"OOF macro-AUC vs labels_v2_final: {exp009_oof_v2final:.4f}  (bar: >= {OOF_AUC_BAR})")
    print(f"OOF macro-AUC native (vs labels_v3_candidate, training-time metric): {exp009_val_native:.4f}")
    print(f"Gold-58 macro-AUC (tripwire): {exp009_gold58:.4f}")
    print(f"Mean per-label OOF-prediction correlation with exp006: {corr_mean:.4f}  (bar: < {CORR_BAR})")

    auc_pass = exp009_oof_v2final >= OOF_AUC_BAR
    corr_pass = corr_mean < CORR_BAR
    print(f"\nOOF criterion: {'PASS' if auc_pass else 'FAIL'}")
    print(f"Correlation criterion: {'PASS' if corr_pass else 'FAIL'}")

    if auc_pass and corr_pass:
        verdict = "CLEARS THE BAR — run greedy ensemble re-selection."
    else:
        verdict = "DOES NOT CLEAR THE BAR — no re-selection."
    print(f"\nVERDICT: {verdict}")

    print("\n=== Per-label: OOF vs labels_v2_final, and correlation with exp006 ===")
    per_label_df = pd.DataFrame({
        "label": TARGETS,
        "exp009_OOF_v2final": [exp009_per_label_v2final[t] for t in TARGETS],
        "corr_with_exp006": [corr_per_label.get(t, float("nan")) for t in TARGETS],
    })
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 4)
    print(per_label_df.to_string(index=False))

    out_path = root / "exp009_acceptance_report.csv"
    per_label_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
