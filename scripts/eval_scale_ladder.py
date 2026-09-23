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

"""Scale-ladder comparison: exp002 (DINOv2-S) vs exp003 (DINOv2-B) vs exp004 (DINOv3-S, archived
dropped in EXPERIMENTS.md 2026-09-08 — kept here only as a placeholder row if its artifacts ever
reappear), per the 2026-09-08 decision rule: keep a variant only if OOF macro gain >= +0.004, or a
weak label (Synovitis/Contusion/MCL/Lateral OA) gains >= +0.015 with no OOF macro loss.

Also produces an exp002+exp003 ensemble read (rank-mean AND prob-mean, CLAUDE.md §11: "average of
per-model per-label ranks... implement both, compare on OOF") to see if DINOv2-B adds diversity
value even if its solo macro doesn't clear the keep bar on its own.

Folds/experiments with no summary.csv (incomplete run) are silently skipped so this can be re-run
as experiments land rather than requiring everything to be done at once.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WEAK_LABELS = ["Synovitis", "Contusion", "MCL", "Lateral OA"]

EXPERIMENTS = {
    "exp002_dinov2s_slot_v2labels": "exp002 (DINOv2-S, baseline)",
    "exp003_dinov2b_slot_v2labels": "exp003 (DINOv2-B)",
    "exp004_dinov3s_slot_v2labels": "exp004 (DINOv3-S, DROPPED — see EXPERIMENTS.md)",
}

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


def load_full_oof(exp_dir: Path) -> pd.DataFrame | None:
    """Concatenate every fold's oof.parquet — each fold covers disjoint studies (its own val
    split), so the union is one full non-gold OOF set with no duplicates."""
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/oof.parquet"))]
    return pd.concat(parts, ignore_index=True) if len(parts) == 5 else None


def load_gold_fold_ensemble(exp_dir: Path) -> pd.DataFrame | None:
    """Average each fold's gold-58 predictions together (fold-ensemble) before cross-model
    blending — each fold's model saw a different train split, so averaging across folds first is
    the correct within-experiment ensemble, matching how the notebook ensembles at inference."""
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/gold58_predictions.parquet"))]
    if len(parts) != 5:
        return None
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby("study_uid", as_index=False)[TARGETS].mean()


def load_truth(rsna_root: Path, labels_parquet: Path) -> tuple[dict, dict]:
    gold_df = pd.read_csv(rsna_root / "data" / "raw" / "train.csv").rename(
        columns={"StudyInstanceUID": "study_uid"}
    )
    gold_df = gold_df[gold_df[TARGETS[0]].notna()]
    gold_truth = {r["study_uid"]: np.array([float(r[c]) for c in TARGETS]) for r in gold_df.to_dict("records")}

    soft_df = pd.read_parquet(labels_parquet)
    val_truth = {
        r["study_uid"]: np.array([1.0 if r[f"{c}__probability"] >= 0.5 else 0.0 for c in TARGETS])
        for r in soft_df.to_dict("records")
    }
    return gold_truth, val_truth


def macro_auc(pred_df: pd.DataFrame, truth: dict) -> tuple[float, dict]:
    matched = pred_df[pred_df["study_uid"].isin(truth)].reset_index(drop=True)
    truth_matrix = np.stack([truth[uid] for uid in matched["study_uid"]])
    per_label = {}
    for i, col in enumerate(TARGETS):
        y_true = truth_matrix[:, i]
        y_score = matched[col].to_numpy()
        per_label[col] = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else float("nan")
    valid = [v for v in per_label.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan"), per_label


def rank_mean_blend(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    merged = df_a.merge(df_b, on="study_uid", suffixes=("_a", "_b"))
    out = merged[["study_uid"]].copy()
    for c in TARGETS:
        rank_a = merged[f"{c}_a"].rank(pct=True)
        rank_b = merged[f"{c}_b"].rank(pct=True)
        out[c] = (rank_a + rank_b) / 2
    return out


def prob_mean_blend(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    merged = df_a.merge(df_b, on="study_uid", suffixes=("_a", "_b"))
    out = merged[["study_uid"]].copy()
    for c in TARGETS:
        out[c] = (merged[f"{c}_a"] + merged[f"{c}_b"]) / 2
    return out


def macro_row(df: pd.DataFrame, label: str) -> dict:
    row = {
        "experiment": label, "n_folds": len(df),
        "OOF_macro": df["val_macro_auc"].mean() if len(df) else float("nan"),
        "gold58_macro": df["gold58_macro_auc"].mean() if len(df) else float("nan"),
    }
    for t in WEAK_LABELS:
        row[f"OOF_{t}"] = df[f"val_{t}"].mean() if len(df) else float("nan")
    return row


def per_label_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        row = {"label": t, "label_engine_LOO_AUC": LABEL_ENGINE_LOO_AUC[t]}
        for exp_name, df in dfs.items():
            row[f"{exp_name}_OOF"] = df[f"val_{t}"].mean() if len(df) else float("nan")
            row[f"{exp_name}_gold58"] = df[f"gold_{t}"].mean() if len(df) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rsna_root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent))
    root = rsna_root / "artifacts"
    pd.set_option("display.width", 220)
    pd.set_option("display.precision", 4)

    dfs = {}
    rows = []
    for exp_name, label in EXPERIMENTS.items():
        df = load_summaries(root / exp_name)
        dfs[exp_name] = df
        rows.append(macro_row(df, label))
        status = f"{len(df)}/5 folds have final results" if len(df) < 5 else "complete (5/5)"
        print(f"{label}: {status}")

    summary = pd.DataFrame(rows)
    print("\n=== Macro comparison ===")
    print(summary.to_string(index=False))

    baseline = summary.iloc[0]
    if baseline["n_folds"] == 5:
        print("\n=== Decision rule vs exp002 baseline (keep if OOF macro >= +0.004, "
              "or a weak label >= +0.015 with no OOF macro loss) ===")
        for _, row in summary.iloc[1:].iterrows():
            if row["n_folds"] < 5:
                print(f"{row['experiment']}: incomplete ({int(row['n_folds'])}/5 folds) — skipping decision")
                continue
            macro_gain = row["OOF_macro"] - baseline["OOF_macro"]
            weak_gains = {t: row[f"OOF_{t}"] - baseline[f"OOF_{t}"] for t in WEAK_LABELS}
            best_weak_label, best_weak_gain = max(weak_gains.items(), key=lambda kv: kv[1])
            keep = macro_gain >= 0.004 or (best_weak_gain >= 0.015 and macro_gain >= 0)
            print(f"{row['experiment']}: macro_gain={macro_gain:+.4f}, "
                  f"best weak-label gain={best_weak_label} {best_weak_gain:+.4f} -> "
                  f"{'KEEP' if keep else 'DROP'}")

    print("\n=== Per-label: exp002 vs exp003 vs exp004 (OOF / gold-58) vs label engine LOO ===")
    print(per_label_table(dfs).to_string(index=False))

    out_path = root / "scale_ladder_report.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # exp002 + exp003 ensemble (only once both have full 5-fold OOF)
    exp002_dir = root / "exp002_dinov2s_slot_v2labels"
    exp003_dir = root / "exp003_dinov2b_slot_v2labels"
    oof_002, oof_003 = load_full_oof(exp002_dir), load_full_oof(exp003_dir)
    if oof_002 is None or oof_003 is None:
        print("\nexp002+exp003 ensemble: skipped (exp003 OOF not complete yet — 0 folds have oof.parquet)"
              if oof_003 is None else "\nexp002+exp003 ensemble: skipped (exp002 OOF missing — unexpected)")
    else:
        _, labels_parquet_env = None, rsna_root / "artifacts" / "labels" / "labels_v2_final.parquet"
        gold_truth, val_truth = load_truth(rsna_root, labels_parquet_env)

        print("\n=== exp002+exp003 ensemble — OOF (vs report labels) ===")
        for name, blend_fn in [("prob-mean", prob_mean_blend), ("rank-mean", rank_mean_blend)]:
            blended = blend_fn(oof_002, oof_003)
            macro, per_label = macro_auc(blended, val_truth)
            print(f"{name}: macro={macro:.4f}  (exp002 solo {dfs['exp002_dinov2s_slot_v2labels']['val_macro_auc'].mean():.4f}, "
                  f"exp003 solo {dfs['exp003_dinov2b_slot_v2labels']['val_macro_auc'].mean() if len(dfs['exp003_dinov2b_slot_v2labels']) else float('nan'):.4f})")

        gold_002, gold_003 = load_gold_fold_ensemble(exp002_dir), load_gold_fold_ensemble(exp003_dir)
        if gold_002 is not None and gold_003 is not None:
            print("\n=== exp002+exp003 ensemble — gold-58 (fold-ensembled per model, then blended) ===")
            for name, blend_fn in [("prob-mean", prob_mean_blend), ("rank-mean", rank_mean_blend)]:
                blended = blend_fn(gold_002, gold_003)
                macro, per_label = macro_auc(blended, gold_truth)
                print(f"{name}: macro={macro:.4f}")
