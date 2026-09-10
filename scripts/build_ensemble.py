"""Ensemble assembly (2026-09-09 instruction) — greedy forward selection over image-model
candidates, selecting on OOF macro-AUC vs the ORIGINAL labels_v2_final report labels (gold-58 is
reported as a TRIPWIRE only, never used for selection — same non-circularity discipline as
scripts/eval_exp005_acceptance.py). Per-label choice between rank-mean and prob-mean blending at
every step. Reports both:
  - GLOBAL: one member set + per-label method, shared across all 12 labels.
  - PER-LABEL: independent greedy selection (different members allowed per label).
Per-label is only worth the added deployment complexity if it beats global by >= 0.002 macro —
otherwise the instruction is to recommend the simpler global recipe.

Usage:
    python scripts/build_ensemble.py --candidates exp002_dinov2s_slot_v2labels exp003_dinov2b_slot_v2labels exp005_dinov2s_slot_v3labels exp006_dinov2b_slot_v3labels
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PER_LABEL_MIN_GAIN = 0.002


def load_full_oof(exp_dir: Path) -> pd.DataFrame | None:
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/oof.parquet"))]
    if len(parts) != 5:
        return None
    return pd.concat(parts, ignore_index=True).set_index("study_uid")


def load_gold_fold_ensemble(exp_dir: Path) -> pd.DataFrame | None:
    parts = [pd.read_parquet(p) for p in sorted(exp_dir.glob("fold*/gold58_predictions.parquet"))]
    if len(parts) != 5:
        return None
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby("study_uid")[TARGETS].mean()


def load_truth(labels_parquet: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(labels_parquet)
    return {
        r["study_uid"]: np.array([1.0 if r[f"{c}__probability"] >= 0.5 else 0.0 for c in TARGETS])
        for r in df.to_dict("records")
    }


def load_gold_truth(gold_csv: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(gold_csv).rename(columns={"StudyInstanceUID": "study_uid"})
    df = df[df[TARGETS[0]].notna()]
    return {r["study_uid"]: np.array([float(r[c]) for c in TARGETS]) for r in df.to_dict("records")}


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else float("nan")


def build_target_matrix(oof_dict: dict[str, pd.DataFrame], target: str, study_uids) -> pd.DataFrame:
    return pd.DataFrame({name: df.loc[study_uids, target] for name, df in oof_dict.items()},
                         index=study_uids)


def prob_mean_score(matrix: pd.DataFrame, members: list[str], y_true: np.ndarray) -> float:
    return safe_auc(y_true, matrix[members].mean(axis=1).to_numpy())


def rank_mean_score(matrix: pd.DataFrame, members: list[str], y_true: np.ndarray) -> float:
    ranks = matrix[members].rank(pct=True)
    return safe_auc(y_true, ranks.mean(axis=1).to_numpy())


def best_method_score(matrix: pd.DataFrame, members: list[str], y_true: np.ndarray) -> tuple[str, float]:
    p, r = prob_mean_score(matrix, members, y_true), rank_mean_score(matrix, members, y_true)
    return ("prob-mean", p) if p >= r else ("rank-mean", r)


def greedy_forward_global(target_matrices: dict[str, pd.DataFrame], truths: dict[str, np.ndarray],
                           candidates: list[str]) -> dict:
    selected: list[str] = []
    current_macro = float("-inf")
    history = []
    remaining = list(candidates)
    while remaining:
        best_c, best_macro, best_per_label = None, float("-inf"), None
        for c in remaining:
            trial = selected + [c]
            per_label = {t: best_method_score(target_matrices[t], trial, truths[t]) for t in TARGETS}
            macro = float(np.mean([v[1] for v in per_label.values()]))
            if macro > best_macro:
                best_macro, best_c, best_per_label = macro, c, per_label
        if best_macro <= current_macro:
            break
        selected.append(best_c)
        current_macro = best_macro
        history.append({"added": best_c, "macro": best_macro})
        remaining.remove(best_c)
    final_per_label = {t: best_method_score(target_matrices[t], selected, truths[t]) for t in TARGETS}
    return {"members": selected, "macro": current_macro, "per_label": final_per_label, "history": history}


def greedy_forward_per_label(target_matrices: dict[str, pd.DataFrame], truths: dict[str, np.ndarray],
                              candidates: list[str]) -> dict[str, dict]:
    results = {}
    for t in TARGETS:
        selected: list[str] = []
        current = float("-inf")
        remaining = list(candidates)
        while remaining:
            best_c, best_score = None, float("-inf")
            for c in remaining:
                _, score = best_method_score(target_matrices[t], selected + [c], truths[t])
                if score > best_score:
                    best_score, best_c = score, c
            if best_score <= current:
                break
            selected.append(best_c)
            current = best_score
            remaining.remove(best_c)
        method, score = best_method_score(target_matrices[t], selected, truths[t])
        results[t] = {"members": selected, "method": method, "auc": score}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()

    rsna_root = Path(os.environ.get("RSNA_ROOT", "."))
    root = rsna_root / "artifacts"
    labels_v2_final = root / "labels" / "labels_v2_final.parquet"

    oof_dict, gold_dict = {}, {}
    for name in args.candidates:
        oof = load_full_oof(root / name)
        gold = load_gold_fold_ensemble(root / name)
        if oof is None or gold is None:
            print(f"SKIPPING {name}: incomplete (missing 5-fold oof/gold58 parquet)")
            continue
        oof_dict[name] = oof
        gold_dict[name] = gold
    candidates = list(oof_dict.keys())
    print(f"Candidate pool: {candidates}")

    common_studies = None
    for df in oof_dict.values():
        common_studies = df.index if common_studies is None else common_studies.intersection(df.index)
    print(f"{len(common_studies)} studies common to all candidates' OOF")

    truth_v2final = load_truth(labels_v2_final)
    gold_truth = load_gold_truth(rsna_root / "data" / "raw" / "train.csv")

    study_uids = [u for u in common_studies if u in truth_v2final]
    truths = {t: np.array([truth_v2final[u][i] for u in study_uids]) for i, t in enumerate(TARGETS)}
    target_matrices = {t: build_target_matrix(oof_dict, t, study_uids) for t in TARGETS}

    print(f"\n=== Solo OOF macro (vs labels_v2_final) per candidate ===")
    for name in candidates:
        per_label = {t: safe_auc(truths[t], target_matrices[t][name].to_numpy()) for t in TARGETS}
        print(f"{name}: {np.mean(list(per_label.values())):.4f}")

    print("\n=== GLOBAL greedy forward selection ===")
    global_result = greedy_forward_global(target_matrices, truths, candidates)
    for step in global_result["history"]:
        print(f"  + {step['added']:45s} -> macro {step['macro']:.4f}")
    print(f"Final members: {global_result['members']}")
    print(f"Final macro (vs labels_v2_final): {global_result['macro']:.4f}")

    print("\n=== PER-LABEL greedy forward selection ===")
    per_label_result = greedy_forward_per_label(target_matrices, truths, candidates)
    per_label_macro = float(np.mean([v["auc"] for v in per_label_result.values()]))
    for t in TARGETS:
        r = per_label_result[t]
        print(f"  {t}: members={r['members']} method={r['method']} auc={r['auc']:.4f}")
    print(f"Per-label macro: {per_label_macro:.4f}")

    gain = per_label_macro - global_result["macro"]
    print(f"\nPer-label vs global macro gain: {gain:+.4f} "
          f"-> {'per-label WORTH the complexity' if gain >= PER_LABEL_MIN_GAIN else 'RECOMMEND GLOBAL (simpler, gain < 0.002)'}")

    print("\n=== Recipe table (GLOBAL, chosen) ===")
    recipe_rows = []
    for t in TARGETS:
        method, auc_val = global_result["per_label"][t]
        recipe_rows.append({"label": t, "members": ",".join(global_result["members"]),
                             "method": method, "OOF_auc_vs_v2final": auc_val})
    recipe_df = pd.DataFrame(recipe_rows)
    print(recipe_df.to_string(index=False))

    print("\n=== gold-58 TRIPWIRE (not used for selection) — global recipe ===")
    gold_study_uids = [u for u in gold_dict[candidates[0]].index if u in gold_truth]
    gold_target_matrices = {
        t: pd.DataFrame({name: gold_dict[name].loc[gold_study_uids, t] for name in candidates},
                         index=gold_study_uids)
        for t in TARGETS
    }
    gold_truths = {t: np.array([gold_truth[u][i] for u in gold_study_uids]) for i, t in enumerate(TARGETS)}
    gold_per_label = {}
    for t in TARGETS:
        method, _ = global_result["per_label"][t]
        score_fn = prob_mean_score if method == "prob-mean" else rank_mean_score
        gold_per_label[t] = score_fn(gold_target_matrices[t], global_result["members"], gold_truths[t])
    gold_macro = float(np.mean(list(gold_per_label.values())))
    print(f"gold-58 macro (global recipe, applied as-is): {gold_macro:.4f}")
    for t in TARGETS:
        print(f"  {t}: {gold_per_label[t]:.4f}")

    out_path = root / "ensemble_recipe.csv"
    recipe_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
