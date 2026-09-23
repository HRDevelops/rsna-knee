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

"""Build folds.parquet: 5-fold GroupKFold on report_cluster_id (+ multilabel stratification once
labels exist). Gold-58 is always quarantined — excluded from every fold (CLAUDE.md §7).

Two modes:
  - No --labels given (today's situation — Phase 1 label engine hasn't produced labels_v1 yet):
    group-only balanced split. Clusters are assigned to folds by a greedy longest-processing-time
    (LPT) bin-packing on cluster size, which balances fold SIZE but does NOT balance label
    prevalence (there's nothing to stratify on yet). This is fold assignment v1 — expect to rerun.
  - --labels given (labels_v1.parquet or later, wide format with `{target}__probability` columns
    per scripts/label_reports.py): greedy iterative stratification. Clusters are processed in
    descending "rarity score" order (rare-label-weighted, Sechidis-et-al style) and each is
    assigned to whichever fold minimizes the resulting per-label positive-rate imbalance, subject
    to every study in a cluster landing in the same fold.

Usage:
    # today (group-only, rerun once labels_v1 exists):
    python scripts/build_folds.py --clusters artifacts/report_clusters.parquet \
        --gold data/raw/train.csv --out artifacts/folds.parquet

    # once labels_v1.parquet exists:
    python scripts/build_folds.py --clusters artifacts/report_clusters.parquet \
        --gold data/raw/train.csv --labels artifacts/labels/labels_v1.parquet \
        --out artifacts/folds.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_FOLDS = 5
TARGET_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def load_gold_uids(gold_csv: Path) -> set[str]:
    df = pd.read_csv(gold_csv).rename(columns={"StudyInstanceUID": "study_uid"})
    return set(df.loc[df[TARGET_COLUMNS[0]].notna(), "study_uid"])


def assign_folds_group_only(clusters: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """LPT bin-packing: sort clusters by size descending, assign each to the currently-smallest fold."""
    cluster_sizes = clusters.groupby("report_cluster_id").size().sort_values(ascending=False)
    fold_totals = np.zeros(N_FOLDS, dtype=int)
    cluster_to_fold: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    order = cluster_sizes.index.tolist()
    # break ties among equal-size clusters randomly so we don't always favor the first alphabetically
    rng.shuffle(order)
    order = sorted(order, key=lambda c: -cluster_sizes[c])
    for cluster_id in order:
        fold = int(np.argmin(fold_totals))
        cluster_to_fold[cluster_id] = fold
        fold_totals[fold] += cluster_sizes[cluster_id]
    return clusters["report_cluster_id"].map(cluster_to_fold)


def rarity_weights(labels_df: pd.DataFrame) -> pd.Series:
    prob_cols = [f"{t}__probability" for t in TARGET_COLUMNS]
    binarized = (labels_df.set_index("study_uid")[prob_cols] >= 0.5).astype(int)
    binarized.columns = TARGET_COLUMNS
    global_rate = binarized.mean()
    # inverse-frequency weight per label, applied per study as sum over its positive labels
    inv_freq = 1.0 / global_rate.clip(lower=1e-3)
    return binarized, inv_freq


def assign_folds_stratified(clusters: pd.DataFrame, labels_df: pd.DataFrame, seed: int = 42) -> pd.Series:
    binarized, inv_freq = rarity_weights(labels_df)
    merged = clusters.merge(binarized, left_on="study_uid", right_index=True, how="left")
    merged[TARGET_COLUMNS] = merged[TARGET_COLUMNS].fillna(0)

    cluster_label_sums = merged.groupby("report_cluster_id")[TARGET_COLUMNS].sum()
    cluster_sizes = merged.groupby("report_cluster_id").size()
    # Process LARGEST clusters first (as in assign_folds_group_only's proven LPT approach), not
    # rarest-first. A single large cluster (e.g. the 209-study site-template) MUST land whole in
    # one fold; if it's processed last, after many small clusters have already filled folds
    # unevenly, that one placement can blow out fold-size balance with no way to correct it
    # afterward (confirmed empirically: rarity-first ordering produced only 3 populated folds out
    # of 5, sizes 1100/1182/2067). Rarity (inv_freq) still matters for WHICH fold a cluster lands
    # in — it weights the label-imbalance term below so getting rare-label counts (MCL, Lateral OA,
    # Baker's) right across folds matters more than getting common-label counts exactly right.
    order = cluster_sizes.sort_values(ascending=False).index.tolist()
    inv_freq_arr = inv_freq[TARGET_COLUMNS].to_numpy()
    # Scale so the label term and the size term start out comparable in magnitude (both roughly
    # O(size_target) once summed) — otherwise one term drowns out the other regardless of the
    # structural fix below.
    size_target = len(merged) / N_FOLDS
    label_weight = size_target / max((cluster_label_sums.sum().to_numpy() * inv_freq_arr).sum(), 1e-6)

    fold_label_counts = np.zeros((N_FOLDS, len(TARGET_COLUMNS)))
    fold_sizes = np.zeros(N_FOLDS)
    cluster_to_fold: dict[str, int] = {}

    for cluster_id in order:
        c_labels = cluster_label_sums.loc[cluster_id].to_numpy()
        c_size = cluster_sizes.loc[cluster_id]
        best_fold, best_score = None, None
        for f in range(N_FOLDS):
            trial_counts = fold_label_counts[f] + c_labels
            trial_size = fold_sizes[f] + c_size
            # Minimize the RESULTING load, not distance from an absolute target. Distance-from-
            # target is a trap here: early on every fold is far below target, so adding to an
            # already-larger fold looks "better" (closer to target) than growing an empty one —
            # confirmed empirically, it collapsed 5 folds down to 3. Minimizing resulting size
            # directly reproduces proper LPT bin-packing (always favors the currently smallest
            # fold), with the rarity-weighted label term as a secondary nudge on TOP of that.
            score = trial_size + label_weight * (trial_counts * inv_freq_arr).sum()
            if best_score is None or score < best_score:
                best_score, best_fold = score, f
        cluster_to_fold[cluster_id] = best_fold
        fold_label_counts[best_fold] += c_labels
        fold_sizes[best_fold] += c_size

    return clusters["report_cluster_id"].map(cluster_to_fold)


def write_summary(folds: pd.DataFrame, mode: str, out_path: Path) -> None:
    lines = [f"# Fold assignment summary ({mode})\n"]
    counts = folds.groupby("fold").size()
    lines.append("| Fold | n studies |")
    lines.append("|---|---|")
    for f, n in counts.items():
        lines.append(f"| {f} | {n} |")
    lines.append(f"\nGold-58 quarantined (fold = -1): {int((folds['fold'] == -1).sum())} studies.\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None,
                         help="labels_v1.parquet — omit to fall back to group-only balanced folds")
    parser.add_argument("--out", type=Path, default=Path("artifacts/folds.parquet"))
    parser.add_argument("--out-report", type=Path, default=Path("docs/FOLDS.md"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clusters = pd.read_parquet(args.clusters)
    gold_uids = load_gold_uids(args.gold)

    non_gold = clusters[~clusters["study_uid"].isin(gold_uids)].reset_index(drop=True)

    if args.labels is not None and args.labels.exists():
        labels_df = pd.read_parquet(args.labels)
        fold_assignment = assign_folds_stratified(non_gold, labels_df, seed=args.seed)
        mode = "group + multilabel-stratified"
    else:
        fold_assignment = assign_folds_group_only(non_gold, seed=args.seed)
        mode = "group-only (no labels yet — RERUN with --labels once labels_v1.parquet exists)"

    non_gold = non_gold.copy()
    non_gold["fold"] = fold_assignment

    gold_rows = clusters[clusters["study_uid"].isin(gold_uids)].copy()
    gold_rows["fold"] = -1

    folds = pd.concat([non_gold, gold_rows], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(args.out, index=False)
    write_summary(folds, mode, args.out_report)

    print(f"Mode: {mode}")
    print(folds.groupby("fold").size())
    print(f"Wrote {args.out} and {args.out_report}")


if __name__ == "__main__":
    main()
