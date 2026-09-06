"""Near-duplicate report clustering via MinHash-LSH on character 5-gram shingles.

Why this exists (CLAUDE.md §6 step 1.1, §7, risk register): near-identical/templated reports
(same site, same boilerplate phrasing) must share a CV fold, or OOF macro-AUC gets inflated by
template leakage — the model "memorizes" a site's phrasing rather than generalizing.

Implementation notes: uses a from-scratch numpy MinHash (32-bit universal hashing, banded LSH for
candidate generation, union-find for connected components) instead of the `datasketch` package —
deliberately avoids adding a new pip dependency while a separate, unrelated pip install (torch/
vLLM) may be running concurrently in the same venv (concurrent installs into one venv can corrupt
its site-packages metadata).

Usage:
    python scripts/cluster_reports.py \
        --reports data/raw/train.csv \
        --out artifacts/report_clusters.parquet \
        --out-report docs/REPORT_CLUSTERS.md
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

MERSENNE_PRIME = (1 << 31) - 1
NUM_PERM = 64
NUM_BANDS = 16
ROWS_PER_BAND = NUM_PERM // NUM_BANDS
SHINGLE_SIZE = 5  # WORDS, not characters — character 5-grams are too generic across any language's
                  # common short substrings and caused unrelated reports to band-collide (verified
                  # empirically: an initial char-gram run produced a 2134-study "cluster" of totally
                  # unrelated Spanish/German/English reports). Word 5-grams are the standard choice
                  # for near-duplicate/template detection (Broder-style w-shingling).
JACCARD_VERIFY_THRESHOLD = 0.5  # LSH candidates are re-checked against this before merging


def normalize_report(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[date\]", " ", text)
    text = re.sub(r"\d+([.,]\d+)?", "#", text)  # numbers vary per patient, wording doesn't
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    words = text.split(" ")
    words = [w for w in words if w]
    if len(words) < k:
        return {text} if text else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def shingle_hash_array(shingle_set: set[str]) -> np.ndarray:
    if not shingle_set:
        return np.empty(0, dtype=np.int64)
    hashes = [
        int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "big")
        for s in shingle_set
    ]
    return np.array(hashes, dtype=np.int64)


def build_hash_functions(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE_PRIME, size=NUM_PERM, dtype=np.int64)
    b = rng.integers(0, MERSENNE_PRIME, size=NUM_PERM, dtype=np.int64)
    return a, b


def minhash_signature(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.full(NUM_PERM, MERSENNE_PRIME, dtype=np.int64)
    vals = (a[:, None] * x[None, :] + b[:, None]) % MERSENNE_PRIME
    return vals.min(axis=1)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def cluster_reports(reports_df: pd.DataFrame) -> pd.DataFrame:
    a, b = build_hash_functions()
    signatures = np.zeros((len(reports_df), NUM_PERM), dtype=np.int64)
    for i, text in enumerate(reports_df["Report"]):
        norm = normalize_report(str(text))
        sh = shingles(norm)
        x = shingle_hash_array(sh)
        signatures[i] = minhash_signature(x, a, b)

    uf = UnionFind(len(reports_df))
    candidate_pairs: set[tuple[int, int]] = set()
    for band_i in range(NUM_BANDS):
        band = signatures[:, band_i * ROWS_PER_BAND:(band_i + 1) * ROWS_PER_BAND]
        buckets: dict[tuple, list[int]] = {}
        for i, row in enumerate(band):
            key = tuple(row.tolist())
            buckets.setdefault(key, []).append(i)
        for idxs in buckets.values():
            if len(idxs) > 1:
                first = idxs[0]
                for other in idxs[1:]:
                    candidate_pairs.add((min(first, other), max(first, other)))

    # LSH banding only proposes CANDIDATES — verify each against the full minhash signature's
    # approximate Jaccard before merging, or a handful of spurious single-band collisions (common
    # with short/generic shingles) can chain unrelated documents into one giant component.
    for i, j in candidate_pairs:
        approx_jaccard = float((signatures[i] == signatures[j]).mean())
        if approx_jaccard >= JACCARD_VERIFY_THRESHOLD:
            uf.union(i, j)

    roots = [uf.find(i) for i in range(len(reports_df))]
    root_to_cluster_id = {}
    cluster_ids = []
    for r in roots:
        if r not in root_to_cluster_id:
            root_to_cluster_id[r] = f"cluster_{len(root_to_cluster_id):05d}"
        cluster_ids.append(root_to_cluster_id[r])

    out = reports_df[["study_uid"]].copy()
    out["report_cluster_id"] = cluster_ids
    out["cluster_size"] = out.groupby("report_cluster_id")["report_cluster_id"].transform("count")
    return out


def write_report(clusters: pd.DataFrame, out_path: Path) -> None:
    sizes = clusters.drop_duplicates("report_cluster_id")["cluster_size"].sort_values(ascending=False)
    n_clusters = len(sizes)
    n_singletons = int((sizes == 1).sum())
    lines = [
        "# Report near-duplicate clusters\n",
        f"{len(clusters)} studies -> **{n_clusters} clusters** "
        f"({n_singletons} singletons, {n_clusters - n_singletons} clusters with >1 member).\n",
        "## Largest clusters (top 20)\n",
        "| cluster_size | n_clusters_this_size |" if False else "",
    ]
    lines = [l for l in lines if l != ""]
    lines.append("| Cluster size | Count of clusters this size |")
    lines.append("|---|---|")
    size_counts = sizes.value_counts().sort_index(ascending=False)
    for size, count in size_counts.head(20).items():
        lines.append(f"| {size} | {count} |")
    lines.append("")

    top_clusters = clusters.drop_duplicates("report_cluster_id").nlargest(10, "cluster_size")
    lines.append("## 10 largest individual clusters\n")
    lines.append("| Cluster ID | Size | Share of dataset |")
    lines.append("|---|---|---|")
    for _, row in top_clusters.iterrows():
        pct = 100.0 * row["cluster_size"] / len(clusters)
        lines.append(f"| {row['report_cluster_id']} | {row['cluster_size']} | {pct:.1f}% |")
    lines.append("")

    largest_share = top_clusters["cluster_size"].max() / len(clusters) if len(top_clusters) else 0
    if largest_share > 0.05:
        lines.append(f"**Flag: largest cluster is {100*largest_share:.1f}% of the dataset** — "
                      f"verify this isn't a mega-template that will starve one fold of diversity; "
                      f"consider capping per-cluster influence or splitting it before GroupKFold "
                      f"if a single cluster dominates a fold's composition.\n")
    else:
        lines.append(f"No single cluster exceeds 5% of the dataset — no dominant-template risk "
                      f"flagged at cluster level.\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/report_clusters.parquet"))
    parser.add_argument("--out-report", type=Path, default=Path("docs/REPORT_CLUSTERS.md"))
    args = parser.parse_args()

    df = pd.read_csv(args.reports).rename(columns={"StudyInstanceUID": "study_uid"})
    df = df[["study_uid", "Report"]]

    clusters = cluster_reports(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_parquet(args.out, index=False)
    write_report(clusters, args.out_report)
    print(f"{len(df)} studies -> {clusters['report_cluster_id'].nunique()} clusters. "
          f"Wrote {args.out} and {args.out_report}")


if __name__ == "__main__":
    main()
