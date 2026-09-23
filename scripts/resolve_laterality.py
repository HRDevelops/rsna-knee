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

"""Combine DICOM Laterality tags + report-extracted side into one final per-study side decision.

Precedence (per user decision, 2026-09-06): DICOM tag majority (when internally consistent across
a study's series) > report-extracted side > classifier fallback (not trained yet — those studies
are just listed here as needing it, per EXECUTION_PLAN Phase 2.3).

Inputs:
  - artifacts/decode_audit.parquet   (scripts/decode_audit.py; per-series tag_Laterality)
  - artifacts/study_side.csv         (scripts/extract_side.py; per-study report-derived side)

Usage:
    python scripts/resolve_laterality.py \
        --decode-audit artifacts/decode_audit.parquet \
        --study-side artifacts/study_side.csv \
        --out artifacts/final_side.csv \
        --out-report docs/LATERALITY_RESOLUTION.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_tag_value(v: str) -> str | None:
    v = (v or "").strip().upper()
    if v in ("R", "RIGHT"):
        return "R"
    if v in ("L", "LEFT"):
        return "L"
    if v == "B":
        return "BILATERAL"
    return None


def per_study_tag_side(decode_audit: pd.DataFrame) -> pd.DataFrame:
    decode_audit = decode_audit.copy()
    decode_audit["norm_side"] = decode_audit["tag_Laterality"].apply(normalize_tag_value)

    rows = []
    for study_uid, group in decode_audit.groupby("study_uid"):
        values = sorted(set(v for v in group["norm_side"] if v is not None))
        if len(values) == 0:
            rows.append({"study_uid": study_uid, "tag_side": None, "tag_consistent": None,
                         "tag_values_seen": ""})
        elif len(values) == 1:
            rows.append({"study_uid": study_uid, "tag_side": values[0], "tag_consistent": True,
                         "tag_values_seen": values[0]})
        else:
            rows.append({"study_uid": study_uid, "tag_side": "CONFLICT", "tag_consistent": False,
                         "tag_values_seen": ",".join(values)})
    return pd.DataFrame(rows)


def resolve(tag_side_df: pd.DataFrame, report_side_df: pd.DataFrame) -> pd.DataFrame:
    merged = tag_side_df.merge(report_side_df, on="study_uid", how="outer")
    merged["tag_side"] = merged["tag_side"].where(merged["tag_side"].notna(), None)
    merged["side_report"] = merged["side_report"].fillna("UNKNOWN")

    final_sides, sources = [], []
    for _, row in merged.iterrows():
        tag_side = row["tag_side"]
        report_side = row["side_report"]
        if tag_side in ("R", "L"):
            final_sides.append(tag_side)
            sources.append("tag")
        elif tag_side == "BILATERAL":
            final_sides.append("BILATERAL")
            sources.append("tag_bilateral")
        elif tag_side == "CONFLICT":
            if report_side in ("R", "L"):
                final_sides.append(report_side)
                sources.append("conflict_resolved_by_report")
            else:
                final_sides.append(None)
                sources.append("conflict_unresolved")
        elif report_side in ("R", "L"):
            final_sides.append(report_side)
            sources.append("report")
        else:
            final_sides.append(None)
            sources.append("none_needs_classifier")

    merged["final_side"] = final_sides
    merged["source"] = sources
    return merged


def compute_agreement(merged: pd.DataFrame) -> tuple[int, int, float]:
    comparable = merged[
        merged["tag_side"].isin(["R", "L"]) & merged["side_report"].isin(["R", "L"])
    ]
    n = len(comparable)
    if n == 0:
        return 0, 0, float("nan")
    n_agree = int((comparable["tag_side"] == comparable["side_report"]).sum())
    return n_agree, n, n_agree / n


def write_report(merged: pd.DataFrame, out_path: Path) -> None:
    n_agree, n_comparable, rate = compute_agreement(merged)
    source_counts = merged["source"].value_counts()

    lines = ["# Laterality resolution — DICOM tag x report cross-check\n"]
    lines.append(f"Tag/report agreement on the {n_comparable} studies where BOTH a consistent tag "
                  f"side and an unambiguous report side exist: **{n_agree}/{n_comparable} "
                  f"({100*rate:.1f}%)**.\n" if n_comparable else "No overlap to compare.\n")

    lines.append("## Final side source breakdown\n")
    lines.append("| Source | n studies |")
    lines.append("|---|---|")
    for src, n in source_counts.items():
        lines.append(f"| {src} | {n} |")
    lines.append("")

    conflicts = merged[merged["source"].isin(["conflict_resolved_by_report", "conflict_unresolved"])]
    lines.append(f"## Tag-conflict studies ({len(conflicts)} total)\n")
    lines.append("**Read this before treating these as resolved or as data errors.** A study-level "
                  "L/R conflict across series is exactly what you'd see from a genuinely BILATERAL "
                  "exam (both knees imaged in one study) — which breaks the whole "
                  "one-side-per-study assumption the flip-right-knees pipeline (CLAUDE.md §5) is "
                  "built on. Almost all of these have no report-stated side either (report=UNKNOWN), "
                  "consistent with a bilateral study rather than a mislabel. **Recommendation: "
                  "handle these per-SERIES, not per-study** — look up each series' own "
                  "Laterality tag and flip/route it individually, rather than forcing one "
                  "study-wide side decision. Don't silently pick a side and move on.\n")
    lines.append("| study_uid | tag_values_seen | report_side | resolution |")
    lines.append("|---|---|---|---|")
    for _, row in conflicts.iterrows():
        lines.append(f"| {row['study_uid']} | {row['tag_values_seen']} | {row['side_report']} | "
                      f"{row['source']} |")
    lines.append("")

    unresolved = merged[merged["source"] == "none_needs_classifier"]
    lines.append(f"## Studies with NO side source ({len(unresolved)}) — need the trained classifier\n")
    lines.append(f"These {len(unresolved)} studies have neither a usable DICOM Laterality tag nor "
                  f"a report-extracted side. They depend entirely on the report-supervised side "
                  f"classifier (EXECUTION_PLAN Phase 2.3) once it's trained on the {source_counts.get('tag', 0) + source_counts.get('report', 0) + source_counts.get('conflict_resolved_by_report', 0)} "
                  f"studies that already have a side label from tag/report.\n")

    bilateral = merged[merged["source"] == "tag_bilateral"]
    if len(bilateral):
        lines.append(f"## Bilateral-tagged studies ({len(bilateral)}) — special-case, do not "
                      f"laterality-normalize blindly\n")
        for uid in bilateral["study_uid"]:
            lines.append(f"- {uid}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-audit", type=Path, required=True)
    parser.add_argument("--study-side", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/final_side.csv"))
    parser.add_argument("--out-report", type=Path, default=Path("docs/LATERALITY_RESOLUTION.md"))
    args = parser.parse_args()

    decode_audit = pd.read_parquet(args.decode_audit)
    report_side_df = pd.read_csv(args.study_side)

    tag_side_df = per_study_tag_side(decode_audit)
    merged = resolve(tag_side_df, report_side_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    write_report(merged, args.out_report)

    n_agree, n_comparable, rate = compute_agreement(merged)
    print(f"Agreement on overlap: {n_agree}/{n_comparable} ({rate:.1%})" if n_comparable else "No overlap")
    print(merged["source"].value_counts())
    print(f"Wrote {args.out} and {args.out_report}")


if __name__ == "__main__":
    main()
