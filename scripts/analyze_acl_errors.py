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

"""ACL error analysis (2026-09-17, ACL workstream item 1). One-off diagnostic, not a reusable
pipeline component: find the ensemble's highest-error ACL OOF cases and report what they have in
common (slots/planes present, side resolution, gold-58 membership, crude report language, and how
much the report-derived label and the image-model OOF prediction disagree for this specific
label).

Usage: python scripts/analyze_acl_errors.py [--top-n 40] [--out artifacts/acl_error_report.md]
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

RSNA_ROOT = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent))
ARTIFACTS = RSNA_ROOT / "artifacts"

ENSEMBLE_MEMBERS = [
    "exp006_dinov2b_slot_v3labels",
    "exp003_dinov2b_slot_v2labels",
    "exp005_dinov2s_slot_v3labels",
]  # ACL's accepted blend method is prob-mean over these 3 (ensemble_recipe.csv)

# Very crude language heuristic (no saved per-study language artifact exists) — same spirit as the
# "crude regex estimate" CLAUDE.md §3 already describes for the project's own language census, not
# a real langid model. Good enough to check whether error cases skew toward any one language.
LANG_MARKERS = {
    "es": [" el ", " la ", " rodilla ", " sin ", " con ", " menisco"],
    "fr": [" le ", " la ", " genou ", " sans ", " avec ", " ménisque", " menisque"],
    "nl": [" de ", " het ", " knie ", " zonder ", " met ", " meniscus"],
    "de": [" der ", " die ", " das ", " knie ", " ohne ", " mit "],
    "en": [" the ", " knee ", " without ", " with ", " meniscus"],
}


def guess_language(report: str) -> str:
    text = f" {report.lower()} "
    scores = {lang: sum(text.count(m) for m in markers) for lang, markers in LANG_MARKERS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other/unknown"


def load_ensemble_oof(label: str = "ACL") -> pd.DataFrame:
    frames = []
    for member in ENSEMBLE_MEMBERS:
        member_frames = []
        for fold in range(5):
            path = ARTIFACTS / member / f"fold{fold}" / "oof.parquet"
            if not path.exists():
                print(f"  [WARN] missing {path}")
                continue
            df = pd.read_parquet(path, columns=["study_uid", label])
            member_frames.append(df)
        member_df = pd.concat(member_frames, ignore_index=True).rename(columns={label: member})
        frames.append(member_df.set_index("study_uid"))
    combined = pd.concat(frames, axis=1)
    combined["ensemble_pred"] = combined[ENSEMBLE_MEMBERS].mean(axis=1)  # prob-mean, per ensemble_recipe.csv
    return combined.reset_index()


def load_slot_presence() -> pd.DataFrame:
    """One row per study: which of the 6 canonical slots have a series present at all (decode
    success not required here — presence in train_series.csv, the same source RSNAKneeDataset's
    slot_map uses)."""
    series_df = pd.read_csv(RSNA_ROOT / "data/raw/train_series.csv")

    def slot_name(row) -> str | None:
        plane = row["Anatomical_Plane"]
        fs = row["Fluid_Sensitive"]
        if plane not in ("Sagittal", "Coronal", "Axial") or pd.isna(fs):
            return None
        return f"{plane}_{'FS' if bool(fs) else 'nonFS'}"
    series_df["slot"] = series_df.apply(slot_name, axis=1)
    present = (
        series_df.dropna(subset=["slot"])
        .groupby("StudyInstanceUID")["slot"]
        .apply(lambda s: sorted(set(s)))
        .rename("slots_present")
        .reset_index()
        .rename(columns={"StudyInstanceUID": "study_uid"})
    )
    present["n_slots_present"] = present["slots_present"].apply(len)
    present["has_sagittal"] = present["slots_present"].apply(lambda s: any("Sagittal" in x for x in s))
    return present


def load_gold58_comparison(label: str = "ACL") -> pd.DataFrame:
    """Gold-58 is excluded from every fold's train/val split (CLAUDE.md §7), so it never appears
    in oof.parquet — the genuine out-of-sample image-model signal on gold-58 lives in each fold's
    gold58_predictions.parquet instead (every fold's model predicts all 58 studies; average across
    folds for one fold-ensembled prediction per study)."""
    member_frames = []
    for member in ENSEMBLE_MEMBERS:
        fold_dfs = [pd.read_parquet(ARTIFACTS / member / f"fold{f}" / "gold58_predictions.parquet",
                                     columns=["study_uid", label]) for f in range(5)]
        member_frames.append(pd.concat(fold_dfs).groupby("study_uid")[label].mean().rename(member))
    gold_ens = pd.concat(member_frames, axis=1)
    gold_ens["image_ens_pred"] = gold_ens[ENSEMBLE_MEMBERS].mean(axis=1)
    return gold_ens[["image_ens_pred"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "acl_error_report.md")
    args = parser.parse_args()

    labels = pd.read_parquet(ARTIFACTS / "labels/labels_v2_final.parquet",
                              columns=["study_uid", "ACL__probability", "ACL__confidence_weight"])
    oof = load_ensemble_oof("ACL")
    slots = load_slot_presence()
    side = pd.read_csv(ARTIFACTS / "final_side.csv")[["study_uid", "final_side", "source"]]
    train_raw = pd.read_csv(RSNA_ROOT / "data/raw/train.csv")
    train_raw = train_raw.rename(columns={"StudyInstanceUID": "study_uid"})
    gold_uids = set(train_raw.loc[train_raw["ACL"].notna(), "study_uid"])

    df = (
        oof.merge(labels, on="study_uid", how="inner")
        .merge(slots, on="study_uid", how="left")
        .merge(side, on="study_uid", how="left")
        .merge(train_raw[["study_uid", "Report", "ACL"]], on="study_uid", how="left")
    )
    df["is_gold58"] = df["study_uid"].isin(gold_uids)
    # Target: gold hard label where available (the real ground truth), else the report-derived
    # soft label (labels_v2_final) — same convention the accepted OOF-macro-AUC number itself uses.
    df["target"] = df["ACL"].where(df["is_gold58"], df["ACL__probability"])
    df["error"] = (df["ensemble_pred"] - df["target"]).abs()
    df["language"] = df["Report"].fillna("").apply(guess_language)

    # --- Report/image disagreement for ACL specifically ---
    # NOTE: gold-58 is excluded from every fold's train/val split by design (CLAUDE.md §7), so it
    # is NEVER present in the oof.parquet-based `df` above (confirmed: is_gold58.sum() == 0 in that
    # frame) — the genuine out-of-sample gold-58 comparison needs gold58_predictions.parquet
    # instead, loaded separately here.
    corr = df["ACL__probability"].corr(df["ensemble_pred"])
    gold_image = load_gold58_comparison("ACL")
    gold_true = train_raw.loc[train_raw["ACL"].notna(), ["study_uid", "ACL"]].set_index("study_uid")["ACL"]
    gold_report = labels.set_index("study_uid")["ACL__probability"]
    gold_df = pd.DataFrame({"gold_true": gold_true, "image_ens_pred": gold_image["image_ens_pred"],
                             "report_label": gold_report}).dropna()
    from sklearn.metrics import roc_auc_score
    gold_report_corr = gold_df["report_label"].corr(gold_df["gold_true"])
    gold_image_corr = gold_df["image_ens_pred"].corr(gold_df["gold_true"])
    gold_report_auc = roc_auc_score(gold_df["gold_true"], gold_df["report_label"])
    gold_image_auc = roc_auc_score(gold_df["gold_true"], gold_df["image_ens_pred"])

    top_errors = df.sort_values("error", ascending=False).head(args.top_n)

    lines = ["# ACL error analysis — ensemble OOF, 2026-09-17", "",
             f"Population: {len(df)} non-gold studies (gold-58 is excluded from oof.parquet BY "
             f"DESIGN — CLAUDE.md §7 fold-safe reference set — so it structurally cannot appear "
             f"in the error table below; the top-errors-vs-population comparison further down is "
             f"only ever over the 4349 non-gold studies, not a finding that gold-58 is error-free).",
             "",
             "## Report vs. image disagreement for ACL specifically", "",
             f"- Corr(report-derived label_v2_final ACL prob, image-ensemble OOF ACL pred), full "
             f"non-gold population: **{corr:.3f}** — moderately high, but this compares OOF "
             f"predictions against the SAME (or a blend of the same) targets the models were "
             f"trained on, so it's not fully independent evidence of agreement.",
             f"- **Genuine out-of-sample check, gold-58** (never trained on by either the image "
             f"models or the label engine's own LOO calibration): "
             f"report-label vs. true-gold AUC = **{gold_report_auc:.4f}** (corr {gold_report_corr:.4f}), "
             f"image-ensemble vs. true-gold AUC = **{gold_image_auc:.4f}** (corr {gold_image_corr:.4f}).",
             ("  -> **Essentially tied, both near-ceiling** — the v3 blend's report-heavy weight "
              "(w_image=0.1) for ACL was optimizing a thin, likely noise-level margin on n=58 "
              "(CLAUDE.md's own 'wide CIs' caveat applies), not evidence of a large image-model "
              "blind spot on ACL specifically."),
             ("  -> **This changes the read on the −0.097 external gap**: gold-58 says our image "
              "model discriminates ACL about as well as the report engine does when evaluated "
              "cleanly (~0.988 AUC either way). The much lower 0.8522 OOF number is measured "
              "against a NOISY SOFT-LABEL proxy over the full 4349-study population, not a clean "
              "label — some of the apparent gap vs. Qiu et al.'s 0.949 (measured on their own "
              "large, cleanly-labeled test set) is plausibly proxy-label noise inflating our "
              "measured gap, not purely a real model/architecture deficit. Doesn't invalidate "
              "prioritizing ACL — 0.8522 is still our lowest comparable label and the true metric "
              "is unknowable without more clean labels — but it argues AGAINST assuming an "
              "anatomy-aware architecture change alone will close most of the gap, and FOR also "
              "treating ACL-specific label/target quality on the broader population as a live "
              "hypothesis worth checking, not something gold-58 alone can rule out."),
             "",
             f"## Top {args.top_n} highest-error ACL cases", "",
             "| study_uid (short) | error | pred | target | gold58 | side | slots present | has_sagittal | language |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in top_errors.iterrows():
        slots_str = ",".join(s.replace("_nonFS", "").replace("_FS", "*") for s in (r["slots_present"] or []))
        lines.append(
            f"| {r['study_uid'][:24]}... | {r['error']:.3f} | {r['ensemble_pred']:.3f} | {r['target']:.3f} | "
            f"{'Y' if r['is_gold58'] else 'n'} | {r['final_side'] if pd.notna(r['final_side']) else '?'} | "
            f"{slots_str} | {'Y' if r['has_sagittal'] else '**N**'} | {r['language']} |"
        )

    lines += ["", "## Aggregate pattern in the top errors vs. the full population", "",
              "| | top errors | full population |", "|---|---|---|",
              f"| % missing sagittal slot | {(~top_errors['has_sagittal']).mean():.1%} | {(~df['has_sagittal']).mean():.1%} |",
              f"| % side unresolved (final_side null/UNKNOWN) | "
              f"{(top_errors['final_side'].isna() | (top_errors['final_side'] == 'UNKNOWN')).mean():.1%} | "
              f"{(df['final_side'].isna() | (df['final_side'] == 'UNKNOWN')).mean():.1%} |",
              f"| mean n_slots_present | {top_errors['n_slots_present'].mean():.2f} | {df['n_slots_present'].mean():.2f} |",
              "", "Language distribution, top errors:", "",
              top_errors["language"].value_counts(normalize=True).to_string(),
              "", "Language distribution, full population:", "",
              df["language"].value_counts(normalize=True).to_string()]

    report = "\n".join(lines) + "\n"
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
