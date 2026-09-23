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

"""Extract which knee (left/right) each report is about, from the report text itself.

CLAUDE.md §5: "reports give near-free side labels in every language" — this is the source data
for (a) the report-supervised side classifier fallback (EXECUTION_PLAN Phase 2.3) and (b) cross-
checking against the DICOM Laterality tag (scripts/resolve_laterality.py), which only covers 50%
of studies (docs/DECODE_AUDIT.md).

Coverage is necessarily uneven: patterns below are hand-built for the 5 languages CLAUDE.md calls
out by name (es/en/fr/nl/de, ~2/3 of the corpus) plus best-effort patterns for 6 more
(pt/it/ru/uk/tr/pl). The remaining "other" bucket (CLAUDE.md §3: ~1,491 reports, a dozen+
languages/sites) will mostly fall through to side="unknown" here — that's expected, not a bug;
those studies rely on the DICOM tag or the trained classifier instead.

Usage:
    python scripts/extract_side.py --reports data/raw/train.csv --out artifacts/study_side.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Each pattern is (side, compiled regex). Patterns require the laterality word within a few
# tokens of the knee word (either order) to cut false positives from unrelated uses of
# "right"/"left" etc. elsewhere in the report.
_KNEE = {
    "en": "knee", "es": "rodilla", "fr": "genou", "nl": "knie", "de": "knie",
    "pt": "joelho", "it": "ginocchio", "pl": "kolano",
}

_PATTERNS: list[tuple[str, re.Pattern]] = []


def _add(side: str, *phrases: str) -> None:
    for phrase in phrases:
        _PATTERNS.append((side, re.compile(phrase, re.IGNORECASE | re.UNICODE)))


# English
_add("R", r"\bright\W{1,15}knee\b", r"\bknee\W{1,15}right\b", r"\bright[- ]sided knee\b")
_add("L", r"\bleft\W{1,15}knee\b", r"\bknee\W{1,15}left\b", r"\bleft[- ]sided knee\b")

# Spanish
_add("R", r"\brodilla\w*\W{1,15}derech\w*\b", r"\bderech\w*\W{1,15}rodilla\w*\b")
_add("L", r"\brodilla\w*\W{1,15}izquierd\w*\b", r"\bizquierd\w*\W{1,15}rodilla\w*\b")

# French
_add("R", r"\bgenou\w*\W{1,15}droit\w*\b", r"\bdroit\w*\W{1,15}genou\w*\b")
_add("L", r"\bgenou\w*\W{1,15}gauche\w*\b", r"\bgauche\w*\W{1,15}genou\w*\b")

# Dutch
_add("R", r"\bknie\w*\W{1,15}recht\w*\b", r"\brecht\w*\W{1,15}knie\w*\b", r"\brechterknie\b")
_add("L", r"\bknie\w*\W{1,15}link\w*\b", r"\blink\w*\W{1,15}knie\w*\b", r"\blinkerknie\b")

# German
_add("R", r"\bknie\w*\W{1,15}recht\w*\b", r"\brecht\w*\W{1,15}knie\w*\b")
_add("L", r"\bknie\w*\W{1,15}link\w*\b", r"\blink\w*\W{1,15}knie\w*\b")

# Portuguese
_add("R", r"\bjoelho\w*\W{1,15}direit\w*\b", r"\bdireit\w*\W{1,15}joelho\w*\b")
_add("L", r"\bjoelho\w*\W{1,15}esquerd\w*\b", r"\besquerd\w*\W{1,15}joelho\w*\b")

# Italian
_add("R", r"\bginocchio\w*\W{1,15}destr\w*\b", r"\bdestr\w*\W{1,15}ginocchio\w*\b")
_add("L", r"\bginocchio\w*\W{1,15}sinistr\w*\b", r"\bsinistr\w*\W{1,15}ginocchio\w*\b")

# Polish
_add("R", r"\bkolano\w*\W{1,15}praw\w*\b", r"\bpraw\w*\W{1,15}kolano\w*\b")
_add("L", r"\bkolano\w*\W{1,15}lew\w*\b", r"\blew\w*\W{1,15}kolano\w*\b")

# Turkish
_add("R", r"\bsağ\W{1,15}diz\b", r"\bdiz\W{1,15}sağ\b")
_add("L", r"\bsol\W{1,15}diz\b", r"\bdiz\W{1,15}sol\b")

# Russian / Ukrainian (Cyrillic) — best-effort
_add("R", r"прав\w*\s*колен\w*", r"колен\w*\s*прав\w*")
_add("L", r"лев\w*\s*колен\w*", r"колен\w*\s*лев\w*")
_add("R", r"прав\w*\s*коліно\w*", r"коліно\w*\s*прав\w*")
_add("L", r"лів\w*\s*коліно\w*", r"коліно\w*\s*лів\w*")


def extract_side(report: str) -> tuple[str, str]:
    matches_r = [p.pattern for side, p in _PATTERNS if side == "R" and p.search(report)]
    matches_l = [p.pattern for side, p in _PATTERNS if side == "L" and p.search(report)]
    if matches_r and not matches_l:
        return "R", matches_r[0]
    if matches_l and not matches_r:
        return "L", matches_l[0]
    if matches_r and matches_l:
        return "AMBIGUOUS", f"R:{matches_r[0]} | L:{matches_l[0]}"
    return "UNKNOWN", ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/study_side.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.reports).rename(columns={"StudyInstanceUID": "study_uid"})
    results = df["Report"].astype(str).apply(extract_side)
    df["side_report"] = [r[0] for r in results]
    df["evidence_phrase"] = [r[1] for r in results]

    out_df = df[["study_uid", "side_report", "evidence_phrase"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    counts = out_df["side_report"].value_counts()
    print(counts)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
