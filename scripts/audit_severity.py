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

"""exp017 step 1 (zero-GPU): does the existing LLM extraction preserve a per-finding severity GRADE?

The extraction schema (configs/rubric_v*.yaml) stores, per finding: present (yes/no/uncertain), probability,
severity_phrase (free-text quote, in the report's own language) and evidence_quote — there is no ordinal
grade field. This script measures how much grade information the free-text phrase carries, using a
multilingual keyword lexicon (en/es/fr/de/nl/it/pt), so the decision "can we build ordinal targets without a
new LLM pass" rests on numbers, not on impression.

    python scripts/audit_severity.py --out artifacts/severity_audit.md
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Lexicon, checked most-severe-first. Deliberately conservative, multilingual; substring match on lowercase.
LEX = {
    "severe": r"severe|sévère|severo|severa|grave|schwer|ernstig|hochgradig|marked|marcad|ausgepr|massiv|"
              r"extens|importante|abundant|large|grande|grote|groot|complete|completa|complet|volledig|"
              r"vollständig|total|high.?grade|grade\s*(3|4|iii|iv)|grado\s*(3|4|iii|iv)|stade\s*(3|4|iii|iv)|"
              r"graad\s*(3|4)|\bii?i\b.?\bgrad",
    "moderate": r"moderat|modéré|moderad|matig|mäßig|mässig|maessig|mittelgradig|\bmittel|"
                r"grade\s*(2|ii)\b|grado\s*(2|ii)\b|stade\s*(2|ii)\b|graad\s*2",
    "mild": r"mild|slight|small|leve\b|leves\b|ligero|ligera|pequeñ|léger|légère|legere|faible|discret|"
            r"licht|gering|klein|lieve|lieve|hafif|grade\s*(1|i)\b|grado\s*(1|i)\b|stade\s*(1|i)\b|graad\s*1",
    "trace": r"trace|minimal|minim|physiolog|vestig|\bspoor|spuren|fisiológ|infime|zeer\s*weinig",
}


def grade_of(phrase: str) -> str:
    p = (phrase or "").lower()
    for g in ("severe", "moderate", "mild", "trace"):
        if re.search(LEX[g], p):
            return g
    return "none_found" if p.strip() else "empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = Path(os.environ.get("RSNA_ROOT", Path(__file__).resolve().parent.parent.parent))
    lab = root / "artifacts/labels"
    qwen = pd.read_parquet(lab / "qwen2.5-72b-instruct-awq_v2.parquet").set_index("study_uid")
    llama = pd.read_parquet(lab / "llama-3.3-70b-instruct-awq.parquet").set_index("study_uid")
    v3 = pd.read_parquet(lab / "labels_v3_candidate.parquet").set_index("study_uid")
    raw = pd.read_csv(root / "data/raw/train.csv", usecols=["StudyInstanceUID", "Report"]).set_index("StudyInstanceUID")

    L = ["# exp017 step 1 — severity-field audit (zero-GPU)", "",
         "Schema fields per finding: `present` {yes,no,uncertain}, `probability`, `severity_phrase` (free text, "
         "report language), `evidence_quote`, `imputed`. **No ordinal-grade field exists.** Grade below = "
         "multilingual keyword lexicon applied to `severity_phrase` (most-severe-first).", ""]

    for name, df in (("Qwen2.5-72B (v2)", qwen), ("Llama-3.3-70B", llama)):
        L += [f"## {name}", "",
              "| label | n present=yes | n uncertain | phrase non-empty | grade found (any) | mild | trace | moderate | severe | none_found |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t in TARGETS:
            yes = df[df[f"{t}__present"] == "yes"]
            unc = int((df[f"{t}__present"] == "uncertain").sum())
            g = yes[f"{t}__severity_phrase"].map(grade_of)
            n = max(len(yes), 1)
            c = g.value_counts()
            ne = 1 - c.get("empty", 0) / n
            anyg = sum(c.get(k, 0) for k in ("mild", "trace", "moderate", "severe")) / n
            L.append(f"| {t} | {len(yes)} | {unc} | {ne:.0%} | {anyg:.0%} | {c.get('mild', 0) / n:.0%} | "
                     f"{c.get('trace', 0) / n:.0%} | {c.get('moderate', 0) / n:.0%} | {c.get('severe', 0) / n:.0%} | "
                     f"{c.get('none_found', 0) / n:.0%} |")
        L.append("")

    L += ["## Mild-only fraction among current (v3) positives", "",
          "v3 positive = `labels_v3_candidate` probability >= 0.5 (the deployed training target's binarisation). "
          "Grade from the Qwen v2 phrase. `mild+trace` = phrase carries a mild/trace word; `graded` = any grade word.", "",
          "| label | v3 positives | graded | mild+trace | mild+trace of graded | moderate | severe |", "|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        pos = v3.index[v3[f"{t}__probability"] >= 0.5]
        g = qwen.loc[pos, f"{t}__severity_phrase"].map(grade_of)
        graded = g.isin(["mild", "trace", "moderate", "severe"])
        mt = g.isin(["mild", "trace"])
        L.append(f"| {t} | {len(pos)} | {graded.mean():.0%} | {mt.mean():.0%} | "
                 f"{(mt.sum() / max(graded.sum(), 1)):.0%} | {(g == 'moderate').mean():.0%} | {(g == 'severe').mean():.0%} |")

    L += ["", "## Sample phrases (v3 positives, Qwen v2; 6 random per label)", ""]
    for t in TARGETS:
        pos = v3.index[v3[f"{t}__probability"] >= 0.5]
        s = qwen.loc[pos, f"{t}__severity_phrase"].sample(min(6, len(pos)), random_state=0).tolist()
        L.append(f"- **{t}**: " + " | ".join(repr(x[:70]) for x in s))
    rep = "\n".join(L) + "\n"
    print(rep)
    if args.out:
        args.out.write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
