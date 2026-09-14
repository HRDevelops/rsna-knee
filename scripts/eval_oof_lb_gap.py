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

"""OOF <-> public-LB gap analysis (CLAUDE.md §7, EXPERIMENTS.md "OOF<->LB gap analysis" template,
2026-09-10) — run this ONCE, right after probe #1 (the ensemble) is scored on the public
leaderboard. Prints the interpretation and writes a report; does NOT edit EXPERIMENTS.md itself
(that file is hand-curated, append-only by convention — paste the report's markdown table + verdict
into it manually).

The bar is pre-registered here (not tuned after seeing the LB score, which would defeat the
point): |gap| < 0.02 -> validation is trustworthy, proceed to probe #2 as planned. |gap| >= 0.02 ->
stop, do not burn probe #2, investigate site-level leakage / prevalence shift / label-engine
ceiling (see CLAUDE.md §12 and this script's --gap docstring) before any further submission.

Usage:
    python scripts/eval_oof_lb_gap.py --lb-score 0.8621
    python scripts/eval_oof_lb_gap.py --lb-score 0.8621 --gold58-macro-auc 0.8744 --out artifacts/oof_lb_gap_report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Pre-registered reference (EXPERIMENTS.md 2026-09-09/10, ensemble_recipe_v1.yaml's greedy
# selection run) — the ensemble's OOF macro-AUC vs labels_v2_final. This is fixed at the value
# recorded BEFORE any LB score was seen; do not update it after the fact.
ENSEMBLE_OOF_MACRO_AUC = 0.8744
GAP_TRUST_THRESHOLD = 0.02


def interpret(oof: float, lb: float) -> dict:
    gap = oof - lb
    trustworthy = abs(gap) < GAP_TRUST_THRESHOLD
    if trustworthy:
        verdict = "TRUSTWORTHY — proceed to probe #2 (efficiency notebook) as planned."
        detail = None
    elif gap > 0:
        verdict = "STOP — OOF overstates LB by >= 0.02. Do not burn probe #2 yet."
        detail = (
            "OOF > LB by more than the trust threshold. In order of likelihood (CLAUDE.md §12):\n"
            "  1. Site-level/report-template leakage — check whether GroupKFold's report-template/\n"
            "     site clustering (CLAUDE.md §7) actually kept near-duplicate reports and same-site\n"
            "     studies together; a leak here inflates OOF relative to a genuinely held-out LB\n"
            "     population.\n"
            "  2. Prevalence shift — CLAUDE.md §3 warns train/public-LB/private-LB prevalences may\n"
            "     differ; AUC is prevalence-invariant per label so this shouldn't by itself cause a\n"
            "     gap, but confirm no per-label AUC computation accidentally became threshold-\n"
            "     sensitive.\n"
            "  3. Label-engine ceiling — check the gold-58 tripwire AUC (pass --gold58-macro-auc): if\n"
            "     it also looks off relative to what individual experiment entries recorded, the\n"
            "     label engine (not the fold construction) is the more likely culprit."
        )
    else:
        verdict = "STOP — LB beats OOF by >= 0.02. Investigate before treating this as a win."
        detail = (
            "A negative gap (LB > OOF) is not automatically good news — run the same checklist as\n"
            "the OOF > LB case; it can indicate a fold-construction issue in the other direction\n"
            "(OOF made harder than it should be) just as easily as a real generalization win."
        )
    return {"oof": oof, "lb": lb, "gap": gap, "trustworthy": trustworthy, "verdict": verdict, "detail": detail}


def render_report(result: dict, gold58_macro_auc: float | None) -> str:
    lines = [
        "# OOF <-> LB gap analysis — probe #1 (ensemble)",
        "",
        "| | value |",
        "|---|---|",
        f"| OOF macro-AUC (reference, pre-registered) | {result['oof']:.4f} |",
        f"| Public LB macro-AUC | {result['lb']:.4f} |",
        f"| Gap (OOF - LB) | {result['gap']:+.4f} |",
        f"| Trust threshold | < {GAP_TRUST_THRESHOLD} |",
    ]
    if gold58_macro_auc is not None:
        lines.append(f"| Gold-58 macro-AUC (tripwire, for context) | {gold58_macro_auc:.4f} |")
    lines += ["", f"**Verdict: {result['verdict']}**"]
    if result["detail"]:
        lines += ["", result["detail"]]
    lines += ["", "---", "Paste this table + verdict into docs/EXPERIMENTS.md's",
              "\"## OOF<->LB gap analysis\" section (that file is append-only by convention — this",
              "script does not edit it directly)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lb-score", type=float, required=True, help="public leaderboard macro-AUC for probe #1")
    parser.add_argument("--oof-score", type=float, default=ENSEMBLE_OOF_MACRO_AUC,
                         help=f"OOF reference to compare against (default: the pre-registered {ENSEMBLE_OOF_MACRO_AUC})")
    parser.add_argument("--gold58-macro-auc", type=float, default=None,
                         help="optional gold-58 tripwire AUC from the actual submitted run, for context in the report")
    parser.add_argument("--out", type=Path, default=None, help="write the markdown report here (optional)")
    args = parser.parse_args()

    result = interpret(args.oof_score, args.lb_score)
    report = render_report(result, args.gold58_macro_auc)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
