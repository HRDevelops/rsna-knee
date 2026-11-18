# Publishability audit (2026-11-18)

Written after reading the 1st–10th and 13th place solution writeups (Kaggle, read 2026-11-18) and re-checking our own
records (`docs/EXPERIMENTS.md`, `docs/research/*`). Figures: `docs/figures/final/fig1–fig4`.

## Verdict

**Not publishable as it stands**, in any of the three framings it could take:
- a methods paper (MICCAI, TMI);
- a negative-results paper (MIDL, MELBA);
- a "what matters for knee MRI" analysis.

**One realistic route exists:** a short **negative-result paper** at MELBA or a MICCAI 2027 workshop. It needs two
fixes:
- **evaluation against expert labels**, via Kaggle late submissions;
- **a competitive baseline**, trained on image-read labels.

Neither fix fits before the course deadline (5 Dec). Treat this as a 2027 option, not a current claim.

## What we actually have

| Contribution | Evidence quality | Novel after the writeups? |
|---|---|---|
| Pre-registered geometry experiments, two causal controls (donor swap, β-off), two STOPs | Strong *internally*: paired runs, rules frozen before running | Partly. See R4: other teams used geometry as an *input representation* and it seemed to help |
| Preprocessing ladder B0–B5: all data changes macro by < seed noise, shifts per-label strengths | Fold 0, 2 seeds | Weak: an engineering note |
| ACL forensics plus the pre-registered triplet test (wider spacing hurts ACL) | Fold 0, 2 seeds, paired bootstrap | Weak: a narrow and pipeline-specific result |
| Label-noise profile: training labels mostly *miss* positives (η⁻ ≫ η⁺) | Gold-58 only, n⁺ as low as 9, already reused | **No.** The 2nd-place writeup reports the same pattern (experts 240 positives vs reports 154; PF OA 21 vs 2), and every top team acted on it |
| Phase 0A registration audit and blinded review | Inconclusive by its own gates | No |

## Why a reviewer would reject it now

**R1. The base model is far from the state of the art.**
- Private 0.912 (rank 3,970 / 6,425) vs 0.957–0.960 for the top 10, and 0.942 for the public-notebook plateau.
- A reviewer reads "geometry does not help" as "geometry does not help *a weak pipeline*".
- Fig. 1b shows why the gap exists. The 4th-place team used the **same architecture** as ours (CoAtNet-2 window-MIL,
  three adjacent slices). On one fold it scored 0.925 private with report labels and 0.952 with VLM image-read labels.
  So the gap is mostly *labels*, not architecture, which makes a matched rerun on strong labels feasible.

**R2. The evaluation labels are the noisy training labels.**
- Every Δ we report is fold-validation AUC against `labels_v3` binarized: report-derived, missing 22–55% of gold
  positives on 7 labels.
- A geometry effect on exactly those image-only findings would be invisible or penalised.
- The geometry null therefore has an untested alternative explanation: **measurement insensitivity**.
- Gold-58 cannot settle it: n = 58, CIs about ±0.1 per label, and already reused.

**R3. Statistical scope is thin.**
- Fold 0 only, 2 seeds.
- The noise is large relative to the effects: the macro-AUC bootstrap SD is about 0.004, and per-label seed
  differences reach 0.022.
- An equivalence claim needs the whole CI inside a practical margin, across folds.

**R4. Outside evidence cuts against a broad reading.**
- **1st place.** A 6-channel variant that resamples the other series "in patient coordinates onto the current series'
  slices" gave "gain more than backbone ensemble".
- **7th place.** Tokens carry their 3D position in mm in a common knee frame.
- **6th place.** The whole model is built on a fitted 3D knee frame.
- None of these is a controlled ablation, but a reviewer will ask why geometry seemed to help them and not us.
- Our honest answer is narrow: *a Mahalanobis attention bias between separately acquired planes* added nothing.
  Geometry as a resampling or canonical-frame input was not tested.

**R5. LLM-use policy.**
- Design and analysis were developed with heavy LLM assistance (Claude Code; ChatGPT review rounds).
- MIDL 2027 prohibits LLM-generated research designs and analyses, so it is excluded.
- MELBA and most journals require disclosure. Disclosure is fine, but it must be complete.

**R6. Novelty of the label findings is gone.** Image-reading VLM labellers are now public, with code and label files
(1st, 2nd, 7th place). Our report-only engine is a baseline, not a contribution.

## The one route that could work (a 2027 short paper)

**Claim:** *in a competitive multi-planar knee-MRI classifier evaluated on expert image labels, an explicit
cross-plane footprint-correspondence bias gives no measurable gain; donor-geometry and β-off controls show why.*

**Requirements (all needed):**
1. **Expert-label evaluation.** Score B and G by Kaggle **late submission**.
   - The private split is about 1,300 studies graded by radiologists from images, which removes R2.
   - Late submissions are scored: a commenter under the 1st-place writeup reports a late submission at 0.956 private.
   - **To check:** that the competition rules allow late-submission scores in publications.
2. **Competitive labels.** Retrain B and G on a public image-read label set. Candidates:
   - 7th place, "RSNA Knee LLM Labels": gold-58 0.952;
   - the 1st-place team's labels.

   Target a matched baseline of about 0.95 private. **To check:** each label set's license and whether it may be
   used in a publication.
3. **Folds and seeds.** 5 folds × 2 seeds, paired, with an **equivalence** analysis (paired-bootstrap CI of Δgeo
   inside ±0.004) on both fold-validation and private-LB scores. About 40 runs × 2.6 h ≈ 100 H100-h, plus Kaggle
   GPU time for 4–8 late submissions.
4. **A new pre-registration** saying this cannot reverse the earlier STOPs, and stating its own margins.
5. **One geometry-as-input arm** (resampling other planes onto each series, as in 1st place). This answers R4 and
   turns the paper from "one mechanism failed" into "where geometry helps and where it does not".
6. **Full AI-use disclosure; target MELBA** or a MICCAI 2027 workshop, not MIDL.

**Chance of acceptance after these fixes:** moderate for a workshop or MELBA short paper, if the private-LB
equivalence holds or the geometry-as-input arm separates. Without fixes 1–3: low.

## What this means for the course presentation (24 / 26 Nov)

The project is a solid course result **as an honest engineering-and-research story**:
- a working end-to-end system;
- a documented diagnosis of why it trails the winners (labels; Fig. 1);
- rigorous pre-registered negative experiments (Figs. 2–3);
- a label-noise analysis that agrees with what the winners found (Fig. 4).

Present it that way. Do not claim publication-level novelty.
