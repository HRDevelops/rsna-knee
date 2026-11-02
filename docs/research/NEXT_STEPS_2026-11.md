# Decisions and plan after the geometry line closed (2026-11-02)

Based on review round 6 (`REVIEW_ROUND6_RESULTS.md` and the reviewer's reply), with the reviewer's external claims
checked.

## Decisions

| Item | Decision |
|---|---|
| Geometry research line | **Closed permanently** for this project. No IPE, jitter, kNN, all-series bias, other $\beta$ form, token grid or registration variant. A future geometry project would need a different mechanism or task (for example localization or segmentation) |
| How the negative result is stated | Narrowly: *explicit acquisition-footprint correspondence, implemented as a Mahalanobis cross-plane attention bias, did not provide useful diagnostic information in this multi-planar knee-MRI classification pipeline.* Not "geometry never helps MRI" |
| Two failure modes | Final-stage tokens: the pathway is dormant ($\beta$-off ΔAUC 0.0000). Local tokens: the pathway is active (median bias about −3.5 logits) but uninformative (donor swap ΔAUC about 0) |
| Phase 0A | Stays INCONCLUSIVE. The blinded review suggests real few-mm offsets in stable pairs and confirms the gross MI tail is optimizer failure |

## Publication facts (checked 2026-11-02)

- **MIDL 2027.** The author instructions allow LLMs "only for minor language edits". Generating "research ideas, study
  designs, analyses, results, or substantive text with an LLM is not allowed". The full-paper deadline is
  **4 December 2026**, one day before the course report.
  - This project's design and analyses were developed with heavy LLM assistance (Claude Code and ChatGPT review rounds).
  - So: no MIDL submission without first asking the programme chairs, and the history is never omitted. Not before the
    course deadline in any case.
- **MELBA** welcomes generalizable negative results and requires disclosure of AI use (per the reviewer's summary of its
  policy; to verify before any submission). It is a possible later target, if the null is characterized across folds.
- **Course report.** Check the course's own AI-use policy before submission.

## Plan for the next five weeks (rough split: 60% report and write-up, 30% labels, 10% ACL)

### 1. Course report, due 2026-12-05 (start now)

- **Source:** `docs/report/main.tex`, the existing skeleton.
- **Story:** strong competition model (private 0.912) → data audit (32.5% of slices used) → preprocessing ladder (all
  data, macro within noise, per-label shifts) → geometry audit (inconclusive) → pre-registered geometry experiments with
  causal controls (negative) → lessons.
- **Form:** written so it can later become a manuscript.

### 2. Optional: null characterization for a publication

Only if a paper is wanted. It needs its own pre-registration stating that it cannot reverse either STOP.
- **Runs:** remaining folds 1–4, seeds 42 / 43, frozen settings, for $B_{\text{cap}}$ ↔ $G_{\text{final}}$ and
  $B_{\text{local}}$ ↔ $G_{\text{local}}$. About 8 runs × 4 folds ≈ 50–70 H100-h.
- **Analysis:** equivalence / practical-null reasoning.
  - Does the paired-bootstrap CI of $\Delta_{\text{geo}}$ exclude +0.004? Strong equivalence means the whole CI lies
    inside [−0.004, +0.004].
  - The same for the donor swap against 0.001.
- External data (OAI) is bonus evidence, not a minimum.

### 3. Labels: the next research question

- **Question:** how much performance is lost because the supervision is noisy, finding-dependent and report-derived?
- **Model:** a per-finding noise-transition (observation) model with an unchanged architecture:
  $P(\tilde y_c=1\mid x)=(1-\eta^-_c)p_c+\eta^+_c(1-p_c)$.
- **Caution:** gold-58 has already influenced many decisions. Use it as prior or exploratory information for
  $\eta^\pm_c$, not for another search. A new small expert-read subset would be the right estimation set.
- **First step:** a short design note and pre-registration, then a fold-0 pilot on the frozen window-MIL B5 pipeline.

### 4. ACL forensics (cheap; no training yet)

From existing fold-0 summaries (per seed):

| Rung | B0 | B1 | B2 | B3 | B4 | B5 |
|---|---|---|---|---|---|---|
| ACL, seed 42 | 0.9533 | 0.9400 | 0.9198 | 0.9209 | 0.9093 | 0.9303 |
| ACL, seed 43 | 0.9317 | 0.9247 | 0.9199 | 0.9197 | 0.9124 | 0.9176 |

**Findings.**
- The drop already starts at B1, which changed pixels only (−0.013 / −0.007), and continues at B2, native windows
  (−0.020 / −0.005).
- The two B0 seeds differ by 0.022 on ACL alone, as large as the whole drop. The direction is consistent; the
  magnitude is within per-label seed noise.

**Before any targeted run:**
- Compute paired per-label bootstrap CIs for B0 → B1 and B1 → B2 from the existing OOF predictions.
- Only if those exclude 0, test fixed physical triplet spacing (centre ± 4 or 6 mm) against native adjacency on the
  frozen pipeline.

### 5. Deferred

- 5-fold B5 training: only if late Kaggle scoring is confirmed, or the report or a paper needs it.

**Still needed from the user:** RSNA data terms (before anything with images is published or pushed); late-submission
scoring; OAI access; course AI-use policy.
