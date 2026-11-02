# Label-noise transition model: design draft for review (2026-11-03)

Status: **draft, not pre-registered, nothing trained.** Written for the next review round.

## What the training labels look like against gold-58 (descriptive)

Training targets are `labels_v3_candidate` soft probabilities; fold-validation "truth" is the same file binarized at
0.5. Against the 58 image-read gold studies (binarized at 0.5). Caveat: v3's per-label blend weights were chosen by
LOO on these same 58 studies, so the v3 rates are optimistic.

| Target | Gold + | False-positive rate η⁺ | False-negative rate η⁻ | Agreement |
|---|---|---|---|---|
| ACL | 24 | 0.06 (2/34) | 0.08 (2/24) | 0.93 |
| MCL | 9 | 0.04 (2/49) | **0.44 (4/9)** | 0.90 |
| Medial meniscus | 26 | 0.06 (2/32) | 0.12 (3/26) | 0.91 |
| Lateral meniscus | 23 | 0.09 (3/35) | 0.22 (5/23) | 0.86 |
| Medial OA | 15 | 0.05 (2/43) | 0.27 (4/15) | 0.90 |
| Lateral OA | 11 | 0.04 (2/47) | **0.55 (6/11)** | 0.86 |
| PF OA | 21 | 0.05 (2/37) | 0.29 (6/21) | 0.86 |
| Effusion | 35 | 0.13 (3/23) | 0.03 (1/35) | 0.93 |
| Synovitis | 27 | 0.16 (5/31) | 0.30 (8/27) | 0.78 |
| Baker's | 12 | 0.02 (1/46) | 0.25 (3/12) | 0.93 |
| Contusion | 19 | 0.08 (3/39) | **0.42 (8/19)** | 0.81 |
| Fracture | 18 | 0.10 (4/40) | 0.39 (7/18) | 0.81 |

**Pattern.** The noise is mostly **missed positives** (η⁻ ≫ η⁺ for 10 of 12 labels), not over-calling. That is the
opposite of the original assumption that reports over-call mild disease: the severity-thresholded rubric seems to have
over-corrected. The pre-blend report labels (`labels_v2_final`) show the same pattern, plus a high η⁺ on effusion
(0.39) and contusion (0.26).

## The core design problem: what can measure a gain?

1. **Fold-validation AUC is scored against the same noisy labels.** A model that recovers true positives the labels
   missed is *penalised* there, so fold-val cannot detect the effect we want.
2. **Gold-58 is the only clean set, and it is weak and used.** n = 58, positives as few as 9 (MCL); it already chose the
   v3 blend weights and has steered many decisions. Per-label AUC CIs are about ±0.1.
3. **The forward model is rank-neutral at test time.** $P(\tilde y=1\mid x)=(1-\eta^-)p+\eta^+(1-p)$ is monotone in
   $p$ for each label, so it can only help through the training gradients (which positives the model learns to
   recognise), not by re-scaling predictions.

So a fold-0 pilot on the existing data can show at most a **mechanism** (for example, recall at fixed specificity on
gold-58 moves in the predicted direction for the high-η⁻ labels), not a decision-grade result.

## Options to discuss

- **(A) Fixed-η forward correction.** η from gold-58 (shrunk toward a pooled prior, e.g. Beta-binomial), frozen
  before training; loss = BCE on the forward-corrected probability. Cheap: one config flag. The evaluation problem is
  as above.
- **(B) Learned per-label η with a weak prior** (η regularised toward the gold-58 estimate). Identifiability is poor
  without anchor points; risk of absorbing real signal.
- **(C) New clean evaluation set first.** For example, about 100–150 additional studies, stratified toward the
  high-η⁻ labels (MCL, lateral OA, contusion, fracture, synovitis), read blinded against the same severity criteria.
  This is the only route to a decision-grade answer, but it needs expert readers. Who reads them is the user's call
  (a non-radiologist read would not count as gold).
- **(D) Label-side repair instead of a loss-side model.** Target the missed-positive pattern directly in the labeler
  (for example, relax the "on the fence → negative" rule for MCL / lateral OA / contusion). Tuning it on gold-58 again
  repeats the reuse problem.

## Questions for the reviewer

1. Given (1)–(3), is a fold-0 pilot worth running before a new clean set exists, and if so, what endpoint is honest?
   (A candidate is a pre-registered, gold-58-only, directional mechanism check: Δ recall at the B5 operating point on
   the five high-η⁻ labels, with no adoption decision.)
2. Should η be estimated from v2 (pre-blend, less contaminated by gold-58 selection) or v3 (what is actually trained
   on)?
3. Is (C) the necessary precondition for any publishable claim here?
