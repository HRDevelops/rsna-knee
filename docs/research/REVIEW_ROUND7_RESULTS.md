Thank you for the last review. I followed it: the geometry line is closed; I ran ACL forensics and then the targeted
test; and I drafted the label-noise design but did not train anything. The course report is next. Results, then
questions.

## 1. ACL forensics

**Step 1 (no training).** Paired per-label cluster bootstrap of the ladder: fold 0, seed-averaged, 879 studies /
672 report clusters, 1,000 reps.
- B1 → B2 (native-adjacent windows): ACL **−0.0199 [−0.0442, −0.0021]**.
- B0 → B5 (whole ladder): ACL −0.0207 [−0.0436, −0.0031].
- Other steps: no ACL effect.
- Caution: 72 label × step intervals, so some exclusions of 0 are expected by chance.

**Step 2 (pre-registered before any run).**
- **Design.** Frozen B5 window-MIL with only the in-window triplet spacing changed: centre ± round(Δ / slice spacing)
  native slices, with Δ = 7 mm and Δ = 10.5 mm. B5 itself uses native adjacency (about ±3.5 mm).
- **Runs.** Fold 0, seeds 42 / 43.
- **Rule.** Adopt only if the ACL 95% CI lower bound is > 0 *and* the macro change is ≥ −0.002.

| Arm | Macro | ACL | Δ macro | Δ ACL |
|---|---|---|---|---|
| B5 | 0.9586 | 0.9301 | — | — |
| ±7 mm | 0.9566 | 0.9227 | −0.0020 [−0.0053, +0.0008] | −0.0074 [−0.0221, +0.0073] |
| ±10.5 mm | 0.9552 | 0.9082 | −0.0034 [−0.0059, −0.0013] | **−0.0219 [−0.0384, −0.0065]** |

**Outcome.** No arm qualifies; B5 is unchanged. Wider spacing hurts ACL, monotonically, so the B1→B2 drop is *not*
caused by native adjacency being too narrow. Both arms also lose lateral meniscus (−0.012) and contusion (about
−0.004). My reading: the ladder's ACL drop is within per-label seed noise (the two B0 seeds differ by 0.022 on ACL),
and I plan no more ACL runs.

## 2. Label noise: what the training labels get wrong (descriptive, from gold-58)

Training targets are `labels_v3_candidate` soft probabilities. Fold-validation "truth" is the same labels binarized at
0.5.
- Caution: v3's per-label blend weights were chosen by LOO on these same 58 studies, so the v3 rates are optimistic.
- The rates below are v3 binarized vs gold.

| Target | Gold + | η⁺ (false positives) | η⁻ (missed positives) |
|---|---|---|---|
| ACL | 24 | 0.06 | 0.08 |
| MCL | 9 | 0.04 | **0.44** |
| Medial meniscus | 26 | 0.06 | 0.12 |
| Lateral meniscus | 23 | 0.09 | 0.22 |
| Medial OA | 15 | 0.05 | 0.27 |
| Lateral OA | 11 | 0.04 | **0.55** |
| PF OA | 21 | 0.05 | 0.29 |
| Effusion | 35 | 0.13 | 0.03 |
| Synovitis | 27 | 0.16 | 0.30 |
| Baker's | 12 | 0.02 | 0.25 |
| Contusion | 19 | 0.08 | **0.42** |
| Fracture | 18 | 0.10 | 0.39 |

**Pattern.** The noise is mostly **missed positives** (η⁻ ≫ η⁺ for 10 of 12 labels). That is the opposite of our
original assumption that reports over-call: the severity-thresholded rubric seems to have over-corrected. The pre-blend
report labels (v2) show the same pattern, plus a high η⁺ on effusion (0.39) and contusion (0.26).

**The design problem I see before building anything.**
1. Fold-val AUC is scored against the same noisy labels. A model that recovers the missed positives would be penalised
   there, so fold-val cannot detect the effect we want.
2. Gold-58 is the only clean set, and it is small and used: n = 58, as few as 9 positives (MCL), and it has already
   chosen the v3 blend weights and steered many decisions. Per-label CIs are about ±0.1.
3. The forward model $P(\tilde y=1\mid x)=(1-\eta^-)p+\eta^+(1-p)$ is monotone in $p$ for each label. It can only help
   through the training gradients, not by re-scaling test predictions.

**Options.**
- **(A)** Fixed-η forward correction: η from gold-58, Beta-binomial shrinkage, frozen before training.
- **(B)** Learned η under a prior.
- **(C)** Create a new clean evaluation set first: about 100–150 studies, enriched for the high-η⁻ labels, read
  blinded against the same criteria. This needs expert readers.
- **(D)** Repair the labeler for missed positives. This re-uses gold-58 again.

## Questions

1. Do you agree the ACL line is closed: treat the ladder ACL drop as seed noise, and report the spacing test as a
   negative?
2. Given points 1–3 in §2, is a fold-0 label-noise pilot worth running *before* a new clean set exists? If yes, what
   endpoint is honest? My candidate: a pre-registered, gold-58-only directional mechanism check (Δ recall at the B5
   operating point on the five high-η⁻ labels), explicitly with no adoption decision.
3. Should η come from v2 (pre-blend, less shaped by gold-58 selection) or from v3 (what is actually trained on)?
4. Is (C), a new expert-read set, a necessary precondition for any publishable label-noise claim? If so, what is the
   minimum size and design (enrichment, number of readers, adjudication) you would accept?
5. With about 4.5 weeks to the course report (5 Dec), would you now stop running experiments and write, or is there
   one experiment whose result would materially change the report?
