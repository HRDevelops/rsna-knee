Phase 0A ran as pre-registered (`PLAN_v1_3.md` §5; pre-registration in `docs/EXPERIMENTS.md`, 2026-10-25). Below: the
outcome, what the numbers show, where the audit tool fails, and a proposed next step.

**Final numbers:** 145 non-pilot studies; all 30 shards. Shard 2 hit its 3 h limit and was rerun as job 33518 with
per-study saving.

## Outcome by the pre-registered rule: INCONCLUSIVE

**Phase 0A was inconclusive because the registration audit narrowly failed its own validity gates (G1, G2).
Therefore the apparent G3 failure (80.8% vs 97%, not a narrow miss) cannot be interpreted as a validated estimate of
true cross-series misregistration.** Not relabelled as a pass; not retuned.

Intervals: study-cluster bootstrap, 145 studies, 500 replicates; the pairs within a study are not independent.

| Gate | Bar | Result [95% CI] |
|---|---|---|
| G1 precision: injected 5 mm / 3° recovered within 1 mm and 0.5° (cross-plane pairs) | ≥ 95% | **96.8%** [95.3, 98.0] (n = 1,020), pass |
| G1 precision: injected 10 mm / 5° | ≥ 95% | **94.1%** [92.3, 95.9] (n = 1,020), fail |
| G2 cycle residual, Sag → Cor → Ax → Sag triangles, median | ≤ 1 mm | **1.06 mm** [1.00, 1.14] (n = 488), fail |
| G3 cross-plane pairs within 3 mm (max over 9 probes) and 2° | ≥ 97% | **80.8%** [77.0, 84.5] (n = 1,020), not interpretable |

Median correction: 1.68 mm [1.57, 1.80] and 0.66° [0.61, 0.72].

## What the measured corrections look like (cross-plane pairs; all 3 starts agree in 97.5%)

- **Size.** Correction to the MI optimum: median 1.68 mm / 0.66°. The 90th percentile is 3.93 mm / 1.70°, the 95th
  5.60 mm / 2.45°. At the study centre the median is 1.09 mm; 88.8% of pairs are within 3 mm at the centre and 2°.
- **In footprint units** (6×6 tokens, pair SD ≈ 8–11 mm): d median 0.19, p90 0.45, p95 0.64, p99 2.26. So 95% of pairs
  need less than two-thirds of one footprint SD.
- **Gross cases** (> 10 mm or > 5°): 2.6%.
- **By pair type.** Sag–Cor 78.3% within, Sag–Ax 82.6%, Cor–Ax 83.7%. Same-plane pairs (reported only): 90.5%, median
  0.87 mm.
- **By field strength.** 1.5 T 88.2% within, 3 T 69.4%. Vendors range from 74% to 100%; small vendors have small n.

## Where the audit tool fails (post hoc, labelled as such)

- **Unstable pairs.** 2.5% of cross-plane pairs (n = 25) end in different basins from different starts. They hold most
  of the failures: 44% gross, injected-error recovery 32–40%.
- **Stable pairs.** On the other 97.5%, G1 recovery is 98.2% / 95.7%, gross corrections are 1.6%, and the median is
  1.66 mm / 0.65°.
- **QC overlays of the four worst-flagged pairs** (an informal look, not a rating): the DICOM placement looks aligned in
  all four. The "corrections" (24–136 mm, 10–103°) look like MI optima in the wrong place.
- **Tool resolution.** Cycle residuals of about 1 mm mean the tool's pairwise optima are themselves only consistent to
  about 1 mm. It therefore cannot tell 1–2 mm of real inter-sequence motion from MI bias.
  - One candidate bias is chemical-shift displacement of fat between fat-suppressed and non-suppressed sequences. It is
    larger at 3 T and could contribute to the field-strength difference. Not tested.

## How the tool was built (5 pilot studies, excluded from the gates)

1. **Single-start MI drifted to fake 48 mm corrections.** Fix: a 45 mm region of interest, an overlap guard, and
   multi-start search with the DICOM placement as a candidate.
2. **Mixed sampling objectives.** Fix: one objective over all voxels.
3. **Regular-step gradient descent stalled in the rotation–translation valley.**
   - A benchmark on real pairs (jobs 33416, 33420) showed Powell returns to the MI minimum after every injected error,
     where gradient descent ended 1.7–25 mm off.
   - Fix: Powell.

Synthetic phantom check: the identity is found within 0.12 mm, and injected errors are recovered within 0.01 mm.

## Decision (2026-10-25, after the reviewer's check)

Phase 0A is inconclusive because the audit validity gates did not both pass, so the G3 branches cannot be invoked
mechanically. **As a post-Phase-0A engineering decision** (not a pre-registered branch):

| Item | Decision |
|---|---|
| Formal Phase 0A result | INCONCLUSIVE |
| Retune the registration until it passes | No |
| Claim that G3 failed scientifically | No: G1 and G2 make it uninterpretable |
| Primary $G$ geometry | Raw DICOM coordinates, **no misregistration augmentation** (most offsets are below one support SD: d median 0.19, p95 0.64) |
| Full empirical residual augmentation, gross tail included | **No.** The tail is contaminated by optimizer failure and could teach $\beta_h\to0$ |
| $G_{\text{jitter}}$ | Secondary arm with bounded perturbations in support units, $d\le0.75$ |
| $\Sigma_{\text{reg}}=\sigma^2_{\text{reg}}I$ ($\sigma_{\text{reg}}$ 1–2 mm) added inside $d^2$ | Later ablation, only if $G$ works |
| 40-pair blinded visual check | Yes, in parallel, structured (below). It cannot change the Phase 0A label |
| Field strength | Robustness stratum, not a noise input. A descriptive clustered regression (3 T, vendor, pair type, FS mismatch, thickness, instability) comes first |
| Phase 0B | **GO** (independent of how geometry uncertainty is handled) |
| Phase 1 | Waits until the geometry-noise handling is pre-registered |

**Visual check design.**
- 40 cross-plane pairs: 20 random and 20 from the upper correction tail.
- Balanced across Sag–Cor / Sag–Ax / Cor–Ax and, as far as possible, 1.5 T / 3 T.
- Raw vs MI-corrected placement shown in random order and blinded.
- Scored as raw better / tie / corrected better, using fixed landmarks: the joint line, the posterior femoral condyle
  contour and the tibial plateau.

## Addendum 2026-11-02: blinded 40-pair review (descriptive)

Stable pairs: the MI-corrected placement was preferred in 24 of 30, mostly visible translations, suggesting real few-mm inter-sequence offsets. Unstable or gross pairs: the raw placement was preferred in 9 of 10, confirming the gross tail is optimizer failure. One rater; n = 40. The INCONCLUSIVE label is unchanged. Details are in docs/EXPERIMENTS.md, 2026-11-02.
