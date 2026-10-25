Phase 0A ran as pre-registered (`PLAN_v1_3.md` §5; pre-registration in `docs/EXPERIMENTS.md`, 2026-10-25). Below: the
outcome, what the numbers show, where the audit tool fails, and a proposed next step.

**Status: preliminary.** These numbers cover 140 of the 145 non-pilot studies. The last shard (5 studies) hit its time
limit and is rerunning; the final numbers will replace these.

## Outcome by the pre-registered rule: INCONCLUSIVE

G1 and G2 miss narrowly, and G3 only counts when they pass. I am not relabelling this as a pass.

| Gate | Bar | Result |
|---|---|---|
| G1 precision: injected 5 mm / 3° recovered within 1 mm and 0.5° (cross-plane pairs) | ≥ 95% | **96.7%** (n = 986), pass |
| G1 precision: injected 10 mm / 5° | ≥ 95% | **93.9%** (n = 986), fail |
| G2 cycle residual, Sag → Cor → Ax → Sag triangles, median | ≤ 1 mm | **1.06 mm** (n = 472), fail |
| G3 cross-plane pairs within 3 mm (max over 9 probes) and 2° | ≥ 97% | **80.8%** (n = 986) |

## What the measured corrections look like (cross-plane pairs; all 3 starts agree in 97.6%)

- **Size.** Correction to the MI optimum: median 1.69 mm / 0.66°. The 90th percentile is 3.95 mm / 1.70°, the 95th
  5.70 mm / 2.53°. At the study centre the median is 1.09 mm; 88.6% of pairs are within 3 mm at the centre and 2°.
- **In footprint units** (6×6 tokens, pair SD ≈ 8–11 mm): d median 0.19, p95 0.65, p99 2.36. So 95% of pairs need less
  than two-thirds of one footprint SD.
- **Gross cases** (> 10 mm or > 5°): 2.7%.
- **By pair type.** Sag–Cor 78.4% within, Sag–Ax 82.4%, Cor–Ax 83.9%. Same-plane pairs (reported only): 90.2%, median
  0.84 mm.
- **By field strength.** 1.5 T 87.9% within, 3 T 69.6%. Vendors range from 74% to 100%; small vendors have small n.

## Where the audit tool fails (post hoc, labelled as such)

- **Unstable pairs.** 2.4% of cross-plane pairs end in different basins from different starts. They hold most of the
  failures: 45.8% gross, injected-error recovery 29–38%.
- **Stable pairs.** On the other 97.6%, G1 recovery is 98.1% / 95.5% and gross corrections are 1.7%.
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

## Proposed next step (needs a decision)

1. **Keep the label "inconclusive".** No re-tuning of the audit to reach a pass.
2. **Take the branch the plan already defines for a G3 failure.**
   - Use raw DICOM geometry, and train $G$ (and only geometry, never pixels) with rigid misregistration noise drawn from
     the measured cross-plane corrections (empirical distribution, gross cases included).
   - This is conservative whether the residuals are real motion or MI bias.
   - The dose–response test then measures how sensitive $G$ is to exactly this kind of error.
3. **Optional: a blinded visual check** of about 40 cross-plane pairs at the DICOM placement (20 flagged, 20 random), to
   say whether the 3–6 mm tail is real.
4. **Then Phase 0B:** cache v2 and the preprocessing ladder (GPU).

**Questions:**
1. Do you agree with keeping "inconclusive" and taking the noise-model branch, rather than re-tuning the audit?
2. Is the visual check worth doing before Phase 0B?
3. Should the field-strength difference (3 T 69.6% vs 1.5 T 87.9% within) be modelled (for example, noise scaled by field
   strength), or treated as audit-tool behaviour?
