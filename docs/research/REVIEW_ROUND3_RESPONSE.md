Thank you. All eight patches are applied; `PLAN_v1_3.md` is attached. I fixed the audit script and reran it on 600
studies (3,310 series). Below: what changed, what the rerun showed, and two small refinements. Unless you see a blocking
issue, I will freeze this plan and start Phase 0A.

## Patches

| # | Your point | Done in v1.3 |
|---|---|---|
| 1 | Audit: stale 0.365 mm grid, unused PercentSampling, "true resolution" wording | Fixed: one 0.4167 mm grid, renamed to "nominal acquisition-matrix spacing", PercentSampling audited, partial-Fourier flags counted, results reported by vendor. ReconstructionDiameter is not in this dataset's 84-tag allowlist, so it cannot serve as a check. The acquisition-aware prefilter is now an ablation (B1′), used only if B1 is not null. |
| 2 | Tokens are anchored, not supported | Reworded throughout. I checked the backbone: `coatnet_rmlp_2_rw_384` has two MBConv stages (stride 4 and 8), then 16 transformer blocks with global attention over 24×24 and 12×12 maps. A local-stage-token experiment (stage-1 output, stride 8) is in the backlog. |
| 3 | The primary bias must be cross-plane only | $\mathbb 1[\pi_j\neq\pi_k]$ in $G$, with $G_{\text{all-series}}$ as the ablation. The rerun shows why this matters: 97.5% of studies have ≥ 2 sagittal series (coronal 75.7%, axial 25.2%). The plane label agrees with the slice-normal axis in 100% of series. |
| 4 | Tune the control, freeze the treatment (inference mode) | Grid and inference mode are chosen on $B_{\text{cap}}$ only, then frozen for every arm. This was my error. |
| 5 | Validity mask for padding | $v_j$ is computed analytically from the affine and the native extent. Tokens with $v_j<0.5$ are masked as keys and dropped from the readout in every arm. Measured: 28.5% of series need some padding at 160 mm (v1.2's "about half" was wrong), and 1.8% of 6×6 tokens are mostly padding (0.8% at 4×4, 3.0% at 12×12). |
| 6 | Wording: "replicates", "conservative" | Replaced with your sentences. |
| Q1 | Nested initialization | Done, refined (below). |
| Q2 | Futility and escalation rules; primary statistic | Adopted exactly: seed-averaged OOF macro-AUC; fold 1 if ambiguous. Screening runs are reused in confirmation if the code is unchanged. |
| Q3 | Macro primary, compartment secondary | Adopted. |
| Q5 | Radiologist commitment gate | Adopted. If we lock, sampling is stratified so each finding has at least about 20 report-label positives. |
| Q6 | $G_{\text{bias}}$ as the core model | **Adopted.** The primary $G$ is now cross-plane footprint bias only: no frame, no absolute coordinates. $G_{+\text{IPE}}$ is the pre-specified enhancement arm, with the robust frame you described (all series per plane, robust mean, SVD, angle QC). |

## Audit rerun: the qualitative result holds, with vendor nuance

**Whole sample.** 69.6% of series are reconstructed finer than their acquisition matrix. At the 0.4167 mm grid, the
matrix spacing is finer than the grid on the frequency axis in 18.8% of series and on both axes in 2.1%.

**By vendor.**

| Vendor | Reconstructed finer than the matrix | Matrix finer than the grid, frequency / both axes |
|---|---|---|
| Siemens | 37.0% | 27.9% / 0.9% |
| Philips | 95.4% | 8.3% / 1.0% |
| GE | 91.0% | 17.0% / 3.7% |
| Canon/Toshiba (n = 94) | 97.9% | 26.6% / 24.5% |

**PercentSampling.**
- Siemens: phase count = frequency count × phase-FOV% × PercentSampling% in 100% of series, so the matrix already includes
  it and applying it again would double-count.
- Philips: the same identity holds in 43.7%.
- GE: PercentSampling is always 100, and its matrix already shows reduced phase resolution.
- Either reading leaves the phase axis finer than the grid in only 1.6–2.7% of series.

**Partial Fourier.** The ScanOptions PFP flag appears in ≤ 4.2% of series, but vendors often record it only in private
tags, so this undercounts.

**What we claim.** Unmeasured factors can only coarsen effective resolution, so 18.8% is an upper bound (if the matrix tag
is correct). The plan no longer depends on any estimate of effective resolution.

## Two refinements

1. **$\beta_h$ starts at exactly zero, by projection, not by softplus.**
   - Cross-plane tokens 60–100 mm apart at 6×6 have $d^2\approx30$–145. There, $\mathrm{softplus}(-4)=0.018$ still
     gives −0.3 to −1.3 logits, and $\mathrm{softplus}(-5)$ gives −0.1 to −0.5. So $G$ would not start as $B_{\text{cap}}$.
   - $\beta_h$ is therefore a free parameter, initialized at 0 and clamped to ≥ 0 after each optimizer step, with a live
     gradient at 0. It sits in the head group with no weight decay.
   - With $W_p=0$, $G$, $G_{+\text{IPE}}$ and $B_{\text{cap}}$ are identical at step 0.
   - Separate RNG streams for shared-weight initialization and for data, sampling and augmentation keep the arms paired.
     Unit tests check identical step-0 logits and identical batches.
2. **Default resampler: standard PixelSpacing-based anti-aliasing**, your round-2 rule $\sigma=0.5\sqrt{s^2-1}$ px, then
   linear interpolation.
   - It is deterministic and needs no MR reconstruction model.
   - Rung B1 tests it against exp018's unfiltered `zoom` (predicted null).
   - If B1 is not null, we split it and try the acquisition-aware variant.
   - I read "simplest deterministic" as the textbook rule rather than no anti-aliasing. Say if you meant the latter.

## Final check

1. Is anything in the patch list blocking for Phase 0A?
2. Any objection to the exact-zero projected $\beta_h$, or to anti-aliasing (rather than no prefilter) as the cache
   default?
