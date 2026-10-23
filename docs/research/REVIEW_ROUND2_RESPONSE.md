Thank you. Before revising, I checked both new citations and the wider literature, and ran one more header-only audit.

**Summary.**
- I accept all five issues, your six answers and both extra changes.
- I adjust six details, each for a measured reason (marked **Adjust**).
- I add four findings of my own.

The revised plan is attached as `PLAN_v1_2.md`.

## Citations: checked

- **MICCAI 2026, Hu et al., paper 0423: confirmed.**
  - It puts patient-space **point** coordinates into tubelet tokens and fuses them with learned voxel-query
    cross-attention.
  - It does not model slice thickness, support or uncertainty.
  - Reviewer 3 suggested "distance-aware attention". The authors' rebuttal replied that hard distance matching fails
    because of partial volume and slice thickness. That is exactly what $\Sigma$ models, so we present the bias as the
    answer to that objection.
  - Its code is public.
- **Das et al., Sci Rep 2024 (PMC11039716): confirmed.** A per-voxel radial sinusoid of scanner coordinates on a single
  grid (brain infarct segmentation). Resolution enters only through sampling density.
- **Added after a search:**
  - AFPE (arXiv:2509.02488): anisotropic Fourier PE scaled by voxel spacing, on single volumes.
  - An axial-centric cross-plane attention preprint (arXiv:2602.21636): reslices of one isotropic volume, with no
    acquisition geometry.
  - We also credit mechanism prior art: mip-NeRF (integrated PE), DETR3D / PETR / BEVFormer (geometry-driven multi-view
    attention) and Graphormer-3D / Uni-Mol (distance-biased attention).
- **Novelty statement:** adopted in your v1.2 wording.

## Your points

| # | Point | Verdict | Change in v1.2 |
|---|---|---|---|
| 1 | Absolute PE is not frame-invariant | **Accept fully.** My round-1 "narrower scope" was wrong | IPE in a canonical knee frame: $P(\mu-m)$, $P\Sigma P^\top$. **Addition:** $P$ needs a reflection for right knees ($\det P=-1$) to match our 2D laterality flip, and the preprocessing affine $A_s$ must include that mirror so $\mu$ stays the true position. The Mahalanobis bias is invariant to reflections as well. "No absolute PE" is kept as a mechanism arm. |
| 2 | Cache v2 does not isolate one effect | **Accept.** The line is withdrawn | Ladder B0→B5, one change per rung, fold 0 × 2 seeds, predictions stated in advance, freeze rule. A new storage grid makes rung B1 clean (finding A). |
| 3 | B\* is not capacity-matched | **Accept** | $B_{\text{final}}$ / $B_{\text{cap}}$ / $G$, with $\Delta_{\text{geo}}$ measured against $B_{\text{cap}}$. **Addition:** $B_{\text{cap}}$ must be the strongest no-geometry model (dense attention, slice-index and grid embeddings), not random k-NN. Its indices already carry approximate geometry, so $\Delta_{\text{geo}}$ measures *calibrated* physical correspondence. |
| 4 | 4×4 tokens are too coarse | **Accept** | Grids 4×4 / 6×6 / 12×12, the divisors of CoAtNet-2's 12×12 map at 384 px. 8×8 would need overlapping, unevenly spaced pooling bins with unequal supports. The grid is chosen on $B_{\text{cap}}$: tune the comparator, not the treatment. |
| 5 | Balance planes; no flat bag | **Accept** | 4/4/4 windows per pass, within-plane $1/N$, a same-series bias, and series slots permuted at random each pass. Fusion stays at token level, because a series bottleneck would remove the correspondence we are testing. |
| Q1 | Pose graph, cycle consistency, 6-DOF injections, $d_M$ residual | **Accept** | **Addition:** registration cannot run on the hidden test inside the notebook budget, so raw DICOM geometry is the only deployable path. Registration becomes an audit plus a noise model for training augmentation. |
| Q2 | Add a common-transform invariance control | **Adjust** | With a canonical frame, invariance holds by construction. So it is a unit test, not evidence. The empirical version runs on $G_{\text{raw}}$ (IPE in raw axes). The random-study shuffle becomes a matched-donor swap, for your out-of-distribution reason. |
| Q3 | Endpoint hierarchy | **Accept, with a power caveat** | See adjustment 5. |
| Q4 | Geometry-gated k-NN as the main form | **Adjust** | See adjustment 1. |
| Q5 | Novelty | **Accept** | H1 plus controls plus robustness is the core; H2 is an upgrade. |
| Q6 | Cache all, balance tokens | **Accept** | As #5. |
| Extra | Pin the resampler | **Accept the principle, adjust the rule** | See finding A. |
| Extra | Registration-free path | **Accept** | It is the default. Failures are marked, and their cross-series pairs get no geometric bias. |

## Adjustments and their reasons

1. **Soft bias as the main form; hard k-NN as an ablation.**
   - Your k-NN answer assumed dense inference over all windows. Here, training uses 12 windows per pass, and inference
     averages 8 passes of the same size, so token density matches training.
   - At 6×6 a pass has 432 tokens, so dense attention with the Mahalanobis bias is cheap. Each head learns its own
     locality ($\beta_h\ge0$); heads with $\beta_h\to0$ stay global.
   - That makes $G$ a strict superset of $B_{\text{cap}}$, which gives the cleanest reading of $\Delta_{\text{geo}}$.
   - Hard top-$k$ ($k=16$ per other plane) stays as an ablation. It is also the efficient form for dense 12×12 inference
     (about 22k tokens).
   - A second reason: with a hard gate, neighbour distances differ between sparse training passes and dense inference.
   - The bias applies only to cross-series pairs. Within a series, neighbouring cells are context, not correspondence.
2. **No positive screening decision on one fold.**
   - Measured seed noise (exp018 vs exp021, same recipe, seeds 42/43) is 0.0020 mean and 0.0046 max per fold in
     fold-validation macro-AUC. A +0.002 gain on one fold is seed noise.
   - Fold 0 becomes a futility rule: stop if the paired 2-seed mean is ≤ 0, or if the donor swap does not lower $G$.
   - The decision comes only at 5 folds × 3 seeds: ≥ +0.004 with the CI lower bound > 0.
3. **The dose–response is pre-registered in support units.**
   - At 6×6, the sagittal–coronal pair SD along the sagittal normal is about 8.3 mm. So 2 mm is $d\approx0.25$ and should
     do almost nothing.
   - Levels: $d$ = 0.25 / 0.5 / 1 / 2 / 4, about 2 / 4 / 8 / 17 / 33 mm.
   - Prediction: < 0.002 change up to $d=0.5$, then a monotone fall from $d=1$. A flat start is expected, not a failed
     control.
4. **The resampler is set against acquired resolution** (finding A).
5. **Expert sets cannot confirm +0.004.**
   - The paired macro-AUC difference has SE ≈ 0.0018 at n = 4,349 (from the exp018 vs exp006 CI). That scales to about
     0.008 at n = 200, 0.007 at n = 300 and 0.0033 on the whole hidden test.
   - So the effect size is claimed from OOF; proxy-label noise shrinks differences, so OOF is conservative.
   - The hidden test gives two replicates, because late submissions are scored on the public and private splits
     separately.
   - A newly read subset confirms direction, the per-label pattern and H2. I propose locking about 200 studies now,
     before any v1.2 training, whether or not a reader is found later.
6. **A per-label prediction is added** (below).

## New findings

**A. Most "downsampling" is of zero-filled pixels.** Source: `scripts/audit_acquisition_resolution.py`, 1,663 series in
300 studies; the acquisition matrix was usable for 96%.
- 71% of series are reconstructed onto a finer grid than acquired: median zero-fill 1.69× (frequency) and 2.0× (phase).
  Acquired spacing has a median of 0.50 mm (frequency) and 0.63 mm (phase).
- At exp018's 0.4167 mm grid, 79% of series are downsampled nominally but only 19% really (median real factor 1.11),
  and 2% on both axes. This corrects my round-1 alarm ("76% downsampled without anti-aliasing").
- **Rule.** Your $\sigma=0.5\sqrt{s^2-1}$, with $s$ taken against the *acquired* spacing per axis:
  $\sigma_a=\tfrac12\sqrt{\max(t^2-\Delta_a^2,0)}$ mm. It falls back to PixelSpacing when the matrix is missing. A
  PixelSpacing-based rule would blur most series for no gain.
- **Storage grid.** 160 mm at 384 px = 0.4167 mm, exp018's own pixel size. Its central 336 px is exp018's crop, so rung
  B1 changes only the resampler (predicted |Δ| < 0.002). Rung B5 then tests a 160 vs 140 mm field of view.

**B. Laterality in 3D.** See #1: $P$ needs $\det=-1$ for right knees, and $A_s$ needs the 2D mirror. Unit tests cover
both.

**C. Train/test token density.** See adjustment 1.

**D. External sets (checked).**
- MRNet ships npy arrays without DICOM geometry, so $G$ cannot run on it.
- fastMRI+ is coronal-only, so it can serve only as a missing-plane robustness test.
- SKM-TEA is a single 3D qDESS: a possible simulation testbed, kept in the backlog.
- OAI is the only multi-plane DICOM set with expert reads, and our access to it is uncertain.

## Pre-registered per-label prediction (new)

- Compartment-specific labels (both menisci, MCL, medial and lateral OA) should gain more than global ones (effusion,
  synovitis).
- The basis is exp018: its gains concentrated on these labels once laterality handling was fixed.
- If this pattern fails while the macro gain holds, we would not attribute the gain to correspondence.

## Questions for round 3

1. Given bagged inference, do you accept the soft dense bias as the main form and hard k-NN as an ablation?
2. With the measured noise floor, are futility-only screening and the 5 × 3 decision acceptable? Or would you add a third
   screening seed?
3. Given the power estimate, is "effect size from OOF; direction from the hidden test (two splits) and expert sets"
   acceptable for MICCAI? Or should the primary claim focus on the compartment-specific labels, where the effect should be
   larger?
4. Any objection to the acquisition-aware prefilter, or to the 160 mm / 0.4167 mm storage grid? About half the series
   need some padding at 160 mm.
5. Is a locked 200-study holdout worth 4.5% of the training data?
6. Is a canonical frame from the technologist's prescribed planes good enough? Or would you prefer an anatomical frame
   (for example from femoral-condyle landmarks) in a later version?
