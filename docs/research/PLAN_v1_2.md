# Research plan v1.2 — acquisition-support-aware cross-plane fusion for multi-planar knee MRI (draft for review)

Supersedes `PLAN_v1_1.md` after review round 2 (2026-10-24). Nothing has been trained. The only new measurement is a
header-only audit (§1.3). Every number is measured in this repo (source given) or marked as an estimate.

**Core claim (v1.2).** Represent every multi-planar MRI feature as a physical 3D support distribution
$(\boldsymbol\mu,\Sigma)$, propagated through the actual DICOM acquisition and our preprocessing transform. Then let
cross-plane attention favour observations whose supports are physically compatible. The claim is about **acquisition
support**, not about patient coordinates as such: point coordinates in multi-plane MRI are already published (§1.5).

---

## 0. What changed since v1.1

| v1.1 | v1.2 | Why |
|---|---|---|
| Absolute IPE in raw patient axes | IPE in a study-local canonical knee frame: $P(\boldsymbol\mu-\mathbf m)$, $P\Sigma P^\top$, with $P=R_{\text{knee}}F_{\text{side}}$ | Raw axes expose leg pose to the network (round 2, #1). $F_{\text{side}}$ is a reflection for right knees (new, §2.3) |
| "exp018 cache vs cache v2 isolates the data effect" | **Withdrawn.** Ladder B0→B5, one change per rung, fold 0, 2 seeds, predictions stated first (§4) | Cache v2 changed six things at once (#2) |
| B\* = exp018 architecture on cache v2 | Three objects: $B_{\text{final}}$ (ladder winner), $B_{\text{cap}}$ (same token transformer, no geometry; **primary comparator**), $G$. $\Delta_{\text{geo}}=\mathrm{AUC}(G)-\mathrm{AUC}(B_{\text{cap}})$ | B\* was a different architecture (#3) |
| 4×4 tokens | 4×4 / 6×6 / 12×12 (divisors of CoAtNet-2's 12×12 map at 384 px), chosen on $B_{\text{cap}}$; start 6×6 | 35–40 mm cells are too coarse (#4). Tune the comparator, not the treatment |
| Every series in one flat bag | Plane-balanced window budget, within-plane $1/N_{\text{series}}$, same-series bias, random per-pass series slots (§2.5) | Sequence multiplicity must not become a prior (#5) |
| Register every series to one reference | Pose graph with cycle consistency, 6-DOF injections, residuals also in $d_M$ units; registration-free by default (§5, 0A) | Round 2, Q1 and the registration-free path |
| Shuffle geometry from a random study | Matched-donor swap, common-transform invariance (a unit test, §3.2), dose–response in support units with a predicted flat region | Round 2, Q2, and the support SDs of §2.2 |
| Hidden test as "confirmatory" | Development: paired 5-fold OOF. Independent benchmark: hidden test (public and private splits separately). Expert sets: direction, per-label pattern, H2 (§6) | Round 2, Q3, plus our power estimate (§6.2) |
| Per-plane k-NN as the main attention | Soft Mahalanobis bias over all tokens of a pass, so $G\supseteq B_{\text{cap}}$; hard per-other-plane top-$k$ as an ablation | Passes stay small (432 tokens at 6×6), so dense attention is cheap (§2.4) |
| "Area or Gaussian prefilter", 140 mm at 384 px | Pinned acquisition-aware prefilter; storage grid 160 mm at 384 px = 0.4167 mm, exp018's own pixel size | 71% of series are zero-fill interpolated (§1.3) |
| Screening gate +0.002 on one fold | Fold-0 futility rule on 2 paired seeds; decision only at 5 folds × 3 seeds | Measured seed noise: 0.0020 mean / 0.0046 max per fold (§1.4) |

## 1. Facts this plan rests on

### 1.1 Geometry (header-only, 809 series in 150 studies; `scripts/check_series_geometry.py`)

- Anisotropy about 10×: slice thickness median 3.0 mm vs in-plane pixels 0.31 mm.
- 0% of studies share one FrameOfReferenceUID (re-keyed), yet 100% have a common 3D box intersection.
- Series-centre spread within a study: median 8.5 mm, p90 15.2 mm, p99 46.0 mm.

### 1.2 Data usage of the current cache (`scripts/audit_data_usage.py`)

- Uses 32.5% of 819,078 slices and 84.3% of 24,371 series.
- Its "adjacent" windows are adjacent only after subsampling: about 40 mm thick in coronal non-FS.
- The centred 140 mm crop drops a median 23% of the image area.

### 1.3 Acquired vs reconstructed in-plane resolution (new; `scripts/audit_acquisition_resolution.py`)

1,663 series in 300 random studies. AcquisitionMatrix and phase direction were usable for 96.2%.

| | p10 | median | p90 |
|---|---|---|---|
| Reconstructed pixel (PixelSpacing), mm | 0.224 | 0.312 | 0.500 |
| Acquired spacing, frequency axis, mm | 0.376 | 0.498 | 0.682 |
| Acquired spacing, phase axis, mm | 0.463 | 0.625 | 0.833 |
| Zero-fill factor, frequency / phase | 1.00 / 1.25 | 1.69 / 2.00 | 2.05 / 2.86 |

- **Zero-fill interpolation is the norm.** 70.8% of series are reconstructed onto a finer grid than acquired
  (zero-fill > 1.05 on the frequency axis).
- **Little real resolution is lost at exp018's grid.** At 0.4167 mm, 79.1% of series are downsampled *nominally*
  (PixelSpacing finer than the grid) but only 18.8% *really* (acquired frequency spacing finer than the grid; median
  real factor 1.11), and 2.1% on both axes.
- **Correction to round 1.** We reported "76% of series downsampled with no anti-aliasing". That is true for
  PixelSpacing, but it loses little information. We therefore predict that the resampler change alone (rung B1) moves
  AUC by less than the noise floor.
- **Through-plane and protocol.** Slice thickness: median 3.0 mm (p10 2.71, p90 4.0). Gap: median 0.49 mm
  (p90 1.0). 3D sequences: 3.7%. Field strength: 1.5 T 60%, 3 T 36%. Vendors: Siemens 41%, Philips 35%, GE 18%,
  Canon/Toshiba 5%.

### 1.4 Noise floor (`docs/EXPERIMENTS.md`, 2026-09-25)

exp018 and exp021 are the same recipe with seeds 42 and 43.

- Per fold, after top-3 averaging, fold-validation macro-AUC differs by 0.0020 mean / 0.0046 max; gold-58 by
  0.0046 / 0.0067.
- 5-fold OOF: −0.0011 [−0.0031, +0.0008].
- A between-architecture difference at n = 4,349 has a paired cluster-bootstrap SE of about 0.0018 (exp018 vs exp006:
  +0.0135 [+0.0100, +0.0171]).

### 1.5 Prior art (checked 2026-10-24)

| Work | Geometry | Acquisition support / thickness | Fusion | Domain |
|---|---|---|---|---|
| Hu et al., MICCAI 2026, *Geometry-Aware SSL for Whole-Heart 3D+t* | Patient-space **point** coordinates in tubelet tokens | Not modelled | Learned voxel-query cross-attention. A reviewer suggested distance-aware attention; the rebuttal argued hard distance matching fails because of partial volume and slice thickness | Cardiac cine SAX/LAX, self-supervised |
| Das et al., Sci Rep 2024 (PMC11039716) | Per-voxel radial sinusoid of scanner coordinates | Implicit (sampling density only) | Single grid, no multi-plane | Brain infarct segmentation |
| Jabareen et al., 2025 (AFPE, arXiv:2509.02488) | Anisotropic Fourier-feature PE scaled by voxel spacing | Spacing only | Single volume | X-ray, CT, echo |
| CoPAS, Qiu et al., Nat Commun 2024 | None (rotated voxel arrays) | No | Learned co-plane attention | Knee MRI, different 12 findings |
| Park et al., 2026 (arXiv:2602.21636) | None (reslices of one isotropic volume) | No | Axial-centric cross-plane attention | MedMNIST3D |
| mip-NeRF; DETR3D / PETR / BEVFormer; Graphormer-3D / Uni-Mol | Integrated Gaussian PE; 3D points projected into views; distance-biased attention | Gaussian frustum (mip-NeRF) | Geometry-driven attention | Vision, molecules |

**What is ours:**
- MR acquisition support as a closed-form Gaussian $(\boldsymbol\mu,\Sigma)$ per token: slice profile, in-plane cell,
  actual triplet spread and orientation.
- A Mahalanobis compatibility bias across separately acquired planes. It answers the partial-volume objection in the
  MICCAI 2026 rebuttal directly.
- A causal validation (§3).

## 2. Method (first model = classification only)

### 2.1 Geometry through the whole chain

Native pixel $(u,v)$ of slice $i$ in series $s$ maps to
$\mathbf x=\mathbf o_{s,i}+u\,\delta^c_s\mathbf r_s+v\,\delta^r_s\mathbf c_s$. Cache v2 stores the exact 2D affine
$A_s$ from network pixels to native pixels.

**New:** $A_s$ includes the laterality mirror. Coronal and axial right knees are flipped horizontally; sagittal right
knees have their slice order reversed, which changes indices but not positions. So $\boldsymbol\mu$ is always the true
patient (LPS) position of the content the network sees. A unit test checks that tokens of mirrored series reproject onto
the same native pixels as the unmirrored computation.

### 2.2 Token support

v1.1's moment-matched triplet mixture, unchanged:

$$\Sigma_l=\tfrac{(\ell^c)^2}{12}\mathbf r\mathbf r^\top+\tfrac{(\ell^r)^2}{12}\mathbf c\mathbf c^\top+\tfrac{\tau^2}{12}\mathbf n\mathbf n^\top,\qquad
\boldsymbol\mu_j=\tfrac13\textstyle\sum_l\boldsymbol\mu_l,\qquad
\Sigma_j=\tfrac13\sum_l\big[\Sigma_l+(\boldsymbol\mu_l-\boldsymbol\mu_j)(\boldsymbol\mu_l-\boldsymbol\mu_j)^\top\big].$$

- **Inputs.** $\ell$ is the token cell size from $A_s$ and the grid ($160/g$ mm at a 160 mm field of view); $\tau$ is
  SliceThickness.
- **Thin-slice series** (spacing < 2 mm, mostly 3D): windows take every $k=\mathrm{round}(3\,\text{mm}/\text{spacing})$-th
  native slice, so window thickness is comparable across series. $\Sigma$ uses the actual positions.
- **Caveat.** The through-plane part of $\Sigma$ is physical. The in-plane part is the nominal pooling cell: CoAtNet-2's
  stages 3–4 attend globally within a slice, so a token's effective receptive field is wider than its cell.
  Ablation: a learned per-head scale on the in-plane part.
- **Typical sizes** (sagittal–coronal pair, 3.5 mm slice spacing, 3 mm thickness). Pair SD along the sagittal normal is
  11.9 / 8.3 / 4.9 mm at 4×4 / 6×6 / 12×12. Along the shared superior–inferior axis it is 16.3 / 10.9 / 5.4 mm.

### 2.3 Canonical frame and integrated positional encoding

**Frame construction.**
- $F_{\text{side}}=\mathrm{diag}(-1,1,1)$ for right knees and $I$ for left knees, which matches the 2D normalization to
  a left-knee frame.
- $R_{\text{knee}}$ is built from the reflected series orientations: $\mathbf e_1$ = sagittal normal; $\mathbf e_2$ =
  coronal normal orthogonalised against $\mathbf e_1$; $\mathbf e_3=\mathbf e_1\times\mathbf e_2$. Signs are fixed to the
  patient axes, with fallbacks when a plane is missing.
- So $P=R_{\text{knee}}F_{\text{side}}$ has $\det P=-1$ for right knees.
- $\mathbf m$ is the centre of the study's common 3D box intersection, which exists for 100% of studies (§1.1).

$$\mathbf h'_j=\mathbf h_j+W_p\,\gamma\big(P(\boldsymbol\mu_j-\mathbf m),\,P\Sigma_jP^\top\big),$$

$\gamma$ is mip-NeRF's integrated encoding, with frequencies axis-aligned in the canonical frame and wavelengths
4–256 mm.

- **Why this frame.** Technologists prescribe oblique planes relative to anatomy (sagittal along the ACL, coronal
  parallel to the posterior condyles), so $R_{\text{knee}}$ approximates an anatomical frame.
- **Residual pose** is handled by ±5° frame jitter in training.
- **No absolute orientation in the metadata MLP**, so the invariance of §3.2 holds.

### 2.4 Attention

For all token pairs of one pass:

$$a_{jk}=\frac{\mathbf q_j^\top\mathbf k_k}{\sqrt d}\;-\;\mathbb 1[s_j\neq s_k]\,\frac{\beta_h}{2}\,d_{jk}^2\;+\;b_h\,\mathbb 1[s_j=s_k],
\qquad d_{jk}^2=\Delta_{jk}^\top(\Sigma_j+\Sigma_k+\epsilon I)^{-1}\Delta_{jk},\qquad \beta_h=\mathrm{softplus}(\theta_h)\ge0 .$$

- **Cross-series pairs only.** The geometric term applies only between different series, where correspondence is the
  question. Within a series, neighbouring cells are context, not correspondence.
- **Multi-contrast pairs.** Same-plane sequences with the same slice prescription have $d\approx0$ at matching cells, so
  multi-contrast fusion at one location comes for free.
- **Invariance.** $d^2$ is invariant to rigid transforms and reflections, so it needs no canonical frame.
- **$G$ contains $B_{\text{cap}}$.** Heads with $\beta_h\to0$ stay global, so $B_{\text{cap}}$ is a special case of $G$.
  $\theta_h$ starts small, so $G$ starts close to $B_{\text{cap}}$. The learned $\beta_h$ are reported; if they all go to
  zero, geometry is unused.
- **Same-series term.** $b_h$ encodes identity, not geometry, and is in both arms.
- **Ablation: hard gate.** Keep only the $k=16$ nearest tokens per other plane by $d$ (round 2's recommendation). This is
  also the efficient form for dense 12×12 inference over all windows (about 22k tokens).

### 2.5 Token budget and sampling

- **Budget per pass:** $K=12$ windows per study, exp018's training budget.
  - Plane-balanced: 4 sagittal, 4 coronal, 4 axial.
  - A missing plane's share goes to the other planes.
  - Within a plane, each series gets a share $\propto1/N_{\text{series}}$.
  - Series slot embeddings are permuted at random each pass.
  - At 6×6 this gives 432 tokens per pass.
- **Inference**, two modes:
  - *Bagged:* the mean of $R=8$ random plane-balanced passes, which keeps training's token density.
  - *Dense:* one pass over all windows, about 150 per study in cache v2 (estimate).
  - Each arm uses its better mode on fold-0 validation.
  - Density matters more for $G$, because in a dense pass the nearest cross-plane neighbours are closer than in
    training.
- **Storage:** every series stays on disk; only the tokens that enter the transformer are balanced.

### 2.6 Readout and loss

Per-finding attention-MIL over tokens (exp018's head, applied to tokens). Loss: confidence-weighted BCE on the soft v3
labels. No localization loss in the first model.

### 2.7 Analysis-only outputs

Evidence centroids as in v1.1, for meniscal tear, contusion, fracture and Baker's cyst only.

## 3. Model objects and causal tests

### 3.1 Objects

| Object | Definition | Answers |
|---|---|---|
| $B_{\text{final}}$ | exp018 architecture (window-MIL) on the frozen ladder data | Does the whole proposal beat our strongest practical baseline? |
| $B_{\text{cap}}$ | Token transformer of §2.4–2.6 without $(\boldsymbol\mu,\Sigma)$. It has plane, series-slot, slice-index (after laterality normalization) and 2D-grid embeddings, the metadata MLP and dense attention | **Primary comparator.** Its indices already carry approximate geometry, so $\Delta_{\text{geo}}$ measures calibrated physical correspondence beyond index geometry |
| $G$ | $B_{\text{cap}}$ + canonical IPE + Mahalanobis bias | $\Delta_{\text{geo}}=\mathrm{AUC}(G)-\mathrm{AUC}(B_{\text{cap}})$ |
| Mechanism arms (fold 0) | $G_{\text{bias}}$ (no IPE), $G_{\text{IPE}}$ (no bias), $G_{k\text{NN}}$ (hard gate), $G_{\text{raw}}$ (IPE in raw patient axes) | Which channel carries a gain, if any |

$B_{\text{cap}}$ and $G$ share initialization (ImageNet), schedule, data and sampling. Seeds are paired: identical window
draws and identical initialization of shared weights.

### 3.2 Causal tests (inference only: geometry changes, pixels do not)

1. **Matched-donor swap.** Each study gets the geometry of a donor study matched on plane composition, side, field of
   view, slice count and sagittal obliquity. The geometry stays in distribution, but its link to the image content is
   broken. Prediction: $G$ falls below $B_{\text{cap}}$.
2. **Dose–response in support units.** Rigidly perturb all series of one plane: translation along its normal and
   in-plane, and rotation about $\mathbf m$.
   - Levels: $d$ = 0.25, 0.5, 1, 2, 4 for the median cross-plane pair. At 6×6 that is about 2, 4, 8, 17 and 33 mm along
     the sagittal normal.
   - Pre-registered: change < 0.002 macro for $d\le0.5$, then a monotone fall from $d\ge1$.
   - A flat curve at 2–4 mm is therefore expected, not a failure.
3. **Common-transform invariance.** Apply one rigid transform to all series, with side held fixed. With the canonical
   frame this holds by construction, so it is a **unit test** (logits equal within 1e-4), not evidence. The empirical
   version runs on $G_{\text{raw}}$.

### 3.3 Pre-registered per-label prediction

Compartment-specific findings should gain more than global ones: mean $\Delta_{\text{geo}}$ over both menisci, MCL,
medial OA and lateral OA > mean over effusion and synovitis (paired bootstrap).

- **Reason:** exp018's gains concentrated on these labels once laterality handling was fixed (`docs/EXPERIMENTS.md`
  2026-09-21).
- **If the pattern fails** while the macro gain holds, we do not attribute the gain to correspondence.

## 4. Cache v2 and the preprocessing ladder

### 4.1 Cache v2 (pinned)

- **Grid.** 160 mm field of view at 384 px = 0.4167 mm/px, exactly exp018's pixel size.
  - Its central 336 px is exp018's 140 mm crop on exp018's own grid, so the ladder can change one thing at a time.
  - At this pixel size, 81% of series lose no acquired resolution (§1.3).
  - About half the series need some padding at 160 mm (median field of view 160 mm).
- **Resampler (acquisition-aware, deterministic).** For each image axis $a$ (frequency or phase, from
  InPlanePhaseEncodingDirection), take the acquired spacing $\Delta_a$ = field of view$_a$ / acquired samples. Apply the
  Gaussian prefilter
  $$\sigma_a=\tfrac12\sqrt{\max(t^2-\Delta_a^2,\,0)}\ \text{mm},\qquad t=0.4167\ \text{mm},$$
  then linear interpolation through $A_s$.
  - This is round 2's $\sigma=0.5\sqrt{s^2-1}$, with the source resolution set to the acquired spacing instead of the
    zero-filled pixel.
  - Without an AcquisitionMatrix (3.8% of series), the rule falls back to PixelSpacing, i.e. round 2's rule as given.
  - No prefilter when upsampling.
- **Slices.** Every series and every native slice: 819,078 × 384² B ≈ 120 GB uint8. The 6–94% span stays the default
  window range.
- **Normalization.** Percentiles 1 and 99.5 per series, over its in-span slices inside the central 140 mm.
- **Per-slice record.**
  - Geometry: IPP, IOP, spacing, thickness, $A_s$ (including the mirror).
  - Acquisition: acquired matrix and phase direction; TR, TE, TI, flip angle, field strength, vendor and sequence tags.
  - Missingness flags for every field.
- **Unit tests.** Reprojection to DICOM ≤ 0.5 px on 100 random slices, including mirrored right knees. The central-crop
  geometry must match cache v1 for the same series.

### 4.2 Ladder (fold 0, seeds 42 and 43, exp018 architecture and training budget)

| Rung | Change (one at a time) | Prediction |
|---|---|---|
| B0 | exp018 as trained. Already exists (exp018 and exp021 fold 0); no new compute | — |
| B1 | Cache v2 pixels; same series, same 64 slices, same windows; 140 mm crop | \|Δ\| < 0.002 (§1.3) |
| B2 | + native-adjacent windows from all in-span slices of the same 5 series; exp018's plane mix (51.9 / 29.6 / 18.5% of windows) | No sign predicted |
| B3 | + every series (within-plane $1/N$); same plane mix | No sign predicted |
| B4 | + plane-balanced budget (1/3 each) | No sign predicted |
| B5 | + 160 mm field of view (full 384 px; no 336→384 resize, so anatomy is also 12.5% smaller at the network input) | No sign predicted |

- **Freeze rule.** Keep each rung unless it lowers the 2-seed mean by more than 0.004 (twice the mean seed gap). A
  rejected rung is reverted and the ladder continues from the rung before it.
- **Use of the result.** The frozen pipeline is $B_{\text{final}}$'s data, and $B_{\text{cap}}$ and $G$ use it too.
- **B1 still bundles small changes:** resampler, normalization range, and an exact affine instead of `zoom`. All are
  predicted to be small; if B1 exceeds the noise floor, we split it.

## 5. Phases and gates

| Phase | Content | Gate |
|---|---|---|
| **0A Geometry validity** | 150 studies, CPU. Pose graph over one series per plane family (10 pairs and 10 triangles per study). 6-DOF Mattes MI with SimpleITK, starting from the DICOM transform. Injected errors of 5 mm / 3° and 10 mm / 5°. Residuals reported in mm and in $d$ at 6×6 | **Optimizer precision:** injections recovered within 1 mm / 0.5° in ≥ 95% of cases; median cycle residual ≤ 1 mm. **Raw DICOM geometry:** if ≥ 97% of series pairs need corrections ≤ 3 mm / 2°, stay registration-free and mark the rest (no geometric bias for their cross-series pairs). Otherwise train with misregistration noise drawn from the measured residuals. **Never on the hidden test:** registration cannot run inside the notebook budget (about 1,300 studies × 10 pairs) |
| **0B Cache v2 + ladder** | §4 | Unit tests pass; ladder frozen by its rule |
| **0C Grid and inference mode** | $B_{\text{cap}}$ at 4×4 / 6×6 / 12×12, fold 0 × 2 seeds; dense vs bagged inference | Choose the best $B_{\text{cap}}$ grid on fold-0 validation and freeze it for $G$. Report $G$ at the other grids as a sensitivity analysis |
| **1A/1B Screening** | $G$ vs $B_{\text{cap}}$ with paired seeds, plus the mechanism arms; fold 0 | **Futility only.** Stop if the 2-seed mean $\Delta_{\text{geo}}\le0$, or if the donor swap lowers $G$ by < 0.002 (geometry unused). No positive decision on one fold: +0.002 equals the mean seed gap |
| **1C Confirmation** | $B_{\text{final}}$, $B_{\text{cap}}$ and $G$ at 5 folds × 3 seeds; the tests of §3.2; the per-label prediction of §3.3 | H1 holds if $\Delta_{\text{geo}}\ge+0.004$ OOF macro with paired cluster-bootstrap 95% CI lower bound > 0, the causal tests behave as pre-registered, and gold-58 drops by no more than 0.007 (the seed max) |
| **1D Hidden test** | Late submissions of $B_{\text{cap}}$ and $G$, if scoring works | Public and private splits reported separately, against a reference SE from the OOF bootstrap rescaled to each split's size |
| **2 Evidence** | Evidence centroids (v1.1) | Side accuracy clearly above chance on the hand-checked subset |
| **3 Robustness** | Missing plane, simulated thick slices, vendor, field strength; fastMRI+ coronal-only | Reported as found |

## 6. Evaluation

### 6.1 Endpoints

| Role | Endpoint |
|---|---|
| Development (primary, inferential) | Paired 5-fold OOF macro-AUC against report labels, 3 seeds, cluster bootstrap over report clusters |
| Independent benchmark | RSNA hidden test: about 1,300 expert-labelled studies, aggregate macro-AUC only. The public and private splits act as two replicates |
| Expert external | OAI (MOAKS reads, multi-plane DICOM) if access works; the locked RSNA subset if a radiologist reads it (§6.3) |
| Secondary | Gold-58, used only as a tripwire (it has been reused many times) |
| Robustness | fastMRI+ (coronal only), protocol strata, simulated protocol shifts |

- DeLong tests only on expert binary labels, with Holm correction.
- Cluster bootstrap for proxy labels; 3 seeds.
- Minimum effect: +0.004 macro OOF.

### 6.2 What the external sets can and cannot show (checked 2026-10-24)

| Dataset | Multi-plane | DICOM geometry | Labels | Use |
|---|---|---|---|---|
| OAI | Yes (sagittal IW-FS, coronal IW, sagittal DESS and reformats) | Yes | Expert MOAKS reads | The only external test of H1. Access needs a US login.gov account, and non-US users report failures |
| MRNet | Yes (3 planes) | **No** (npy arrays, 256²) | Extracted from reports | $G$ cannot run; $B_{\text{final}}$ only |
| fastMRI+ | **No** (coronal PD with or without fat suppression) | Single series | Expert boxes, 22 categories | Single-plane robustness (missing-plane condition) |
| SKM-TEA | **No** (one 3D qDESS) | Yes | Report-derived, with 3D boxes | Possible testbed: simulate thick orthogonal stacks with exact geometry (backlog) |

**Power (estimate from §1.4).**
- A paired macro-AUC difference has SE ≈ 0.0018 at n = 4,349.
- Scaled to other sizes, that is ≈ 0.008 at n = 200, ≈ 0.007 at n = 300 and ≈ 0.0033 on the whole hidden test.
- So expert sets of feasible size cannot confirm a +0.004 effect. They can confirm direction, the per-label pattern
  (§3.3) and H2.
- The effect size is therefore claimed from OOF. Noise in the proxy labels shrinks differences, so OOF is conservative.

### 6.3 Locked subset (proposal; needs the user's decision)

- **What.** Before any v1.2 training, lock about 200 non-gold training studies, sampled as whole report clusters and
  stratified by site and label prevalence.
- **Rule.** Exclude them from all training and selection.
- **Use.** If a radiologist reads them blind (12 findings plus 3D points for H2), they become an expert set. If not,
  they remain a clean holdout.
- **Cost.** About 4.5% of training data, the same for every arm.
- **Before starting:** check the RSNA data terms for re-annotation and publication.

## 7. Hypotheses and title

- **H1 (primary):** acquisition-support-aware correspondence improves multi-planar diagnosis over a capacity-matched
  no-geometry control on the same data, with the per-label pattern of §3.3.
- **H2:** evidence centroids agree with report-stated sides and structures, and with expert 3D points. Conditional: a
  claim only if it passes.
- **H3 (exploratory):** pivot-shift contusion footprint, only if H2 passes.
- **H4:** the dose–response follows the pre-registered curve in $d$, and under protocol shift (missing plane, thicker
  slices) $G$ degrades no more than $B_{\text{cap}}$.

**Title:** *Support-Aware Cross-Plane Attention: Fusing Anisotropic Multi-Planar Knee MRI by Physical Acquisition
Geometry.*

## 8. Compute (estimates; one exp018 fold ≈ 1 h 40 m on an H100)

| Block | Cost |
|---|---|
| 0A pose-graph registration | ≈ 15 CPU-hours (≈ 2,100 registrations, including injections) |
| Cache v2 build | ≈ 1 h wall on 20 CPU shards; ≈ 120 GB (quota ≈ 214 GB free) |
| 0B ladder (5 rungs × 2 seeds × fold 0) | ≈ 18 H100-h |
| 0C grid and inference mode (3 grids × 2 seeds) | ≈ 12 H100-h |
| 1A/1B screening ($G$ + 4 mechanism arms, × 2 seeds) | ≈ 20 H100-h |
| 1C confirmation (3 objects × 5 folds × 3 seeds) | ≈ 90 H100-h |
| CoPAS run (public code) | ≈ 20 H100-h |
| 2 Evidence / 3 Robustness | ≈ 15 / ≈ 30 H100-h |
| **Total** | **≈ 205 H100-hours** plus CPU |

## 9. Open questions for the reviewer (round 3)

1. **Bias form.** Given bagged inference, is a soft dense Mahalanobis bias the right main form ($G\supseteq B_{\text{cap}}$),
   with hard k-NN as an ablation?
2. **Screening.** With the measured noise floor, are futility-only screening and the 5 × 3 decision enough? Or would you
   add a third screening seed?
3. **Claims and power.** Given §6.2, is "effect size from OOF, direction from the hidden test (two splits) and expert
   sets" acceptable for MICCAI? Or should the primary claim be restricted to the compartment-specific labels, where the
   effect should be larger?
4. **Preprocessing.** Any objection to the acquisition-aware prefilter, or to the 160 mm / 0.4167 mm storage grid?
5. **Locked holdout.** Is a locked 200-study holdout worth 4.5% of the training data?
6. **Canonical frame.** Is a frame from the technologist's prescribed planes good enough? Or would you prefer an
   anatomical frame (for example from femoral-condyle landmarks) in a later version?
