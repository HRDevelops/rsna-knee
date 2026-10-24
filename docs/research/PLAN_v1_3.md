# Research plan v1.3 — acquisition-support-aware cross-plane fusion for multi-planar knee MRI (frozen for Phase 0)

Supersedes `PLAN_v1_2.md` after review round 3 (2026-10-24). The reviewer green-lit v1.2 subject to eight patches; all
eight are applied here. Nothing has been trained. The only new measurement is the corrected header audit (§1.3).
Every number is measured in this repo (source given) or marked as an estimate.

**Final amendments (review round 4, 2026-10-24: "GO, start Phase 0A"; no further plan review):**
1. Phase 0A gates apply to cross-plane pairs and to triangles with one series from each plane. Same-plane pairs and
   mixed triangles are reported but not gated (§5).
2. $\beta_h$ gets exact projected optimization: its own SGD without momentum (§2.4).
3. The reference cache has no added prefilter (exact affine + linear). Ordinary and acquisition-aware anti-aliasing
   are explicit ablations (§4).

**Scientific question.** Does representing anisotropic MRI observations by their physical acquisition footprint
improve cross-plane fusion?

**Core claim.** Each multi-planar MRI token is *anchored* to an acquisition footprint $(\boldsymbol\mu,\Sigma)$,
propagated through the actual DICOM acquisition and our preprocessing transform. Tokens from different anatomical
planes communicate more strongly when their footprints are physically compatible. The claim is not about patient
coordinates as such: point coordinates in multi-plane MRI are already published (§1.5).

---

## 0. What changed since v1.2

| v1.2 | v1.3 | Source |
|---|---|---|
| Audit had a stale 0.365 mm grid, called matrix spacing "true acquired resolution", and read PercentSampling without using it | Fixed and rerun on 600 studies. One 0.4167 mm grid; "nominal acquisition-matrix spacing"; PercentSampling, ScanOptions (partial Fourier) and a vendor breakdown (§1.3). ReconstructionDiameter is not in this dataset's tag allowlist | Round 3, #1 |
| Acquisition-aware prefilter as the cache default | Default: standard PixelSpacing-based anti-aliasing (round 2's rule). The acquisition-aware prefilter is an ablation, used only if rung B1 is not null | #1, Q4 |
| "A token's support is $(\boldsymbol\mu,\Sigma)$" | "Each token is **anchored** to an acquisition footprint $(\boldsymbol\mu,\Sigma)$". A local-stage-token mechanism experiment goes in the backlog | #2 |
| Bias on all cross-series pairs | **Primary: different anatomical planes only**, $\mathbb 1[\pi_j\neq\pi_k]$. All-series bias becomes an ablation | #3 |
| Each arm picks its own inference mode | Grid **and** inference mode chosen on $B_{\text{cap}}$ only, then frozen for every arm | #4 (blocking) |
| No padding handling | Per-token validity $v_j$ from $A_s$. Tokens with $v_j<0.5$ are removed from attention keys and from the readout, identically in every arm | #5 |
| "Two replicates"; "OOF is conservative" | "Two held-out leaderboard partitions"; OOF effect stated as a proxy-label effect (§6) | #6 |
| Primary $G$ = canonical IPE + bias | **Primary $G$ = cross-plane bias only** (no absolute coordinates, no frame estimator). $G_{+\text{IPE}}$ is the pre-specified enhancement arm | Q6 |
| $\beta_h=\mathrm{softplus}(\theta_h)$ with small init; $W_p$ random | $\beta_h\ge0$ by projection, initialized at **exactly 0**; $W_p=0$. Common random numbers; step-0 identity unit test (§2.4) | Q1 (exact zero is our refinement) |
| Futility: stop if 2-seed mean ≤ 0 | Stop only if both seeds ≤ 0 **and** mean ≤ −0.002, or if the donor swap moves $G$ by < 0.001 in both seeds. If ambiguous, run fold 1 with the same seeds (§5) | Q2 |
| Primary statistic undefined | Seed-averaged prediction per held-out study → one OOF macro-AUC per arm. Seed-level deltas are reported as stability | Q2 |
| Per-label pattern | Macro-AUC stays primary. Compartment vs global becomes the mechanistic secondary endpoint | Q3 |
| Locked holdout proposed unconditionally | Only with a radiologist commitment; rare-positive enrichment | Q5 |
| Frame from one series per plane | Robust frame: all reliable series per plane, sign-aligned, robust mean, SVD orthonormalization, angle QC (used only by $G_{+\text{IPE}}$) | Q6 |
| "About half the series need padding at 160 mm" | **28.5%**. 1.8% of 6×6 tokens are mostly padding | Audit rerun |

## 1. Facts this plan rests on

### 1.1 Geometry (header-only, 809 series in 150 studies; `scripts/check_series_geometry.py`)

- Anisotropy about 10×: slice thickness median 3.0 mm vs in-plane pixels 0.31 mm.
- 0% of studies share one FrameOfReferenceUID (re-keyed), yet 100% have a common 3D box intersection.
- Series-centre spread within a study: median 8.5 mm, p90 15.2 mm, p99 46.0 mm.

### 1.2 Data usage of the current cache (`scripts/audit_data_usage.py`)

- Uses 32.5% of 819,078 slices and 84.3% of 24,371 series.
- Its "adjacent" windows are adjacent only after subsampling: about 40 mm thick in coronal non-FS.
- The centred 140 mm crop drops a median 23% of the image area.

### 1.3 Header audit, corrected (`scripts/audit_acquisition_resolution.py`)

3,310 series in 600 random studies. AcquisitionMatrix and phase direction were usable for 96.2%.

"Nominal acquisition-matrix spacing" = field of view along an axis / matrix samples on that axis. It is not effective
resolution. Partial Fourier, sampling percentage, parallel imaging, k-space apodization and reconstruction filters are
invisible or only partly visible in the headers, and each can only make the effective resolution coarser. So, if the
matrix tag is correct, the share below is an **upper bound** on series whose effective resolution is finer than our grid.

| Vendor (series) | Reconstructed finer than the matrix (×>1.05, frequency axis) | Pixel finer than 0.4167 mm | Matrix spacing finer than the grid: frequency / both axes | Median matrix spacing, frequency / phase (mm) |
|---|---|---|---|---|
| **All (3,185)** | **69.6%** | **77.9%** | **18.8% / 2.1%** | 0.498 / 0.625 |
| Siemens (1,361) | 37.0% | 56.6% | 27.9% / 0.9% | 0.469 / 0.625 |
| Philips (1,114) | 95.4% | 97.5% | 8.3% / 1.0% | 0.500 / 0.602 |
| GE (601) | 91.0% | 87.0% | 17.0% / 3.7% | 0.500 / 0.714 |
| Canon/Toshiba (94) | 97.9% | 100% | 26.6% / 24.5% | 0.500 / 0.548 |
| Fujifilm/Hitachi (15) | 66.7% | 60.0% | 0% / 0% | 0.500 / 0.597 |

- **PercentSampling.**
  - Siemens: present in 100%, median 75 [60, 100]. The matrix phase count equals frequency count × phase-FOV% × PercentSampling% in **100%** of series, so the matrix already includes it; applying it again would double-count.
  - Philips: median 85 [75, 100]; the same identity holds in 43.7%.
  - GE: always 100, so uninformative. Its matrix shows reduced phase resolution directly (ratio median 0.70).
  - Canon/Toshiba: absent. Fujifilm: 0, which is invalid.
  - Either reading leaves the phase axis finer than the grid in only 2.7% (matrix as given) or 1.6% (PercentSampling applied on top).
- **Partial Fourier.** The ScanOptions flag PFP is set in ≤ 4.2% of series per vendor. Vendors often record it only in private tags, so this undercounts.
- **Reading.** The qualitative result holds: most images are reconstructed on a finer grid than their acquisition matrix, and at 0.4167 mm the matrix spacing is finer than the grid on one axis for ≤ 18.8% of series (2.1% on both). The headers cannot give effective resolution, so no core design choice depends on an estimate of it (§4.1).
- **Planes.** The Anatomical_Plane label agrees with the dominant axis of the slice normal in 100% of series. Studies with ≥ 2 series in the same plane: sagittal 97.5%, coronal 75.7%, axial 25.2%; 100% have this in some plane. So same-plane multi-contrast pairs are almost universal (relevant to §2.4).
- **Padding on the 160 mm grid.** 28.5% of series have a field of view < 160 mm on some axis. Tokens that are mostly padding ($v<0.5$): 0.8% / **1.8%** / 3.0% at 4×4 / 6×6 / 12×12.
- **Protocol.**
  - Slice thickness: median 3.0 mm (p10 2.5, p90 4.0). Gap: median 0.5 mm (p90 1.0).
  - 3D sequences: 4.0%.
  - Field strength: 1.5 T 60.6%, 3 T 38.9%.
  - Vendors: Siemens 41.1%, Philips 34.0%, GE 19.7%, Canon/Toshiba 4.7%, Fujifilm/Hitachi 0.5%.

### 1.4 Noise floor (`docs/EXPERIMENTS.md`, 2026-09-25)

exp018 and exp021 are the same recipe with seeds 42 and 43.

- Per fold, after top-3 averaging, fold-validation macro-AUC differs by 0.0020 mean / 0.0046 max; gold-58 by 0.0046 / 0.0067.
- 5-fold OOF: −0.0011 [−0.0031, +0.0008].
- A between-architecture difference at n = 4,349 has a paired cluster-bootstrap SE of about 0.0018 (exp018 vs exp006: +0.0135 [+0.0100, +0.0171]).

### 1.5 Prior art (checked 2026-10-24)

| Work | Geometry | Acquisition support / thickness | Fusion | Domain |
|---|---|---|---|---|
| Hu et al., MICCAI 2026, *Geometry-Aware SSL for Whole-Heart 3D+t* | Patient-space **point** coordinates in tubelet tokens | Not modelled | Learned voxel-query cross-attention. A reviewer suggested distance-aware attention; the rebuttal argued that hard distance matching fails because of partial volume and slice thickness | Cardiac cine SAX/LAX, self-supervised |
| Das et al., Sci Rep 2024 (PMC11039716) | Per-voxel radial sinusoid of scanner coordinates | Implicit (sampling density only) | Single grid, no multi-plane | Brain infarct segmentation |
| Jabareen et al., 2025 (AFPE, arXiv:2509.02488) | Anisotropic Fourier-feature PE scaled by voxel spacing | Spacing only | Single volume | X-ray, CT, echo |
| CoPAS, Qiu et al., Nat Commun 2024 | None (rotated voxel arrays) | No | Learned co-plane attention | Knee MRI, a different set of 12 findings |
| Park et al., 2026 (arXiv:2602.21636) | None (reslices of one isotropic volume) | No | Axial-centric cross-plane attention | MedMNIST3D |
| mip-NeRF; DETR3D / PETR / BEVFormer; Graphormer-3D / Uni-Mol | Integrated Gaussian PE; 3D points projected into views; distance-biased attention | Gaussian frustum (mip-NeRF) | Geometry-driven attention | Vision, molecules |

**What is ours:**
- Per-token acquisition footprints $(\boldsymbol\mu,\Sigma)$ in closed form: slice profile, in-plane cell, actual triplet spread, orientation.
- Separately acquired planes.
- A footprint-distance bias between planes. It answers the partial-volume objection in the MICCAI 2026 rebuttal directly.
- Causal validation (§3).

## 2. Method (first model = classification only)

### 2.1 Geometry through the whole chain

Native pixel $(u,v)$ of slice $i$ in series $s$ maps to $\mathbf x=\mathbf o_{s,i}+u\,\delta^c_s\mathbf r_s+v\,\delta^r_s\mathbf c_s$.
Cache v2 stores the exact 2D affine $A_s$ from network pixels to native pixels.

- **Laterality mirror.** $A_s$ includes it: coronal and axial right knees are flipped horizontally. Sagittal right knees
  have their slice order reversed, which changes indices but not positions. So $\boldsymbol\mu$ is always the true
  patient (LPS) position of the content the network sees.
- **Unit test (mirror).** Tokens of mirrored series must reproject onto the same native pixels as the unmirrored
  computation.
- **Validity.** $A_s$ and the native Rows/Columns also give each token's validity analytically:
  $v_j$ = share of the token cell covered by real image rather than padding. No mask image is stored.

### 2.2 Token anchor footprint

$$\Sigma_l=\tfrac{(\ell^c)^2}{12}\mathbf r\mathbf r^\top+\tfrac{(\ell^r)^2}{12}\mathbf c\mathbf c^\top+\tfrac{\tau^2}{12}\mathbf n\mathbf n^\top,\qquad
\boldsymbol\mu_j=\tfrac13\textstyle\sum_l\boldsymbol\mu_l,\qquad
\Sigma_j=\tfrac13\sum_l\big[\Sigma_l+(\boldsymbol\mu_l-\boldsymbol\mu_j)(\boldsymbol\mu_l-\boldsymbol\mu_j)^\top\big].$$

- **What it means.** The footprint says where a token is **anchored**, not where all the information in its feature
  came from.
- **Receptive field.** CoAtNet-2 (`coatnet_rmlp_2_rw_384`) has two MBConv stages (stride 4 and 8; 48×48 at 384 px),
  then 16 transformer blocks with **global** attention over 24×24 and 12×12 maps. A final-stage token can therefore carry
  information from the whole slice.
- **Through-plane vs in-plane.** The through-plane part of $\Sigma$ is physical (slice profile and actual slice
  positions). The in-plane part is the nominal pooling cell.
- **Inputs.** $\ell$ = cell size from $A_s$ and the grid ($160/g$ mm); $\tau$ = SliceThickness.
- **Thin-slice series** (spacing < 2 mm): windows take every $k=\mathrm{round}(3\,\text{mm}/\text{spacing})$-th native slice.
- **Typical pair SDs** (sagittal–coronal, 3.5 mm spacing, 3 mm thickness):
  - along the sagittal normal: 11.9 / 8.3 / 4.9 mm at 4×4 / 6×6 / 12×12;
  - along the shared superior–inferior axis: 16.3 / 10.9 / 5.4 mm.

### 2.3 Canonical frame (used only by the enhancement arm $G_{+\text{IPE}}$)

**Side reflection.** $F_{\text{side}}=\mathrm{diag}(-1,1,1)$ for right knees, $I$ for left, which matches the 2D
left-knee normalization.

**Robust $R_{\text{knee}}$:**
- For each plane, collect the normals of all series with valid geometry.
- Sign-align them to the patient axis.
- Take a robust mean (the principal eigenvector of $\sum\mathbf n\mathbf n^\top$).
- Set $\mathbf e_1$ = sagittal mean and $\mathbf e_2$ = coronal mean, then build $\mathbf e_3=\mathbf e_1\times\mathbf e_2$
  and orthonormalize by SVD (the nearest rotation).
- QC: record within-plane angular spread and the pre-orthonormalization angle between $\mathbf e_1$ and $\mathbf e_2$;
  flag disagreement > 10°.
- Fallbacks when a plane is missing.

**Encoding.** With $P=R_{\text{knee}}F_{\text{side}}$ and $\mathbf m$ = centre of the study's common box intersection:

$$\mathbf h'_j=\mathbf h_j+W_p\,\gamma\big(P(\boldsymbol\mu_j-\mathbf m),\,P\Sigma_jP^\top\big),\qquad W_p(0)=0 .$$

- $\gamma$ is mip-NeRF's integrated encoding: axis-aligned in the canonical frame, wavelengths 4–256 mm.
- Training applies ±5° frame jitter.
- No absolute orientation enters the metadata MLP.

### 2.4 Attention

For all valid token pairs of one pass:

$$a_{jk}=\frac{\mathbf q_j^\top\mathbf k_k}{\sqrt d}\;+\;b_h\,\mathbb 1[s_j=s_k]\;-\;\mathbb 1[\pi_j\neq\pi_k]\,\frac{\beta_h}{2}\,d_{jk}^2,
\qquad d_{jk}^2=\Delta_{jk}^\top(\Sigma_j+\Sigma_k+\epsilon I)^{-1}\Delta_{jk}.$$

- **Planes, not series.** $\pi$ is the anatomical plane. The primary bias acts only between *different planes*, which is
  what the claim is about. Same-plane multi-contrast pairs (≥ 2 sagittal series in 97.5% of studies) get no geometric
  term in $G$. The all-series variant is the ablation $G_{\text{all-series}}$.
- **Invariance.** $d^2$ is invariant to rigid transforms and reflections, so the primary $G$ needs no frame and no
  absolute coordinates.
- **Nested at initialization.**
  - $\beta_h\ge0$ is a free parameter, initialized at exactly 0 and projected to $\beta_h\leftarrow\max(\beta_h,0)$ after
    each optimizer step.
  - Its gradient at 0 is live, so geometry can switch on.
  - It has its own optimizer: projected SGD without momentum, $\beta_h\leftarrow\max(0,\beta_h-\eta\,\partial L/\partial\beta_h)$.
    The projection is then exact; AdamW plus clamping would keep stale momentum pointing outside the feasible region.
  - $\eta$ is fixed before Phase 1 from a gradient-scale check ($\beta$ must be able to move within the first epoch),
    never from AUC.
  - Why not a small softplus init: cross-plane pairs 60–100 mm apart at 6×6 have $d^2\approx30$–145. There,
    $\mathrm{softplus}(-4)=0.018$ gives −0.3 to −1.3 logits and $\mathrm{softplus}(-5)$ gives −0.1 to −0.5, so $G$ would
    not start as $B_{\text{cap}}$.
  - With $\beta_h=0$ and $W_p=0$, $G$, $G_{+\text{IPE}}$ and $B_{\text{cap}}$ are identical at step 0.
- **Common random numbers.** Separate generators for shared-weight initialization and for data order, window sampling
  and augmentation, so the extra geometry parameters never shift either stream.
  - Unit test: identical logits at step 0 on a fixed batch ($\max|z_G-z_B|<10^{-6}$ in fp32).
  - Unit test: identical batch composition across arms for the first 100 steps.
- **Validity.** Keys with $v_k<0.5$ are masked for every query and excluded from the readout, the same way in all arms.
  This is 1.8% of tokens at 6×6.
- **Same-series term.** $b_h$ is identity, not geometry, and is in all arms.
- **Ablation: hard gate.** $G_{k\text{NN}}$ keeps only the $k=16$ nearest tokens per other plane by $d$.

### 2.5 Token budget and sampling

- **Budget per pass:** $K=12$ windows per study.
  - Plane-balanced: 4 sagittal, 4 coronal, 4 axial.
  - A missing plane's share goes to the other planes.
  - Within a plane, each series gets a share $\propto1/N_{\text{series}}$.
  - Series slot embeddings are permuted at random each pass.
  - At 6×6 this gives 432 tokens per pass.
- **Inference.** Two modes:
  - *Bagged:* the mean of $R=8$ random plane-balanced passes.
  - *Dense:* one pass over all windows (about 150 per study; estimate).
- **Mode selection.** The mode is chosen on $B_{\text{cap}}$ only and then frozen for every arm (§5, 0C).
- **Storage.** Every series stays on disk.

### 2.6 Readout and loss

Per-finding attention-MIL over valid tokens. Loss: confidence-weighted BCE on the soft v3 labels. No localization loss.

### 2.7 Analysis-only outputs

Evidence centroids (v1.1), for meniscal tear, contusion, fracture and Baker's cyst only.

## 3. Model objects and causal tests

### 3.1 Objects

| Object | Definition | Role |
|---|---|---|
| $B_{\text{final}}$ | exp018 architecture (window-MIL) on the frozen ladder data | Practical baseline |
| $B_{\text{cap}}$ | Token transformer of §2.4–2.6 with no footprints. It has plane, series-slot, slice-index (after laterality normalization) and 2D-grid embeddings, the metadata MLP, dense attention and validity masking | **Primary comparator.** Indices carry approximate geometry, so $\Delta_{\text{geo}}$ measures calibrated physical correspondence |
| **$G$** | $B_{\text{cap}}$ + cross-plane footprint bias | **Primary model.** $\Delta_{\text{geo}}=\mathrm{AUC}(G)-\mathrm{AUC}(B_{\text{cap}})$ |
| $G_{+\text{IPE}}$ | $G$ + canonical IPE | Pre-specified enhancement: $\mathrm{AUC}(G_{+\text{IPE}})-\mathrm{AUC}(G)$ |
| $G_{\text{all-series}}$, $G_{k\text{NN}}$ | Bias on all cross-series pairs; hard top-16 gate | Mechanism arms (fold 0) |
| Backlog | Local-stage tokens: CoAtNet stage-1 output (stride 8, before the global-attention stages) vs final-stage tokens | Run if $G$ fails on final-stage tokens, or to strengthen the paper |

The hierarchy is $B_{\text{cap}}\rightarrow G\rightarrow G_{\text{all-series}}$, plus $G\rightarrow G_{+\text{IPE}}$. All arms share
initialization, schedule, data, sampling, grid, inference mode and the RNG streams.

### 3.2 Causal tests (inference only: geometry changes, pixels do not)

1. **Matched-donor swap.** Each study gets the geometry of a donor matched on plane composition, side, field of view,
   slice count and sagittal obliquity. Prediction: $G$ falls below $B_{\text{cap}}$.
2. **Dose–response in support units.** Rigidly perturb all series of one plane.
   - Levels: $d$ = 0.25, 0.5, 1, 2, 4 for the median cross-plane pair, about 2, 4, 8, 17 and 33 mm at 6×6.
   - Pre-registered: change < 0.002 macro for $d\le0.5$, then a monotone fall from $d\ge1$.
3. **Common-transform invariance.** One rigid transform applied to all series, with side held fixed. The primary $G$ is
   invariant by construction, and $G_{+\text{IPE}}$ through its canonical frame. So this is a **unit test** (logits equal
   within 1e-4), not evidence.

### 3.3 Mechanistic secondary endpoint (pre-registered)

The prediction: mean $\Delta_{\text{geo}}$ over both menisci, MCL, medial OA and lateral OA > mean over effusion and
synovitis (paired bootstrap).

- **Why expected:** exp018's gains concentrated on these labels once laterality handling was fixed.
- **Role:** macro-AUC says whether the model works; this pattern says whether it works for the proposed reason.
- **If it fails** while the macro gain holds, we do not attribute the gain to correspondence.

## 4. Cache v2 and the preprocessing ladder

### 4.1 Cache v2 (pinned)

- **Grid.** 160 mm field of view at 384 px = 0.4167 mm/px, exp018's pixel size. Its central 336 px is exp018's 140 mm crop
  on exp018's own grid.
- **Resampler (default): exact affine + linear interpolation through $A_s$, no added prefilter.**
  - This is the closest reproduction of exp018 (`zoom`, order 1), with an exact affine.
  - Linear interpolation already low-passes. Extra blur is an explicit, tested choice, not part of the reference cache.
  - Reason: about 70% of series are reconstructed finer than their acquisition matrix (§1.3), so a filter driven by
    PixelSpacing would mostly blur interpolated pixels.
- **Anti-aliasing ablations** (temporary exp018-layout caches, ≈ 30 GB each, deleted afterwards):
  - $B1_{\text{AA}}$: Gaussian prefilter $\sigma=(s-1)/2$ source pixels, scikit-image's default, with
    $s=t/\text{PixelSpacing}$ and $t=0.4167$ mm.
  - $B1_{\text{acq}}$ (optional): $\sigma$ from the nominal acquisition-matrix spacing.
  - Round 2's $\sigma=0.5\sqrt{s^2-1}$ is a Gaussian-PSF-matching heuristic, not a library default. It smooths
    considerably more: 0.45 vs 0.17 px at $s=1.34$.
- **Slices.** Every series and every native slice: 819,078 × 384² B ≈ 120 GB uint8. The default window range stays 6–94%.
- **Normalization.** Percentiles 1 and 99.5 per series, over its in-span slices inside the central 140 mm.
- **Per-slice record.** IPP, IOP, spacing, thickness, $A_s$ (including the mirror), native extent (for validity),
  acquisition matrix, phase direction, PercentSampling, ScanOptions, TR, TE, TI, flip angle, field strength, vendor and
  sequence tags, with missingness flags.
- **Unit tests.** Reprojection ≤ 0.5 px on 100 random slices, including mirrored right knees; central-crop geometry
  equal to cache v1 for the same series; analytic validity checked against rendered padding.

### 4.2 Ladder (fold 0, seeds 42 and 43, exp018 architecture and training budget)

| Rung | Change (one at a time) | Prediction |
|---|---|---|
| B0 | exp018 as trained. Already exists (exp018 and exp021 fold 0) | — |
| B1 | Cache v2 pixels (exact affine + linear, no added prefilter); same series, slices and windows; 140 mm crop | \|Δ\| < 0.002 |
| $B1_{\text{AA}}$ (ablation) | B1 + anti-aliasing, $\sigma=(s-1)/2$ | Adopted only if it raises the 2-seed mean by > 0.004; otherwise reported only |
| B2 | + native-adjacent windows from all in-span slices of the same 5 series; exp018's plane mix (51.9 / 29.6 / 18.5%) | No sign predicted |
| B3 | + every series (within-plane $1/N$); same plane mix | No sign predicted |
| B4 | + plane-balanced budget (1/3 each) | No sign predicted |
| B5 | + 160 mm field of view (full 384 px; no 336→384 resize, so anatomy is also 12.5% smaller at the network input) | No sign predicted |

- **Freeze rule.** Keep each rung unless it lowers the 2-seed mean by more than 0.004 (twice the mean seed gap). A
  rejected rung is reverted.
- **If B1 is not null.** Split it into its remaining parts: exact affine vs `zoom` grid mapping, and the normalization
  range.

## 5. Phases and gates

| Phase | Content | Gate |
|---|---|---|
| **0A Geometry validity** | 150 studies, CPU. Pose graph over one series per plane family (10 pairs and 10 triangles per study). 6-DOF Mattes MI with SimpleITK, starting from the DICOM transform. Injected errors of 5 mm / 3° and 10 mm / 5°. Residuals in mm and in $d$ at 6×6 | **G1 precision:** injected errors on cross-plane pairs recovered within 1 mm / 0.5° in ≥ 95% at each level. **G2 cycles:** median residual of Sag → Cor → Ax → Sag triangles ≤ 1 mm. **G3 raw geometry:** ≥ 97% of **cross-plane** pairs need corrections ≤ 3 mm and ≤ 2° (max over 9 probe points) → registration-free, with pairs over tolerance marked (no geometric bias for them). Otherwise train with misregistration noise drawn from the measured cross-plane corrections. Same-plane pairs and mixed triangles are reported, not gated. G1 and G2 must pass for the audit to count. **Never on the hidden test** (notebook budget). Code: `scripts/phase0a_registration.py`, `scripts/phase0a_report.py`, synthetic check `tests/test_phase0a_geometry.py` |
| **0B Cache v2 + ladder** | §4 | Unit tests pass; ladder frozen |
| **0C Grid and inference mode** | $B_{\text{cap}}$ at 4×4 / 6×6 / 12×12, fold 0 × 2 seeds; each run evaluated both dense and bagged | Choose grid and mode by $B_{\text{cap}}$ fold-0 validation, then **freeze for all arms**. $G$ under the other mode or grids is reported only as sensitivity |
| **1A/1B Screening** | $G$ vs $B_{\text{cap}}$ (paired seeds 42/43, fold 0); mechanism arms $G_{+\text{IPE}}$, $G_{\text{all-series}}$, $G_{k\text{NN}}$ (descriptive) | **Stop** if both paired seeds have $\Delta_{\text{geo}}\le0$ **and** mean ≤ −0.002, or if the donor swap moves $G$ by < 0.001 in both seeds. **Go** if both seeds > 0, mean ≥ +0.002 and the donor swap ≥ 0.001 in both. **Otherwise** run fold 1 with seeds 42/43 and continue only if the 4-run mean is > 0 and the donor swap shows geometry is used |
| **1C Confirmation** | $B_{\text{final}}$, $B_{\text{cap}}$, $G$ at 5 folds × 3 seeds; $G_{+\text{IPE}}$ at 5 × 3 as the pre-specified secondary. Screening runs are reused if the code is unchanged | **Primary statistic:** the 3-seed mean prediction per held-out study, all folds concatenated → one OOF macro-AUC per arm; paired cluster bootstrap for $\Delta_{\text{geo}}$. H1 holds if $\Delta_{\text{geo}}\ge+0.004$, the 95% CI lower bound is > 0, the causal tests behave as pre-registered, and gold-58 drops by no more than 0.007. Seed-level deltas are reported as stability |
| **1D Hidden test** | Late submissions of $B_{\text{cap}}$ and $G$, if scoring works | Public and private splits reported separately as two held-out leaderboard partitions, against a reference SE from the OOF bootstrap rescaled to each split's size |
| **2 Evidence** | Evidence centroids | Side accuracy clearly above chance on the hand-checked subset |
| **3 Robustness** | Missing plane, simulated thick slices, vendor (§1.3 strata), field strength; fastMRI+ coronal-only | Reported as found |

## 6. Evaluation

### 6.1 Endpoints and wording

| Role | Endpoint |
|---|---|
| Primary | Macro-AUC. Development estimate: paired OOF as defined in §5 (1C) |
| Mechanistic secondary | Compartment vs global $\Delta_{\text{geo}}$ (§3.3) |
| Independent benchmark | RSNA hidden test (about 1,300 expert-labelled studies; aggregate only). Public and private splits are two held-out leaderboard partitions of the same test distribution, not independent replicates |
| Expert external | OAI (MOAKS, multi-plane DICOM) if access works; the locked RSNA subset if a radiologist commits (§6.3) |
| Secondary | Gold-58, used only as a tripwire |
| Robustness | fastMRI+ (coronal only), protocol strata, simulated protocol shifts |

**Wording rules.**
- *"OOF against report-derived labels is the highest-powered paired development endpoint, but its effect size is a
  proxy-label effect and is not assumed to equal the effect under expert image-review labels."*
- Report-label errors can be pathology-, severity- and finding-correlated. They can therefore shrink, inflate or reverse a
  difference.
- Manuscript: *"The effect magnitude is estimated on high-powered paired OOF evaluation against report-derived labels.
  Independent expert-labelled benchmarks assess whether the direction transfers to image-review labels."*

**Tests.** DeLong only on expert binary labels (Holm correction); cluster bootstrap for proxy labels.

### 6.2 External sets (checked 2026-10-24) and power

| Dataset | Multi-plane | DICOM geometry | Labels | Use |
|---|---|---|---|---|
| OAI | Yes | Yes | Expert MOAKS reads | The only external test of H1. Access needs a US login.gov account, and non-US users report failures |
| MRNet | Yes | **No** (npy arrays, 256²) | Extracted from reports | $B_{\text{final}}$ only |
| fastMRI+ | **No** (coronal PD with or without fat suppression) | Single series | Expert boxes, 22 categories | Missing-plane robustness |
| SKM-TEA | **No** (one 3D qDESS) | Yes | Report-derived, with 3D boxes | Possible simulation testbed (backlog) |

**Power.** A paired macro-AUC difference has SE ≈ 0.0018 at n = 4,349. That scales to ≈ 0.008 at n = 200 and ≈ 0.0033 on
the whole hidden test. Expert sets of feasible size can test direction, catastrophic per-label failure, the
compartment pattern and H2; they cannot confirm +0.004.

### 6.3 Locked subset: radiologist commitment gate

- **Lock only if** expert reading is realistically likely within the project.
- **Then lock now**, before any v1.3 model exists:
  - 200 non-gold studies;
  - sampled as whole report clusters;
  - stratified, with oversampling so each finding has at least about 20 report-label positives (fracture, contusion,
    Baker's cyst, MCL and lateral OA are the rare ones).
- **Otherwise, do not lock.** A pseudo-label holdout adds little beyond cross-validation.
- **Before any re-annotation**, check the RSNA data terms.

## 7. Hypotheses and title

- **H1 (primary):** cross-plane footprint-compatibility attention improves multi-planar diagnosis (macro-AUC) over a
  capacity-matched no-geometry control on the same data. The compartment pattern (§3.3) is its mechanistic test.
- **H1b (secondary):** a canonical integrated positional encoding adds to $G$.
- **H2:** evidence centroids agree with report-stated sides and structures and with expert 3D points. Conditional: a
  claim only if it passes.
- **H3 (exploratory):** pivot-shift contusion footprint, only if H2 passes.
- **H4:** the dose–response follows the pre-registered curve, and under protocol shift $G$ degrades no more than
  $B_{\text{cap}}$.

**Title:** *Support-Aware Cross-Plane Attention: Fusing Anisotropic Multi-Planar Knee MRI by Physical Acquisition
Geometry.*

## 8. Compute (estimates; one exp018 fold ≈ 1 h 40 m on an H100)

| Block | Cost |
|---|---|
| 0A pose-graph registration | ≈ 15 CPU-hours |
| Cache v2 build | ≈ 1 h wall on 20 CPU shards; ≈ 120 GB (quota ≈ 214 GB free) |
| 0B ladder (5 rungs × 2 seeds) | ≈ 18 H100-h |
| 0C grid and inference mode (3 grids × 2 seeds) | ≈ 12 H100-h |
| 1A/1B screening ($G$ + 3 mechanism arms, × 2 seeds; + fold 1 if ambiguous) | ≈ 16 (+ 8) H100-h |
| 1C confirmation (4 objects × 5 folds × 3 seeds, minus reused runs) | ≈ 110 H100-h |
| CoPAS run (public code) | ≈ 20 H100-h |
| 2 Evidence / 3 Robustness | ≈ 15 / ≈ 30 H100-h |
| **Total** | **≈ 230 H100-hours** plus CPU |

## 9. Status and next step

- **Review status.** The reviewer recommends stopping plan review after these patches and starting Phase 0A.
- **What 0A is.** CPU-only registration on about 150 studies on the cluster, about 15 CPU-hours. No GPU work.
- **Start condition.** It starts when the user approves.
- **Open user decisions:**
  - OAI access (login.gov);
  - whether a radiologist can read a locked subset (this decides §6.3);
  - RSNA data terms for re-annotation and publication;
  - whether late submissions are scored.
