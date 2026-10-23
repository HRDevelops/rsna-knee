> **Superseded by `PLAN_v1_2.md` (2026-10-24, after review round 2).** Kept unchanged below as the record of what was
> reviewed. Known issues in this version:
> - the absolute positional encoding uses raw patient axes (not frame-invariant, and no reflection for right knees);
> - §3 claims exp018-cache vs cache v2 isolates the data-usage effect, but cache v2 changes six things at once;
> - B\* is not capacity-matched to the method;
> - the 4×4 grid was not tested;
> - all series enter as one flat bag;
> - the "76% downsampled without anti-aliasing" figure is nominal, because most of those pixels are zero-filled
>   (v1.2 §1.3).

# Research plan v1.1 — patient-space correspondence for multi-planar knee MRI (draft for review)

Supersedes `PLAN_v1.md` after review round 1 (2026-10-23). Nothing has been run yet except metadata audits. Every number
is measured in this repo (source given) or marked as an estimate.

**Core claim (narrowed):** use DICOM patient coordinates and acquisition support to fuse orthogonal knee-MRI features
at their physical intersections. Correspondence comes from geometry, not from learned attention between rotated volumes.

---

## 0. What changed since v1

| v1 item | v1.1 | Why |
|---|---|---|
| "The planes live in one physical frame" | **Candidate** common coordinates, verified by image registration before use (Phase 0.1) | 0% of series share a FrameOfReferenceUID; box overlap does not prove registration |
| World coordinates computed from raw PixelSpacing | Full preprocessing affine carried per slice: resample, crop, pad, network resize | Our network never sees the DICOM grid (this was a blocking bug in the formulation) |
| Canonical frame $\mathrm{diag}(\pm1,1,1)$ | Study-local knee basis from the series orientations, used only for interpretation and side labels | The core attention bias uses relative distances, which do not depend on the frame |
| Triplet covariance with a symmetric $\pm g$ | Moment-matched mixture of the three **actual** slice footprints | Sampled triplets are not symmetric |
| Bias $\log\mathcal N(\Delta;0,\Sigma_j+\Sigma_k)$ | Mahalanobis term only; any resolution preference becomes a separate, ablated term | $\log\mathcal N$ hides $-\tfrac12\log\lvert\Sigma\rvert$, which favours thin-slice tokens |
| "Attention gives a 3D lesion location" | "3D **evidence centroid** from class-specific attention", claimed only after validation | Attention is not guaranteed to be causal or faithful |
| $\mathcal L_{\text{geo}}$ in the first model | **Removed** from the first model; later only with safeguards | It can collapse onto an easy shared landmark such as the notch |
| ACL and MCL as point findings | Point centroids only for meniscal tears, contusion, fracture, Baker's cyst; ACL and MCL later as principal-axis regions | Ligaments are elongated structures |
| MR signal-equation embedding | Measured acquisition metadata + missingness masks → small MLP; the signal model stays only as an ablation | Signal equations are underdetermined (echo trains, fat-sat, coils, vendor reconstruction) |
| INR explicit-3D arm | Journal backlog | Separate research problem |
| Phase 0: 96 slices + SWA + mirror | Replaced by **cache v2** (data usage) and a matched baseline | The public "SWA" is three selected epochs averaged (`discussion/746792`); we already average our top 3 |
| "Gap to 0.942 is mostly data, labels, density" | "Plausible contributors; their effects are confounded" | The public author's own gains are small and confounded (+0.001 density, +0.005 OAI, +0.001 mirror) |
| Primary endpoint: OOF against report labels | Endpoint hierarchy (§5) | Report labels are our own pseudo ground truth |
| CoPAS: "same 12 findings", "no code" | **Corrected.** Different 12 categories; code is public | Read from the full text (PMC11368947) |

## 1. Facts this plan rests on (measured 2026-10-23)

**Geometry** (header-only check of 809 series in 150 studies; `scripts/check_series_geometry.py`):

- Anisotropy: slice thickness median 3.0 mm vs in-plane 0.31 mm, so about 10×.
- 0% of studies have one shared FrameOfReferenceUID.
- 100% of studies have a common 3D box intersection across their series.
- Series-centre spread within a study: median 8.5 mm, p90 15.2 mm, p99 46.0 mm.
- 3D acquisitions are about 3.6% of series.

**Data usage of the current cache** (`scripts/audit_data_usage.py`, metadata only):

| | Value |
|---|---|
| Series used | 84.3% (20,539 / 24,371). Dropped: axial 1,491, sagittal 1,420, coronal 921 |
| Slices used (unique) | **32.5%** of 819,078; 38.0% within the series that are used |
| Subsampling per slot (native in span ÷ kept) | Sag-FS 1.44, Sag 1.86, Cor-FS 2.17, Cor 3.25, Ax 2.33 |
| Consequence | A "3-adjacent-slice" window is 3 *sampled* slices: about 40 mm thick in coronal non-FS (slice-spacing audit, `docs/EXPERIMENTS.md`) |
| In-plane | 76% of series are downsampled to the 0.417 mm grid (median 1.29×, p90 1.85×) by `scipy.ndimage.zoom(order=1)`, with **no anti-alias prefilter** |
| Field of view | Median 160 mm. A centred 140 mm crop drops a median 23% of the image area; 5.7% of series are padded |
| Storage | Raw 531 GB; current cache 30 GB; all in-span slices at 384² uint8 ≈ 106 GB (quota has ≈ 214 GB free) |

**CoPAS (Qiu et al., Nat Commun 2024), read in full:**

- It already argues that interpolating the thick axis adds no information, and uses orthogonal planes instead.
- Its "cross-plane volumes" are made by rotating each voxel array, with no patient coordinates. Correspondence is learned
  by cross-attention, and the crop is centred using a meniscus segmentation.
- Its 12 categories differ from RSNA's (they include PCL, LCL, plica, fat pad and retinaculum).
- Code: github.com/zqiuak/CoPAS. It can be run as a baseline.

## 2. Method (first model = classification only)

### 2.1 Geometry through the whole preprocessing chain

For slice $i$ of series $s$: $\mathbf o_{s,i}$ (ImagePositionPatient), $\mathbf r_s,\mathbf c_s$ (ImageOrientationPatient), and
$\mathbf n_s=\mathbf r_s\times\mathbf c_s$. Native pixel $(u,v)$ (column, row) maps to

$$\mathbf x=\mathbf o_{s,i}+u\,\delta^c_s\,\mathbf r_s+v\,\delta^r_s\,\mathbf c_s .$$

Cache v2 stores, per series, the exact 2D affine $A_s$ from network pixels $(u',v')$ to native pixels: resample scale
(including rounding), crop and pad offsets, and the network resize. Then

$$[u\;v\;1]^\top=A_s\,[u'\;v'\;1]^\top .$$

A token is one cell of a $4\times4$ adaptive-pooled feature map of one triplet. Its in-plane centre and its physical cell
size $(\ell^c_s,\ell^r_s)$ in mm both come from $A_s$, never from raw PixelSpacing.

### 2.2 Token support from the actual triplet

Slice $l\in\{1,2,3\}$ of the triplet has centre $\boldsymbol\mu_l$ (through $A_s$ and its own $\mathbf o_{s,i_l}$) and footprint

$$\Sigma_l=\tfrac{(\ell^c_s)^2}{12}\mathbf r_s\mathbf r_s^\top+\tfrac{(\ell^r_s)^2}{12}\mathbf c_s\mathbf c_s^\top+\tfrac{\tau_s^2}{12}\mathbf n_s\mathbf n_s^\top .$$

The token's moments are those of the equally weighted mixture:

$$\boldsymbol\mu_j=\tfrac13\textstyle\sum_l\boldsymbol\mu_l,\qquad
\Sigma_j=\tfrac13\sum_l\big[\Sigma_l+(\boldsymbol\mu_l-\boldsymbol\mu_j)(\boldsymbol\mu_l-\boldsymbol\mu_j)^\top\big].$$

This handles unequal gaps, oblique planes and asymmetric triplets. In cache v2, triplets are native-adjacent slices
(§3), so the normal spread is the true one: about 3–4 mm, not about 40 mm.

### 2.3 Acquisition-aware encoding and cross-plane attention

$$\mathbf h'_j=\mathbf h_j+W_p\,\gamma(\boldsymbol\mu_j-\mathbf m,\Sigma_j),\qquad
\gamma(\boldsymbol\mu,\Sigma)=\big[\sin(\boldsymbol\omega_k^\top\boldsymbol\mu)\,e^{-\frac12\boldsymbol\omega_k^\top\Sigma\boldsymbol\omega_k},\ \cos(\cdot)\,e^{-\frac12\boldsymbol\omega_k^\top\Sigma\boldsymbol\omega_k}\big]_k$$

This is the integrated positional encoding: it damps frequencies finer than the slice resolves along the thick axis.
$\mathbf m$ is the study centre.

$$a_{jk}=\frac{Q_jK_k^\top}{\sqrt d}-\frac{\beta_h}{2}\,\Delta_{jk}^\top(\Sigma_j+\Sigma_k+\epsilon I)^{-1}\Delta_{jk},
\qquad \Delta_{jk}=\boldsymbol\mu_j-\boldsymbol\mu_k,\quad \beta_h\ge 0 .$$

Neighbours are restricted per **other** plane: the $k$ nearest sagittal→coronal, sagittal→axial, and so on, so one dense
plane cannot fill the neighbour set. A resolution-preference term $-\tfrac{\beta'_h}{2}\log\lvert\Sigma_k\rvert$ is a separate
ablation, off by default.

Readout is per-finding attention-MIL, as in exp018. Classification loss only: confidence-weighted BCE on the soft v3
labels.

### 2.4 Outputs used as analysis only (not losses)

The per-plane class-attention moments $(\boldsymbol\mu^{(\pi)}_c,\Sigma^{(\pi)}_c)$ and the precision-weighted
**evidence centroid** $\boldsymbol\mu^\star_c$, for meniscal tear, contusion, fracture and Baker's cyst only.

### 2.5 Controls that make the claim causal

1. **Capacity-matched no-geometry control.** Same cross-plane transformer, same tokens, same neighbour budget (but
   random, not geometric), slice-index positions, no geometric bias.
2. **Geometry shuffle.** Each series gets the geometry of a random other study. Gains must disappear.
3. **Dose–response.** Inject test-time misregistration of 2 / 5 / 10 mm into one plane. AUC must fall monotonically if
   geometry is really used.

## 3. Cache v2 (data usage; prerequisite for both the baseline and the method)

- **Every series and every native slice in the 6–94% span.** No slot cap and no subsampling. About 106 GB uint8 at 384².
- **Anti-aliased resampling** (area or Gaussian prefilter) to a fixed physical grid. Start at 140 mm and 384 px
  (0.365 mm); test 160 mm later.
- **Per-slice record:** ImagePositionPatient, ImageOrientationPatient, spacing, thickness, the affine $A_s$, and the
  acquisition metadata (TR, TE, TI, flip angle, field strength, sequence tags, vendor) with missingness flags.
- **Windows** are 3 native-adjacent slices around a sampled centre. Training samples random centres per plane; inference
  is dense.

**B\* (matched baseline)** = exp018's model on cache v2, with native-adjacent triplets and the same token set and budget
as the method, minus the geometry.

- The first comparison, exp018 cache vs cache v2 on fold 0, measures the data-usage effect alone.
- All later claims are against B\* and control (1).

## 4. Phases and gates

| Phase | Content | Gate to continue |
|---|---|---|
| **0.1 Alignment** | Rigid 6-DOF mutual-information registration of every series to a reference series. Start from the DICOM transform, ≥ 100 studies, CPU. Positive control: inject a 10 mm shift and check it is recovered. | Median residual ≤ 3 mm, p95 ≤ 5 mm, gross failures (> 10 mm or > 5°) < 3% of pairs, with an MI peak at the DICOM transform. Otherwise estimate a per-series rigid $T_s$ and store it in cache v2. |
| **0.2 Cache v2** | Build with full geometry; unit tests that the affine reprojects to DICOM within 0.5 px | Reprojection test passes on 100 random slices |
| **0.3 B\*** | exp018 vs B\* on fold 0 | Report the effect, whatever its sign; B\* becomes the reference |
| **1 Core** | Method vs B\* vs control (1), fold 0, then 5 folds × 3 seeds; controls (2) and (3) | Continue if ≥ +0.002 OOF over **both** B\* and control (1) on fold 0, and control (2) erases the gain |
| **2 Evidence** | Evidence-centroid validation (§5); only then $\mathcal L_{\text{geo}}$ with safeguards (positive finding, ≥ 2 planes with high evidence, stop-gradient consensus); ACL/MCL as principal axes | Side accuracy clearly above chance on a hand-checked subset |
| **3 Robustness** | Metadata conditioning; vendor, field strength, missing plane, simulated thick slices | Reported as found |
| Backlog | INR explicit 3D; signal-model ablation; H3 biomechanics (needs Phase 2 to pass) | — |

## 5. Evaluation

| Role | Endpoint |
|---|---|
| Development | 5-fold OOF macro-AUC against report labels (binarised v2), paired cluster bootstrap over report clusters |
| **Confirmatory** | The RSNA hidden test (expert image-review labels, about 1,300 studies) through late submission, **if** scoring is confirmed. The page shows a "Late Submission" button (2026-10-23), but we have no host statement. Only macro-AUC is returned, and submissions are rationed. Fallback: expert-labelled external sets on overlapping findings. |
| Secondary | Gold-58: reused many times by us and by the public notebooks, so not independent |
| Generalization | MRNet (ACL, meniscus), fastMRI+ (**projected 2D** localization, coronal only), SKM-TEA (to check); OAI only if access works (it needs a US login.gov account, and non-US users report failures) |
| Robustness | Manufacturer, 1.5 T vs 3 T, missing plane, physics-simulated thicker slices |

DeLong only on expert binary labels, with Holm correction. Cluster bootstrap for proxy-label comparisons. 3 seeds.
Pre-registered minimum effect: +0.004 macro OOF, about one bootstrap SD.

**Evidence-centroid validation (H2):**
- Report-derived side and structure (LLM), with 100 reports checked by hand.
- A hand-annotated subset of RSNA studies if an annotator is available.
- Projected 2D hits on fastMRI+ boxes.
- Note: 3DReasonKnee (arXiv:2510.20967) links MOAKS reads to OAI MRI series and might give 3D references, but it
  sits behind OAI access. To check.

## 6. Hypotheses and title

- **H1 (primary):** explicit patient-space correspondence improves multi-planar diagnosis over a capacity-matched
  learned-attention control on the same data.
- **H2:** class-attention evidence centroids agree with report-stated sides and structures, and with projected external
  boxes. Conditional; it is a claim only if it passes.
- **H3 (exploratory):** the pivot-shift contusion footprint. Only if H2 passes.
- **H4:** degrades less under misregistration-free protocol shifts (thicker slices, missing plane), and degrades under
  injected misregistration (the dose–response control).

**Title:** *Patient-Space Correspondence for Multi-Planar Knee MRI: Uncertainty-Aware Cross-Plane Attention in DICOM
Coordinates.*

## 7. Compute (estimates)

| Block | Cost |
|---|---|
| Phase 0.1 registration | ~2 CPU-hours on the ZEN4 nodes |
| Cache v2 build | ~1 h wall over 20 CPU shards, ~106 GB |
| B\* fold 0 | ~2–3 H100-hours |
| Phase 1 | ~150 H100-hours (fold-0 screening plus 5 folds × 3 seeds × 3 arms) |
| Phase 2 | ~15 H100-hours (inference + LLM extraction) |
| Phase 3 | ~50 H100-hours |
| **Total** | **≈ 220 H100-hours** |

## 8. Open questions for the reviewer

1. Is the Phase 0.1 registration design (mutual information, start at the DICOM transform, injected-shift positive
   control, the gates above) adequate, given 3 mm slices limit precision along the normal?
2. Is the matched control (1), plus the shuffle (2) and dose–response (3), sufficient to make H1 causal?
3. Is the endpoint hierarchy acceptable for MICCAI when the confirmatory test set returns only an aggregate macro-AUC?
4. With 4×4 tokens per triplet and dense native-adjacent windows, inference is about 1–3k tokens per study. Is per-plane
   k-NN attention the right efficiency choice, or is a cross-plane line-intersection sampler cleaner?
5. Given CoPAS and PETR-style 3D position encodings, is "DICOM-geometry correspondence with acquisition support" enough
   novelty for MICCAI, or does it need H2 to pass to be competitive?
