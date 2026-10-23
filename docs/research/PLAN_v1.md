> **Superseded by `PLAN_v1_1.md` (2026-10-23, after review round 1).** Kept unchanged below as the record of what was
> reviewed. Known errors in this version: it assumes a common coordinate frame that is not yet verified, uses raw
> PixelSpacing after preprocessing, and states that CoPAS covers the same 12 findings and has no public code. Both CoPAS
> statements are wrong.

# Research plan v1 — tri-planar, acquisition-aware fusion for knee MRI (draft for review)

Status: proposal, nothing run yet. Written 2026-10-23 after the competition closed. Every number below is either
measured in this repo (source given) or marked as an estimate.

---

## 0. Where we stand (short)

- **Data.** RSNA Knee Abnormality Detection (Kaggle 2026). 4,407 training studies, 24,371 series, 819,078 DICOM slices.
  12 binary findings; macro ROC-AUC. Only **58** studies have expert labels. Every study has a free-text report, in 12+
  languages, and there are no reports at test time. The hidden test set is about 1,300 studies.
- **What we did.**
  1. Label engine: Qwen2.5-72B and Llama-3.3-70B read the reports; Platt calibration chosen by leave-one-out on the 58;
     one noisy-student round (v3).
  2. DINOv2 slot model: public 0.893.
  3. 2.5D window-MIL with CoAtNet-2 (64 slices, 140 mm field of view, 336 px input upsampled to 384, laterality fix,
     2 seeds × 5 folds). Public **0.929**, private **0.912**: rank 3,970 of 6,425.
- **What the leaderboard teaches** (from the public notebook that scored 0.958 public / 0.942 private, read 2026-10-23):
  the strong public stack is a rank blend of several authors' models. The core model is a 2.5D CoAtNet-2 trained
  **with extra OAI knees**, at **96 slices of native 384 px**, with **SWA**. It is blended with a self-supervised
  "OrthoFoundation" encoder, uses **left/right mirror test-time augmentation**, and its blend weights were chosen on the same
  58 gold studies. 1,322 teams sit at exactly private 0.942; the top is **0.960**.

  Our gap to 0.942 is therefore likely to be mostly **data, labels and input density**, not the core architecture.
  A paper must first close or explain that gap with a matched baseline (Phase 0).

## 1. The scientific gap, made concrete with our data

Measured in `artifacts/decode_audit.parquet` and a header-only check of 809 series in 150 studies (2026-10-23):

| Fact | Value |
|---|---|
| Slice thickness, median (5–95%) | 3.0 mm (2.0–4.0) |
| In-plane spacing, median (5–95%) | 0.31 mm (0.17–0.56) |
| Anisotropy (thickness ÷ in-plane), median / p90 | **10.2× / 15.8×** |
| 2D vs 3D acquisitions (sample) | 754 vs 29 series (3D ≈ 3.6%) |
| Geometry tags present (position, orientation, spacing, thickness) | 100% |
| Series in a study sharing one FrameOfReferenceUID | **0%** (UIDs re-keyed at anonymisation) |
| Studies whose series boxes share a common 3D intersection | **100%** |
| Series-centre spread within a study, median / p90 / p99 | 8.5 / 15.2 / 46.0 mm |
| Sequence parameters (TR, TE, flip angle, sequence type) present | 95% |

**Reading.** Every plane is sharp in two directions and about 10 times blurrier in the third, and the blurry direction
is different for each plane. The three planes live in one physical coordinate frame, even though the UIDs no longer
prove it. Current pipelines, ours included, throw this geometry away: slices are an ordered list per series, and the
planes are only combined by learned attention. **Proposal: fuse the planes where they physically intersect, and model
how thick each slice really is.**

## 2. Physics-informed: what is feasible here, and what is not

| Idea | Verdict | Why |
|---|---|---|
| Biomechanical PINN (contact stress, cartilage elastostatics, load sharing) as a training loss | **Not recommended** | The MRI is acquired supine and unloaded, so no load or deformation is observed. There are no segmentations of cartilage, menisci or ligaments, so there is no domain to solve a PDE on. There is no stress ground truth, so a PDE-residual loss cannot be validated, and reviewers will ask exactly that. The labels (for example synovitis or Baker's cyst) are not mechanical quantities. |
| MRI **acquisition** physics: slice profile, anisotropic sampling, scanner geometry | **Core of this plan** | Exact, known per series from DICOM, and testable. |
| MRI **contrast** physics: signal equation from TR/TE/TI/flip angle | **Secondary component** | Gives a physically meaningful sequence description across 16+ sites and vendors, instead of a fragile "fluid-sensitive" flag. |
| Physically correct low-resolution simulation (k-space truncation, slab averaging) | **Augmentation** | The correct physics of a coarser protocol; this is where cross-site robustness is tested. |
| Biomechanics as a **testable clinical hypothesis**, not a loss | **Yes (H3)** | Pivot-shift ACL injuries leave a known bone-contusion "footprint" (lateral femoral condyle and posterolateral tibial plateau; Sanders et al., *RadioGraphics* 2000). Test whether the model recovers this pattern from image-level labels alone. |

A biomechanical PINN only becomes defensible with segmented anatomy (for example OAI-ZIB), finite-element ground truth and
loading assumptions. That would be a separate project, not a diagnosis paper.

## 3. Method: TriPlane-MIL (working name)

### 3.1 Every feature has a physical location and an uncertainty

For series $s$, slice $i$: position $\mathbf{o}_{s,i}$ (ImagePositionPatient), row and column directions
$\mathbf{r}_s,\mathbf{c}_s$ (ImageOrientationPatient), normal $\mathbf{n}_s=\mathbf{r}_s\times\mathbf{c}_s$, spacings
$(\delta^r_s,\delta^c_s)$, thickness $\tau_s$. $g_s$ is the spacing between the slices that actually form a window after
our slice sampling. That is often 2–4× the native spacing: coronal and axial series are subsampled to fit the 64-slice
budget (`docs/EXPERIMENTS.md`, slice-spacing audit). A pixel at column $u$ and row $v$ sits at

$$\mathbf{x}_{s,i}(u,v)=\mathbf{o}_{s,i}+u\,\delta^c_s\,\mathbf{r}_s+v\,\delta^r_s\,\mathbf{c}_s .$$

Canonical knee frame: $\tilde{\mathbf{x}}=\mathbf{M}_\ell(\mathbf{x}-\mathbf{m})$, where $\mathbf{m}$ is the study
centre and $\mathbf{M}_\ell=\mathrm{diag}(\pm1,1,1)$ mirrors right knees, so $+x$ is always lateral.

A token is one cell of the backbone feature map of one 3-slice window. Its footprint is a Gaussian
$\mathcal{N}(\boldsymbol\mu_j,\Sigma_j)$ whose covariance encodes the acquisition:

$$\Sigma_j=\tfrac{(S\delta^c_s)^2}{12}\,\mathbf{r}_s\mathbf{r}_s^\top+\tfrac{(S\delta^r_s)^2}{12}\,\mathbf{c}_s\mathbf{c}_s^\top+\Big(\tfrac{\tau_s^2}{12}+\tfrac{2g_s^2}{3}\Big)\mathbf{n}_s\mathbf{n}_s^\top ,$$

where $S$ is the feature stride. The terms come from: a rectangular slice profile of width $\tau_s$ (variance $\tau^2/12$),
three slices at offsets $\{-g,0,g\}$ (variance $2g^2/3$), and the in-plane cell size.

### 3.2 Acquisition-aware positional encoding

We use the expected Fourier features of the footprint (the integrated positional encoding of mip-NeRF, Barron et al. 2021),
which is exact for a Gaussian:

$$\gamma(\boldsymbol\mu,\Sigma)=\Big[\sin(\boldsymbol\omega_k^\top\boldsymbol\mu)\,e^{-\frac12\boldsymbol\omega_k^\top\Sigma\boldsymbol\omega_k},\;\cos(\boldsymbol\omega_k^\top\boldsymbol\mu)\,e^{-\frac12\boldsymbol\omega_k^\top\Sigma\boldsymbol\omega_k}\Big]_{k=1}^{K}.$$

Frequencies finer than what the slice physically resolves are damped along the thick axis. The model cannot pretend to
know a position more precisely than the scan allows. As far as we know, this has not been used for anisotropic
multi-plane MRI (to verify in the literature review).

### 3.3 Cross-plane geometric attention

Tokens from all planes attend to each other, with a bias equal to the log overlap of their footprints:

$$a_{jk}=\frac{(W_Q\mathbf{h}_j)^\top W_K\mathbf{h}_k}{\sqrt d}+\beta_h\log\mathcal{N}\!\big(\boldsymbol\mu_j-\boldsymbol\mu_k;\,\mathbf{0},\,\Sigma_j+\Sigma_k+\epsilon I\big),$$

since $\int\mathcal{N}(\mathbf{x};\boldsymbol\mu_j,\Sigma_j)\,\mathcal{N}(\mathbf{x};\boldsymbol\mu_k,\Sigma_k)\,d\mathbf{x}=\mathcal{N}(\boldsymbol\mu_j-\boldsymbol\mu_k;\mathbf{0},\Sigma_j+\Sigma_k)$.

A sagittal token and a coronal token that image the same tissue (their slabs intersect) get a high bias. $\beta_h\ge0$ is
learned per head, so heads can be local or global. Attention is restricted to the $k$ nearest tokens by Mahalanobis
distance, which keeps the cost linear.

### 3.4 Per-finding pooling gives a 3D location for free

Per-finding attention-MIL as now: $\alpha_c(j)=\mathrm{softmax}_j\,\mathbf{w}_c^\top\tanh(V\mathbf{h}'_j)$ and
$\hat y_c=\sigma(\mathbf{u}_c^\top\sum_j\alpha_c(j)\mathbf{h}'_j+b_c)$.

Within plane $\pi\in\{\text{sag},\text{cor},\text{ax}\}$, with $\alpha_c$ renormalised over that plane's tokens, the
attention defines a 3D location distribution with moments

$$\boldsymbol\mu^{(\pi)}_c=\sum_{j\in\pi}\alpha^{(\pi)}_c(j)\,\boldsymbol\mu_j,\qquad
\Sigma^{(\pi)}_c=\sum_{j\in\pi}\alpha^{(\pi)}_c(j)\big[\Sigma_j+(\boldsymbol\mu_j-\boldsymbol\mu^{(\pi)}_c)(\boldsymbol\mu_j-\boldsymbol\mu^{(\pi)}_c)^\top\big].$$

Each plane pins the location sharply in its two in-plane axes and loosely along its normal. **Triangulation** is the
precision-weighted fusion of the planes:

$$\Lambda_c=\sum_\pi(\Sigma^{(\pi)}_c)^{-1},\qquad \boldsymbol\mu^\star_c=\Lambda_c^{-1}\sum_\pi(\Sigma^{(\pi)}_c)^{-1}\boldsymbol\mu^{(\pi)}_c .$$

### 3.5 Losses

$$\mathcal{L}=\underbrace{\textstyle\sum_c w_c\,\mathrm{BCE}(\hat y_c,\tilde y_c)}_{\text{soft v3 labels, confidence-weighted}}
+\lambda_{\text{geo}}\,\mathcal{L}_{\text{geo}}+\lambda_{\text{loc}}\,\mathcal{L}_{\text{loc}} .$$

**Geometric consistency** is the multi-view "reprojection error", applied only to focal findings (ACL, MCL, both menisci,
contusion, fracture, Baker's cyst). Diffuse findings (effusion, synovitis, OA) are not forced to a point:

$$\mathcal{L}_{\text{geo}}=\sum_{c\in\mathcal{F}}\tilde y_c\sum_\pi\Big[(\boldsymbol\mu^{(\pi)}_c-\boldsymbol\mu^\star_c)^\top(\Sigma^{(\pi)}_c)^{-1}(\boldsymbol\mu^{(\pi)}_c-\boldsymbol\mu^\star_c)+\eta\log\det\Sigma^{(\pi)}_c\Big].$$

The $\log\det$ term stops the trivial solution of spreading attention everywhere. $\eta$ is small and gets an ablation.

**Weak localization from text (optional).** Our LLMs extract the side (medial or lateral) and the structure (femur,
tibia, patella; anterior or posterior horn) that a report gives for a finding. With $+x$ lateral:

$$P_c(\text{lateral})=\Phi\!\big(\mu^\star_{c,x}/\sigma^\star_{c,x}\big),\qquad
\mathcal{L}_{\text{loc}}=\sum_{c:\,z_c\text{ known}}\mathrm{BCE}\big(P_c(\text{lateral}),z_c\big).$$

This assumes the study centre $\mathbf{m}$ sits near the middle of the knee (the intercondylar notch). The menisci lie
about 20–40 mm either side of it, so a centring error under about 10 mm keeps the side correct. Phase 0 checks this on
the hand-checked subset: medial-meniscus locations must fall at $x<0$.

### 3.6 MR contrast embedding (secondary component)

For canonical tissues $t$ (fluid, cartilage, fat, muscle, marrow, ligament) with literature $(PD_t,T_{1,t},T_{2,t})$ at the
series' field strength, compute the expected signal from the sequence type (ScanningSequence / SequenceVariant):

- spin echo: $S_t=PD_t(1-e^{-TR/T_{1,t}})e^{-TE/T_{2,t}}$
- inversion recovery: $S_t=PD_t\,|1-2e^{-TI/T_{1,t}}+e^{-TR/T_{1,t}}|\,e^{-TE/T_{2,t}}$
- gradient echo: $S_t=PD_t\sin\alpha\frac{1-E_1}{1-E_1\cos\alpha}e^{-TE/T_{2,t}^*}$, with $E_1=e^{-TR/T_{1,t}}$

The normalised log-signal vector $\mathbf{f}_s$ conditions the encoder (FiLM) and replaces the fluid-sensitive flag.
The relaxation constants are approximate, so this is an inductive bias, not a simulation; the ablation decides whether
it stays.

### 3.7 Optional explicit-3D arm (answers "do we need a real 3D volume?")

Fit a per-study implicit neural representation $f_\theta(\mathbf{x},c)$ through the slice forward model
$\hat y_{s,i,p}=\int \mathrm{PSF}_s(\mathbf{x}-\mathbf{x}_{s,i,p})\,f_\theta(\mathbf{x},c_s)\,d\mathbf{x}$, with per-series
rigid corrections (as in NeSVoR, Xu et al., IEEE TMI 2023). Reconstruct isotropic 0.5 mm volumes and run a 3D or 2.5D
model on them.

Gate: leave-one-slice-out reconstruction must beat trilinear interpolation. Cost: about 1–2 GPU-minutes per study, so
roughly 100–200 GPU-hours (estimate). Run it only if Phase 2 shows that geometry matters.

## 4. Hypotheses and title

**Clinical hypothesis.** Each 2D knee MRI plane resolves anatomy sharply in only two of three directions. Fusing the
orthogonal planes where they physically intersect, with the acquisition's true resolution, will:

- **H1 (diagnosis):** improve AUC on side- and compartment-specific findings (menisci, compartment OA, contusion,
  fracture) over a matched 2.5D model;
- **H2 (localization):** localize abnormalities in 3D without any localization labels, agreeing with report-stated
  locations and with external bounding boxes;
- **H3 (biomechanics):** recover the pivot-shift contusion footprint in ACL-torn knees from image-level labels alone;
- **H4 (robustness):** degrade less under physically simulated protocol changes (thicker slices, missing plane), across
  vendors and field strengths.

**Title options.**

1. *Triangulating Knee Pathology: Acquisition-Aware Tri-Planar Fusion for Weakly Supervised MRI Diagnosis and Label-Free
   3D Localization*
2. *Geometry Is Free Supervision: Physics-of-Acquisition Priors for Multi-Planar Knee MRI*

## 5. Validation protocol

### Phase 0 — matched strong baseline and gap analysis (do first; 1–2 weeks)

1. **Alignment check.** Do sagittal and coronal intensity profiles agree along their intersection lines (about 50
   studies)? Accept if the median residual offset is ≤ 3 mm; otherwise add per-series rigid registration.
2. **Baseline B\*.** Our exp018, upgraded with the public lessons that need no outside data:
   - 96 native-384 slices
   - SWA
   - mirror test-time augmentation (swap medial and lateral outputs)

   Each upgrade gets one fold-0 ablation. B\* is the baseline for every later claim.
3. **External-data effect** (needs OAI access, see §8): B\* trained with and without OAI. This sizes how much of the
   0.912 → 0.942 gap is data rather than method.

### Phase 1 — core method on fold 0, then 5 folds × 3 seeds (2–3 weeks)

| Ablation | Question |
|---|---|
| B\* (index positions only) | reference |
| + world-coordinate positional encoding (plain Fourier) | does geometry help at all? |
| + integrated encoding (thickness-aware) | does acquisition physics help beyond geometry? |
| + cross-plane geometric attention | does fusing at intersections help? |
| + $\mathcal{L}_{\text{geo}}$ (with and without $\log\det$) | does consistency help, or just sharpen attention? |
| + $\mathcal{L}_{\text{loc}}$ (text locations) | does weak location supervision help AUC? |
| + contrast embedding vs fluid flag vs learned sequence embedding | physics vs learned |
| laterality fix off | how much of the gain is just the mirror? |
| explicit 3D arm (INR volume + 3D model) vs naive isotropic resampling + 3D CNN | implicit vs explicit 3D |

**Baselines.**
- MRNet-style per-plane CNN (Bien et al. 2018)
- our DINOv2 slot model
- B\*
- a naive 3D CNN on resampled volumes
- CoPAS-style learned co-plane attention: re-implemented from the paper, since no code was found (Qiu et al., *Nat
  Commun* 2024)

### Statistics (pre-registered before any result)
- **Primary endpoint:** macro-AUC on 5-fold out-of-fold predictions against report labels, plus gold-58, and the RSNA
  hidden test via Kaggle late submissions (if late submissions are open; to verify).
- **Comparisons:** paired cluster bootstrap over report clusters (macro), DeLong per finding with Holm correction,
  3 seeds, and a pre-set minimum effect of +0.004 macro (≈ 1 bootstrap SD).

### Localization (H2, H3)
- **Text-derived locations.** LLM extraction of side and structure for all 4,407 reports. Before using them, spot-check
  100 reports by hand.
- **Metrics:** side accuracy of $\boldsymbol\mu^\star_c$, compartment accuracy, and the agreement between planes
  (median triangulation residual in mm).
- **External boxes:** fastMRI+ pathology annotations (Zhao et al., *Sci Data* 2022; coronal knee). Pointing game and
  box-hit rate.
- **H3:** among contusion-positive studies, the share whose contusion location falls in the lateral femoral condyle or
  posterolateral tibial plateau, ACL-positive vs ACL-negative. Use a permutation test, and cross-check against the
  contusion sites the reports state.

### External validation (data access needed)

| Cohort | Overlapping labels | Use |
|---|---|---|
| MRNet (Stanford; 3 planes) | ACL tear, meniscal tear | zero-shot AUC |
| fastMRI + fastMRI+ boxes | meniscal tear, bone-marrow edema, effusion, … (coronal) | localization, single-plane robustness |
| OAI (MOAKS subset) | effusion-synovitis, BML, meniscal tear, OA | AUC on overlapping labels; optional extra training data |
| KneeMRI (sagittal only) | ACL | missing-plane robustness |

### Robustness (H4)
- Subgroups by Manufacturer and by 1.5 T vs 3 T (tags present in about 95% of series).
- One plane removed at test time.
- Physically simulated thicker slices (slab averaging) and lower in-plane resolution (k-space truncation).

## 6. Compute and timeline (estimates)

| Block | H100-hours | Calendar |
|---|---|---|
| Phase 0 (header pass on CPU, B\* ablations, OAI arm if available) | 40–60 | 1–2 weeks |
| Phase 1 screening (fold 0) | 40 | 1–2 weeks |
| Phase 1 full (5 folds × 3 seeds, main ablations) | 120–150 | 2 weeks |
| LLM location extraction (vLLM, 72B) | 10 | 1–2 days |
| Optional INR arm | 100–200 | 2 weeks |

Without the INR arm this is about 250 GPU-hours. A MICCAI 2027 submission (deadline usually early March; check) is
realistic. A journal version (TMI or MedIA) would add the external cohorts and the INR arm.

## 7. Risks and go/no-go gates

| Risk | Gate or mitigation |
|---|---|
| Geometry adds little AUC, because 2.5D CNNs already learn plane context | Fold-0 gate: continue only if ≥ +0.002 OOF over B\*, or if localization is clearly better. The paper can stand on H2/H3 if H1 is null; report it honestly. |
| Patients move between series, so planes misalign | Phase 0 alignment check, then per-series rigid registration if needed |
| Report-derived locations are noisy or sparse | Hand spot-check; use them only as weak supervision and for evaluation |
| Novelty overlap (CoPAS; multi-view 3D position encoding as in PETR, Liu et al. 2022) | Full literature review before writing; read CoPAS end to end |
| Data use: RSNA data terms for publication; OAI / fastMRI / MRNet agreements | Verify the terms before starting; you sign the agreements |
| Gold-58 is over-used (by us and by public notebooks) | Report it, but never select on it; use the hidden test via late submissions for the final numbers |

## 8. Decisions needed from you

1. **Direction:** acquisition physics and geometry as the core, with biomechanics as a hypothesis test (H3) rather than a
   PINN loss. Agree, or do you want the biomechanics route regardless?
2. **Data access** (you have to apply yourself): OAI (NIMH Data Archive), fastMRI (NYU) + fastMRI+ annotations,
   MRNet (Stanford AIMI). Which ones?
3. **Check on Kaggle:** whether late submissions are open, and what the competition data terms say about publication.
4. **Start Phase 0 now** (alignment check + B\*): about 50 GPU-hours.

## References to verify while writing

Bien et al., *PLOS Med* 2018 (MRNet) · Qiu et al., *Nat Commun* 15:7637, 2024 (CoPAS) · Barron et al., ICCV 2021
(mip-NeRF) · Xu et al., IEEE TMI 2023 (NeSVoR) · Liu et al., ECCV 2022 (PETR) · Zhao et al., *Sci Data* 2022 (fastMRI+) ·
Sanders et al., *RadioGraphics* 2000 (bone-contusion patterns) · Hunter et al., *Osteoarthritis Cartilage* 2011 (MOAKS) ·
Ambellan et al., *MedIA* 2019 (OAI-ZIB).
