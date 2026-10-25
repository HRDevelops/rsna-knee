# Phase 1 pre-registration: cross-plane footprint geometry (G) vs B_cap

Committed 2026-10-26, **before any forward pass of G that produces a validation or gold-58 number**. Until this file is
committed, G may only be run in smoke tests that check shapes, finite gradients, the exact $G = B_{\text{cap}}$ identity
at $\beta = 0$, footprint transforms, right-knee mirroring, padding masks and donor-swap mechanics.

Context:
- Plan: `PLAN_v1_3.md`, with its amendments.
- Phase 0A: **inconclusive** (`PHASE0A_RESULTS.md`). Raw geometry is used as an engineering decision; it is not
  "validated".
- Phase 0B: frozen pipeline B5.
- Phase 0C: frozen 6×6 grid with dense inference (`docs/EXPERIMENTS.md`).

## 1. Frozen for every Phase 1 arm

| Item | Frozen |
|---|---|
| Data pipeline | B5: all series, native-adjacent windows, plane-balanced, 160 mm at 384 px |
| Tokens | 6×6 per window |
| Inference | Dense (all windows, one pass, capped at 192), same for $B_{\text{cap}}$ and every $G$ arm |
| Train/test density mismatch | Accepted as a fixed property (dense was chosen on $B_{\text{cap}}$); not tuned around |
| Training | 12 plane-balanced windows per study per step, 8 studies, 16 epochs, exp018 learning rates and schedule, intensity jitter 0.1 |
| Checkpoint selection | As $B_{\text{cap}}$: top-3 epochs by fold-validation macro-AUC, averaged. Per-epoch scoring is bagged, the same as $B_{\text{cap}}$'s runs |
| Primary comparator | $B_{\text{cap}}$ = the Phase 0C runs `p0c_g6_s42`, `p0c_g6_s43`, dense outputs (fold 0); new runs for other folds |
| Gold-58 | Tripwire only, never a selection input |
| Primary statistic | Seed-averaged held-out predictions → one OOF macro-AUC per arm (concatenated over folds) |
| Primary effect | $\Delta_{\text{geo}} = \mathrm{AUC}(G_{\text{raw}}) - \mathrm{AUC}(B_{\text{cap}})$ |

## 2. Primary model $G_{\text{raw}}$

$B_{\text{cap}}$ plus one term in every attention layer:

$$a_{jk}=\frac{\mathbf q_j^\top\mathbf k_k}{\sqrt d}+b_h\mathbb 1[s_j=s_k]-\mathbb 1[\pi_j\neq\pi_k]\,\frac{\beta_h}{2}\,(\boldsymbol\mu_j-\boldsymbol\mu_k)^\top(\Sigma_j+\Sigma_k+\epsilon I)^{-1}(\boldsymbol\mu_j-\boldsymbol\mu_k),\qquad\epsilon=1\ \text{mm}^2 .$$

- **Geometry source.** Raw DICOM IPP / IOP / PixelSpacing and the exact cache-v2 affine, with the laterality mirror
  folded in. No registration correction and **no geometry augmentation**.
- **Footprint per token.** For a window (3 slices $l$), cell $(a,b)$ of the 6×6 grid, cell side $\ell=160/6$ mm:
  - $\boldsymbol\mu_l$ = patient position of the cell centre on slice $l$;
  - $\Sigma_l=\tfrac{\ell^2}{12}(\mathbf r\mathbf r^\top+\mathbf c\mathbf c^\top)+\tfrac{\tau^2}{12}\mathbf n\mathbf n^\top$, with $\tau$ = SliceThickness;
  - the token's $(\boldsymbol\mu,\Sigma)$ is the moment-matched mixture of the three slices.
- **Plane rule.** No same-plane geometry: $\pi$ = anatomical plane from the series table. **Absolute IPE: off.**
- **$\beta_h \ge 0$ (8 heads × 2 layers).** Initialized at exactly 0. Updated by its own projected SGD without momentum
  or weight decay, $\beta\leftarrow\max(0,\beta-\eta_\beta\nabla_\beta L)$, at a constant $\eta_\beta$.
- **Rule for $\eta_\beta$, fixed now.** Before any validation number, a smoke run of 50 training steps on real batches
  (no evaluation) logs $|\partial L/\partial\beta_h|$. Then $\eta_\beta = 0.5/(N_{\text{steps/epoch}}\times\text{median}|\partial L/\partial\beta_h|)$, so that at the median gradient $\beta$ can move about 0.5 in one
  epoch. The value is written into the config and this ledger before the first real run, and never changed afterwards.
- **Unit tests (must pass before launch).**
  - $\max|z_G - z_{B_{\text{cap}}}| < 10^{-6}$ in fp32 at initialization, for identical shared weights.
  - Identical sampled windows and batch order for paired arms with the same seed (separate data RNG).
  - Footprint $\boldsymbol\mu$ reprojects through the affine to the right patient point, mirrored series included.
  - Padding-masked tokens never receive attention.
  - The donor swap changes $(\boldsymbol\mu,\Sigma)$ and nothing else.

## 3. Screening, escalation, confirmation (unchanged from PLAN_v1_3 §5, made explicit)

- **Fold 0, seeds 42 / 43.** $\Delta_{\text{geo},s}$ = dense fold-val macro-AUC of $G_{\text{raw}}$ minus $B_{\text{cap}}$, same seed.
- **Donor swap.** At inference only, each study's geometry is replaced by that of a donor: a training-fold study with
  the same set of planes and the same side, nearest in total slice count. Within each plane, series are paired in
  slice-count order, cycling if the donor has fewer.
  - $\Delta_{\text{swap},s}=\mathrm{AUC}(G_{\text{correct}})-\mathrm{AUC}(G_{\text{donor}})$.
  - "Geometry is used" means $\Delta_{\text{swap},s}\ge0.001$ for both seeds.
- **Stop** if both $\Delta_{\text{geo},s}\le0$ and their mean is $\le-0.002$, or if geometry is not used.
- **Go** if both $\Delta_{\text{geo},s}>0$, the mean is $\ge+0.002$, and geometry is used.
- **Otherwise** run fold 1 with seeds 42 / 43 (with a fold-1 $B_{\text{cap}}$ pair). Continue only if the 4-run mean is
  $> 0$ and geometry is used. No third seed.
- **Confirmation.** 5 folds × 3 seeds for $B_{\text{cap}}$ and $G_{\text{raw}}$. H1 holds if all of these hold:
  - $\Delta_{\text{geo}}\ge+0.004$;
  - the paired cluster-bootstrap 95% lower bound is $>0$ (report clusters, 1,000 reps);
  - the donor swap and dose–response behave as pre-registered (PLAN_v1_3 §3.2: $<0.002$ change for $d\le0.5$, then a
    monotone fall from $d\ge1$);
  - gold-58 does not drop by more than 0.007.
- **Secondary endpoint.** Compartment-specific labels (both menisci, MCL, medial and lateral OA) gain more than global
  labels (effusion, synovitis).
- **Fold-0 bar, for the record.** $B_{\text{cap}}$ is 0.0025 below window-MIL B5. A +0.004 effect would put $G$ above
  B5 on fold 0.

## 4. Secondary arm $G_{\text{jitter}}$ (not a rescue)

- **Question.** Does modest registration-uncertainty regularization add robustness beyond raw geometry? It never
  replaces $G_{\text{raw}}$ if H1 fails.
- **Perturbation, training only.**
  - With probability 0.5, pick one present plane at random.
  - Draw a random rigid transform: random axis, angle and direction, with the translation and rotation magnitudes drawn
    together.
  - Scale the whole transform (rotation angle and translation together) so that the median cross-plane footprint
    displacement $d$ equals $d_{\text{jitter}}\sim U(0, 0.75)$.
  - It is never sampled from the Phase 0A correction tail.
- Run only after $G_{\text{raw}}$ passes screening; same folds and seeds; reported against $G_{\text{raw}}$ and
  $B_{\text{cap}}$.

## 5. Not in Phase 1 (unchanged plan backlog)

- $G_{+\text{IPE}}$, $G_{\text{all-series}}$, $G_{k\text{NN}}$: mechanism arms after screening, fold 0, descriptive.
- $\Sigma_{\text{reg}}$: only if $G$ works.
- Local-stage tokens: only if $G$ fails on final-stage tokens.

## 6. 40-pair blinded geometry review (descriptive only)

- **Sample.** 40 cross-plane pairs from the 145 non-pilot Phase 0A studies, 10 per stratum:
  - low correction and stable starts;
  - moderate correction;
  - high correction;
  - multi-start disagreement or gross tail.
- **Balance.** Across Sag–Cor / Sag–Ax / Cor–Ax and 1.5 T / 3 T where possible.
- **Display.** The same anatomical crop at the plane intersection, rendered at raw DICOM placement and at the MI
  correction, labelled A/B in random order. The key is kept apart from the rating sheet.
- **Ratings.** Better: A / B / indistinguishable / unrateable. Magnitude: none / subtle / clear / gross. Confidence:
  1–5. Mismatch type: translation / rotation / both / cannot tell.
- **Reported.** P(raw preferred), P(MI preferred), P(tie), overall and per stratum.
- **No effect on decisions.** It changes no model choice and no threshold, and it does not change Phase 0A's
  inconclusive label.
