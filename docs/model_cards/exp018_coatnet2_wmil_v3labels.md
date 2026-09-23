# Model Card: exp018 (5-fold CoAtNet-2 Window-MIL): final submission, both tracks

## Model description
A single model family, averaged over 5 fold checkpoints, that reads a knee MRI study as a bag of
**3-adjacent-slice windows** and pools them with **per-finding attention-MIL**.
- **Backbone:** timm `coatnet_rmlp_2_rw_384`, initialized from `sw_in12k_ft_in1k` (ImageNet-12k
  pretraining, fine-tuned on ImageNet-1k; 72.9 M params, 1024-d features). It is fully fine-tuned, and
  the input is upsampled 336 → 384 px.
- **Input unit:** a window is 3 adjacent slices *of the same slot*, stacked as RGB. A window never
  spans two slots. A study has at most 54 candidate windows (mean 51.5 on the profiling set).
- **Head:** LayerNorm → Linear–Tanh–Dropout–Linear(12), then a softmax over *all* of the study's
  windows, separately for each finding. Each finding pools its own weighted window features into its
  own logit. There is no per-series CLS token and no slot bottleneck: every window competes directly
  for every finding.

Code: `src/rsna_knee/wmil.py` (`WindowMIL`, cache/decode, laterality) and `src/rsna_knee/infer_wmil.py`
(inference driver). Config: `configs/exp018_coatnet2_wmil_v3labels.yaml`.

## Preprocessing
- **Field of view:** 140 mm resampled to 336 × 336 px (the earlier DINOv2 pipeline used 115 mm).
- **Slots:** 5 slots with 64 slices in total: Sag-FS 18, Sag-nonFS 14, Cor-FS 12, Cor-nonFS 8,
  Axial 12. `Fluid_Sensitive` defines FS.
- **Slice span:** slices are drawn from the **6–94%** span of each series.
- **Slice order:** by projecting `ImagePositionPatient` onto the slice normal (geometry, not
  `InstanceNumber`, which disagrees with geometric order in 34.5% of series). Intensity is normalized
  per series to uint8.
- **Laterality normalization (geometric):**
  - Coronal and axial images are mirrored horizontally for right knees.
  - Sagittal series are put in ascending patient-x and their **slice order** is reversed for right
    knees. The sagittal medial–lateral axis is the slice axis, not the image plane.
  - **Side source:** a validated side table if available, else the DICOM tag, else a geometric
    image-centre rule (|median x| ≥ 20 mm; x < 0 ⇒ right).
  - The geometric rule agrees with the tag side on 99.0% (n = 2,178) and the report side on 99.1%
    (n = 712). It resolves 99.4% of the 1,491 studies that had no validated side
    (`artifacts/exp018_cache_audit.md`).

## Intended use
The final Kaggle submission for **both** the main leaderboard and the efficiency track
(`notebooks/submit-exp018.ipynb`; see `docs/SUBMISSION_FINAL.md`). Research artifact for a
competition, not a medical device, and not validated for clinical use.

## Training data
All 4,407 training studies, 5-fold GroupKFold by report cluster (the same folds as every other
experiment). Targets are the project's own soft labels, **`labels_v3_candidate`**: an LLM report-label
engine plus one noisy-student round (`docs/METHOD_WRITEUP.md` §4). The 58 gold-labeled studies are
never trained on and are **never used for epoch or model selection**. No external data, no third-party
knee-trained checkpoints, and no third-party labels are used. The public dreaddevelopment kernels were
an architectural reference only.

## Training procedure
- **Loss and optimizer:** BCE with `pos_weight = clip((1−p)/p, 1, 10)` (p = train prevalence, clipped
  to [0.03, 0.7]). AdamW with backbone LR 3e-5 and head LR 1e-3, weight decay 0.02. OneCycle schedule
  with 15% warm-up, 16 epochs.
- **Batching and precision:** batch of 8 studies × 12 randomly sampled windows. Grad-clip 3, bf16,
  gradient checkpointing.
- **Augmentation:** ±10% intensity jitter only; no flips or rotations.
- **Epoch selection:** the top 3 epochs by fold-validation macro-AUC, with weights averaged.
- **Compute and seeds:** one seed per fold, about 1 h 41 min per fold on one H100. These are the
  reference recipe's hyperparameters, untuned.
- **Gate:** fold 0 had a pre-registered kill rule (gold-58 < 0.880 would stop the run). It passed with
  0.9063.

## Evaluation results

| Metric | exp018 | Reference |
|---|---|---|
| OOF macro-AUC (vs `labels_v2_final`, 5 folds) | **0.8816** | exp006 (best previous solo) 0.8681; Δ +0.0135, paired cluster-bootstrap 95% CI [+0.0100, +0.0171] |
| Gold-58 macro-AUC (fold-ensembled, n = 58) | **0.9091** | exp006 0.8897; deployed 3-member ensemble 0.8953 |
| **Public LB** (Kaggle submission 56460279) | **0.9290** | deployed ensemble 0.8930; exp008 0.8850 |
| Efficiency score `AUC/(0.5−0.95) + s/32400` | **−2.0362** | exp008 −1.9452. Assumes maxAUC 0.95; runtime is a *projection* of ~15 min on T4 × 2 |

- **Per-fold gold-58:** 0.906 / 0.901 / 0.905 / 0.906 / 0.893.
- **Where the gains are (Δ OOF vs exp006):** MCL +0.080, Lateral Meniscus +0.031, Lateral OA +0.017,
  ACL +0.016 (CIs exclude 0). Effusion −0.006 and Synovitis −0.006 are flat to slightly down.
- **Fold-count sensitivity (gold-58, averaged over all subsets):** −0.0067 for 1 fold, −0.0023 for 2,
  −0.0007 for 3, −0.0002 for 4. At 2 GPUs, decode rather than the model bounds runtime, so all 5 folds
  cost little extra.
- **Inference parity:** on HPC, the Kaggle inference path matches the training-path predictions with
  per-label Spearman 0.988–0.9999 (60 studies, 0 errors).

## Limitations and biases
- **The gain is not attributed to any one change.** exp018 bundles a new backbone family,
  adjacent-slice windows, the window-MIL head, a 140 mm FOV, the 6–94% span and corrected laterality.
  No ablation isolates which change is responsible.
- **OOF is measured against report-derived labels,** which disagree with image review about 18% of
  the time. It understates the LB here (gap −0.047) and is not comparable across pipelines.
- **Gold-58 is n = 58** (SE ≈ 0.03): a tripwire, not a precise estimate.
- **One seed per fold;** seed variance is unmeasured.
- **Runtime is projected, not measured,** for the full hidden test set on T4 (H100 profiling ×
  documented T4 factors). The placeholder commit run took about 1 minute.
- **Laterality errors on the ~1% of studies** where the geometric side rule is wrong would swap the
  medial/lateral labels for those studies.
- **Fluid/synovial labels** (Effusion, Synovitis) did not improve and may be slightly worse than the
  DINOv2 ensemble.

## Deployment notes
Deploy **only** the scored artifacts:
- `rsna-knee-exp018-code` from 2026-09-21 (`infer_wmil.py` 8,245 B, untimed GPU-worker join);
- `rsna-knee-exp018-weights` (5 × 292,862,052 B).

Code versions `2ec038d`–`d1165b5^` of `infer_wmil.py` carry a timed join that silently writes label
priors on T4. See `docs/SUBMISSION_FINAL.md` §4.

## Ethical considerations
Research artifact for a Kaggle competition, not a medical device. See exp003's model card for the shared
ethical-considerations statement.
