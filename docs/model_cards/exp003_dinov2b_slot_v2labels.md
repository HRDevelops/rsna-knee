# Model Card: exp003 (DINOv2 ViT-B/14, report labels)

## Model description
2.5D slot model for RSNA Knee Abnormality Detection. Per-series DINOv2 ViT-B/14 encoder (`vit_base_patch14_dinov2.lvd142m`, timm) with a slice-attention transformer over 24 uniformly sampled slices per series, feeding a cross-slot transformer study head with per-target attention pooling over up to 6 canonical anatomical slots ({Sagittal, Coronal, Axial} x {fluid-sensitive, non-fluid-sensitive}). Outputs 12 independent per-label probabilities (sigmoid, not softmax). Architecture: `src/rsna_knee/model.py:SlotModel`.

## Intended use
Ensemble member (`ensemble_recipe_v1.yaml`) for the RSNA Knee Abnormality Detection Kaggle competition — predicts probability of 12 knee MRI findings (ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral/PF OA, Effusion, Synovitis, Baker's cyst, Contusion, Fracture) from DICOM MRI series alone (no report text at inference time). Not validated for clinical use, diagnosis, or any use outside this competition's image-only research setting.

## Training data
4,407 knee MRI studies (RSNA Knee Abnormality Detection dataset). Soft labels from `labels_v2_final.parquet` — the report-derived label engine's final calibrated output (per-label choice among {Qwen2.5-72B-only, Llama-3.3-70B-only, 50/50 blend}, selected by strict leave-one-out AUC on the 58-study image-derived gold reference; see `docs/label_blend_v2_final.md`). This is the **original** report-based label generation, prior to the image+report noisy-student iteration used by later experiments.

## Training procedure
5-fold GroupKFold (grouped by near-duplicate report template / site cluster, multilabel-stratified within groups), 25 epochs, AdamW, per-parameter-group gradient clipping (backbone vs. head clipped separately — the fix for a fold-collapse failure mode found during exp001, see `docs/EXPERIMENTS.md` 2026-09-07), pre-norm from-scratch transformer blocks with small-gain initialization, EMA (decay 0.999) weights deployed. Confidence-weighted BCE loss (per-study, per-label label-engine confidence). Config: `configs/exp003_dinov2b_slot_v2labels.yaml`.

## Evaluation results
- OOF macro-AUC vs. `labels_v2_final` (native validation target): **0.8536**
- gold-58 macro-AUC (58-study image-derived reference, never trained on): **0.8714 ± 0.0084**
- Kept over the ViT-S variant (exp002) by the 2026-09-09 scale-ladder decision rule: macro gain +0.0055, MCL weak-label gain +0.0227.

## Limitations and biases
- Trained on report-derived labels which are known to systematically over-call mild disease relative to image-derived ground truth (see CLAUDE.md §2) — this label generation predates the image+report blend correction used by exp005/006/008.
- Reports span ~12+ languages across ~16-19 sites; label quality varies by language (the underlying LLM label engine's agreement with gold-58 differs per label and was not separately audited per language for this specific model).
- Side (left/right) resolution at test time is DICOM-tag-only with a dual-orientation-average fallback for unresolved cases — no report-supervised side classifier was ever trained (see `docs/LATERALITY_RESOLUTION.md`), so laterality-swap error remains a residual risk on ambiguous-tag studies.
- Not evaluated on any population outside this competition's site/scanner mix.

## Ethical considerations
Research artifact for a Kaggle competition, not a medical device. Predictions should never be used for real clinical decision-making. Ground truth itself reflects "on the fence" cases graded negative by a severity-thresholding convention chosen for this competition, not a clinical diagnostic standard.
