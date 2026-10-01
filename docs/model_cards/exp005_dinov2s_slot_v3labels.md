# Model Card: exp005 (DINOv2 ViT-S/14, noisy-student labels)

## Model description
2.5D slot model, identical architecture to exp003 but with a smaller per-series encoder: DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`, timm). Same slice-attention transformer + cross-slot study head + per-target attention pooling over 6 canonical slots. Architecture: `src/rsna_knee/model.py:SlotModel`.

## Intended use
Ensemble member (`ensemble_recipe_v1.yaml`) for the RSNA Knee Abnormality Detection Kaggle competition. Image-only inference (no report text at test time). Not validated for clinical use.

## Training data
Same 4,407 studies as exp003, but trained on `labels_v3_candidate.parquet` — the "noisy-student" round-1 label generation: a per-label weighted blend of the report label engine (`labels_v2_final`) and exp002's (the ViT-S baseline trained on `labels_v2_final`) image-model OOF predictions, weight chosen by strict leave-one-out AUC on gold-58 within a pre-registered per-label direction (image-heavy for 10 labels where the image model already beat the report engine; report-heavy for ACL/MCL). See `docs/label_blend_v3_candidate.md`. This is the label generation that all of exp005/006/008 build on.

## Training procedure
Identical recipe to exp003 (5-fold GroupKFold, 25 epochs, AdamW, per-group grad clipping, pre-norm + small-gain init, EMA, confidence-weighted BCE) — only the backbone size and training-label source differ. Config: `configs/exp005_dinov2s_slot_v3labels.yaml`.

## Evaluation results
- OOF macro-AUC vs. the **original** `labels_v2_final` (an independent check, not the label source this model was trained/tuned against): **0.8631**
- gold-58 macro-AUC: **0.8815 ± 0.0055**
- This label generation (v3) was accepted as an upgrade over v2 only after passing a pre-registered two-criterion acceptance test (must beat exp002 on BOTH the OOF-vs-original-labels metric above AND gold-58, not gold-58 alone — guards against the blend circularly overfitting to the 58-study reference it was calibrated on). Both criteria passed; see `docs/EXPERIMENTS.md` 2026-09-09.

- **Public LB (inferred):** the scored ensemble submission (56273364, 0.893) was most likely exp005 alone after the T4 runtime fallback. On Kaggle's 2 × T4 the
  3 members measure ~59 s/study (~21 h for the test set), so the notebook's time-budget ladder almost certainly
  fell to its last rung, exp005 by itself. Not confirmed: the chosen rung is not visible for the hidden-test run.

## Limitations and biases
- The v3 label blend was itself calibrated against gold-58 (58 studies) — while the acceptance test specifically guards against pure circularity, residual optimism relative to a truly held-out test set cannot be fully ruled out at this sample size.
- A second blend round (v4, blending exp005's own OOF back into v3) was tested and found to **regress** performance (-0.012 macro LOO on gold-58) — label iteration was stopped at v3 for this reason (`docs/EXPERIMENTS.md` 2026-09-09). exp005 should not be re-distilled against its own output.
- Same laterality/language/site limitations as exp003 (see that model card).

## Ethical considerations
Research artifact for a Kaggle competition, not a medical device. See exp003's model card for the shared ethical-considerations statement.
