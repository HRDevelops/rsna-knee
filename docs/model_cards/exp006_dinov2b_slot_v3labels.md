# Model Card: exp006 (DINOv2 ViT-B/14, noisy-student labels)

## Model description
2.5D slot model — DINOv2 ViT-B/14 backbone (same as exp003) trained on the v3 noisy-student label generation (same labels as exp005). Combines exp003's backbone choice with exp005's label generation; the strongest solo model in the ensemble. Architecture: `src/rsna_knee/model.py:SlotModel`.

## Intended use
Ensemble member (`ensemble_recipe_v1.yaml`), the highest-weighted/best-solo member — selected FIRST by the greedy forward ensemble-selection procedure. Image-only inference. Not validated for clinical use.

## Training data
Same 4,407 studies, `labels_v3_candidate.parquet` (see exp005's model card for the label generation's provenance).

## Training procedure
Identical recipe to exp003/exp005 (5-fold GroupKFold, 25 epochs, AdamW, per-group grad clipping, pre-norm + small-gain init, EMA, confidence-weighted BCE). Config: `configs/exp006_dinov2b_slot_v3labels.yaml`.

## Evaluation results
- OOF macro-AUC vs. the original `labels_v2_final`: **0.8681** (best solo model among all 4 deployed image models)
- gold-58 macro-AUC: **0.8806 ± 0.0044**
- Kept over exp005 by the ViT-B-vs-ViT-S scale-ladder rule applied a second time on the v3 labels (macro gain +0.0050, clean win, see `docs/EXPERIMENTS.md` 2026-09-09) — confirms the same backbone-size finding from exp002-vs-exp003 held after the label upgrade too.
- Selected as ensemble member #1 by `scripts/build_ensemble.py`'s greedy forward selection; final 3-member ensemble ({exp006, exp003, exp005}) reaches macro **0.8744** OOF / **0.8960** gold-58 tripwire — a real gain over any solo model, not just the biggest backbone winning by itself.

## Limitations and biases
Same label-generation and laterality/language/site limitations as exp005 (see that model card) — exp006 shares the v3 label source.

## Ethical considerations
Research artifact for a Kaggle competition, not a medical device. See exp003's model card for the shared ethical-considerations statement.
