# Model Card: exp008 (DINOv2 ViT-S/14, distilled — efficiency track)

## Model description
Compact 2.5D slot model for the competition's efficiency track (CLAUDE.md §8/§4). Same DINOv2 ViT-S/14 backbone family as exp002/exp005, but reduced resolution (T=16 slices per series, down from 24; 320px, down from 384) for faster inference. Architecture unchanged otherwise: slice-attention transformer + cross-slot study head + per-target attention pooling. `src/rsna_knee/model.py:SlotModel`.

## Intended use
Standalone efficiency submission (`notebooks/submit-efficiency.ipynb`) — a single compact model intended to score within ~0.015 macro-AUC of the full 3-member ensemble at a fraction of the inference cost, for the competition's `AUC/runtime` efficiency-track scoring formula (CLAUDE.md §4). Not an ensemble member; not validated for clinical use.

## Training data
Same 4,407 studies. Trained with a joint objective: BCE against `labels_v3_candidate.parquet` (the accepted noisy-student label generation, same as exp005/exp006) **plus** a KL-divergence term against `ensemble_teacher_v1.parquet` — the accepted 3-member ensemble's ({exp006, exp003, exp005}, per-label prob-mean/rank-mean per `ensemble_recipe_v1.yaml`) own blended soft predictions over all 4,407 studies (OOF for the ~4,349 non-gold studies, fold-ensembled for the 58 gold studies — the standard non-circular construction used throughout this project's label-blend scripts). This is knowledge distillation from the full ensemble into a single compact student.

## Training procedure
5-fold GroupKFold (same folds as every other experiment), 25 epochs, AdamW, per-group grad clipping, pre-norm + small-gain init, EMA. Loss: `0.5 * BCE(labels_v3) + 0.5 * KL(teacher || student)` — binary KL per independent per-label sigmoid output, not a categorical/softmax KL (`scripts/train.py:kl_bce_mix_loss`). A per-batch non-finite-loss guard skips (rather than applies) any batch whose loss goes NaN/Inf, added after an isolated single-fold divergence during development (`docs/EXPERIMENTS.md` 2026-09-09/10). Config: `configs/exp008_dinov2s_slot_distilled.yaml`.

## Evaluation results
- OOF macro-AUC vs. the original `labels_v2_final` (all 5 folds): **0.8708** — only **-0.0036** off the full 3-member ensemble's 0.8744, comfortably inside the pre-registered 0.015 acceptance target, at roughly half the per-series resolution/slice count.
- gold-58 macro-AUC: **0.8882 ± 0.0032** — the tightest cross-fold spread of any deployed model, and nominally the highest gold-58 point estimate, though gold-58's n=58 makes fine-grained ranking between models on this metric noisy.
- Fold-count sensitivity (checked on gold-58, the only set where every fold's model predicts the same studies): averaging over ALL possible subsets of each size (not just the best one), mean delta vs. the full 5-fold average is -0.0060 (1 fold), -0.0030 (2), -0.0014 (3), -0.0004 (4) — a small, gradual cost, not a cliff. As of 2026-09-11 the efficiency notebook picks fold count by minimizing the actual CLAUDE.md §4 efficiency-track formula (in-notebook-measured decode/model timing + this delta table), not a fixed choice or a wall-clock-budget ladder — in practice this lands on 4 or 5 folds, both essentially equivalent under the formula since DICOM decode dominates per-study cost so heavily that fold count barely moves total runtime. See `docs/SUBMISSION_RUNBOOK.md` §4 for healthy calibration ranges.

## Limitations and biases
- Distilled from a teacher (the 3-member ensemble) that itself has the label-generation limitations described in exp005/exp006's model cards.
- Lower spatial/temporal resolution (320px/16 slices vs. 384px/24 slices) than every other deployed model — may lose sensitivity on small or subtle findings that depend on fine detail across more slices; not separately audited per label for this specific failure mode.
- One training run (job 24704) hit an isolated NaN divergence on a single fold before the batch-level guard was added; the guard has not yet been stress-tested against a recurrence, only confirmed not to trigger on the clean relaunch.

## Ethical considerations
Research artifact for a Kaggle competition, not a medical device. See exp003's model card for the shared ethical-considerations statement.
