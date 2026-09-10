# RSNA Knee Abnormality Detection — Method Writeup (skeleton)

Winners' obligation draft (CLAUDE.md, competition rules): a public write-up accompanying the
open-sourced code + weights + video, due on/before the final submission deadline. This is a
**skeleton** — section headers and pointers to the source material each section should draw from,
not final prose. Fill in once the final leaderboard result is known and the submission is locked.

## 1. Summary

- [ ] One-paragraph summary: final approach, final rank/score, single biggest lever.
- [ ] Team/solo, compute used (HPC cluster spec, total GPU-hours if known).

## 2. Competition overview

- [ ] Task, metric (macro ROC-AUC over 12 targets), data scale (4,407 train studies, ~1,300 test).
- Source: CLAUDE.md §1.

## 3. The central insight: image-derived vs. report-derived labels

- [ ] The ~18% systematic disagreement between report-derived and image-derived gold labels, and
  why this made the label engine (not the image model) the priority workstream.
- Source: CLAUDE.md §2, `docs/EXECUTION_PLAN.md`.

## 4. Label engine

### 4.1 Report clustering + fold construction
- [ ] Near-duplicate/site-template clustering (word n-gram + LSH), the character-n-gram bug found
  and fixed, GroupKFold construction.
- Source: `docs/REPORT_CLUSTERS.md`, `docs/EXPERIMENTS.md` 2026-09-06 entries, `scripts/build_folds.py`.

### 4.2 LLM label extraction (v1, v2)
- [ ] Rubric design, per-label severity-threshold hypotheses, the two-model ensemble (Qwen2.5-72B,
  Llama-3.3-70B), rubric v2 changes (Contusion/Fracture hierarchy, Synovitis conditional prior
  reverted), parse-failure hardening.
- Source: `configs/rubric_v1.yaml`, `configs/rubric_v2.yaml`, `docs/gold_agreement_v1.md`,
  `docs/gold_agreement_v2.md`, `docs/gold_agreement_llama_v2.md`.

### 4.3 Calibrated blend (labels_v2_final)
- [ ] Platt calibration, the sklearn-regularization bug that broke rare-label calibration, strict
  leave-one-out per-label strategy selection ({Qwen-only, Llama-only, blend}), macro LOO AUC 0.823.
- Source: `docs/label_blend_v2_final.md`, `scripts/build_labels_v2_final.py`.

### 4.4 Noisy-student round (labels_v3_candidate)
- [ ] Blending the image model's own OOF predictions back into the label engine, pre-registered
  per-label direction (image-heavy vs. report-heavy), macro LOO AUC 0.904 — and the round-2 blend
  (v4) that regressed, showing where the iteration usefully stopped.
- Source: `docs/label_blend_v3_candidate.md`, `docs/label_blend_v4_candidate.md`,
  `scripts/build_labels_v3_candidate.py`, `scripts/build_labels_v4_candidate.py`.

## 5. Image model architecture

- [ ] 2.5D slot design: per-series DINOv2 ViT encoder + slice-attention transformer, 6 canonical
  slots, cross-slot transformer study head, per-target attention pooling. Laterality
  normalization (flip-to-canonical + laterality-swap augmentation with paired-label swap).
- Source: CLAUDE.md §5, `src/rsna_knee/model.py`, `src/rsna_knee/dataset.py`.
- [ ] The 3D architecture experiment (exp007) and why it was killed — the diversity-vs-accuracy
  tradeoff, and the Effusion/Synovitis correlation anomaly worth remembering.
- Source: `docs/EXPERIMENTS.md` 2026-09-10 (exp007 epitaph entry), `src/rsna_knee/model_3d.py`.

## 6. Training stability

- [ ] The fold-collapse root cause (single global gradient clip vs. per-parameter-group clip),
  diagnosed via `--debug-grad-norms`, acceptance-tested 10/10.
- Source: `docs/EXPERIMENTS.md` 2026-09-07 entries.

## 7. Scale ladder and ensemble

- [ ] DINOv2 ViT-S vs. ViT-B, tested twice (once per label generation) — same result both times.
  Why DINOv3 and a native 3D encoder didn't make the cut.
- [ ] Greedy forward ensemble selection on OOF-vs-original-labels macro (gold-58 as tripwire
  only), global vs. per-label member selection, the "rank-mean has no population to rank against
  at test time" problem and its empirical-CDF fix.
- [ ] The laterality-swap TTA experiment that was measured and rejected.
- Source: `docs/EXPERIMENTS.md` 2026-09-09 entries, `scripts/build_ensemble.py`,
  `scripts/eval_tta_lateral_swap.py`, `ensemble_recipe_v1.yaml`.

## 8. Efficiency-track model

- [ ] Distillation design (KL+BCE mix against the ensemble teacher), resolution/slice reduction,
  the fold-count-vs-runtime tradeoff analysis (small, gradual cost — not a cliff).
- Source: `docs/EXPERIMENTS.md` 2026-09-09/10 entries, `docs/model_cards/exp008_dinov2s_slot_distilled.md`.

## 9. Infrastructure lessons (worth including — these cost real time and are reusable)

- [ ] The quota-guard story: routine-vs-emergency checkpoint floors, the "started already below
  the floor" failure mode, and the pre-flight check that fixed it.
- [ ] The Kaggle inference debugging arc: nested `/kaggle/input` mount layout, dataset zip
  extraction not preserving the package folder, GPU/PyTorch compute-capability mismatches.
- Source: `docs/EXPERIMENTS.md` (search "QUOTA GUARD" and the 2026-09-08/09 Kaggle notebook entries).

## 10. What didn't work / was tried and dropped

- [ ] labels_v4 (round-2 noisy student) — regressed, stopped at v3.
- [ ] exp007 (3D ResNet-50) — killed at 40% trained, ambiguous diversity signal.
- [ ] Laterality-swap TTA — measured OOF-negative.

## 11. Final leaderboard result

- [ ] OOF -> public LB -> private LB gap, both submissions (max-AUC ensemble + efficiency track),
  final rank.

## 12. Reproducibility

- [ ] Pointer to the repo, `CLAUDE.md`/`docs/EXECUTION_PLAN.md` as the source of truth, config-per-experiment
  convention, how to rebuild from `docs/EXPERIMENTS.md`'s chronological log.
