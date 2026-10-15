# CV course progress presentation (5 minutes, instructor's 5-slide structure)

File: `docs/presentation/progress_presentation.pptx` (built by `scripts/build_progress_deck.py`; speaker notes are in the file).
Speaker notes total 513 words: about 3.8 min at 135 words/min, 4.3 min at 120.

## 1. Introduction
*Required:* Project title; expected output and application; figure of the system (`docs/figures/system_overview.png`).

- Title: Weak Supervision and Window-Level Attention for Knee-MRI Abnormality Detection (RSNA Knee Abnormality Detection, Kaggle 2026).
- Expected output: 12 probabilities per knee-MRI study, from images only, scored by macro ROC-AUC.
- Application: research decision support, not a medical device; delivered as a Kaggle code competition (main track + efficiency track).

**Speaker notes**

> Hi, I'm Husnain. My project is on the RSNA knee MRI challenge on Kaggle. Given a knee MRI study, the system outputs twelve probabilities, one per finding: ACL, MCL, meniscus tears, arthritis, effusion and others. It uses images only and is scored by the average ROC-AUC. The application is research decision support, not a medical device. The figure shows the whole system. At the top, language models turn radiology reports into soft training labels, during training only. At the bottom is the image-only model: each study becomes windows of three adjacent slices, a CoAtNet encodes them, and each finding attends over all the windows.

## 2. Dataset
*Required:* Name and URL; size (samples and storage); input format.

- RSNA Knee Abnormality Detection (Kaggle competition): https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- 4,407 training studies (samples), 24,371 series, 819,078 DICOM slices; about 531 GB raw; my preprocessed cache 31.8 GB; hidden test ~1,300 studies.
- Input: raw one-DICOM-per-slice (11-320 slices per series, mostly 512x512, 221 sizes); cache (64, 336, 336) uint8 per study; model input (K<=54, 3, 384, 384); target 12 values.
- Only 58 studies have expert image-review labels; every study has a report, none at test time.

**Speaker notes**

> The data is the RSNA Knee Abnormality Detection competition on Kaggle: 4,407 training studies, 24,371 series, about 819 thousand DICOM slices, roughly 531 gigabytes raw. The test set is hidden, about 1,300 studies. The key problem: only 58 studies have expert labels. Every study has a free-text report, but reports don't exist at test time. I preprocess each study into a 64 by 336 by 336 tensor, 32 gigabytes in total, then into up to 54 windows of three slices at 384 pixels. That is what the model sees.

## 3. Baseline model
*Required:* GitHub/code link; pretrained-model URL; baseline performance.

- Baseline: public Kaggle notebooks by Dread Development (no GitHub repo): inference, training. Backbone library: timm on GitHub. My repo: github.com/HRDevelops/rsna-knee (private until Oct 22).
- Baseline weights (Kaggle dataset, NOT used by me): kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense. My backbone init: timm CoAtNet-2 (ImageNet-12k -> 1k) on Hugging Face.
- Performance: baseline public LB 0.924 (author-reported, not re-run; its gold-58 0.9167 is optimistic because epochs were chosen on those 58). My first model (DINOv2 slots): 0.893.

**Speaker notes**

> The baseline is a public Kaggle notebook by Dread Development: a single CoAtNet-2 with attention over windows of three adjacent slices. There is no GitHub repository; it is a Kaggle notebook with a separate training notebook, and its weights are a Kaggle dataset. All the links are here. I did not use those weights; my backbone starts from public timm ImageNet weights, not knee-trained. The baseline reports 0.924 on the public leaderboard, which I have not re-run. Its 0.9167 on the 58 gold studies is optimistic, because its training keeps the best epochs by AUC on those same 58. My own first model scored 0.893. My code is on GitHub, private until October 22.

## 4. Method
*Required:* Existing implementation and my modifications; evaluation metrics; preliminary results.

- Existing: 5 slots/64 slices, 140 mm crop to 336 px; 3-slice RGB windows into CoAtNet-2 (384 px); per-finding attention; LLM-derived soft labels (4,349 studies); epochs chosen on the 58 gold studies.
- Mine: own label engine (expert-threshold rubric, 2 LLMs, leave-one-out calibration, noisy-student v3); own folds grouped by report cluster, epochs chosen without gold; windows never cross slots + explicit laterality normalisation; 10-checkpoint ensemble; 2-fold fast variant; pre-registered gates; measured T4 runtime.
- Metrics: macro ROC-AUC (out-of-fold on 4,349 studies, gold-58, public leaderboard); efficiency score and runtime; qualitative: attention maps, per-finding ROC curves.
- Preliminary results: mine 5 folds 0.8816 / 0.9091 / 0.929; 10 checkpoints 0.8840 / 0.9111 / 0.929; 2 folds -- / 0.9078 / 0.925 (OOF / gold-58 / public LB).

**Speaker notes**

> The existing implementation fills five slots with 64 slices, crops 140 millimetres, builds three-slice windows, and applies per-finding attention; its labels also come from a language model reading reports. My modifications: my own label engine, with expert thresholds, two language models, and calibration on the 58; my own folds, grouped by report cluster, never choosing epochs on the gold set; windows that never cross slots, with explicit laterality handling; and a ten-checkpoint ensemble plus a fast two-fold version. I did not audit whether the baseline handles laterality. Evaluation is macro AUC, out-of-fold, on the 58 gold studies and on the public leaderboard, plus attention maps. Preliminary: 0.929 versus 0.924, a small difference I am not claiming as an improvement. Private results come October 22.

## 5. Plan
*Required:* Next steps and planned improvements.

- Next steps: Oct 22 private leaderboard; after the close release code, weights and write-up; Nov 23-27 demo; Dec 5 final report.
- Planned improvements (analyses): ablate the four bundled changes; radiologist review of the attention windows; trace the 2/58 prediction mismatch (suspected laterality rule).

**Speaker notes**

> Next steps. October 22: the competition closes and the private leaderboard shows whether these scores hold. Then I release code and weights, give the demo in late November, and submit the report on December 5. Planned improvements are analyses: ablate the four things I changed together to find which one mattered; ask a radiologist to review the attention windows, which are only a best-case illustration now; and trace two studies where deployed predictions differ from training, probably a laterality rule. Thank you.

## Fact sources

| Fact | Source |
|---|---|
| Competition name/URL | `competition_sources` in the notebooks' `kernel-metadata.json`; URL answered HTTP 200 on 2026-10-15 |
| 4,407 studies, 24,371 series, 819,078 DICOM files, 11-320 slices/series, 512x512 most common (24%), 221 sizes | `artifacts/decode_audit.parquet` (sum of `n_files`), `CLAUDE.md` s3 |
| ~531 GB raw | `docs/EXPERIMENTS.md` (`data/raw` 531GB, `du`); NOT re-measured today (HPC unreachable: DNS) |
| Cache 31.8 GB; (64,336,336) uint8; K<=54 windows; res 384 | `docs/EXPERIMENTS.md` exp018 pre-registration; `src/rsna_knee/wmil.py`; attention-extraction log (`res 384`) |
| Baseline design, labels, 0.924, 0.9167 | docstring of `kernels/knee-mri-twelve-findings-from-a-single-model/*.py` (author-reported, not re-run) |
| Baseline selects epochs on the 58 gold studies | `kernels/knee-mri-training-the-twelve-finding-model/*.py` (gold set used as validation, top-k by `gold_auc`) |
| Baseline notebook/dataset/HF/timm URLs | each answered HTTP 200 on 2026-10-15 |
| My results | `docs/EXPERIMENTS.md` (2026-09-21, 09-22, 09-25, 09-26, 10-01); `docs/CV_PROGRESS_PRESENTATION` earlier revision's source table |

## To confirm before presenting

- **GitHub repo is private** until Oct 22: the professor cannot open it. Either add them as a collaborator or keep the slide's 'private until the close' wording.
- **Slide 5's three 'planned improvements' are proposals**, not commitments you made. Edit or delete any you do not intend to do.
- 'Demo Nov 23-27' is taken from your instruction; the slide does not say what the demo shows.
- ~531 GB is the earlier recorded `du` of the raw data folder; it could not be re-measured today.
- The baseline's 0.924 and 0.9167 are what its author reports; I did not re-run it.
- The deck contains no MRI images (the system figure is a diagram), so the competition data terms are not an issue for this file.
