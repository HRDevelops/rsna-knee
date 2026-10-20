# CV course progress presentation (max 5 minutes)

File: `docs/presentation/progress_presentation_v2.pptx` (6 slides: the instructor's five plus one for the models; built by `scripts/fig_deck_assets.py` then `scripts/build_progress_deck.py`). The earlier `progress_presentation.pptx` is superseded.

Speaker notes: 476 words, about 3.5 min at 135 words/min and 4.0 min at 120.

## 1. Introduction
*Covers:* Title; expected output (12 probabilities, images only); application (research decision support); system figure; real MRI strip.

**Speaker notes**

> I'm working on the RSNA knee MRI challenge on Kaggle. A knee study is several image series in three planes. The system reads one study and outputs twelve probabilities, one per finding, like ACL tear, meniscus tears and effusion, from images only, scored by the average ROC-AUC. The use is research decision support, not a medical device. In the diagram, the training labels come from radiology reports, but only during training; at test time only the images are used.

## 2. Dataset
*Covers:* Name + URL; size (4,407 studies, 24,371 series, 819,078 DICOM slices, ~531 GB raw, 31.8 GB cache); input format; the 58-of-4,407 bar drawn to scale.

**Speaker notes**

> The data is the Kaggle competition set: 4,407 training studies, about 819 thousand DICOM slices, roughly 531 gigabytes. The test set is hidden, about 1,300 studies. The problem is the bar, drawn to scale: only 58 studies have expert labels. The rest only have a free-text report, and at test time there are no reports. I turn each study into a 64 by 336 by 336 tensor, then up to 54 windows of three slices.

## 3. Baseline model (my first model)
*Covers:* My baseline = the DINOv2 slot model. GitHub link (private until Oct 22); pretrained-model URLs (DINOv2 ViT-B and ViT-S on Hugging Face); performance (OOF 0.863 / 0.868, public 0.893).

**Speaker notes**

> My baseline is the first model I built, a DINOv2 slot model. Each series goes through a vision transformer and is squashed into one token; a small transformer mixes six slots and twelve queries read the result. It starts from public DINOv2 weights; the code is in my GitHub repository, private until the competition closes. The best version scored 0.868 out-of-fold. My first submission scored 0.893 publicly, though I believe it was the small model alone.

## Everything I trained (extra slide)
*Covers:* All models trained and scored on the same 5 folds (OOF macro-AUC) plus the things tried with no gain.

**Speaker notes**

> Every model I trained, on the same folds. Blue is the DINOv2 slot family. The biggest steps were better labels, 0.849 to 0.863, and a bigger backbone, about 0.005. A newer backbone was worse, a 3D network was too slow, and the tweaks on the right gave nothing beyond noise. Orange is the window-attention family, 0.882 and 0.884. The public leaderboard moved more than this proxy: 0.893 to 0.929.

## 4. Method
*Covers:* Design I started from (public notebook, 0.924 author-reported) vs my four changes; metrics; public-leaderboard comparison.

**Speaker notes**

> The public notebook I started from keeps every window of three adjacent slices and lets each finding attend over them. I rebuilt it and changed four things. Labels: Qwen and Llama read each report; a small calibrator per finding is fit on the 58 expert labels, always scoring a study with a calibrator that never saw it; then one noisy-student round. Folds are grouped by report cluster, and epochs are never chosen on the gold set. Windows never cross slots, and laterality is normalized. Result: 0.929 against 0.924 for the public notebook, a small difference.

## 5. Plan
*Covers:* Dates (Oct 22, Nov 23-27, Dec 5) and three concrete experiments with cost.

**Speaker notes**

> Next steps: on October 22 the private leaderboard shows whether 0.929 holds; then the demo in late November and the report on December 5. I also want to answer three questions with experiments. What caused the jump? The backbone is measured; I will undo the field of view and the laterality fix, five folds each, about 17 GPU-hours. Is the attention real? Delete the top-attended windows and compare with random ones. Does it survive missing data? Drop one plane at test time.

## Plain-language explanations (for questions)

**Which LLM did we use?** Both, per finding. Qwen2.5-72B-Instruct and Llama-3.3-70B each read every report. For each of the
12 findings, the better option was chosen by leave-one-out AUC on the 58 gold studies: the average of both for 8 findings
(ACL, MCL, medial meniscus, medial OA, lateral OA, PF OA, effusion, contusion), Qwen alone for 2 (lateral meniscus, fracture),
Llama alone for 2 (synovitis, Baker's). (`docs/label_blend_v2_final.md`)

**CoAtNet in our case.** CoAtNet is a hybrid image network: the early layers are convolutions (good at local texture), the
later layers are transformer attention (good at global context). We use `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` from timm:
72.9 M parameters, pretrained on ImageNet-12k and fine-tuned on ImageNet-1k at 384 px (generic photos, not knees), and we
fine-tune all of it. One input is a *window*: 3 adjacent MRI slices stacked as the three colour channels. The network turns
each window into a 1,024-number feature; a study has up to 54 windows, and a small attention layer (one per finding) weights
them to produce that finding's probability.

**"Leave-one-out calibrated on the 58 labels."** A language model gives a raw number per finding for each report, and raw
numbers are not trustworthy probabilities. Calibration (Platt scaling) fits a tiny logistic regression that maps the raw
number to a probability, using the studies that have real labels. Leave-one-out means: to score study i, fit that mapping on
the other 57 and apply it to study i; repeat for all 58. Then compute AUC over those 58 honest predictions (macro 0.823). The
labeler is never graded on data it was tuned on, and this is how the per-finding choice (Qwen / Llama / both) was made.

**"Noisy student v3."** Round 1: train an image model (exp002) on the report labels. For every study take its out-of-fold
prediction (from a model that never saw that study). Blend per finding: new label = (1 - w) x report label + w x image
prediction, with w chosen per finding by the same leave-one-out test on the 58 (image-heavy for 10 findings; report-heavy for
ACL and MCL, where reports are better). Train fresh models on these `v3` labels. The image model acts as a teacher whose
labels are imperfect, hence "noisy student". It gave +0.014 OOF. A second round (`v4`) got worse, so I stopped. Caution: the
blend weights were chosen on the same 58, so the 0.904 leave-one-out figure is optimistic; the independent check was OOF
against the original report labels (0.8631 vs 0.8489).

**Did we run only one baseline?** No. My own baseline family: exp002 (ViT-S), exp003 (ViT-B), exp005 (ViT-S, v3 labels),
exp006 (ViT-B, v3), exp004 (DINOv3, worse), exp009 (ConvNeXt-V2), exp007 (3D ResNet, stopped), exp008 (distilled), and the
3-model ensemble. The public Kaggle notebook was never run by me; only its author-reported score (0.924) is used.

## Fact sources

| Fact | Source |
|---|---|
| Competition URL, 4,407 / 24,371 / 819,078, 11-320 slices, mostly 512x512 | `artifacts/decode_audit.parquet`; `CLAUDE.md` s3; notebooks' `kernel-metadata.json` |
| ~531 GB raw (not re-measured; cluster unreachable); 31.8 GB cache | `docs/EXPERIMENTS.md` (`data/raw` 531GB; exp018 pre-registration) |
| Baseline model, OOF 0.8631 / 0.8681, public 0.893 (most likely exp005) | ledger B; `docs/EXPERIMENTS.md` 2026-09-27, 2026-10-01 |
| DINOv2 URLs (HTTP 200 on 2026-10-20) | `configs/exp005_*.yaml`, `exp006_*.yaml` backbone names |
| All models' OOF (chart) | ledger B; exp018/exp021 results; exp022 result |
| Things tried with no gain | ledger C and D; 2026-09-24 (EMA, CoAtNet-3); 2026-09-26 (dense windows) |
| Public notebook design, 0.924 author-reported | docstring of `kernels/knee-mri-twelve-findings-from-a-single-model/*.py` |
| exp022 partial test: 0.8681 slots -> 0.8709 windows -> 0.8816 CoAtNet-2 | ledger B; 2026-09-26 exp022 result |
| Plan costs: ~100 min per fold; 17 GPU-hours = 2 x 5 x ~100 min | exp018 fold-0 record (370 s/epoch x 16) |

## To confirm before presenting

- **The plan slide's three experiments are proposals**, not work already decided. Keep only the ones you will actually do.
- **GitHub repo is private** until Oct 22; the professor cannot open the link. Add them as a collaborator or keep the wording.
- "Demo Nov 23-27" is from your instruction; the slide does not say what the demo shows.
- The 531 GB figure is the earlier recorded `du`; it could not be re-measured.
- The deck contains real MRI slices (intro slide). Course use is very likely fine; check the competition's data terms before it goes anywhere public.
