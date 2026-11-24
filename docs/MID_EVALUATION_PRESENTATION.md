# RSNA Knee Abnormality Detection — Course Mid-Evaluation Presentation

This document contains the complete script, slide content, timing breakdown, and technical defense for the 5-minute Computer Vision semester course project mid-evaluation presentation.

- **Presentation PowerPoint Deck**: [`docs/presentation/mid_evaluation_presentation.pptx`](presentation/mid_evaluation_presentation.pptx)
- **Deck Generator Script**: [`scripts/build_mid_eval_deck.py`](../scripts/build_mid_eval_deck.py)
- **Duration Limit**: Maximum 5 minutes (Strict adherence: calibrated to 4 minutes 45 seconds total delivery time)

---

## Slide 1: Introduction

- **Title**: Weak Supervision & Window-Level Attention for Knee-MRI Abnormality Detection
- **Subtitle**: RSNA Knee Abnormality Detection (Kaggle 2026) · Multilabel 3D Computer Vision
- **Expected Output**:
  - 12 binary knee-MRI pathologies predicted simultaneously per patient study:
    - Ligament Tears: ACL, MCL
    - Meniscal Tears: Medial Meniscus, Lateral Meniscus
    - Osteoarthritis: Medial OA, Lateral OA, PF OA
    - Joint Fluid / Lining: Effusion, Synovitis, Baker's Cyst
    - Bone Damage: Bone Contusion, Fracture
  - Evaluation Metric: Macro-averaged ROC-AUC across all 12 findings.
- **Application**:
  - Research-grade clinical decision support and triage for radiologists interpreting multi-sequence knee MRIs.
  - Deployed in a Kaggle code competition environment (GPU ≤ 9h, internet disabled, dual main and efficiency tracks).
- **System Figure**: [`docs/figures/system_overview.png`](figures/system_overview.png)
  - *Top Lane (Training Only)*: Radiology Reports (4,407 studies, 12+ languages) → Dual-LLM Label Engine (Qwen2.5-72B + Llama-3.3-70B) with clinical severity thresholding → Leave-One-Out Calibration on 58 Gold studies → Soft Labels (v3 Noisy Student).
  - *Bottom Lane (Inference & Training)*: Raw DICOM Series (3–14 per study) → 5 Anatomical Slots (64 slices, 140 mm FOV, 336px, laterality-fixed) → 3-Slice Adjacent RGB Windows (≤ 54 windows) → Fine-tuned CoAtNet-2 backbone → Per-finding Attention-MIL (softmax over ALL windows) → 12 abnormality probabilities.

### Presenter Notes ([0:00 - 0:50] | ~110 words)
> "Good morning, Professor. My semester computer vision project addresses the RSNA Knee Abnormality Detection challenge. Our objective is to predict twelve binary knee-MRI findings—including ACL, MCL, meniscal tears, and osteoarthritis compartments—from 3D DICOM volumes without human intervention. The expected output is a calibrated vector of twelve probabilities evaluated by macro ROC-AUC. In clinical workflows, this serves as an automated triage and second-reader decision support tool. As illustrated in the system diagram, the pipeline has two components: during training, a dual-LLM label engine extracts soft supervisory signals from free-text radiology reports; at inference time, an image-only CoAtNet-2 model processes adjacent slice windows with multi-instance attention to predict all findings directly from DICOM images."

---

## Slide 2: Dataset

- **Name & Direct Access URL**:
  - Name: RSNA Knee Abnormality Detection (Kaggle Competition, 2026)
  - URL: [https://www.kaggle.com/competitions/rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
  - International multi-center cohort across 16–19 hospital sites with 12+ languages.
- **Dataset Scale & Storage Footprint**:
  - Training Cohort: **4,407 patient studies**, 24,371 series, 819,078 DICOM slice images.
  - Hidden Test Cohort: ~1,300 studies evaluated server-side in Kaggle notebook sandbox.
  - Raw Storage Size: **~531 GB** uncompressed DICOM files.
  - Preprocessed Tensor Cache: **31.8 GB** (7.2 MB/study, float-normalized uint8, 16.7× storage reduction).
- **Input Formats & Tensor Dimensions**:
  - Raw DICOM Input: Variable series per study (3–14, median 5); 11–320 slices per series (median 30); mostly 512×512 across 221 distinct image matrix sizes.
  - Preprocessed 5-Slot Cache Tensor: `(N=4407, 64, 336, 336)` uint8 across 5 slots: Sag-FS (18), Sag (14), Cor-FS (12), Cor (8), Axial (12); resampled at 140 mm FOV.
  - Model Input Format: `(Batch, K ≤ 54, 3, 384, 384)` float32 — adjacent 3-slice RGB windows.
  - Target Format: `(Batch, 12)` continuous probability vector in `[0, 1]`.
- **The Core Scientific Challenge**:
  - 4,407 studies have free-text radiology reports, but reports do not exist at test time.
  - **Only 58 studies** carry expert image-review labels (2 MSK radiologists + adjudicator).
  - Reports disagree with expert image ground-truth in **~18%** of cases due to systematic over-calling of mild/trace disease as positive.

### Presenter Notes ([0:50 - 1:45] | ~120 words)
> "Turning to the dataset: we use the official RSNA Knee Abnormality Detection benchmark. The training corpus contains 4,407 patient studies, over 24,000 series, and approximately 819,000 DICOM slices, totaling 531 gigabytes uncompressed. The hidden test set comprises roughly 1,300 studies. Raw inputs are variable-slice DICOM series across axial, coronal, and sagittal planes. We preprocess and normalize each study into a unified 64-slice tensor across five anatomical slots, taking 31.8 gigabytes. The central technical challenge is weak supervision: while all 4,407 studies have radiology reports, only 58 have expert image-review labels. Crucially, reports do not exist at test time, and naive reports disagree with expert image ground-truth in about 18% of cases due to reporting mild, sub-threshold findings as positive."

---

## Slide 3: Baseline Model

- **GitHub Repository & Code Links**:
  - Baseline Architecture: Dread Development public Kaggle kernels (no standalone GitHub repository):
    - [Inference Kernel](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)
    - [Training Kernel](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)
  - Core Backbone Vision Library: `timm` (PyTorch Image Models) on GitHub: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
  - Our Research Implementation: [https://github.com/HRDevelops/rsna-knee](https://github.com/HRDevelops/rsna-knee) (private during competition).
- **Pretrained Model Weights & Initializations**:
  - Baseline Author Weights: Kaggle dataset [raptor-knee-widedense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) (Note: strictly NOT used by our pipeline to avoid test contamination).
  - Our Backbone Initialization: Hugging Face timm CoAtNet-2 ImageNet-12k → 1k: [coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k) (generic natural image weights; zero external knee MRI data used).
- **Baseline Performance & Critical Audit**:
  - Baseline Public Leaderboard AUC: **0.924**.
  - Reported Gold-58 AUC: **0.9167** (Note: this is optimistic and in-sample because the author selected checkpoint epochs directly on those 58 gold studies).
  - Our Initial Baseline (DINOv2 Slot Transformer): **0.893** Public LB (OOF 0.8631, Gold-58 0.8903).
  - Methodological Hygiene: Gold-58 is strictly quarantined as a zero-leak tripwire; our epochs are selected strictly by out-of-fold validation.

### Presenter Notes ([1:45 - 2:40] | ~115 words)
> "Moving to the baseline: the leading public approach was introduced by Dread Development via Kaggle kernels, employing a CoAtNet-2 architecture with window-level attention. It lacks a GitHub repo, but uses the open-source timm vision library. The author-reported baseline achieves 0.924 macro-AUC on the public leaderboard and 0.9167 on the 58 gold studies. However, critical analysis reveals that this 0.9167 is optimistic and in-sample: the author selected checkpoint epochs directly on those 58 gold studies. To ensure strict scientific validity, our own implementation trains from generic ImageNet weights, quarantines the 58 gold studies as a zero-leak tripwire, and selects epochs exclusively using out-of-fold validation. For reference, our initial DINOv2 slice-attention baseline reached 0.893 public AUC."

---

## Slide 4: Method & Preliminary Results

- **Existing Implementation vs. Our Modifications**:
  - *Existing Baseline Implementation*:
    - 5 slots, 64 slices, 140 mm FOV crop to 336px.
    - Adjacent 3-slice RGB windows into CoAtNet-2 (384px); per-finding attention-MIL pooling.
    - Single uncalibrated LLM extraction; epochs chosen directly on the 58 gold studies.
  - *Our Core Technical Modifications*:
    1. **Dual-LLM Severity Label Engine**: Qwen-72B + Llama-70B with clinical severity thresholds; LOO calibration & noisy-student distillation (LOO-AUC 0.823 → 0.904).
    2. **Template-Leak-Free GroupKFold**: MinHash + LSH clustering on report text (3,732 clusters). No template leakage; zero-leak validation; epochs chosen without gold.
    3. **Slot-Isolated Windowing & Laterality Fix**: Windows never cross planes; coronal mirroring + sagittal slice-order reversal resolves 99.4% of right knees correctly.
    4. **Dual-Track Deployment Architecture**: Main track: 10-checkpoint seed ensemble; Efficiency track: 2-fold model (~20.5 min on Kaggle 2×T4, hitting the decode floor).
- **Evaluation Metrics**:
  - Quantitative: Macro ROC-AUC across 12 findings evaluated on OOF (4,349 studies), Gold-58 (unseen 58 studies), and Public LB (~1,300 test studies); Competition Efficiency Metric `AUC / (0.5 - maxAUC) + Runtime / 32400` and T4 wall-clock seconds.
  - Qualitative: Per-finding attention localization maps; per-finding ROC curves.
- **Preliminary Results Table**:

| Model / Configuration | OOF Macro | Gold-58 AUC | Public LB | T4 Runtime | Track Role / Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Public Baseline** (Dread Development) | — | 0.9167* | 0.924 | — | Reference Baseline |
| **Our 1st Model** (DINOv2 Slots, exp005) | 0.8631 | 0.8903 | 0.893 | ~59 s / study | Initial Baseline |
| **exp018: CoAtNet-2 Window-MIL** (5 Folds) | 0.8816 | 0.9091 | **0.929** | ~53 min | Core Architecture |
| **exp021: 10-Checkpoint Ensemble** (2 Seeds) | **0.8840** | **0.9111** | **0.929** | ~100 min | **Final Main Track (Selected)** |
| **exp018: 2-Fold Fast Variant** (Folds {0, 1}) | — | 0.9078 | **0.925** | **~20.5 min** | **Final Efficiency Track (Selected)** |

*\*Note: Baseline Gold-58 AUC is optimistic due to in-sample epoch selection on gold-58. Our models never train or select on gold-58.*

### Presenter Notes ([2:40 - 4:05] | ~165 words)
> "Slide 4 presents our method, architectural modifications, and quantitative results. While the baseline used uncalibrated LLM labels and in-sample epoch selection, we introduced four fundamental improvements: First, a dual-LLM label engine combining Qwen-72B and Llama-70B with clinical severity thresholds, followed by leave-one-out calibration and noisy-student iteration that raised report label quality from 0.823 to 0.904 LOO-AUC. Second, report-template GroupKFold clustering across 3,732 clusters, completely eliminating site leakage. Third, slot-isolated windowing and 3D laterality normalization, correcting coronal mirroring and sagittal slice ordering for right knees across 99.4% of studies. Fourth, a dual-track submission strategy: a 10-checkpoint seed ensemble for the main leaderboard, and a 2-fold model for the efficiency track. As shown in the table, our 10-checkpoint model achieves 0.884 OOF, 0.9111 on gold-58, and 0.929 on the public leaderboard. For efficiency, our 2-fold model matches 0.925 public AUC while running in only 20.5 minutes—hitting the physical DICOM decoding floor on Kaggle's dual T4 hardware."

---

## Slide 5: Plan & Completed Experiments

- **Completed Experiments & Empirical Nulls** (Evidence from Pre-Registered Hypotheses):
  - *Backbone Explorations*: DINOv3 ViT-S was significantly negative (−0.025 OOF); ConvNeXt-V2 was 93% redundant with ViTs; 3D ResNet-50 stalled at 0.693 AUC.
  - *Spatial & Resolution Scaling*: 448 px (vs 384 px) and T=32 slices (vs 24) were strictly null (Δ ≈ −0.0001); dense 87 windows (vs 51) was also null.
  - *Auxiliary Multilingual Text Loss*: BGE-M3 text regression (+0.0036 OOF) failed our pre-registered 2-SD bar (+0.0082) and did not hold on gold-58.
  - *Runtime Profiling*: Proved CoAtNet-2 is model-bound on T4 (~9.7 min/fold), directly motivating the 2-fold efficiency breakthrough.
- **Actionable Planned Improvements**:
  1. **Single-Variable Ablation Study**: Disentangle the 0.893 → 0.929 jump by isolating the individual gains of: CoAtNet backbone, Window-MIL pooling, 140 mm FOV, and laterality fix.
  2. **Clinical Expert Validation of Attention Maps**: Engage an MSK radiologist to grade whether high-attention windows (top-4 per finding) align with true anatomical pathology rather than imaging artifacts.
  3. **TensorRT & GPU DICOM Acceleration**: Compile CoAtNet-2 into FP16 TensorRT / torch.compile with nvJPEG2000 decoding, aiming to run all 5 folds within the 20.5-min efficiency envelope.
  4. **Reconcile 2/58 Deployment Outliers**: Investigate the 2 studies showing minor divergence between training and Kaggle deployment, resolving edge cases in patient orientation tags.
- **Project Milestones & Delivery Roadmap**:
  - *Oct 22, 2026*: Competition closes & Private Leaderboard reveal (evaluates test generalization on ~1,300 hidden cases).
  - *Late Oct 2026*: Open-source release of Apache-2.0 code, weights, and documentation.
  - *Nov 23–27, 2026*: Course project live demonstration showcasing model inference & attention overlays.
  - *Dec 05, 2026*: Final research report submission.

### Presenter Notes ([4:05 - 4:55] | ~115 words)
> "To conclude, Slide 5 outlines our next steps and planned improvements, grounded directly in the experiments we have already completed. In earlier rounds, we rigorously evaluated and ruled out unproductive paths: scaling slice count to 32 or resolution to 448 pixels proved completely null, ConvNeXt backbones were 93% redundant, and multilingual report-text auxiliary losses failed to surpass our pre-registered bar. With these insights, our next steps focus on high-impact validation and deployment: First, conducting a clean single-variable ablation to isolate the individual contributions of the four bundled changes in the 0.929 model. Second, partnering with an MSK radiologist to clinically validate our attention maps against anatomical lesions. Third, deploying FP16 TensorRT compilation to enable full 5-fold ensembling under the 20-minute efficiency budget. Finally, open-sourcing our code and model weights upon competition close on October 22. Thank you, and I look forward to your questions."
