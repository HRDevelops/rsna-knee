# RSNA Knee Abnormality Detection (Kaggle 2026)

Image-only prediction of **12 binary knee-MRI findings** (ACL, MCL, medial/lateral meniscus, medial/
lateral/patellofemoral OA, effusion, synovitis, Baker's cyst, contusion, fracture) from DICOM studies.
Scored by macro-averaged ROC-AUC, as a Kaggle code competition with ≤9h GPU and internet off.

The problem is **weak supervision**. Only 58 of 4,407 training studies carry expert image-review labels.
Every study has a free-text radiology report (12+ languages), but reports are unavailable at test time and
disagree systematically with the expert labels (~18%). This repository contains:
- the report → soft-label engine;
- the image models;
- the Kaggle submission notebooks;
- a full, chronological experiment log.

## Results

| Model | OOF macro-AUC¹ | Gold-58 macro-AUC | **Public LB** | Role |
|---|---|---|---|---|
| **exp018: CoAtNet-2 Window-MIL, 5 folds** | 0.8816 | 0.9091 | **0.9290** | **Final submission, main and efficiency track** |
| DINOv2 2.5D slot ensemble {exp006, exp003, exp005} | 0.8744 | 0.8953 | 0.8930 | Scored fallback |
| exp008: distilled DINOv2 ViT-S (efficiency) | 0.8708 | 0.888 | 0.8850 | Scored fallback |

¹ OOF is scored against report-derived labels, which are a noisy proxy. It understates the leaderboard
for every model and is not comparable across pipelines.

Efficiency track (`AUC/(0.5 − maxAUC) + seconds/32400`, lower is better, maxAUC assumed 0.95): exp018
scores **−2.036** vs exp008's −1.945. Its runtime is ~15–17 min projected for ~1,300 studies on
dual T4. At 2 GPUs, DICOM decode rather than the model bounds runtime, so the most accurate model is
also the efficiency-optimal one.

## How it works

**1. Label engine** (`scripts/label_reports.py`, `configs/rubric_v2.yaml`):
- Two open LLMs (Qwen2.5-72B-Instruct, Llama-3.3-70B, via vLLM) extract structured per-finding
  judgements under a rubric that encodes the experts' severity thresholds.
- A per-label strategy is calibrated on gold-58 with strict leave-one-out, giving `labels_v2_final`.
- One noisy-student round (image-model OOF blended back in) gives **`labels_v3_candidate`**, the
  training target for exp018.
- Folds are GroupKFold by report-template cluster, and gold-58 is never trained on.

**2. exp018: adjacent-slice Window-MIL** (`src/rsna_knee/wmil.py`):

```
DICOM study (3–14 series)
  └─ order slices by ImagePositionPatient · 140 mm FOV → 336 px · 6–94% of each series
     └─ 5 slots, 64 slices: Sag-FS 18 │ Sag 14 │ Cor-FS 12 │ Cor 8 │ Axial 12
        └─ laterality normalized (coronal/axial mirrored, sagittal slice order reversed for right knees)
           └─ windows = 3 adjacent same-slot slices as RGB          (≤ 54 per study)
              └─ CoAtNet-2 (timm coatnet_rmlp_2_rw_384, fully fine-tuned) → 1024-d per window
                 └─ per-finding attention-MIL: softmax over ALL windows, one per finding
                    └─ 12 logits → sigmoid → mean over 5 fold models
```

There is no per-series summary token and no slot bottleneck: every window competes directly for every
finding.

**3. Earlier pipeline** (`src/rsna_knee/model.py`), the DINOv2 2.5D slot transformer: slice-attention →
6 slots → cross-slot transformer → per-finding queries. It is kept for the scored fallbacks and as the
teacher for exp008.

Full method, including negative results: [`docs/METHOD_WRITEUP.md`](docs/METHOD_WRITEUP.md). Model cards:
[`docs/model_cards/`](docs/model_cards/).

## Repository layout

| Path | Contents |
|---|---|
| `src/rsna_knee/` | Library: preprocessing, datasets, models (`model.py`, `wmil.py`), inference drivers (`infer_core.py`, `infer_wmil.py`) |
| `scripts/` | Entry points: label engine, fold building, cache building, training, evaluation, Kaggle staging |
| `configs/` | One YAML per experiment (`exp018_coatnet2_wmil_v3labels.yaml` is the final model) plus LLM rubrics |
| `slurm/` | SLURM job scripts for the HPC runs |
| `notebooks/` | Kaggle submission notebooks (`submit-exp018.ipynb` is the final one) |
| `docs/` | Experiment log (`EXPERIMENTS.md`), method write-up, model cards, submission runbooks |
| `CLAUDE.md` | Project conventions and competition facts (the working spec this repo was built against) |

Data (`data/`), caches, checkpoints and label tables (`artifacts/`) are not in git.

## Quickstart

**Submitting / re-running the final model:** follow
[`docs/SUBMISSION_FINAL.md`](docs/SUBMISSION_FINAL.md). It lists the exact Kaggle notebook, the dataset
versions that produced the scored run, the healthy log line to check, and the code versions that must
never be deployed.

**Reproducing exp018 training** (HPC with SLURM). Paths assume `RSNA_ROOT` holds:
- `data/raw` (the competition data);
- `artifacts/labels/labels_v3_candidate.parquet` and `artifacts/folds.parquet` (from the label
  pipeline).

```bash
# 1. Build the Window-MIL cache (20-shard array; uint8 [64, 336, 336] per study, ~30 GB)
sbatch slurm/cache_build_wmil.sbatch

# 2. Train fold 0 only (the pre-registered gate); after it finishes, the gate script applies the
#    gold-58 kill rule and launches folds 1-4 if it passed
sbatch --array=0 slurm/train_wmil.sbatch configs/exp018_coatnet2_wmil_v3labels.yaml
python scripts/exp018_gate.py

# Direct form for one fold
python scripts/train_wmil.py --config configs/exp018_coatnet2_wmil_v3labels.yaml --fold 0 \
    --out "$RSNA_ROOT/artifacts" --resume auto
```

To build the Kaggle datasets and notebook: `scripts/stage_exp018_kaggle.py`, then
`scripts/build_exp018_notebook.py`. The label engine and folds are rebuilt with
`scripts/label_reports.py`, `scripts/build_labels_v2_final.py`, `scripts/build_labels_v3_candidate.py`,
`scripts/cluster_reports.py` and `scripts/build_folds.py`. The inputs, commands and outputs of each run
are logged in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

**Environment:** Python 3.12, PyTorch 2.x with CUDA, timm ≥ 1.0.26 (Kaggle's image; 1.0.29 on HPC), pydicom
with pylibjpeg / gdcm for JPEG-Lossless and JPEG-2000 DICOMs, pandas, numpy, scikit-learn, and vLLM for
the label engine. There is no pinned lockfile in the repo yet.

## Licence

Code is released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)). The only pretrained
weights used are the generic ImageNet initializations distributed with timm (CoAtNet-2) and DINOv2.
No external medical data, third-party knee-trained checkpoints, or third-party labels are used. The
exp018 design follows the public *dreaddevelopment* Kaggle kernels as an architectural reference (see
`docs/METHOD_WRITEUP.md` §5a). This is a research artifact and not a medical device.
