# EXECUTION PLAN — RSNA Knee Abnormality Detection
Target: top-10 main LB + top-3 efficiency. Today ≈ Sep 1. Final deadline Oct 22 (7.5 weeks).
Public reference to beat: DINOv2-S slot baseline @ 0.809 public.

Time allocation philosophy: ~40% label engine, ~35% image model, ~15% ensembling/distillation,
~10% submission engineering. The label engine is weighted highest because report-derived label
quality is the binding constraint on every image model trained downstream.

---

## PHASE 0 — Infrastructure (Sep 1–4)

**STATUS 2026-09-06: done except the heavy torch/vllm install.** Original sub-steps below are kept
for the record but were superseded by direct recon — see `docs/HPC_WORKFLOW.md`'s "CONFIRMED"
section for the real cluster facts (no `/scratch`, no modules/conda, BeeGFS at `/shared/home`,
`--qos=issai-h100` mandatory on H100, GPU nodes do have internet, H100 80GB confirmed). Data is
already downloaded and extracted at `/shared/home/$USER/rsna-knee/data/raw/` (4407 study dirs
verified). Repo is pushed to `/shared/home/$USER/rsna-knee/code/` via `scripts/push.sh`. venv is
created at `/shared/home/$USER/rsna-knee/venv` (system python3.9, no modules involved) with
pydicom/pylibjpeg/pandas/pyarrow/scikit-learn/PyYAML installed — `python-gdcm` was dropped, it has
no prebuilt wheel for this platform and fails to build (needs cmake, not installed); pydicom +
pylibjpeg's own JPEG/JPEG2000 plugins are sufficient. torch/vllm not yet installed (large,
deliberately deferred until right before they're needed).

0.1 HPC recon (30 min, interactive) — ORIGINAL PLAN, see status note above for what actually applies:
    ssh nu-hpc
    df -h $HOME; quota -s 2>/dev/null; lfs quota -h -u $USER /scratch 2>/dev/null
    module avail Python; module avail cuda; module avail CUDA
    sinfo -p H100 -o "%n %G %m %c %t"        # nodes, gpus, mem, cpus, state
    squeue -p H100 | head                     # queue pressure
    → Record: home quota, scratch quota/path, whether compute nodes have internet
      (curl -sI https://pypi.org from an srun shell). Competition data is likely
      100–300 GB; it MUST live on scratch/project storage, not $HOME.

0.2 Environment on HPC — use `/shared/home/$USER/rsna-knee/venv` + plain pip (no `module load`,
    no conda available on this cluster):
    python3 -m venv /shared/home/$USER/rsna-knee/venv
    source /shared/home/$USER/rsna-knee/venv/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg \
                timm transformers accelerate pandas pyarrow scikit-learn \
                iterative-stratification albumentations wandb vllm
    (Compute nodes DO have internet — confirmed via sbatch job with --qos=issai-h100 — so this
     can run inside a job too, not just on the login node. Skip python-gdcm, see status note.)

0.3 Kaggle CLI on HPC + data download — ALREADY DONE for this project, data lives at
    /shared/home/$USER/rsna-knee/data/raw/ (see docs/HPC_WORKFLOW.md §2 for the commands, kept
    for reference / re-download only).

0.4 Repo + sync loop: DONE. Local machine has no `rsync` (plain Git Bash) — `scripts/push.sh` /
    `pull.sh` auto-fall-back to tar-over-ssh (no delete-sync). `git init` done locally.

0.5 Decode audit job: `scripts/decode_audit.py` written, not yet run against the full dataset —
    next step. Will log transfer syntax, shape, PixelSpacing, SliceThickness, orientation, which
    of the 86 tags actually present (ESPECIALLY Laterality/ImageLaterality/PatientPosition). This
    decides the side-detection route.

Deliverable: data on HPC (✓), env green (lightweight deps ✓, torch/vllm pending),
`docs/DECODE_AUDIT.md` with tag census (pending — run `scripts/decode_audit.py` next).

## PHASE 1 — Label engine v1 (Sep 4–12) ← highest leverage

1.1 Report normalization: strip boilerplate ([DATE], headers), langid per report,
    near-duplicate clustering (MinHash on 5-gram shingles) → `report_cluster_id` for CV groups.
1.2 Rubric v1: one shared JSON-schema prompt, per-label severity rules (CLAUDE.md §2),
    few-shot with 4–6 translated exemplars including tricky negatives
    ("meniscal degeneration w/o tear" → negative; "trace effusion" → negative).
1.3 Run on HPC via vLLM: Qwen2.5-72B-Instruct-AWQ on 1×H100 (~4k reports × ~2k tokens; hours,
    not days). Second opinion: Llama-3.3-70B. Optional third: local Claude API pass on the
    hardest/lowest-agreement 500 reports.
1.4 Score against gold-58: per-label accuracy/F1 + per-language breakdown. Build the
    ERROR LEDGER (every disagreement: report quote, LLM answer, gold, hypothesized cause).
1.5 Iterate rubric per label (expect 2–3 rounds). Exit criteria: no label below ~0.80 gold
    agreement without a documented "reports simply lack this info" explanation
    (likely candidates: Synovitis, Contusion severity).
1.6 Produce `labels_v1.parquet`: per study × 12 → soft prob (ensemble mean of LLM confidences,
    per-label Platt-calibrated on gold-58 LOO) + confidence weight + evidence strings.
1.7 Side extraction from reports (right/left knee in all languages) → `study_side.csv`
    (free supervision for the laterality classifier).

Deliverable: labels_v1, gold agreement table, error ledger, side labels.

## PHASE 2 — Preprocessing cache + CV (Sep 8–14, overlaps Phase 1)

2.1 Cache builder (CPU-array jobs): per series → fixed-spacing resample (target 0.3 mm/px,
    384×384; store also a 512 crop for later), p1–p99.5 clip, uint8 or fp16, stacked npz/zarr.
    Estimate size; if >2× raw, drop to uint8 + 320px variant for the efficiency model.
2.2 Slice-order sanity: sort by ImagePositionPatient projected on normal (NOT InstanceNumber
    alone); assert monotone; log anomalies.
2.3 Laterality: if DICOM tags carry side → done. Else train side classifier
    (tiny ResNet on 3 middle coronal slices, labels from 1.7), validate ≥99.5%, then flip all
    right knees in the cache (store flip flag).
2.4 CV design: 5-fold GroupKFold on report_cluster_id (+ site proxy if discoverable from UID
    prefixes/scanner tags), iterative multilabel stratification on binarized soft labels.
    Gold-58 quarantined from training everywhere.

Deliverable: cached tensors on scratch, folds.parquet, side flags, cache-reader dataloader
benchmarked (must feed H100 >90% util).

## PHASE 3 — Image model v1 (Sep 12–22)

3.1 Reproduce-then-beat: our own slot model, DINOv2 ViT-S/14, T=24 slices, slice-attention +
    cross-slot transformer, BCE on soft labels weighted by labeler confidence, cosine LR,
    EMA, bf16, ~25 epochs. 5 folds on H100 (expect <2h/fold at S/14). Sanity target: OOF vs
    report-labels macro-AUC and gold-58 ≥ public-baseline territory (~0.80 LB) before scaling.
3.2 Scale ladder (one variable at a time, log everything in EXPERIMENTS.md):
    a) ViT-S → ViT-B/14 (→ L/14 if it pays)   b) DINOv2 → DINOv3 weights
    c) T=24→32 slices, 384→448 px             d) per-target attention pooling head
    e) init from MedSigLIP/BiomedCLIP vision tower (compare vs DINO)
3.3 First LB probe ~Sep 18 with best single fold-ensemble. Measure OOF↔LB gap ONCE, then
    budget ≤2 probes/week.
3.4 Parallel diversity track (2nd architecture family for the ensemble):
    3D route — resample study to 3 canonical volumes (one per plane) and run a 3D encoder
    (3D ResNet-50 / SwinUNETR encoder), OR video-transformer over slice sequence.
    Keep whichever is least correlated with the 2.5D model at similar AUC.

Deliverable: v1 OOF ≥ 0.80-LB-equivalent; two architecture families training.

## PHASE 4 — Label engine v2 + noisy student (Sep 20–Oct 3)

4.1 Blend: new target = w·report_prob + (1−w)·image_OOF_prob, w per-label by labeler confidence
    and gold agreement (labels where reports are weak lean on the image model, and vice versa).
4.2 Retrain both families on labels_v2. Tripwire: gold-58 AUC must not drop.
4.3 Optional round 3 only if round 2 clearly paid (>+0.005 OOF).
4.4 Hard-case mining: studies with max disagreement between report and image model → manual
    read of 30–50 evidence quotes; fix rubric or accept noise.
4.5 Pseudo-label the PUBLIC test? NO test images are available offline (hidden rerun) —
    skip; but DO consider external data: OAI, MRNet (Stanford), fastMRI knee, KneeMRI
    (Stajduhar) — all public. MRNet gives ACL/meniscus/abnormal labels for 1,370 exams →
    excellent auxiliary pretraining for ACL/meniscus heads. Verify license = "freely &
    publicly available" per rules; document in the writeup.

## PHASE 5 — Ensemble + distillation (Oct 1–14)

5.1 Ensemble: 2 families × 5 folds × (2 label versions if both good) — prune by greedy
    forward selection on OOF macro-AUC; compare prob-mean vs rank-mean per label (pick per label).
5.2 TTA: laterality-swap (with label swap) + slice-shift; keep only if OOF-positive.
5.3 Efficiency student: ViT-S or EffNetV2-S, T=16, 320px, distilled on ensemble soft preds
    (KL + BCE mix), fp16 + torch.compile; target ≤45 min total runtime incl. decode.
    Decode profiling: multiprocess pydicom w/ pylibjpeg; if JPEG2000 dominates, pre-benchmark
    nvidia DALI / nvjpeg2k path in the Kaggle env.
5.4 Runtime rehearsal: synthetic 1,300-study load test in a Kaggle notebook (copy train
    studies), measure wall-clock incl. cold start, add 25% margin.

## PHASE 6 — Submission engineering + endgame (Oct 10–22)

6.1 Freeze code Oct 15. Weights → Kaggle Datasets (versioned). Both notebooks green on
    synthetic full-size test with time margin.
6.2 Defensive checklist: header exactly matches sample (incl. `Baker's`), one row per test
    study even on decode failure (fallback = label priors), no NaNs, probs clipped to
    (1e-6, 1−1e-6), wall-clock guard with graceful degradation ladder.
6.3 Final selections: (1) max-AUC ensemble notebook, (2) efficiency student notebook —
    uncorrelated failure modes by construction.
6.4 Winners-obligation prep in parallel: repo cleanup, model cards, method writeup skeleton,
    video outline (obligation deadline Nov 5).

---

## Experiment budget (H100-hours, rough)
- Label engine LLM passes: 30–60
- Cache building: CPU-heavy, ~50 CPU-node-hours
- 2.5D training: ~15 configs × 5 folds × 2h ≈ 150
- 3D/diversity family: ~60
- Noisy-student retrains: ~80
- Distillation + misc: ~40
Total ≈ 400 H100-hours → very feasible over 7 weeks even with queue waits; submit long jobs
as fold-parallel array jobs (5 × 1-GPU beats 1 × 5-GPU here).

## Explicit non-goals (do not spend time on)
- Training on Kaggle GPUs (HPC only; Kaggle = inference rehearsal).
- Report-text models at inference (no reports at test time).
- Threshold tuning of any kind (AUC metric).
- Heavy 3D full-resolution models before the 2.5D ladder is exhausted.
- Public-LB chasing (>2 probes/week).
