# CLAUDE.md — RSNA Knee Abnormality Detection (Kaggle, 2026)

This file is the single source of truth for Claude Code working on this project.
Read it fully before writing any code. Goal: **top-10 main leaderboard + top-3 efficiency track**.

---

## 1. Competition in one paragraph

Predict per-study probabilities for **12 binary knee-MRI findings** from DICOM images only.
Metric: **macro-averaged ROC AUC** over the 12 targets (unweighted mean of per-label AUCs).
Code competition: submission is a Kaggle Notebook, **GPU ≤ 9h, internet OFF**, output `submission.csv`.
~1,300 test studies. Final deadline **2026-10-22 23:59 UTC**. Winners must open-source code + weights,
record a video, and publish weights publicly (plan for this from day 1 — clean repo, Apache-2.0).

Targets (exact column order, note the apostrophe in Baker's):
`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`

## 2. THE central fact — weak supervision

- **4,407 training studies. Only 58 carry the 12 labels. All 4,407 have a free-text radiology report.**
- **Reports are NOT provided at test time.** Inference must be image-only.
- Ground truth (train gold + entire test set) was created by **image review**: 2 MSK radiologists +
  adjudicator, **severity-thresholded criteria, "on the fence" graded NEGATIVE**.
- Report-derived labels agree with image-derived gold only **~82%**, and the disagreement is
  **systematic** (reports over-call mild disease; radiologist-annotators applied thresholds).
- Therefore: the quality of our report→label engine largely determines final rank.
  The 58 gold studies are a **calibration/validation set for the labeler**, not training fodder.

### Known systematic report-vs-gold gaps to encode in labeling rubrics
- **Meniscus**: "degeneration / intrasubstance signal / grade 1–2 signal not reaching surface" = NEGATIVE.
  Only tears (signal reaching articular surface, flap, bucket-handle, radial, root, macerated) = POSITIVE.
- **ACL/MCL**: "sprain grade 1", "mucoid degeneration", "intact but edematous" — likely NEGATIVE under
  severity thresholding; partial/complete tear = POSITIVE. Verify empirically against the 58 gold.
- **OA (3 compartments)**: "mild chondropathy / grade 1–2 chondral changes" may be NEGATIVE;
  compartment-specific — map medial TF / lateral TF / patellofemoral separately from cartilage,
  osteophyte, subchondral-change descriptions.
- **Effusion**: "trace/physiologic fluid" = NEGATIVE; "moderate/large effusion" = POSITIVE.
- **Fracture vs Contusion**: contusion = marrow edema without cortical break/fracture line;
  both can be positive together; insufficiency/stress fracture counts as Fracture.
- Rubric thresholds are per-label hypotheses — **tune each against the 58 gold and report per-label
  agreement before trusting**.

## 3. Data facts (verified locally)

- `train.csv`: 4,407 rows. `StudyInstanceUID`, `Report` (always present, 52–4,743 chars, median ~977),
  12 label columns (null except 58 rows; the 58 are fully labeled).
- Report languages (crude regex estimate): es ~908, en ~885, fr ~717, nl ~250, de ~156,
  other ~1,491 (includes uk/ru/pt/it/tr/pl etc. — RSNA says "a dozen languages, 16–19 sites").
- `train_series.csv`: 24,371 series; **5.53 series/study mean, median 5, range 3–14**.
  Planes: Sagittal 9,864 / Coronal 8,609 / Axial 5,898.
- **`Fluid_Sensitive` == `Fat_Suppression` for 100% of train series.** Docs explicitly warn they are
  not necessarily equivalent → treat as potentially decoupled at test time. Never merge the columns;
  build slot logic on `Fluid_Sensitive` primarily and make `Fat_Suppression` a soft feature.
- Gold-58 positive rates: ACL .41, MCL .16, MedMen .45, LatMen .40, MedOA .26, LatOA .19, PFOA .36,
  Effusion .60, Synovitis .47, Baker's .21, Contusion .33, Fracture .31. (Tiny n — wide CIs.)
- DICOMs: `train_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm`, one slice per file, 20–45 slices
  typical (median 30), tail to a few hundred. Mixed transfer syntaxes: Explicit VR LE, **JPEG Lossless,
  JPEG 2000**, Implicit VR LE → decoding needs pydicom + pylibjpeg/gdcm; decode cost is significant,
  cache aggressively. Metadata stripped to an **allowlist of 86 tags** (spacing/orientation survive;
  check whether `Laterality`/`ImageLaterality`/`BodyPartExamined` survive — if not, infer side from
  `ImageOrientationPatient` + anatomy or train a tiny side classifier).
- Test: ~1,300 studies, hidden; public/private split unknown; **prevalences may differ between
  train / public LB / private LB** (stated in the data notice — don't over-tune to public LB).

## 4. Efficiency track (second prize pool, $18k)

`Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`, minimized.
Benchmark = sample_submission score (0.5 macro-AUC), maxAUC = best private AUC → denominator is
negative, so the AUC term is negative and **higher AUC reduces the score**; every 9h of runtime adds
+1.0. Roughly: with maxAUC≈0.95, benchmark 0.5, going 0.90→0.92 AUC gains ~0.044; saving 24 min of
runtime gains the same. **Sweet spot: single distilled model, ~0.5–1h runtime, AUC within ~2 pts of
the big ensemble.** Efficiency submission must be one of our selected/eligible submissions → our two
final picks should be (a) max-AUC ensemble, (b) fast distilled model. This also satisfies the
"uncorrelated failure modes" selection principle.

## 5. Architecture plan (image side)

Per-study input = variable set of series. Canonical **6 slots**:
`{Sagittal, Coronal, Axial} × {Fluid_Sensitive 0/1}`, with presence mask (duplicates within a slot:
pick longest series or average embeddings).

**Stage A — per-series encoder (2.5D):**
- Resample each slice to fixed physical spacing (e.g., 0.3 mm/px crop-or-pad to 384×384 covering
  ~115 mm; ablate spacing), per-series intensity normalization (clip p1–p99.5, z-score).
- Sample T slices per series (e.g., T=24 uniform over the central anatomy; ablate T ∈ {16,24,32}).
- Backbone: fine-tuned **DINOv2/DINOv3 ViT-S/14 then ViT-B/14** (public 0.809 uses DINOv2-S — our
  headroom: bigger backbone, better labels, slice attention). Alternates to ablate: MedSigLIP /
  BiomedCLIP init, ConvNeXt-V2, and a 3D option (SwinUNETR-encoder or 3D ResNet on resampled volumes).
- Slice aggregation: **transformer/attention over slice tokens** (not mean-pool) + positional
  encoding by normalized slice index → series embedding. (RSNA 2023/2024 winning pattern:
  2.5D encoder + sequence model across slices.)

**Stage B — study head:** cross-slot transformer over ≤6 slot embeddings + mask + learned slot
embeddings → 12 logits. Per-target attention pooling (each target queries the slots it cares about:
menisci→Sag/Cor, PF OA→Axial/Sag, Baker's→Sag/Ax fluid-sensitive, etc. — learned, not hard-coded).

**Laterality normalization (critical):** medial/lateral targets (both menisci, Med/Lat OA, MCL) are
mirror-symmetric between left and right knees. Detect knee side per study and **horizontally flip
right knees** so "medial" is always the same image side. Side detection: DICOM orientation tags if
available, else a small classifier trained on studies whose reports state the side
("right knee", "rodilla derecha", "genou droit", "rechter Knie", "knie rechts"...) — reports give
near-free side labels for thousands of studies. Never use vertical flips as augmentation; horizontal
flip only as a *deliberate* laterality-swap aug that also swaps the 4 paired labels.

**Augmentation:** anatomy-preserving — small rotations (±10°), scale (±10%), translation, mild
intensity jitter, coarse dropout; laterality-swap aug with label swap; slice-sampling jitter.
No vertical flip. MixUp/label-smoothing optional given soft labels already smooth.

## 6. Label engine (text side) — the priority workstream

1. **Language ID + dedup**: fingerprint reports; near-duplicate/templated reports must share CV folds.
2. **LLM extraction ensemble**: run 2–3 strong open LLMs on HPC via vLLM (e.g., Qwen2.5-72B-Instruct,
   Llama-3.3-70B, Gemma-3-27B) + optionally Claude API locally. One prompt per study, structured JSON
   out: for each of 12 targets → {present: yes/no/uncertain, severity phrase, evidence quote,
   confidence 0–1}. Rubric embeds the severity thresholds from §2 and instructs
   **"uncertain/on-the-fence → negative-leaning (0.2–0.35), not 0.5"**.
3. **Calibrate against gold-58**: per-label agreement, per-language agreement; iterate rubric per
   label until agreement plateaus (report ≥ .90 target on the labels where reports contain the info;
   accept lower on Synovitis/Contusion which reports under-specify). Keep an error ledger of
   disagreements — many will reveal rubric fixes.
4. **Soft labels**: final training target = calibrated blend of LLM ensemble probabilities
   (e.g., isotonic/Platt per label fit on gold-58 with leave-one-out). Missing-information cases
   get the label-prior, with a per-sample confidence weight used in the loss.
5. **Round-2 (noisy student)**: after first image model, blend image-model OOF probs with report
   probs (report-conf-weighted) → retrain. Iterate once or twice; watch the gold-58 reference AUC
   for degradation (confirmation-bias risk).

## 7. Validation protocol

- 5-fold **GroupKFold** by report-template/site cluster (near-dup reports and same-site studies in
  the same fold), multilabel-stratified within groups (iterative stratification on the soft labels
  binarized at 0.5).
- **Gold-58 is a fold-safe reference set**: exclude from all training; report macro-AUC on it every
  run (noisy, n=58 → use as tripwire not as selector).
- Primary model-selection signal: **OOF macro-AUC against calibrated report labels**, secondary:
  gold-58 AUC, tertiary: public LB (max ~2 probes/week; prevalences shift between splits).
- Expect an OOF↔LB gap; measure it once early, then trust deltas not absolutes.
- Log per-label AUC always — macro metric means a rare label (MCL, Lateral OA, Baker's) moves the
  score as much as ACL; **weakest-label improvements are the cheapest points**.

## 8. Efficiency-model plan

Distill the final ensemble into ONE compact model: ViT-S/14 (or EfficientNetV2-S 2.5D) student
trained on ensemble soft predictions over all 4,407 studies + test-style TTA-free inference.
Runtime budget target: **≤ 45 min** for ~1,300 studies incl. DICOM decode →
fp16, batched decode with multiprocessing, T=16 slices, 320px, no TTA, ONNX or torch.compile.
Profile decode separately — it may dominate; pre-resize on GPU with nvJPEG2000/DALI if needed.

## 9. Kaggle submission mechanics

- Internet OFF → all weights via Kaggle Datasets/Models (`hrdevelops/*` account). Pin wheel files
  for any package not preinstalled (pylibjpeg, pylibjpeg-libjpeg, python-gdcm...).
- Notebook reads `/kaggle/input/rsna-knee-abnormality-detection/{test.csv,test_series.csv,test_series/}`;
  writes `submission.csv` with exact header incl. `Baker's`.
- Rehearse the hidden-test swap: code must handle ~1,300 studies, unseen series counts (3–14),
  possibly decoupled Fluid_Sensitive/Fat_Suppression, missing planes (guard the slot mask),
  corrupt/short series (fallback: fewer slices, or skip series, never crash — a crash = no score).
- Wall-clock guard: time budget tracker; if projected overrun, degrade gracefully (drop TTA →
  drop ensemble members → smallest model) rather than time out.
- Keep two notebooks: `submit-ensemble.ipynb` and `submit-efficiency.ipynb`.

## 10. Compute & environments

- **Local**: Windows + Anaconda, RTX 4090 24GB (`python`, not `py`, inside envs). Use for: EDA,
  prompt-engineering the labeler on samples, small ablations, ONNX/quantization, packaging.
- **HPC** (`ssh nu-hpc`): SLURM, partition `H100`, qos `issai-h100`. Use for: full DICOM
  preprocessing/caching, vLLM label generation (70B models), all serious training, ensembles.
  See `docs/HPC_WORKFLOW.md` for quotas, kaggle CLI, job templates, rsync loop.
- **Kaggle**: RTX PRO 6000 Blackwell 102GB available in notebooks — but training happens on HPC;
  Kaggle only for inference rehearsal + submission.
- Telegram bot notifications for long jobs (existing integration).

## 11. Code conventions for Claude Code

- Repo layout: `src/` (library), `scripts/` (entry points), `configs/` (yaml), `slurm/` (sbatch),
  `notebooks/` (Kaggle submission notebooks), `docs/`, `artifacts/` (gitignored), `logs/`.
- Every training script: resumable checkpoints, deterministic seeds logged, OOF predictions saved
  as parquet, config snapshot saved next to weights, wall-time guard.
- Preprocessing writes a **cache once** (per-series float16 tensors or 8-bit PNG stacks at fixed
  spacing) → all experiments read cache; never decode DICOM inside the training loop.
- One experiment = one yaml in `configs/`, results appended to `docs/EXPERIMENTS.md`
  (date, config, OOF macro-AUC, per-label AUC, gold-58 AUC, LB if probed, notes).
- Audit every rank/sigmoid/aggregation transformation end-to-end (BirdCLEF double-ranking lesson).
- AUC is rank-based: per-label monotone calibration doesn't change score, but **ensembling in rank
  space** (average of per-model per-label ranks) is often more robust than prob averaging — implement
  both, compare on OOF.

## 12. Risk register

| Risk | Mitigation |
|---|---|
| Labeler systematically wrong on a label → ceiling on that AUC | Per-label gold-58 agreement gates; error ledger; per-label rubric iteration |
| Report/site template leakage inflates OOF | Group folds by report-template cluster + site |
| Test Fluid_Sensitive ≠ Fat_Suppression | Slot logic on Fluid_Sensitive; robustness aug: randomly corrupt the flag 5% in training |
| Side detection wrong → paired labels poisoned | Validate side classifier on report-stated-side studies; if uncertain, run both flips and average with label-swap |
| Prevalence shift public→private | Never threshold-tune on LB; AUC is prevalence-invariant per label, keep calibration simple |
| 9h timeout / crash on hidden data | Wall-clock guard, defensive decode fallbacks, rehearse with synthetic 1,300-study load test |
| Winners' obligations (open weights, video) | Apache-2.0 from day 1, training code reproducible from configs |
