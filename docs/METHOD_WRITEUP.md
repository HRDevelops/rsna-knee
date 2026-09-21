# RSNA Knee Abnormality Detection — Method Writeup

Winners' obligation write-up (competition rules) accompanying the open-sourced code, weights, and
presentation video. Drafted 2026-09-15, before the leaderboard is known — every number below is a
real, logged measurement from `docs/EXPERIMENTS.md` except §1's final rank and §11, which are
placeholders pending the actual probes. Update those two sections (and re-check nothing else drifted)
once probe #1/#2 are scored and the competition closes; everything else here is submission-ready.

## 1. Summary

We treat this competition as a **label-quality problem first, a model-architecture problem
second**. Ground truth was built by image review under severity thresholds that report text
systematically disagrees with (~18%, see §3) — so most of our effort (§4) went into an LLM-based
report-label engine, calibrated and iterated against the 58 gold-labeled studies, then improved
further with two rounds of noisy-student blending against our own image model's out-of-fold
predictions. The image side (§5–§7) is a 2.5D per-series ViT encoder with slice- and cross-slot
attention, laterality-normalized, ensembled via greedy forward selection over four candidate
models. Final ensemble OOF macro-AUC (vs. the calibrated report labels) is **0.8744**, distilled
into a single efficiency-track model at **0.8708** (−0.0036, comfortably inside the 0.015 target)
running in **7.5 minutes** per ~1,300 studies on the measured hardware (§8). Both probes are
submitted and scored: **ensemble 0.893 public LB, efficiency 0.885 public LB** (−0.008 delta,
inside the CLAUDE.md §4 efficiency-track target); both OOF↔LB gaps measured trustworthy
(−0.0186, −0.0142, both < the pre-registered 0.02 threshold — see §11 and `docs/EXPERIMENTS.md`).
Final private-leaderboard rank: **pending competition close.**

Solo effort, on a shared university HPC cluster (SLURM, H100 x4/node, BeeGFS shared storage — see
`docs/HPC_WORKFLOW.md`). Total GPU-hours not separately tallocated, but the largest single costs
were the label-engine LLM passes (Qwen2.5-72B-Instruct + Llama-3.3-70B via vLLM, full 4,407-report
corpus, ~3.5h for the first full pass) and the image-model scale ladder / ensemble members (5-fold
training runs of DINOv2 ViT-S/14 and ViT-B/14 at T=24/384px, each roughly 2h/fold on H100). A
parallel 3D-architecture diversity attempt (ResNet-50, §5, §10) was killed at 40% trained
specifically because its per-fold cost (~10.5h) wasn't justified by an ambiguous diversity signal.

## 2. Competition overview

Predict per-study probabilities for 12 binary knee-MRI findings (`ACL, MCL, Medial Meniscus,
Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`)
from DICOM images only — reports are available at train time but explicitly **not** at test time.
Metric: macro-averaged ROC-AUC over the 12 targets. Code competition, GPU ≤9h, internet off. 4,407
training studies (only 58 with direct image-derived gold labels; the rest carry only a free-text
radiology report), ~1,300 hidden test studies. A parallel efficiency track scores
`AUC/(0.5−maxAUC) + RuntimeSeconds/32400` — see §8. Full rules and data facts: `CLAUDE.md` §1–§3.

## 3. The central insight: image-derived vs. report-derived labels

Ground truth (both the 58-study gold set and the entire hidden test set) was produced by **image
review** under severity-thresholded criteria — "on the fence" cases graded negative. Free-text
reports, by contrast, tend to over-call mild disease (a "grade 1 sprain" or "trace effusion" reads
as positive in a report but negative under the image-review threshold). The two sources agree only
about 82% of the time, and the disagreement is *systematic*, not random noise — meniscal
degeneration without a tear, mild chondral change, and physiologic fluid are the recurring false
positives a naive report-labeling approach would produce.

This is why the label engine, not the image architecture, was the priority workstream (~40% of
project time by design, `docs/EXECUTION_PLAN.md`): with only 58 truly gold-labeled studies, the
quality of the report → soft-label pipeline used to train on the other 4,349 studies is the actual
ceiling on everything trained downstream of it.

## 4. Label engine

### 4.1 Report clustering + fold construction

Near-duplicate and site-template detection uses MinHash over **word** 5-gram shingles + LSH, with
a post-LSH approximate-Jaccard verification (threshold 0.5) before union-find merging into
`report_cluster_id`. This replaced a first attempt using **character** 5-grams, which produced one
spurious 2,134-study "cluster" (48% of the dataset) spanning unrelated Spanish/German/English
reports — character n-grams are too generic across languages' common short substrings, and
unverified LSH-band collisions chained transitively into that mega-cluster. The word-shingle +
verification fix produced 3,732 clusters from 4,407 studies, the largest a genuine site template
(209 studies, 4.7% of the dataset). `report_cluster_id` is the GroupKFold grouping key for all 5
folds, keeping near-duplicate reports and same-site studies together so OOF metrics aren't
inflated by template leakage. Gold-58 is quarantined from training in every fold.

### 4.2 LLM label extraction

Two independent open models via vLLM on HPC: **Qwen2.5-72B-Instruct** and **Llama-3.3-70B**, one
prompt per study/report, structured JSON output per target (`present: yes/no/uncertain, severity
phrase, evidence quote, confidence`). The rubric embeds the severity thresholds from §3 directly
("uncertain/on-the-fence → negative-leaning 0.2–0.35, not 0.5") and iterated twice: v1's initial
full run (4,407 studies, 3h32m) had a 2.4% JSON-parse failure rate, traced to the model prepending
an unrequested preamble before the JSON, which then got truncated by too tight a token budget —
fixed by instructing "start immediately with `{`, no preamble" and raising `max_tokens` 1400→2200.
v2 reverted a conditional Synovitis prior back to a flat 0.35/confidence-0.1 default after it made
gold-58 agreement worse, not better.

Per-label, the two models specialize differently: Llama wins MCL (blend LOO-AUC 0.943 vs. Qwen's
0.807), Synovitis (0.711 vs. 0.341 — Qwen alone is *worse than random* on this label), and Baker's
(0.862); Qwen wins Lateral Meniscus (0.812). This per-label split is exactly why the calibrated
blend (§4.3) picks a *strategy per label* rather than a single fixed weighting.

### 4.3 Calibrated blend — `labels_v2_final.parquet`

Per label, chosen among `{Qwen-only, Llama-only, Platt-calibrated blend}` by strict leave-one-out
selection on gold-58 (no label ever sees its own gold-58 row during calibration-fit or strategy
selection). **Macro LOO-AUC of the chosen-per-label strategy: 0.823.** This is the report-only
ceiling — the ~4,349 non-gold studies' training targets for every "v2"-generation image
experiment (exp002/exp003/exp004).

### 4.4 Noisy-student round — `labels_v3_candidate.parquet`

Blends the image model's own OOF predictions (plus fold-ensembled gold-58 predictions) back into
the label engine, per-label weight again chosen by strict LOO on gold-58, within a
**pre-registered direction constraint** fixed before looking at results: image-heavy `[0.5, 1.0]`
for the 10 labels where the image model (exp002) already beat the report engine alone, report-heavy
`[0.0, 0.5]` for ACL/MCL specifically. **Macro LOO-AUC: 0.904** — clearly above both the report
engine alone (0.823) and the image model alone (0.868 gold-58 macro) — real evidence the two
sources are complementary, not just the image model "catching up" to the report engine.
`labels_v3_candidate` became the label generation every image experiment from exp005 onward
trained on, confirmed via an independent (non-circular) acceptance check: OOF macro **against the
original, untouched `labels_v2_final`** improved (+0.047 on ACL alone, the label the report engine
was weakest on).

A second noisy-student round (`labels_v4_candidate`, blending v3 + a newer image OOF) was built
and evaluated but **regressed** — macro LOO-AUC 0.904 → 0.892 — and iteration stopped there. See §10.

## 5. Image model architecture

**Stage A (per-series, 2.5D):** each series is resampled to fixed physical spacing (0.3mm/px,
384×384, uint8 cache), T slices sampled uniformly (T=24 for the ensemble members, T=16 for the
efficiency student), encoded independently through a DINOv2 ViT backbone (`in_chans` replicated
×3, `dynamic_img_size=True` so the pretrained position embedding interpolates to our non-native
input size), then aggregated over the T slice tokens with a small transformer (learned CLS token +
slice-index positional encoding) into one series embedding.

**Stage B (study-level):** up to 6 canonical slots (`{Sagittal, Coronal, Axial} × {Fluid-Sensitive,
non-Fluid-Sensitive}`, presence-masked) get learned slot-identity embeddings, pass through a
cross-slot transformer, then 12 learned per-target query vectors attention-pool over the slot
context — each target learns which slots it actually needs (menisci → sagittal/coronal, Baker's →
sagittal/axial fluid-sensitive, etc.) rather than a hand-coded mapping.

**Laterality normalization:** medial/lateral targets are mirror-symmetric between knees, so every
right knee is flipped to a canonical left-knee convention before encoding (DICOM orientation tags
first, majority vote across a study's series; studies with no resolvable side run through **both**
orientations at inference time and average, with the paired medial/lateral labels swapped on the
mirrored pass). Training augmentation includes a deliberate laterality-swap probability (0.3) that
performs the same flip + label swap as a regularizer, on top of standard rotation/scale/
translation/intensity-jitter/coarse-dropout — never a vertical flip, which has no anatomical
justification here.

**A parallel 3D-architecture diversity attempt (exp007, ResNet-50 over 3 canonical resampled
volumes) was killed at 10/25 epochs (40% trained)**, per a pre-registered rule (correlation > 0.92
→ kill; correlation < 0.85 AND macro ≥ 0.80 → continue to epoch 20). Its OOF correlation with the
2D ensemble (0.444) was genuinely low — real architectural diversity, not noise — but its solo OOF
macro (0.693) never reached the 0.80 bar needed to justify the ~10.5h/fold cost of finding out
whether that correlation would hold up with more training. One detail worth remembering if a
3D/2.5D split is ever revisited: **Effusion (0.85) and Synovitis (0.80) correlated far higher with
the 2D ensemble than every other label (0.21–0.49)** — both are diffuse, whole-joint findings
without the mirror-symmetric structure most other targets have, so a volumetric view may genuinely
converge toward similar features for exactly these two, while diverging more on the
anatomically-lateralized labels.

A second diversity attempt, swapping the Stage-A backbone to **ConvNeXt-V2-Base** (a
fully-convolutional architecture, exp009) while keeping everything else identical to the best ViT-B
member, cleared the accuracy bar (OOF 0.8652 vs. a ≥0.855 threshold) but **failed the correlation
bar** (0.9296 vs. a <0.90 threshold) — too similar to the existing DINOv2 members to add real
ensemble value; not incorporated. See §10.

## 6. Training stability

3 of exp001's first 5 folds trained to near-constant OOF predictions (std 0.005–0.05 per label) vs.
the 2 healthy folds' real per-study variation (std 0.07–0.28) — genuine optimization collapse, not
a data issue (confirmed via an `--eval-raw-too`/OOF-std check, and confidence-weighting ruled out
since `labels_v1` carried no per-sample weights yet). `--debug-grad-norms` instrumentation showed
the backbone's gradient norm wildly spiky (1–25+) while the from-scratch slice-attention/study-head
sat 10–50x smaller and far more stable. **Root cause: a single global `clip_grad_norm_` over the
whole model computes one scale factor from the combined norm, so a backbone spike forces the same
aggressive down-scaling onto the tiny, freshly-initialized head gradients too** — intermittently
starving their learning signal. The fix with the clearest causal evidence: **clip the backbone and
head/slice-agg parameter groups separately** (each to `max_norm=1.0`), applied alongside a broader
stability bundle (pre-norm transformer blocks, small-gain init on from-scratch attention
projections, longer warmup, lower LR, a confidence-weight floor) that's standard practice for a
from-scratch head downstream of a pretrained backbone but wasn't individually ablated — the
per-group clipping fix is the one with direct diagnostic evidence behind it. Acceptance test: 2
seeds × 5 folds = 10 short runs, **10/10 escaped the plateau** with smooth monotonic loss curves.

## 7. Scale ladder and ensemble

**DINOv2 ViT-S/14 → ViT-B/14** was tested twice, on two different label generations, with the same
result both times: exp003 (ViT-B, `labels_v2_final`) beat exp002 (ViT-S) by +0.0055 OOF macro;
exp006 (ViT-B, `labels_v3_candidate`) beat exp005 (ViT-S, same labels) by +0.0050 — both clear the
project's +0.004 "keep the bigger model" bar. **DINOv3 ViT-S/16 was tested once and dropped**:
−0.0252 OOF macro vs. the DINOv2 baseline, no label meaningfully favoring it, Lateral Meniscus and
Baker's the biggest losses (−0.11, −0.05).

**Ensemble assembly**: greedy forward selection on OOF-vs-`labels_v2_final` macro (gold-58 used
only as a tripwire, never for selection), candidate pool `{exp002, exp003, exp005, exp006}`.
Solo macros: exp002 0.8481, exp003 0.8536, exp005 0.8631, exp006 0.8681. **Global-recipe result:
`{exp006, exp003, exp005}`** (exp002 dropped — fully superseded by exp006, its v3-label successor),
**macro 0.8744**, clearly above the best solo model. A per-label recipe variant scored only
+0.0010 over the global recipe — below the pre-set 0.002 bar for the added complexity, so the
simpler global recipe shipped. Gold-58 tripwire on the shipped recipe: 0.8960, consistent with the
OOF story (no sign of selection overfitting to `labels_v2_final` in some gold-58-divergent way).

**Blend method** is prob-mean or rank-mean per label, whichever wins on OOF — but "rank" is a
population statistic with no population to rank against for a single new test-time study. Fixed
with an empirical-CDF lookup: each member's OOF score distribution for a label is stored
(`rank_reference.npz`), and a new prediction's "rank" becomes its percentile within that stored
reference — mathematically the same operation as `pd.Series.rank(pct=True)`, just against a fixed
reference instead of the live batch.

**TTA (laterality-swap dual-pass) was measured and rejected**: real inference passes for all 3
members × 5 folds, blending each member's normal + flipped pass before the ensemble blend, gave
OOF macro 0.8744 → 0.8730 (−0.0014), 8/12 labels regressing. Plausible explanation: the models
already handle laterality directly (flip-to-canonical + swap-augmentation at training time), so a
second mirrored inference pass mostly adds noise from real, non-symmetric imaging artifacts that a
naive horizontal flip doesn't actually cancel.

## 8. Efficiency-track model

**exp008**: DINOv2 ViT-S/14 (same backbone family as the smallest ensemble member), T=16 (down
from 24), 320px (down from 384), distilled from the accepted 3-member ensemble's soft predictions
over all 4,407 studies (`ensemble_teacher_v1.parquet` — OOF for non-gold studies, fold-ensembled
for gold-58, same non-circularity discipline as the label-blend scripts) via a mixed **KL + BCE**
loss: BCE against `labels_v3_candidate`, KL against the teacher, binary-per-label (not a softmax
`nn.KLDivLoss`, since each of the 12 targets is an independent sigmoid output, not one categorical
distribution). `distill_weight: 0.5`, not deeply tuned. **Accepted result** (all 5 folds, no
instability): OOF macro vs. the original `labels_v2_final` **0.8708**, only **−0.0036** off the
full ensemble's 0.8744 — comfortably inside the pre-registered 0.015 target for a model at
roughly 1/9th the per-study compute (1 backbone × T=16/320px vs. 3 backbones × T=24/384px, two of
them ViT-B).

**Fold-count-vs-runtime**: gold-58 is the only set where every fold's model predicts the same
studies, so it's the basis for a fold-count sensitivity check. Averaged over *all* possible
same-size subsets (not just the best pick, to avoid overselling n=58): dropping from 5 folds to 4
costs −0.0004, to 3 costs −0.0014, to 2 costs −0.0030, to 1 costs −0.0060 — **small and gradual,
not a cliff**. The submission notebook therefore picks fold count dynamically by minimizing the
actual efficiency-track formula (`Eff = AUC/(0.5−maxAUC) + RuntimeSeconds/32400`) rather than
hardcoding 5, using in-notebook-measured decode/model timing. In practice, because DICOM decode
dominates this model's per-study cost so heavily (~82% of wall-clock, measured directly — see §9),
the runtime term barely moves between fold counts and the formula ends up choosing 4 or 5 (both
essentially equivalent under it) regardless of hardware.

**Measured runtime, full 1,300-study synthetic load test, decode/inference overlap enabled**:
**447 seconds (7.5 minutes)** on H100, comfortably inside the CLAUDE.md §8 target of ≤45 minutes.
Decode-vs-model profiling motivated a decode fast-path (header-only DICOM ordering + pixel-decode
of only the T slices actually used, instead of decoding a full ~30-slice series and discarding
half of it) — validated against the accurate path (100-study prediction diff: mean 0.0004, max
0.006; full-1300 macro-AUC delta: −0.00004, both far smaller than the fold-count deltas above)
before shipping as the default. T4-hardware projection and the dual-GPU-not-worth-it-here analysis:
`docs/SUBMISSION_RUNBOOK.md` §2, §4.

## 9. Infrastructure lessons

**The quota-guard story.** This project runs on a shared 400GB-per-user BeeGFS quota, and hit it
repeatedly. The first real incident (exp003) lost ~35 GPU-hours: all 5 folds trained cleanly to
epoch 23/25, then the wall-clock guard's "last chance" checkpoint save used the *same* 15GB
routine-save floor as the routine per-epoch save — at ~11.5GB headroom, both were skipped, and the
run exited with zero checkpoint and zero eval. **Root cause: one quota floor applied uniformly to
"skip and try again next epoch" (cheap) and "this is the only chance, ever" (catastrophic) are not
the same risk and shouldn't share a threshold.** Fixed with a lower, purpose-sized emergency floor
(3GB, sized to what a write actually needs) for the last-chance save specifically. A second
incident (exp008's first run) showed the guard could fire on *every single write for an entire
~2h run* without surfacing any differently from an occasional, harmless skip — headroom was
already below the floor *before the job even launched*, just never checked as a go/no-go gate.
Fixed with a loud pre-flight check at the top of training that fails fast in the first lines of the
log instead of silently discovering it hours later. A third incident recurred anyway (exp009,
2026-09-14) even with both fixes in place, because the *freed* headroom after a cleanup pass
(13.9GB) was itself still under the 15GB routine floor — a reminder that "some cleanup" and
"cleared the actual threshold" are different claims worth checking explicitly, not inferring from
"we deleted some large files."

**The Kaggle inference debugging arc.** The first real Kaggle-notebook run needed 7 iterations
before producing a genuine (non-prior-fallback) prediction: dataset-not-ready-yet races, Kaggle's
current nested `/kaggle/input/{datasets,competitions}/<owner>/<slug>/` mount layout (not the old
flat path), a code-dataset zip upload that didn't preserve the package's wrapper folder on
extraction (fixed by locating `__init__.py` wherever it lands and registering that directory as
the module, rather than assuming a fixed on-disk layout), a stale code dataset, and — the one
real, non-packaging bug — **Kaggle silently handing out a Tesla P100 (compute capability sm_60)**
against a PyTorch build whose minimum supported architecture was sm_70: `torch.cuda.is_available()`
returned `True`, every forward pass threw a kernel-image error, and the per-study defensive
fallback quietly produced an all-label-prior submission that would have measured the prior
baseline's AUC, not the model's. Fixed with an explicit `torch.cuda.get_device_capability` check
against the installed build's minimum supported architecture, falling back to CPU (slow, but
caught by the wall-clock guard) rather than trusting `is_available()` alone. The same
zip-extraction-doesn't-preserve-nesting finding resurfaced later while staging weight datasets for
Kaggle (`docs/EXPERIMENTS.md` 2026-09-11) — since every fold's checkpoint shares the filename
`best.pt`, a flattening extraction could silently collide 5 files into 1; weight datasets are now
staged flat (`fold0_best.pt`…`fold4_best.pt`, no subdirectories) specifically to avoid re-triggering
that failure mode with data loss instead of a broken import.

## 10. What didn't work / was tried and dropped (null and negative results)

Negative results are part of the method: the recipe in §5–§8 is what remained after the levers below were
tested against the same folds, the same proxy metric and — for the post-probe ablations — a paired
cluster-bootstrap noise estimate (macro SD ≈ 0.0041; 2 SD = 0.0082). The complete ledger, with hypothesis,
numbers and verdict for every experiment, is the "Consolidated ablation ledger" in `docs/EXPERIMENTS.md`.
The short version: only backbone size (+0.005), noisy-student labels (+0.014) and ensembling (+0.006 over the
best solo) moved OOF beyond noise; every lever tried after that recipe landed is a null (Δ within ±0.004).

- **`labels_v4_candidate`** (round-2 noisy student, blending v3 + a newer image OOF): macro LOO-AUC
  regressed 0.904 → 0.892 — label-engine iteration stopped at v3.
- **exp004** (DINOv3 ViT-S/16): −0.0252 OOF macro vs. the DINOv2 baseline, no label favored it.
- **exp007** (3D ResNet-50): killed at 40% trained — genuine architectural diversity (OOF
  correlation 0.444 with the 2D ensemble) but never reached a usable accuracy bar (solo OOF 0.693).
- **exp009** (ConvNeXt-V2-Base): cleared the OOF bar (0.8652) but failed the diversity bar
  (correlation 0.9296 with exp006, vs. a <0.90 threshold) — accurate, but redundant.
- **Laterality-swap TTA**: measured OOF-negative (−0.0014) on the final ensemble.
- **Per-label ensemble recipe** (vs. the simpler global recipe): only +0.0010 OOF, below the 0.002
  bar for the added complexity.
- **Resolution and slice count** (exp015 448px on a genuinely rebuilt 0.2571mm/px cache; exp010
  T=32): both **null** vs. the same-recipe exp006 baseline (solo OOF macro −0.0001 each; paired
  cluster-bootstrap 95% CI ≈ [−0.0023, +0.0025], so effects larger than ~+0.0025 macro are ruled out).
  Small-finding labels (meniscus, cartilage) did not turn out to be resolution-limited at 384px/T=24.
- **Anatomy-aware ACL init** (exp012, sagittal-biased ACL query): ACL +0.0005 (nothing), MCL −0.0122;
  macro +0.0010, CI spans zero. ACL's low OOF was proxy-label noise, not a vision deficit (gold-58 ACL
  AUC ~0.988 for both report labels and image ensemble), so there was little for a head change to fix.
- **A 6-member re-selected ensemble** {exp012, 003, 006, 015, 010, 005}: +0.0020 OOF over the deployed
  3-member recipe (CI [+0.0011, +0.0027]) — statistically nonzero but in-sample-selected and tiny
  against a ~2.9x inference cost, so not adopted on that evidence alone.
- **Auxiliary report-text objective** (exp016a/b — the one lever that uses the ~977 characters of report text
  per study that a 12-bit label throws away): the image model additionally regresses a frozen multilingual
  text embedding of its own study's report (BGE-M3, all-but-the-top D=5 to strip the language shortcut; targets
  fit per fold on training studies only; a dataset assertion refuses any validation-fold or gold-58 study).
  Pre-registered bar: point macro gain > +0.0082 (2 SD) over exp006. Result: weight 0.1 → +0.0036 (paired 95% CI
  [+0.0012, +0.0061]), weight 0.5 → +0.0012 ([−0.0016, +0.0037]); gold-58 +0.0002 / −0.0015. **Null against the
  bar**: the small OOF-proxy gain for weight 0.1 is not confirmed on the image-reviewed gold-58 set and could be
  the model absorbing report idiosyncrasies that help against report-derived labels; heavier weighting is not
  better. Only Synovitis moved in both arms (≈ +0.011), and it moved the wrong way on gold-58. The contrastive
  (CLIP-style) variant was not run because its cheaper auxiliary counterpart showed no bar-clearing signal.
- **ConvNeXt-V2-B** (exp009): 0.8652 OOF, but correlation 0.9296 with the ViT-B model (diversity bar < 0.90) —
  accurate yet redundant, so it adds nothing an ensemble could use (listed above; repeated here because it
  answers "does a different backbone family help?": no).
- **Severity-aware ordinal targets** (exp017; cumulative P(≥mild/moderate/severe) heads): **dropped on
  reasoning, not tested.** The hypothesis was that report over-calling of mild/trace findings is the remaining
  threshold mismatch with the expert panel. But our rubric already maps mild/grade-1/trace findings to
  p = 0.20–0.35 versus > 0.5 for definite ones, so the soft labels already carry an implicit ordinal signal; the
  extractions store no severity grade (a zero-GPU audit found a grade word in only 8–44% of positives'
  free-text phrases), so a real ordinal target would need a new two-model LLM pass, five folds and a
  leaderboard probe, to be judged on a gold-58 set too coarse to resolve the difference. We report it as an
  untested idea, not a rejected one.
- **A void pilot** (exp013/014, a 400-study 384px-vs-448px comparison) crashed on a harness bug (the subset
  folds omitted the gold-58 studies the trainer evaluates) and was underpowered anyway; its numbers are not
  cited. The resolution question was answered by the full experiment above (exp015).
- **External data** (MRNet, KneeMRI): not used — licence terms did not clearly permit prize use plus
  redistribution of derived weights.

## 11. Final leaderboard result

**Both probes submitted and scored (2026-09-16/17).** Public leaderboard top at time of writing:
~0.95–0.96 macro-AUC.

| | OOF macro-AUC (reference) | Public LB | Gap (OOF − LB) | Verdict |
|---|---|---|---|---|
| Probe #1 — ensemble (3 members) | 0.8744 | **0.8930** | −0.0186 | TRUSTWORTHY (< 0.02) |
| Probe #2 — efficiency (exp008) | 0.8708 | **0.8850** | −0.0142 | TRUSTWORTHY (< 0.02) |

Both gaps are negative (LB beats OOF) and land on the same sign with comparable, sub-threshold
magnitude — read as a mild positive signal on validation-methodology soundness rather than a
coincidence on one run. Interpretation (see `docs/EXPERIMENTS.md`'s "OOF↔LB gap analysis — probe
#1/#2 RESULT" entries for the full checklist): a negative gap isn't automatically good news, but
site/template leakage would push the gap the *opposite* direction, and the most parsimonious
reading is that the true hidden-test labels (image-reviewed) are a cleaner target than our
report-derived OOF proxy (~18% systematic disagreement, §3) — a model that learned real image
pathology can plausibly score higher against the clean labels than our own noisier internal
validation predicts.

Efficiency-track delta: LB 0.885 vs. ensemble LB 0.893 = **−0.008**, comfortably inside the
"within ~2 pts of the big ensemble" target (CLAUDE.md §4) — matches the OOF-predicted delta
(−0.0036) closely enough to trust the distillation held up on real hidden data.

**Gap to the leaderboard top (~6–7 points) is real and being treated as structural, not noise** —
see `docs/EXPERIMENTS.md`'s 2026-09-17 diagnostic entry for the per-label breakdown against an
external published benchmark (Qiu et al. 2024, Nat Commun). That comparison initially flagged ACL
as far behind published external AUCs, but the follow-up (2026-09-18) showed most of that "gap" is
proxy-label noise, not a vision deficit: on the clean gold-58 labels both the report labels and the
image ensemble reach ~0.988 ACL AUC, while OOF is measured against noisy report-derived soft
labels. So OOF systematically *understates* true performance — consistent with both probes beating
their OOF. Remaining effort went to a resolution/slice-count ablation (exp010 T=32, exp015 448px), an
ACL-head initialisation test (exp012), a final ensemble re-selection and an auxiliary report-text objective
(exp016); all were null (§10). MRNet and KneeMRI were both dropped on licensing grounds (no external data).
The modeling phase was closed on 2026-09-21 with the recipe frozen as the 3-member ensemble
{exp006, exp003, exp005} plus the exp008 distilled efficiency model. Final private-leaderboard rank:
**pending competition close.**

## 12. Reproducibility

The repository (`CLAUDE.md` as the single source of truth for project conventions,
`docs/EXECUTION_PLAN.md` for the phased plan this work followed) is structured one-experiment-per-
YAML-config under `configs/`, with every training/labeling run logged chronologically in
`docs/EXPERIMENTS.md` — that file's "Decisions log" section is the authoritative, unedited history
this writeup summarizes; if any number here and `EXPERIMENTS.md` ever disagree, trust
`EXPERIMENTS.md`. To rebuild a specific result: find its config in `configs/`, its data lineage
(which `labels_v*` / `folds.parquet` it used) in the matching `EXPERIMENTS.md` entry, and run
`scripts/train.py --config <path> --fold <n>`. Model cards for the 4 deployed models
(`exp003`, `exp005`, `exp006`, `exp008`) are in `docs/model_cards/`. Code is Apache-2.0 licensed
throughout (see license headers).
