# Submission Runbook — RSNA Knee Abnormality Detection

Step-by-step guide for actually running the two submission notebooks on Kaggle. Written
2026-09-15, after the T4 x2 hardware adaptation (see `docs/EXPERIMENTS.md` 2026-09-11) — numbers
below assume that adaptation is in place. If either notebook has changed since, re-verify against
the notebook source before trusting this document; it is not auto-generated.

Two notebooks, two probes, per the 2026-09-10 plan change (the stale exp002/v7 probe is
permanently skipped):
- **`notebooks/submit-ensemble.ipynb`** — probe #1, max-AUC 3-member ensemble.
- **`notebooks/submit-efficiency.ipynb`** — probe #2, exp008 distilled single-model efficiency entry.

---

## 1. Datasets — slugs and attachment order

All five datasets are staged and ready on HPC at `/shared/home/husnain.rasool/rsna-knee/kaggle_staging/`
(see `docs/EXPERIMENTS.md` 2026-09-11 "Kaggle dataset staging finalized" for how they got there —
`best.pt` stripped to `ema`-only, weights staged flat as `fold0_best.pt`…`fold4_best.pt`, not
`foldN/best.pt` subdirectories). None exist on Kaggle yet as of this writing — first push needs
`kaggle datasets create`, not `version`.

| Slug | Contents | Used by |
|---|---|---|
| `rsna-knee-code` | `src/rsna_knee/*.py` + `ensemble_recipe_v1.yaml` | both notebooks |
| `rsna-knee-exp006-weights` | exp006 (DINOv2 ViT-B/14, `labels_v3_candidate`), 5 folds + `rank_reference.npz` | ensemble |
| `rsna-knee-exp003-weights` | exp003 (DINOv2 ViT-B/14, `labels_v2_final`), 5 folds + `rank_reference.npz` | ensemble |
| `rsna-knee-exp005-weights` | exp005 (DINOv2 ViT-S/14, `labels_v3_candidate`), 5 folds + `rank_reference.npz` | ensemble |
| `rsna-knee-exp008-weights` | exp008 (DINOv2 ViT-S/14, distilled), 5 folds, no `rank_reference.npz` (not an ensemble member) | efficiency |
| `rsna-knee-abnormality-detection` | the competition data itself | both notebooks |

**Attach, in this order, before running either notebook:**
- Ensemble: `rsna-knee-code`, `rsna-knee-exp006-weights`, `rsna-knee-exp003-weights`,
  `rsna-knee-exp005-weights`, `rsna-knee-abnormality-detection`.
- Efficiency: `rsna-knee-code`, `rsna-knee-exp008-weights`, `rsna-knee-abnormality-detection`.

Attachment order doesn't matter to the code itself (`_find_dir` searches `/kaggle/input` by name
substring, not position) — listed in a fixed order here only so a human doing this manually has a
checklist. Cell-1 of each notebook prints the resolved path for every attached dataset and
**asserts** if one is missing — a notebook that fails at cell-1 means an attachment problem, not a
code bug; re-check the attached-datasets panel before debugging further.

**Push commands** (run from wherever `kaggle_staging/` lives, e.g. on HPC):
```bash
kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/code -r zip
kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/rsna-knee-exp003-weights
kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/rsna-knee-exp005-weights
kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/rsna-knee-exp006-weights
kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/rsna-knee-exp008-weights
```
`-r zip` is required for the code dataset (it has one subdirectory, `rsna_knee/`, that the
default `--dir-mode skip` would silently drop). The four weight datasets are flat — no `-r` needed.
**Re-staging after any code change**: refresh `kaggle_staging/code/rsna_knee/*.py` from
`src/rsna_knee/*.py` (a plain `cp`, no restructuring needed) and re-run `kaggle datasets version -p
<dir> -m "<message>"` for the code dataset only — the weight datasets don't change unless a model
is retrained.

---

## 2. GPU hardware pinning

The pinned accelerator is **T4 x2**. Pin it differently per notebook:

| Notebook | Pin | Why |
|---|---|---|
| Ensemble | **T4 x2** | Members are sharded across `cuda:0`/`cuda:1` (greedy-balanced by relative cost — `{exp006, exp005}` on one device, `exp003` alone on the other, a ~4:3 split). Profiled H100 decode:model split is ~68:32 — model inference is a big enough share of per-study cost that a 2nd GPU meaningfully helps (T4 projection: ~102 min on 1 GPU vs. ~62 min on 2 — see §4). |
| Efficiency | **T4 x1** (single GPU) is sufficient | Profiled H100 decode:model split is ~82:18 — decode dominates so heavily that splitting the small model slice across a 2nd GPU saves only ~1-2 minutes out of a ~12-19 min total, not worth the complexity. If Kaggle only offers T4 x2 as a package deal for both notebooks, leaving the 2nd GPU idle here is harmless — the code detects `torch.cuda.device_count()` and simply doesn't use a device it has no work to schedule on. |

Both notebooks' cell-2 GPU-capability check (`torch.cuda.get_device_capability` vs.
`torch.cuda.get_arch_list()`'s minimum) falls back to CPU if the assigned GPU is too old for the
installed torch build — this exists because of a real 2026-09-08 incident where Kaggle handed out
a Tesla P100 (sm_60) instead of the expected accelerator, and `torch.cuda.is_available()` alone
returned `True` while every forward pass silently crashed and fell back to label priors for the
whole run. **If either notebook prints `device: cpu`, stop and re-check the accelerator setting**
— it means this fallback fired, and a CPU run will very likely blow the wall-clock budget or at
minimum score far below what OOF predicts.

---

## 3. Golden path

1. Open the notebook, attach datasets per §1, pin hardware per §2.
2. Run all cells top to bottom. Cell-1 asserts on missing datasets; cell-2 prints device/GPU-count/
   decode-worker/member-device info — **read this line before continuing**, it's the fastest way
   to catch a misconfiguration before burning the whole run.
3. The calibration cell (cell-6, both notebooks) times a small sample (10 studies) and either:
   - **Ensemble**: projects total runtime for each degradation-ladder rung, picks the highest rung
     that fits the 8h budget with 15% margin.
   - **Efficiency**: computes the CLAUDE.md §4 efficiency-track score for each candidate fold count
     (1-5) and picks whichever minimizes it (not "fits a budget" — every fold count fits easily).
   Compare the printed numbers against §4's healthy ranges before letting the full run proceed.
4. The main loop decodes and infers all ~1300 studies, printing progress roughly every 100 studies
   (efficiency) or every 100 studies (ensemble), with running elapsed-time/budget-fraction.
5. The final cell asserts header/row-count/no-NaN/[0,1]-bounds correctness and writes
   `/kaggle/working/submission.csv`.
6. Before clicking Submit: check the final printed line (`n_prior_fallback`, `n_dual_flip`, total
   elapsed) against §5's healthy ranges.

---

## 4. Expected calibration metrics — healthy vs. unhealthy

Calibration numbers below are per-study, measured over the 10-study calibration sample (cell-6).
H100 baselines are real, logged measurements (`docs/EXPERIMENTS.md` 2026-09-10/11); T4 figures are
projected (3.5x model-time multiplier, decode unchanged — decode is CPU-bound and already shown
CPU-count/GPU-choice-insensitive). Treat the T4 numbers as a plausible range, not a guarantee — the
pending 3-sample Kaggle test is the actual ground truth once run.

### Ensemble notebook

| Metric | Healthy range (T4 x2) | Unhealthy — investigate | Likely cause |
|---|---|---|---|
| `device` | `cuda` | `cpu` | GPU-capability fallback fired (old/wrong GPU handed out) — see §2 |
| `n_gpus` | `2` | `1` or `0` | Accelerator not actually pinned to T4 x2; member sharding silently degrades to single-device (still correct, just slower — not a crash, but check before assuming §4's dual-GPU numbers apply) |
| `member device assignment` | two distinct `cuda:0`/`cuda:1` values across the 3 members | all members on the same device | Same as above — `n_gpus<=1` fallback path taken |
| decode s/study | ~2.5 – 4.5s | > 8s | CPU decode workers under-provisioned, or an unrelated Kaggle infra slowdown — decode should NOT depend on which GPU is attached |
| model s/study (3 members x 5 folds combined) | ~2.5 – 5.5s (T4 x2) | > 10s | bf16 fell back to an unsupported/emulated path (should be fp16 — check `AUTOCAST_DTYPE`), or `n_gpus` reported 2 but sharding isn't actually happening |
| projected total (rung 0) | ~55 – 110 min | > 200 min, or the printed rung is NOT 0 | If rung dropped below 0, the projection thinks it won't fit even the generous 8h budget with margin — something is running far slower than expected; do not just accept a lower rung silently, find out why first |

### Efficiency notebook

| Metric | Healthy range (T4 x1) | Unhealthy — investigate | Likely cause |
|---|---|---|---|
| `device` | `cuda` | `cpu` | Same GPU-capability fallback as above |
| `fast_decode` | `True` | `False` | `FAST_DECODE` constant was changed — the validated default is on |
| decode s/study | ~0.4 – 0.9s | > 2s | Fast-decode path not actually active, or CPU under-provisioned |
| model s/study (5-fold) | ~0.5 – 0.8s (T4, includes the 3.5x multiplier) | > 2s | Same bf16/fp16 concern as ensemble, though efficiency has always passed fp16 explicitly so this is less likely here |
| chosen fold count | 4 or 5 (both score within ~0.0005 of each other under the formula — either is fine) | 1 or 2 | Only happens if decode is running MUCH slower relative to model than profiled — worth understanding why, even though the formula's choice is still technically correct for whatever it measured |
| projected total | ~8 – 20 min | > 45 min (the CLAUDE.md §8 target) | Investigate before accepting — this notebook has consistently run in the 7-19 min range in every rehearsal so far |

**Fastest sanity check**: both notebooks' first calibration print line includes the GPU name
(`torch.cuda.get_device_name(0)`). If it doesn't say some variant of "T4", the accelerator setting
didn't take — stop and re-check before the full run burns time on the wrong hardware.

---

## 5. Downgrade / fallback procedures

Two independent layers, from least to most drastic:

**Layer 1 — calibration-based selection** (picks ONE setting for the whole run, applied before any
real inference happens, so scoring stays consistent study-to-study):
- Ensemble: degradation ladder — rung 0 (3 members, no TTA) → rung 1 (2 members: exp006+exp005,
  drops exp003 as the most redundant member) → rung 2 (exp005 only, the smallest/fastest model).
  Picks the highest rung whose projected total fits the 8h budget with 15% margin.
- Efficiency: fold count (1-5) chosen to minimize the CLAUDE.md §4 efficiency-track formula, not a
  budget-fit check — every fold count fits the 8h budget trivially, so this is a pure
  accuracy-vs-runtime-score optimization, not a safety fallback.

**Layer 2 — wall-clock guard, per-study** (`infer_core.WallClockGuard`, active for the entire main
loop as defense-in-depth in case Layer 1's projection was wrong): if elapsed time crosses 95% of
the 8h budget, all remaining studies get the label-prior fallback (`infer_core.prior_row()` —
mean probability per label from the full `labels_v2_final.parquet`) instead of running inference,
guaranteeing the submission still has exactly one valid row per test study rather than timing out
with none. A study whose decode fails outright (corrupt series, unexpected schema) also falls back
to the label-prior row individually, regardless of budget status — this is CLAUDE.md §9's "a crash
= no score" defense, and is expected to fire occasionally (missing planes, decode errors) even on
a completely healthy run.

**Reading the final line**: `Done: N studies, ... n_prior_fallback, n_dual_flip, ... total.`
- `n_prior_fallback` should be **0, or very close to 0** (a handful from genuine per-study decode
  failures is normal; anything in the hundreds means Layer 2 fired, i.e. the run was actually
  running out of budget — go back to §4 and figure out why the calibration projection was wrong).
- `n_dual_flip` (studies with no resolvable DICOM side tag, run through both orientations and
  averaged) should land around **50%** of the test set — every full-1300 rehearsal so far has come
  in at 659-663/1300 (~50.7-51.0%). A number far from that suggests something changed in how side
  tags are being read, not a runtime problem.

**If a run genuinely needs to be aborted and re-tried** (e.g., wrong accelerator pinned): there is
no partial-resume — Kaggle notebook runs are stateless per execution. Fix the setting and re-run
from the top; nothing on disk carries over between attempts.

---

## 6. Post-run checklist before clicking Submit

- [ ] `device: cuda` printed, correct GPU name (T4)
- [ ] Ensemble only: `n_gpus: 2`, member device assignment shows two distinct devices
- [ ] Calibration numbers within §4's healthy ranges
- [ ] Chosen rung/fold-count matches §4's expectation (rung 0 / 4-or-5 folds)
- [ ] Final line: `n_prior_fallback` near 0, `n_dual_flip` near 50%
- [ ] Total elapsed time within §4's healthy range, well under the 8h cap
- [ ] Output cell's `sub_df.head()` looks sane (values in [0,1], no obviously-degenerate rows)
- [ ] `submission.csv` header matches `sample_submission.csv` exactly (asserted automatically, but
      worth a human glance — this is the one thing that silently invalidates an otherwise-perfect
      run if it's ever wrong)

Only once all of the above check out: click Submit.
