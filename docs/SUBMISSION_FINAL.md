# Final Submission State: RSNA Knee Abnormality Detection

Frozen 2026-09-23; selections updated 2026-09-26 (main = exp021, efficiency = exp018 2-fold). Verified with the Kaggle CLI on the same day (`kaggle competitions submissions`,
`kaggle datasets files`/`download`, `kaggle kernels pull`). History and reasoning are in
`docs/EXPERIMENTS.md`; this page covers what is selected and what to do if something has to change.

## 1. Selections

| Role | Submission ref | Public LB | Notebook (Kaggle) | Datasets | Runtime, 2 × T4 (measured) |
|---|---|---|---|---|---|
| **Main leaderboard** | **56556092** (2026-09-25 17:07 UTC): exp021, 10-checkpoint seed ensemble | **0.929** | `hrdevelops/submit-exp021` v1 | `rsna-knee-exp018-code`, `rsna-knee-exp018-weights`, `rsna-knee-exp021-weights`, competition data | ~100 min (10 × ~9.7 min/fold) |
| **Efficiency track** | **56560771** (2026-09-25 21:48 UTC): exp018, 2 folds {0, 1} | **0.925** | `hrdevelops/submit-exp018-2fold` v1 | `rsna-knee-exp018-code`, `rsna-knee-exp018-weights`, competition data | **~20.5 min** |
| Fallback: exp018 5-fold | 56460279 (2026-09-22 09:22 UTC) | 0.929 | `hrdevelops/submit-exp018` | `rsna-knee-exp018-code`, `rsna-knee-exp018-weights` | ~53 min |
| Fallback: ensemble | 56273364 (`ensamble_notebook` v3) | 0.893 | `hrdevelops/ensamble-notebook` | `rsna-knee-code`, `rsna-knee-exp006/003/005-weights` | — |
| Fallback: efficiency | 56297755 (`rsna-knee-submit-efficiency` v2) | 0.885 | `hrdevelops/rsna-knee-submit-efficiency` | `rsna-knee-code`, `rsna-knee-exp008-weights` | — |

All are `COMPLETE` with a score. The CLI cannot show or set final selections, so **make the selections in the
Kaggle UI** (My Submissions → select).

**How the two picks were made (both rules fixed BEFORE the probe they decided):**
- **Main = exp021.** It displayed 0.929, a tie with exp018 at 3-decimal precision (exp018's true score is
  anywhere in [0.9285, 0.9295)). Rule: a displayed tie or better selects the 10-checkpoint ensemble, because
  its OOF gain is real (+0.0024, 95% CI [+0.0014, +0.0033]) and the private leaderboard is what counts.
- **Efficiency = exp018 2-fold.** Rule: select it if its public LB displays >= 0.903. It scored **0.925**. On
  Kaggle's own 2 × T4 the 5-fold exp018 run is model-bound at **~53 min** (not the 15.3 min once projected);
  2 folds run in **~20.5 min** (measured, `hrdevelops/rsna-knee-profile-2fold`). Efficiency score
  (maxAUC 0.95 assumed, public AUC): 2-fold **−2.018** vs 5-fold **−1.966**.

### 1a. exp021 10-checkpoint seed ensemble (main-track pick)

| | |
|---|---|
| Kaggle notebook | `hrdevelops/submit-exp021`, **version 1** (commit run COMPLETE: `3 predicted, 0 on prior, 0 errors`) |
| Code | the scored `rsna-knee-exp018-code` (`infer_wmil.py` 8,245 B), unchanged |
| Weights | scored `rsna-knee-exp018-weights` (seed 42, 5 files) + `rsna-knee-exp021-weights` (seed 43, 5 × 292,865,482 B) |
| Evidence | OOF 0.8840 vs 0.8816 (+0.0024, 95% CI [+0.0014, +0.0033]); gold-58 0.9111 vs 0.9091 (CI spans 0) |
| Runtime | ~100 min on 2 × T4 (the measured ~9.7 min per checkpoint × 10); far inside the 9 h cap |

### 1b. exp018 2-fold (efficiency pick)

| | |
|---|---|
| Kaggle notebook | `hrdevelops/submit-exp018-2fold`, **version 1** (commit run COMPLETE: `weight files: ['fold0_best.pt', 'fold1_best.pt']`, `3 predicted, 0 on prior, 0 errors`) |
| Built from | the scored `submit-exp018.ipynb` with ONE code cell changed (`scripts/build_exp018_2fold_notebook.py`) |
| Code / weights | the scored datasets, unchanged; decode threads unchanged (`cpu_count() − 1`; a 4th thread measured no faster) |
| Folds | {0, 1}, **pre-declared** (not chosen on a score); gold-58 0.9078 (5-fold 0.9091; mean of all 10 pairs 0.9067) |

Deploy only version 1 of each notebook (the commit-checked ones).

## 2. Why the scored exp018 artifact is internally consistent

- **Notebook:** the code cells of `hrdevelops/submit-exp018` (pulled 2026-09-23) are identical to
  `notebooks/submit-exp018.ipynb` at HEAD. There is no `os._exit` in it. Accelerator: `NvidiaTeslaT4`.
  Attached: `rsna-knee-exp018-code`, `rsna-knee-exp018-weights`, `rsna-knee-abnormality-detection`.
- **Code dataset `rsna-knee-exp018-code`:** created 2026-09-21 18:38 UTC, before the scored submission
  and before commit `2ec038d`. Checked against git:

  | file | size | matches |
  |---|---|---|
  | `infer_wmil.py` | 8,245 B | `e9944c6` (pre-`2ec038d`), untimed `w.join()` |
  | `wmil.py` | 16,609 B | the only version git has (`d407107`); CRLF line endings, md5-identical to the HPC source |
  | `preprocess.py`, `__init__.py` | | identical to HEAD |

- **Weights `rsna-knee-exp018-weights`:** `fold0_best.pt` … `fold4_best.pt`, 292,862,052 B each,
  created 2026-09-21 18:56 UTC. These are the only uploaded version, and they predate the submission.

## 3. If a resubmission is ever needed

**exp018 (the only thing that should ever be resubmitted):**
1. Open `hrdevelops/submit-exp018` in Kaggle. **Do not edit any cell.**
2. In the Input panel, confirm `rsna-knee-exp018-code` is the 2026-09-21 version (`infer_wmil.py`
   8,245 B) and `rsna-knee-exp018-weights` has 5 × 292,862,052 B files. If Kaggle offers "update to
   new version", **decline**.
3. Settings: accelerator **GPU T4 x2**, internet **off**.
4. Save Version → **Save & Run All (commit)**. Healthy log on the 3-row placeholder:
   `[exp018] wrote /kaggle/working/submission.csv: 3 rows, 3 predicted, 0 on prior, 0 errors`, then
   `Wrote ...: 3 rows, all checks passed.` The run status must be **Complete**, not Error. About 1–2 min.
5. Submit that version from the notebook's Output / "Submit to competition". Scoring runtime on ~1,300
   studies, MEASURED on Kaggle's 2 × T4 (2026-09-26): ~53 min for 5 folds, ~20.5 min for the 2-fold
   efficiency notebook, ~100 min for exp021's 10 checkpoints. All are far inside the 9h cap. (The ~17 min
   once quoted here was an H100 projection with a T4 factor that turned out ~3.6× too optimistic.)
6. If `on prior` is above 0 on the placeholder, or the status is Error: **do not submit**. Keep the
   existing 56460279 selected.

**Second slot, if a distinct pick is required:** resubmit exp018 unchanged using the steps above. If a
different model is required, use the ensemble (56273364, 0.893). It is already scored, so it can be
selected as-is. Do not re-run it.

**Fallback notebooks (ensemble / efficiency), only if a re-run is ever forced:** as of 2026-09-23,
`rsna-knee-code` has a newer version whose only change is `infer_core.py` (20,034 → 23,103 B: the
`prefetch_decode` exit-safety fix). That fix is proven on HPC but **has never run on Kaggle**. For a
re-run, keep the dataset version the scored notebook version pinned (the 2026-09-16 upload) unless there
is a specific reason to take the fix.

## 4. Known-bad versions: never deploy

| Artifact | Version | Why |
|---|---|---|
| `src/rsna_knee/infer_wmil.py`, `infer_blend018.py` | commit `2ec038d` up to (not including) `d1165b5` | `w.join(timeout=30)`: on T4, studies still queued after the timeout are silently written as label priors (Kaggle v4: 2/3 on prior, 0 errors), and GPU threads keep running into shutdown |
| `rsna-knee-blend018-code` | v3 (and any version) | carries the timed join above; the blend never completed a scored run |
| `hrdevelops/submit-blend018-ipynb` | v2, v3 | hung 9h / 10h at teardown after writing the submission; never scored |
| `hrdevelops/submit-blend018-ipynb` | v4 | ends with `os._exit(0)`: papermill `DeadKernelError`, so the run is `ERROR` |
| `scripts/hard_exit_tail.py` | any | same `os._exit(0)`: **never add it to a submission notebook** |
| `rsna-knee-exp018-code` rebuilt from HEAD | any | HEAD's `infer_wmil.py` (try/finally, `cancel_futures`, untimed join) is believed correct but has **never run on Kaggle**; the scored 8,245 B version is the only one to deploy |

Rule of thumb: **deploy only artifacts that have produced a scored submission.** If any new code must go
out, it must first complete a Kaggle commit with `0 on prior` on the placeholder, and finish with status
Complete.
