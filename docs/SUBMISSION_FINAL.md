# Final Submission State: RSNA Knee Abnormality Detection

Frozen 2026-09-23. Verified with the Kaggle CLI on the same day (`kaggle competitions submissions`,
`kaggle datasets files`/`download`, `kaggle kernels pull`). History and reasoning are in
`docs/EXPERIMENTS.md`; this page covers what is selected and what to do if something has to change.

## 1. Selections

| Role | Submission ref | Public LB | Notebook (Kaggle) | Datasets |
|---|---|---|---|---|
| **Main leaderboard** | **56460279** (2026-09-22 09:22 UTC) | **0.929** | `hrdevelops/submit-exp018` | `rsna-knee-exp018-code`, `rsna-knee-exp018-weights`, competition data |
| **Efficiency track** | same: **56460279** | 0.929 | same | same |
| Fallback: ensemble | 56273364 (`ensamble_notebook` v3) | 0.893 | `hrdevelops/ensamble-notebook` | `rsna-knee-code`, `rsna-knee-exp006/003/005-weights` |
| Fallback: efficiency | 56297755 (`rsna-knee-submit-efficiency` v2) | 0.885 | `hrdevelops/rsna-knee-submit-efficiency` | `rsna-knee-code`, `rsna-knee-exp008-weights` |

All four are `COMPLETE` with a score. The CLI cannot show or set final selections, so **make the
selections in the Kaggle UI** (My Submissions → select).

exp018 serves both tracks because it is the best-scoring model and also minimizes the efficiency
formula (Eff −2.036 vs exp008's −1.945; `scripts/exp018_efficiency_table.py`). If Kaggle requires two
*distinct* selected submissions, or will not count one submission for both tracks, pick the **second
slot** by the rule in §3; do not change the first.

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
5. Submit that version from the notebook's Output / "Submit to competition". Projected scoring runtime
   on ~1,300 studies: ~17 min on T4 x2, ~33 min on T4 x1 (`EXPERIMENTS.md` 2026-09-22 timing check;
   a worst-case decode-worker assumption gave ~83 min). All are far inside the 9h cap.
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
