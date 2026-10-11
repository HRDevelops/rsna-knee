# CV course progress presentation — Oct 15, 5 minutes

Project: RSNA Knee Abnormality Detection (Kaggle 2026): 12 findings from knee MRI, scored by macro-AUC. Solo project.
**Five slides plus one backup, ~3:55 of speaking** (speaker notes total 525 words: 3.9 min at 135 words/min,
4.2 min at 125 wpm, so under 4:30 either way). Every number is a logged measurement from `docs/EXPERIMENTS.md`; the section for each
is in the "Number sources" table at the end. Figures are in `docs/figures/` (PNG for slides, PDF for the report).
Revised 2026-10-11 after review: explicit mine-vs-borrowed line, lighter slide 5, ROC grid moved to a backup slide,
closing milestones line, notes re-timed.

Status when presenting: modeling is closed and the freeze is Oct 15. **One model family (exp018's CoAtNet-2
Window-MIL) supplies both final picks:** accuracy = exp021, 10 checkpoints (public 0.929); efficiency = exp018 on
2 folds (public 0.925, ~20.5 min measured). The private-leaderboard rank is unknown until the competition closes
(Oct 22).

## Three honesty points to say out loud (they change what the numbers mean)

1. **The 0.893 was most likely ONE small model, exp005, not the 3-model ensemble.** The notebook picks its
   configuration from a time budget; the 3 models measure ~59 s/study on Kaggle's T4s (~21 h), so it almost
   certainly fell back to exp005 alone. Inferred, not confirmed (the hidden-test log is not visible).
2. **The 0.893 -> 0.929 jump bundled four changes** (backbone family, window input, 140 mm FOV, corrected
   laterality). No ablation separates them, so the talk does not credit any one of them.
3. **Gold-58 is n = 58** (95% CI about +/-0.03 on the macro-AUC), and **OOF is scored against noisy report-derived
   labels.** The leaderboard is the only clean external number, and it is the public split only.

## Slide plan at a glance

| # | Slide | Time | Figure |
|---|---|---|---|
| 1 | The problem: strong images, weak labels, and the label engine | 0:00-0:45 | none (build a 3-bar graphic: 4,407 / 58 / 0) |
| 2 | The path and the plateau | 0:45-1:30 | `ablation_path_and_nulls.png` |
| 3 | Diagnosis, the formulation change, and what is mine vs borrowed | 1:30-2:25 | `formulation_before_after.png` |
| 4 | What the model looks at | 2:25-3:10 | `attention_windows_gold58_slide.png` |
| 5 | Results, lessons, and what is next | 3:10-4:05 | 3-row table (no figure) |
| B1 | Backup (not presented): per-label ROC | on request | `roc_gold58_exp018_exp021.png` |

## Slide 1 — The problem: strong images, weak labels (0:00-0:45)

**On the slide**
- Input: a knee MRI study = 3-14 series (sagittal / coronal / axial, fat-suppressed or not). Output: 12
  probabilities. Score = macro ROC-AUC.
- Bar graphic: **4,407** training studies -> **58** with expert image-review labels, **4,407** with a free-text
  report (12+ languages), **0** reports at test time. Reports agree with the experts only ~82% of the time.
- Label engine: two open LLMs (Qwen2.5-72B, Llama-3.3-70B) read each report against a rubric of the experts'
  thresholds -> calibrated on the 58 (leave-one-out) -> one noisy-student round -> `v3` labels. A second round
  regressed, so I stopped.

**Say (~110 words)**
"I predict twelve knee-MRI findings, scored by macro AUC. The catch is the labels: of 4,407 training studies, only
58 have expert image-review labels. Every study has a radiology report, but there are none at test time, and the
reports agree with the experts only about 82% of the time, because they over-call mild disease. So I built a label
engine. Two open language models read each report against a rubric of the experts' thresholds. I calibrate on the
58 real labels, let the image model clean the labels once, and stop there, because a second round made things
worse. Better labels alone were worth about plus 0.014 AUC."

## Slide 2 — The path and the plateau (0:45-1:30)

**Figure:** `docs/figures/ablation_path_and_nulls.png` (panel A the leaderboard path, panel B the tested changes).

**On the slide**
- A: public DINOv2-S baseline notebook 0.809 (third-party, the starting point, **not my submission**) -> my
  DINOv2 slot pipeline **0.893** -> Window-MIL **0.929**. Public top ~0.95-0.96 at the time.
- B: each change as a paired OOF delta with its 95% interval. Resolution, slice count, head initialisation and a
  report-text auxiliary loss all sit inside noise (or below the pre-registered +0.0082 bar). Hollow = null.
- Early gains that did work (OOF): ViT-S -> ViT-B +0.005, v3 labels +0.014, ensembling +0.006.

**Say (~95 words)**
"My first image model squeezes each series into one token, then mixes six slots. Its public score was 0.893. One
correction: that notebook picks its configuration from a time budget, and on Kaggle's GPUs the three-model ensemble
would need about 21 hours, so it almost certainly ran only its smallest model, one ViT-S. I can't confirm that.
Then I tried every lever I had: resolution, more slices, a better head, a report-text loss, bigger ensembles, each
against a bar I set in advance. Nothing cleared it. So I concluded the labels were the ceiling. That was wrong."

## Slide 3 — Diagnosis, the formulation change, and what is mine vs borrowed (1:30-2:25)

**Figure:** `docs/figures/formulation_before_after.png`.

**On the slide**
- Audit of 18 public kernels: a public 20-model DINOv2 ensemble also stalls at **0.891**, so the ceiling belonged
  to the model family. One single CoAtNet-2 with attention over slice windows scored **0.924**.
- My own audit found a laterality defect: sagittal right knees were mirrored front-to-back instead of
  medial-lateral, and 34% of studies (1,491) had no side normalisation. A geometric rule resolved 99.4% of them
  (99% agreement with validated sides).
- exp018: 140 mm FOV -> windows of 3 adjacent slices -> fully fine-tuned CoAtNet-2 -> **per-finding attention
  over all windows**. A pre-registered kill rule. **Public 0.929.**
- **What's mine vs borrowed (boxed line on the slide):** *Borrowed:* the window-MIL design, from a public Kaggle
  kernel (cited on the slide). *Mine:* the label engine, the labels, the folds, the laterality fix, the
  re-implementation, and all the training. *No third-party weights or labels.*
- Caveat on the slide: four things changed together; none is isolated.

**Say (~110 words)**
"Instead of another ablation, I looked outside my own pipeline. The DINOv2 family stalled near 0.89 everywhere,
even in a public twenty-model ensemble, and one very different design didn't. Mine compresses each series to one
vector; the other keeps every window of three adjacent slices and lets each finding attend over all of them. To be
clear about what's mine: that window design follows a public notebook, which I cite. The label engine, the labels,
the folds, the laterality fix, the re-implementation and all the training are mine, with no third-party weights or
labels. It scored 0.929. I changed four things at once, so I can't say which one mattered."

## Slide 4 — What the model looks at (2:25-3:10)

**Figure:** `docs/figures/attention_windows_gold58_slide.png` (three rows; the full six-row version and the
all-58-studies slot-share heatmap `attention_by_slot_gold58.png` are for the report and for questions).

**On the slide**
- For each finding, the four most-attended windows and the attention over all windows (exp021, mean of 10
  checkpoints). ACL attends to sagittal fat-suppressed slices, medial meniscus to coronal, Baker's cyst to axial.
- How the studies were chosen: **a fixed rule, before any map was viewed**: the most confidently detected
  gold-positive study per finding. This is a best case, not a hit rate. The three rows shown are a presentation
  choice, picked to cover three planes.
- The unselected view (all 58 gold studies): ACL puts **62%** of its attention on sagittal FS slices (30% would be
  uniform); Baker's puts **45%** on axial (20% uniform); effusion 57% sagittal FS. Attention is broad: the top 4
  windows hold only 22-48% in these examples, and a finding pools over roughly 16-40 effective windows.

**Say (~100 words)**
"Because each finding pools over windows, I can see where it looks. I chose these three studies by a rule fixed
before I viewed any map: the most confident gold-positive case per finding. So this is a best case, not a hit rate.
The ACL pools mostly from sagittal fat-suppressed slices, the medial meniscus from coronal, the Baker's cyst from
axial. Across all 58 gold studies, the ACL puts 62% of its attention on sagittal fat-suppressed slices, against 30%
if it were uniform. That shows where the pooling draws from. It's not proof of localization, and no radiologist has
checked these windows."

## Slide 5 — Results, lessons, and what is next (3:10-4:05)

No figure: the per-label ROC grid is on backup slide B1.

| | OOF | Gold-58 (fold-ens.) | Public LB | Runtime, Kaggle 2 x T4 |
|---|---|---|---|---|
| DINOv2 slot pipeline (most likely exp005 alone, T4 fallback) | 0.8631 | 0.8903 | 0.893 | 3 members ~21 h measured |
| **exp021, 10 checkpoints (final, accuracy)** | **0.8840** | **0.9111** [0.883, 0.936] | **0.929** | ~100 min |
| **exp018, 2 folds (final, efficiency)** | n/a | 0.9078 | **0.925** | **~20.5 min** (score -2.018 vs -1.966 for 5 folds) |

- The leaderboard beat OOF every time (about **+0.030** for the slot pipeline, corrected; **+0.045** for exp021),
  because OOF is scored against noisy report labels. Gold-58 is n = 58, so read it as a sketch.
- Lessons: (1) a run of null ablations shows a model *family* is saturated, not where the ceiling is; (2) measure
  runtime on the target hardware: my borrowed speed factors were off by 3.6x (CoAtNet) and ~13x (DINOv2);
  (3) **a runtime fallback can silently change which model is scored**: log the rung that was chosen.
- **Next (closing line): private leaderboard Oct 22 -> demo Nov 23-27 -> final report Dec 5.**

**Say (~105 words)**
"Final picks: ten checkpoints for accuracy, public 0.929, and two folds for speed, 0.925 in about twenty minutes.
The leaderboard beat my validation score every time, because I validate against noisy report labels. Three lessons.
One: six null results showed my model family was saturated, not that the problem was. Two: measure runtime on the
real hardware. I projected fifteen minutes and measured fifty-three. Three: a runtime fallback can silently change
which model you're scoring, which is why my 0.893 is probably one small model. Log the choice. Next: the private
leaderboard on October 22, a demo in late November, and the final report on December 5."

## Backup slide B1 — Per-label ROC (not presented; for questions)

**Figure:** `docs/figures/roc_gold58_exp018_exp021.png` (OOF version: `roc_oof_exp018_exp021.png`).

- Gold-58, exp021: strongest medial meniscus and Baker's 0.986, ACL 0.979, effusion 0.978; weakest PF OA 0.824,
  lateral OA 0.828, lateral meniscus 0.834. (n = 58: wide intervals, printed on the figure.)
- ACL is the clearest case of noisy proxy labels: OOF 0.857 but gold-58 0.979.
- Rows cut from slide 5, available if asked: exp018 5 folds (OOF 0.8816, gold-58 0.9091, public 0.929, ~53 min);
  distilled DINOv2-S efficiency probe (OOF 0.8708, gold-58 0.888, public 0.885; fold count unknown).

## Likely questions (prep)

- *Which change in exp018 made the difference?* Unknown: backbone, windows vs slots, 140 mm FOV and laterality all
  changed together. The per-label pattern (MCL +0.080, lateral meniscus +0.031 OOF) hints at laterality and the
  wider FOV, but that is a hint, not a result. DINOv2-B inside the window pipeline (exp022) got OOF 0.8709 vs
  0.8816, so the backbone is not the whole story.
- *How much of this is yours?* The window-MIL design follows a public kernel (cited; reimplemented here). The label
  engine, labels, folds, laterality fix, every training run, all ablations, the efficiency work and the evaluation
  protocol are mine. No third-party weights or labels; only ImageNet/DINOv2 initialisations from timm.
- *Does the attention show it finds the lesion?* Not shown. It shows which windows each finding pools from, and
  it is broad (a finding pools over ~16-40 effective windows). The six studies were chosen by confidence, not
  by whether the map looked right, but no radiologist has reviewed them.
- *Why is ACL's OOF so low but its gold-58 so high?* The OOF labels come from reports, which over-call mild ACL
  findings; on the 58 image-review labels the same model scores 0.979. The ACL work stopped for that reason.
- *Why not train on the 58 labels?* Too few; they calibrate the label engine and validate.
- *How do you know OOF is not leaking?* Folds are grouped by report cluster, and the leaderboard *beat* OOF
  (leakage would push the other way).
- *Did you blend with the older model?* The blend never scored: two Kaggle runs hung at teardown. A later offline
  test passed on OOF with the 3-model ensemble but that ensemble needs ~21 h on T4; with the single small model it
  fell below the pre-set bar (+0.0009 vs +0.0020), so it was closed.
- *Closest prior work?* Qiu et al., Nature Communications 15:7637 (2024), co-plane attention across MRI sequences for
  twelve knee abnormalities; it is cited in the report's related work.

## Number sources (all in `docs/EXPERIMENTS.md` unless noted)

| Number | Where |
|---|---|
| 4,407 / 58 / ~82% report-gold agreement; labels v3 +0.014 OOF; v4 regressed | CONSOLIDATED ABLATION LEDGER (2026-09-21), sections A and B; `CLAUDE.md` section 2 |
| 0.809 public baseline (third-party notebook) | `CLAUDE.md` section 5 |
| 0.893 = DINOv2 slot pipeline, most likely exp005 alone; OOF 0.8631 (3-member 0.8744); ~59 s/study, ~21 h | 2026-09-27 and 2026-10-01 entries |
| Null ablations and their intervals | Ledger section C and D; 2026-09-20 results block |
| Public audit (18 kernels, 0.891 / 0.924); laterality defect (34%, 1,491 studies, 99.4%) | 2026-09-21 audit and exp018 pre-registration entries |
| exp018 OOF 0.8816, gold-58 0.9091; exp021 OOF 0.8840 (+0.0024 [+0.0014, +0.0033]), gold-58 0.9111 [0.8834, 0.9359] | 2026-09-21 exp018 result; 2026-09-25 exp021 result |
| exp018 2-fold 0.925, ~20.5 min, efficiency -2.018 vs -1.966 | 2026-09-26 probes #5/#6 and efficiency entries |
| LB-minus-OOF gaps: +0.030 (slot, corrected), +0.047 (exp018), +0.045 (exp021 = 0.929 - 0.8840) | 2026-10-01 correction; 2026-09-22; computed |
| exp022 OOF 0.8709 (gate FAIL) | 2026-09-26, section B |
| exp005 re-gate +0.0009 vs +0.0020 bar | 2026-10-01 re-gate result |
| Attention shares (62% / 45% / 57%), effective windows | `docs/figures/attention_slot_mass_gold58.csv` (`scripts/fig_attention.py`) |
| Per-label AUCs (gold-58, OOF) | printed on `roc_gold58_exp018_exp021.png`, `roc_oof_exp018_exp021.png` |
| Milestones: private LB Oct 22; demo Nov 23-27; report Dec 5 | course schedule, as given |

## Items to verify before Oct 15

- The exp018/exp021 runtimes were measured on Kaggle's 2 x T4 on 120-150 training studies and extrapolated to
  ~1,300 studies.
- The attention figures contain real training MRI slices from the competition data. Course use is very likely fine;
  read the competition's data-use terms before the images go into a public Kaggle write-up or GitHub.
- Whether any private-leaderboard information is published before Oct 15.
- Team size: the slides say "I" throughout, which assumes a solo project. If there are teammates, the talk needs to
  say what each did.
