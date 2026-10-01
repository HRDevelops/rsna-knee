# CV course progress presentation — Oct 15, 5 minutes

Project: RSNA Knee Abnormality Detection (Kaggle 2026): 12 findings from knee MRI, scored by macro-AUC.
Every number below is a logged measurement from `docs/EXPERIMENTS.md`. Rewritten 2026-09-24 for the full
four-phase arc. Timing budget: 5 slides × ~55 s ≈ 4:35, leaving ~25 s of slack.

Status when presenting: modeling and code frozen (freeze Oct 15). **One model family (exp018's CoAtNet-2
Window-MIL) supplies both final picks:** accuracy = exp021, 10 checkpoints (public 0.929); efficiency = exp018 on
2 folds (public 0.925, ~20 min measured). The final private-leaderboard rank is not yet known (the
competition closes Oct 22).

## Outline (what the five minutes say)

1. **Problem + the twist** (0:00–0:55): 58 real labels, 4,407 reports, no reports at test time.
2. **Phase 1: weak supervision** (0:55–1:50): LLM label engine and a noisy-student round → `labels_v3`.
3. **Phase 2: the DINOv2 slot model and its ceiling** (1:50–2:50): 0.893 (most likely exp005 alone), then six null ablations.
4. **Phases 3–4: audit → exp018** (2:50–3:50): what the audit found, and the window-MIL model that
   broke the ceiling.
5. **Results + lessons** (3:50–4:35): the final table, one model family for both tracks, and two of my
   own conclusions that measurement overturned.

## Slide 1 — The problem: strong images, weak labels

**On the slide**
- Input: a knee MRI study is 3–14 series (sagittal / coronal / axial, fat-suppressed or not), with
  20–45 slices each.
- Output: 12 probabilities (ACL, MCL, medial/lateral meniscus, 3 OA compartments, effusion, synovitis,
  Baker's cyst, contusion, fracture). Score = macro ROC-AUC.
- The catch (a bar graphic): 4,407 training studies → **58 with true labels**, **4,407 with a free-text
  report** (12+ languages), **0 reports at test time**. Reports agree with the experts only ~82% of the
  time, because they over-call mild disease.

**Say (≈55 s):** "Most of the labels I'd need don't exist. What I have is text: radiology reports in a
dozen languages that mostly, but systematically not always, agree with what the expert panel called
positive. So it starts as a label-quality problem."

## Slide 2 — Phase 1: weak supervision via LLM report labelling

**On the slide** (flow diagram)
- Report → two open LLMs (Qwen2.5-72B, Llama-3.3-70B; the rubric encodes "mild / on-the-fence =
  negative") → per-finding probability → calibrated on the 58 gold studies (leave-one-out) → `v2` soft
  labels.
- Noisy-student round: blend the labels with the image model's own out-of-fold predictions → **`v3`**.
  A second round (`v4`) **regressed**, so I stopped.
- Guardrails: folds grouped by report cluster (near-duplicate reports never split across folds); the
  gold-58 studies are never trained on.
- Gain from better labels alone: **+0.014 OOF macro**, same architecture.

**Say (≈55 s):** "Two LLMs read every report against a rubric of the experts' thresholds. I calibrate
against the 58 real labels, let the image model clean the labels once, and stop there because a second
round made it worse."

## Slide 3 — Phase 2: the DINOv2 2.5D slot model, and a ceiling

**On the slide**
- Architecture: DINOv2 ViT slice encoder → transformer across slices → one CLS token per series →
  **6 slots** {sag, cor, ax} × {fluid-sensitive or not} → cross-slot transformer → 12 per-finding
  attention queries.
- Progress (OOF macro vs proxy labels): ViT-S **0.849** → ViT-B **0.854** → + v3 labels **0.868** →
  3-model ensemble **0.874**. **Public LB 0.893** for the DINOv2 slot pipeline (most likely exp005 alone after the T4 runtime fallback), with a distilled
  efficiency model at 0.885.
- Then **six bootstrap-tested ablations, all null** (noise SD ≈ 0.004, bar +0.0082):

  | Ablation | Δ OOF |
  |---|---|
  | 448 px | −0.0001 |
  | 32 slices | −0.0001 |
  | ACL head init | +0.0010 |
  | report-text auxiliary loss ×2 | +0.0036 / +0.0012 |
  | 6-model re-selected ensemble | +0.0020 |

  ConvNeXt-V2 was redundant (corr 0.93); a 3D ResNet was killed.

**Say (≈60 s):** "This model family got me to 0.893. (On Kaggle's T4s the notebook's time budget almost certainly
dropped it to its smallest member, exp005, so that score is one ViT-S model.) Then I tested every remaining lever properly, with
pre-registered bars, the same folds and bootstrap noise, and all of them were null. At the time I
concluded the ceiling was the labels. That turned out to be wrong."

## Slide 4 — Phases 3–4: the audit, and exp018

**On the slide**
- **Audit** (18 public kernels, plus my own cache audit):
  - A public 20-model DINOv2 ensemble also stalls at 0.891, so the ceiling is the *model family*.
  - One single CoAtNet-2 with attention over slice windows reaches 0.924.
  - Things I had never varied: my crop was **115 mm** vs the public 130–140 mm, and every finding
    only ever saw a per-series summary.
  - My own audit found a laterality defect: sagittal right knees were flipped front-to-back rather
    than medial-lateral, and 34% of studies had no side normalization. A geometric rule fixed it,
    with 99% agreement.
- **exp018** (diagram): 140 mm, 336 px → windows of 3 adjacent slices as RGB (≤54 per study) →
  fully fine-tuned CoAtNet-2 → **per-finding attention over all windows** (no series bottleneck).
  My labels, my folds, pre-registered kill rule.

**Say (≈60 s):** "Instead of another ablation, I looked at what everyone else was doing. The DINOv2
family was stuck everywhere, and one very different design wasn't. I re-implemented it on my own labels
with a pre-registered kill rule, and fixed a laterality bug I found along the way. I changed several
things at once, so I can't tell you which one mattered: that's the honest caveat."

## Slide 5 — Results + lessons

| | OOF | Gold-58 | Public LB | Runtime (T4 × 2) |
|---|---|---|---|---|
| DINOv2 slot pipeline (most likely exp005 alone after the T4 runtime fallback) | 0.8631 (exp005) | 0.8903 | 0.893 | 3 members: ~21 h measured; exp005 rung: shorter |
| DINOv2 distilled (old efficiency pick) | 0.8708 | 0.888 | 0.885 | minutes |
| **exp021: CoAtNet-2 Window-MIL, 2 seeds × 5 folds (final, accuracy)** | **0.8840** | **0.9111** | **0.929** | ~100 min |
| **exp018 on 2 folds (final, efficiency)** | — | 0.9078 | **0.925** | **~20 min (measured)** |
| exp018, 5 folds | 0.8816 | 0.9091 | 0.929 | ~53 min (measured) |
| Leaderboard top, for scale | — | — | ~0.95–0.96 | — |

- **+0.036 LB** from one model family.
- **Efficiency needed a real measurement.** I had projected 15 min from H100 timing; on Kaggle's own T4s the
  5-fold run takes ~53 min and is model-bound. Cutting to 2 folds brings it to ~20 min for 0.004 AUC, which
  wins the efficiency formula (−2.018 vs −1.966).
- The LB beat OOF every time, because OOF is scored against noisy report labels.
- Lessons:
  - A run of null ablations tells you *that* family is saturated, not what the ceiling is.
  - Survey outside your own pipeline before concluding "label ceiling".
  - Measure runtime on the target hardware; a borrowed speed factor was off by 3.6× (CoAtNet) and ~13× (DINOv2).
  - A runtime fallback ladder can silently change which model gets scored: log the chosen rung.

**Say (≈45 s):** "One model family does both jobs: ten checkpoints for accuracy, two for speed. Two lessons:
six null results proved my model family was saturated, not that the problem was; and a runtime I had projected
at 15 minutes measured 53 on the real hardware. Next is open-sourcing code, weights and the write-up."

## Likely questions (prep)

- *Why not train directly on the 58 labels?* Too few; they are used for calibration and validation
  only.
- *Which change in exp018 made the difference?* Unknown: backbone, windows vs slots, 140 mm FOV and
  laterality all changed together, and no ablation separates them. The per-label pattern (MCL +0.080,
  lateral meniscus +0.031) hints at laterality and the wider FOV, but that is a hint, not a result.
- *Isn't exp018 just a copy of a public notebook?* It is the same design, re-implemented, trained on my
  own labels and folds, with no third-party weights or labels, and epoch selection that never used the
  gold set (the reference selected on it).
- *How do you know OOF gains aren't leakage?* Folds are grouped by report cluster, and the LB *beat* OOF
  (leakage would push the other way).
- *Did you ensemble exp018 with the old model?* I tried a blend. It hung on Kaggle's runner twice and
  never scored, so I dropped it.

## Items to verify before Oct 15

- exp018/exp021 runtimes are MEASURED on Kaggle's 2 × T4 machine on 150 training studies and extrapolated to
  ~1,300; the old DINOv2 ensemble's ~62 min is still an H100 projection. Say "projected" for that one.
- The final selection state and any private-LB result, if published before Oct 15.
