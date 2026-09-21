# CV course progress presentation — Oct 15, 5 minutes

Project: RSNA Knee Abnormality Detection (Kaggle 2026) — 12 findings from knee MRI, macro-AUC.
Every number below is a logged measurement from `docs/EXPERIMENTS.md` (ablation ledger, 2026-09-21).
Timing budget: 5 slides x ~55 s = ~4:35, leaving ~25 s slack. Status when presenting: modeling closed,
recipe frozen; final private-leaderboard rank not yet known (competition closes Oct 22).

## Outline (what the five minutes say)

1. **Problem + the twist** (0:00–0:55): 12 binary findings per study, ROC-AUC averaged over labels; only 58 of
   4,407 training studies have real labels — all 4,407 have a free-text radiology report, but no report at test
   time. The reports disagree with the expert image-review labels ~18% of the time, systematically.
2. **Weak supervision via LLM labels** (0:55–1:50): an LLM label engine turns reports into soft labels.
3. **From a slice-CNN baseline to the 2.5D slot model** (1:50–2:50): the architecture and what each step bought.
4. **Results** (2:50–3:40): OOF and leaderboard table, two submissions (accuracy + efficiency).
5. **What didn't work + next steps** (3:40–4:35): the null-result ablation table; what is left.

## Slide 1 — The problem: strong images, weak labels

**On the slide**
- Input: a knee MRI study = 3–14 series (sagittal / coronal / axial, fat-suppressed or not), 20–45 slices each.
- Output: 12 probabilities (ACL, MCL, medial/lateral meniscus, 3 OA compartments, effusion, synovitis, Baker's
  cyst, contusion, fracture). Score = macro ROC-AUC.
- The catch (a bar graphic): 4,407 training studies → **58 with true labels**, **4,407 with a free-text report**
  (12+ languages), **0 reports at test time**. Report-vs-expert agreement only ~82%: reports over-call mild disease.

**Say (≈55 s):** "Most of the labels I'd need don't exist. What I have is text — radiology reports, in a dozen languages —
that mostly but not systematically agrees with what the expert panel later called positive. So this is a
label-quality problem first and an architecture problem second."

## Slide 2 — Weak supervision: LLM report labelling

**On the slide** (a flow diagram)
- Report → two open LLMs (Qwen2.5-72B, Llama-3.3-70B; rubric encodes "mild / on-the-fence = negative") →
  per-finding probability → calibrated against the 58 gold studies (leave-one-out) → soft labels `v2`.
- Noisy-student round: blend labels with the image model's own out-of-fold predictions → `v3` (accepted only after
  it beat the previous labels on two independent checks). A second round (`v4`) **regressed** (−0.012) → stopped.
- Guardrails: 5-fold CV grouped by report cluster (near-duplicate reports never split across folds); the 58 gold
  studies are never trained on — a tripwire, not a selector.
- Gain from better labels alone: **+0.014 OOF macro** (same architecture, v2 → v3).

**Say (≈55 s):** "Two LLMs read every report; a rubric tells them the experts' bar for 'positive'. I check them against
the 58 real labels, then let the image model help clean the labels once — and I stop there, because a second
round made it worse."

## Slide 3 — Model progression: per-slice CNN → 2.5D slot model

**On the slide** (architecture diagram, left to right)
- Starting point in the literature: **MRNet-style** — a 2D CNN on each slice, pool over slices, one model per
  plane. (Not run by us — no external data, so no baseline number is claimed; it is the design we extend.)
- Ours: DINOv2 ViT slice encoder → transformer across slices (learned position) → **6 slots**
  {sag, cor, ax} x {fluid-sensitive or not} with a presence mask → cross-slot transformer → **12 per-finding
  attention queries** → 12 logits. Right knees flipped so "medial" is always the same side.
- Measured steps (OOF macro, proxy labels): ViT-S 2.5D **0.849** → ViT-B **0.854** → + v3 labels **0.868** →
  3-model ensemble **0.874**.

**Say (≈60 s):** "The standard public knee baseline looks at one slice at a time and pools. I keep that idea but let the model
attend across slices, then across series, and let each finding pick the series it cares about — so Baker's cyst
looks at sagittal, patellofemoral arthritis at axial. Bigger backbone gave half a point, better labels gave one and a half."

## Slide 4 — Results

| | OOF macro-AUC (proxy) | Public LB | Runtime |
|---|---|---|---|
| 3-model ensemble ({ViT-B v3, ViT-B v2, ViT-S v3}) | 0.8744 | **0.893** | ~62 min projected, T4 x2 (from H100 profiling) |
| Distilled single ViT-S (efficiency track) | 0.8708 | **0.885** | 7.5 min on H100 (T4 run: check log) |
| Leaderboard top (for scale) | — | ~0.95–0.96 | — |

- The leaderboard beat my own validation by 0.014–0.019: my OOF is measured against noisy report-derived labels,
  so it *understates* true performance (on the 58 clean labels the ACL AUC is ~0.99).
- Efficiency track: one distilled model within **0.008** of the ensemble for a fraction of the compute.
- Honest gap: ~6–7 points to the top. I treat that as structural, not tuning noise.

**Say (≈50 s):** "Two submissions: the accuracy one scores 0.893; a single small model distilled from the ensemble scores
0.885 and runs in minutes, which is what the efficiency prize rewards. The board beat my validation, which is
consistent with my validation labels being the noisy side."

## Slide 5 — Ablation of nulls, and what's next

**On the slide** — after the recipe landed, seven principled levers, all null (Δ OOF macro vs the base model;
noise SD ≈ 0.004):

| Lever | Hypothesis | Δ macro | Verdict |
|---|---|---|---|
| 448 px (rebuilt cache) | small findings are resolution-limited | −0.0001 | null |
| 32 slices (vs 24) | slice-count-limited | −0.0001 | null |
| ACL-specific head init | ACL is a head deficit | +0.0010 | null (ACL low = label noise) |
| Report-text auxiliary loss | reports hold signal beyond 12 labels | +0.0036 / +0.0012 | null vs bar (+0.0082) |
| ConvNeXt-V2 backbone | different family adds diversity | — | redundant (corr 0.93) |
| 3D ResNet | architectural diversity | 0.693 solo | killed |
| 6-model ensemble | more members | +0.0020 | not adopted (2.9x cost) |

Also *not tested, by decision*: ordinal severity targets (labels already encode mild vs definite).

- Takeaway: the plateau is a *label* ceiling, not a resolution or architecture ceiling.
- Next: freeze and submit (both notebooks), final-selection by Oct 22, then open-source code + weights + video.
  Re-open modelling only for a merger partner or a new insight.

**Say (≈55 s):** "After the recipe stopped improving I tested every remaining lever properly — pre-registered bars, same folds,
noise estimated by bootstrap. All null. That's informative: more resolution, more slices and smarter heads don't help
because the ceiling is the labels. Next steps are engineering: lock both submissions and release the code."

## Likely questions (prep)

- *Why not train directly on the 58 labels?* Too few; used them as calibration/validation only.
- *Why ViT/DINOv2?* Strong general features; ViT-B beat ViT-S (+0.005), DINOv3 and ConvNeXt did not beat DINOv2.
- *How do you know the OOF gap isn't leakage?* Folds grouped by report cluster; the LB *beat* OOF (leakage would
  push the other way); gold-58 never trained on.
- *Is the 0.893 vs top 0.95 gap a bug?* Diagnosed per label against a published benchmark; most of the apparent
  ACL deficit was proxy-label noise; the remainder is treated as unresolved.
- *MRNet baseline number?* Not run (external-data licence); the progression numbers start at our first working
  2.5D model.

## Items to verify before Oct 15 (I have not confirmed these)

- Ensemble runtime on Kaggle: the ~62 min T4 x2 figure on slide 4 is a projection from H100 profiling, not a
  read-out of the Kaggle run log — replace with the notebook's logged runtime.
- Efficiency notebook runtime on a single T4: you quoted ~13 min; our logs show 7.5 min on H100 and a 12–19 min T4
  projection — use the Kaggle log's number.
- Final selection state and any private-LB result, if published before Oct 15.
