# Winners' Obligation Video — Outline

Outline for the competition presentation video (winners' obligation deadline: Nov 5, per
`docs/EXECUTION_PLAN.md` §6.4). Target length **8–10 minutes**. First written 2026-09-15 and rewritten
2026-09-24 around the full four-phase arc, which ends with exp018 as the final submission for both
tracks. Companion document: `docs/METHOD_WRITEUP.md`. This outline cites its section numbers rather
than repeating content. Every number here is logged in `docs/EXPERIMENTS.md`; the only open item is
the private-LB rank.

Recording notes: screen-record code, plots and tables where noted. A talking-head segment is optional.

**The story in one line:** better labels got us to a ceiling, a public-benchmark audit showed the
ceiling was the model family rather than the labels, and one structurally different model broke
through: +0.036 on the leaderboard, and two of its folds also make the efficiency pick.

---

## 1. Hook + problem framing (~45s)

- 12 binary knee-MRI findings, macro-AUC, code competition (≤9h GPU, internet off).
- The twist: only **58 of 4,407** training studies have expert labels; all 4,407 have a free-text
  report, in 12+ languages; there are **no reports at test time**. Reports agree with the expert image
  review only ~82% of the time, and the disagreement is *systematic* (reports over-call mild disease).
- On screen: the four-phase timeline (below) as a roadmap graphic.
- (Writeup: §2, §3)

## 2. Phase 1: the weak-supervision engine (~2 min)

- Two open LLMs, **Qwen2.5-72B-Instruct** and **Llama-3.3-70B** (vLLM on HPC), extract structured JSON
  per finding. A rubric encodes the experts' severity thresholds ("on the fence" = negative-leaning).
  Show one rubric snippet and one extraction.
- Report clustering for leak-free folds: the character-n-gram bug that merged 48% of the dataset into
  one bogus cluster, and the word-shingle + verification fix (3,732 real clusters, largest 4.7%).
- Calibrated per-label blend on gold-58 (strict leave-one-out) gives `labels_v2_final`, macro LOO-AUC
  0.823. **Noisy-student round** (image-model OOF blended back in) gives **`labels_v3`**, macro LOO-AUC
  **0.904**. A second round regressed (0.904 → 0.892), so iteration stopped.
- Payoff: **+0.014 OOF macro** from labels alone, same architecture.
- (Writeup: §4)

## 3. Phase 2: the DINOv2 2.5D slot transformer and the 0.893 ceiling (~1.5 min)

- Architecture diagram: per-series DINOv2 ViT → slice-attention transformer → 6 slots
  {sag, cor, ax} × {fluid-sensitive or not} → cross-slot transformer → 12 per-finding attention
  queries.
- The scale ladder: ViT-S → ViT-B (+0.005, replicated twice); then a 3-member ensemble (OOF 0.8744)
  and a distilled ViT-S efficiency model (0.8708). **Public LB 0.893 / 0.885.**
- The ceiling: **six bootstrap-tested ablations, all null** against a pre-registered bar (noise SD
  ≈ 0.004):

  | Ablation | Δ OOF |
  |---|---|
  | 448 px | −0.0001 |
  | 32 slices | −0.0001 |
  | ACL-specific head init | +0.0010 |
  | report-text auxiliary loss, weight 0.1 | +0.0036 |
  | report-text auxiliary loss, weight 0.5 | +0.0012 |
  | 6-model re-selected ensemble | +0.0020 |

  Plus the diversity attempts that failed their own bars: ConvNeXt-V2 was redundant (corr 0.93), and a
  3D ResNet was killed at 40% trained.
- Honest beat: at this point **we concluded the plateau was a label ceiling**. That conclusion turned
  out to be wrong, which is the next section.
- (Writeup: §5–§8, §10)

## 4. Phase 3: the diagnostic audit (~1.5 min)

- Instead of another ablation, a survey of 18 public kernels (`artifacts/public_benchmark_audit.md`).
  What it showed:
  - **The DINOv2-slot family plateaus everywhere:** a public 20-member DINOv2 ensemble sits at 0.891,
    next to our 0.893. So the ceiling belongs to the model family, not to our labels.
  - **One structurally different design breaks it:** a single CoAtNet-2 with attention-MIL over
    adjacent-slice windows reaches 0.924 solo.
  - **Concrete differences we had never varied:** public pipelines crop **130–140 mm**; ours cropped
    **115 mm**, so the outer compartments (collateral ligaments, lateral meniscus) can fall at or past
    the crop edge. And our slot model compresses each series into one CLS token before any finding
    sees it, so a lesion visible on 2–3 of 30 slices has to survive that compression ("focal slice
    dilution"). The window-MIL design lets each finding attend to individual windows instead.
- Our own audit, in the exp018 cache build, found a real laterality defect in the deployed pipeline:
  - sagittal right knees were mirrored **anterior–posterior** rather than medial–lateral;
  - **34% of studies** had no normalized side at all.
  A geometric side rule fixed both: it resolves 99.4% of those studies and agrees 99% with validated
  sides.
- Framing to say out loud: these were **hypotheses the audit identified, not proven causes**. exp018
  changed them all at once, and no ablation separates them (see §7).
- (Writeup: §5a; `docs/EXPERIMENTS.md` 2026-09-21 audit + exp018 pre-registration)

## 5. Phase 4: the exp018 breakthrough (~2 min)

- Architecture: 140 mm FOV at 336 px; 5 slots / 64 slices from the 6–94% span; windows of 3 adjacent
  same-slot slices as RGB (≤54 per study) → fully fine-tuned **CoAtNet-2** → **per-finding
  attention-MIL over all windows** (no series bottleneck). Our labels (`labels_v3`), our folds,
  corrected laterality. The public kernel was an architecture reference only; no third-party weights
  or labels.
- Discipline beat: the design and a kill rule were **pre-registered** (fold-0 gold-58 < 0.880 ⇒ stop).
  Fold 0 scored 0.9063, and epoch selection never touched gold-58.
- Results:

  | | OOF | Gold-58 | Public LB |
  |---|---|---|---|
  | DINOv2 3-member ensemble | 0.8744 | 0.8953 | 0.893 |
  | **exp018, single model family, 5 folds** | **0.8816** | **0.9091** | **0.929 (+0.036)** |

  The gains sit on the mirror-symmetric labels: MCL +0.080 and Lateral Meniscus +0.031 OOF.
- **Efficiency track, decided by measurement:** I had projected the 5-fold run at ~15 min from H100 timing
  and a borrowed 3.5× T4 factor. A profiling notebook on Kaggle's own 2 × T4 measured **~53 min**: CoAtNet-2 on a
  T4 is ~12.6× slower than on an H100, and the run is model-bound (~9.7 min per fold). So fold count is the lever:
  **2 folds run in ~20 min at 0.925 public (Eff −2.018 vs −1.966 for 5 folds)**. Good beat: "measure on the
  target hardware".
- **Main track:** a second seed of the same recipe (exp021, 10 checkpoints) adds +0.0024 OOF; 0.929 public.
- Two follow-ups that did NOT help, worth one sentence: a DINOv2-B backbone in the same pipeline (weaker, and
  every blend weight lowers OOF), and denser windows at inference (null).
- (Writeup: §5a, §11; model card `docs/model_cards/exp018_coatnet2_wmil_v3labels.md`)

## 6. Engineering honesty beat (~1 min, optional)

Pick one:
- **The blend that never scored** (recommended; it is the most instructive):
  - A rank-space blend of exp018 with the old ensemble hung at teardown on Kaggle twice (9h, 10h)
    after writing a valid submission.
  - We "root-caused" it to a real thread bug that turned out **not to be on the blend's code path**,
    and the fix introduced a timed join that silently wrote label priors on slower GPUs.
  - Lesson: confirm the suspect code actually ran, run a pre-fix control, and assert on *prediction
    content*, not just row counts. The blend was dropped under a pre-agreed guardrail.
- **Or the quota guard:** a near-full shared-storage quota silently skipped checkpoint saves and cost
  ~35 GPU-hours once.
- (Writeup: §9)

## 7. Results + close (~1 min)

- Final selections: **main = exp021** (10 checkpoints, 0.929); **efficiency = exp018 on 2 folds** (0.925, ~20 min).
  exp018 5-fold (0.929), the DINOv2 ensemble (0.893) and exp008 (0.885) are scored fallbacks. **Private LB / final rank: fill in once the
  competition closes.**
- OOF understated the LB for every probe (gaps −0.019, −0.014, −0.047): validation labels are the noisy
  side, not leakage.
- What we would do with more time: ablate exp018's bundled changes (FOV, windows vs slots,
  laterality) to find out which one mattered. Its Effusion and Synovitis did not improve over DINOv2.
- Repo, weights and writeup pointer (Apache-2.0; `README.md`, `docs/METHOD_WRITEUP.md`).

---

## Recording checklist

- [ ] Screen captures:
  - [ ] report-cluster fix numbers, rubric snippet, gold-58 LOO-AUC table (Phase 1);
  - [ ] slot-model diagram and the null-ablation table (Phase 2);
  - [ ] the public-audit summary and the laterality before/after orientation correlations
        (coronal 0.766 → 0.965) (Phase 3);
  - [ ] the exp018 window-MIL diagram, gate report, and results table (Phase 4).
- [ ] Make code and terminal text readable at video resolution.
- [ ] Runtime target 8–10 min. If a cut runs long, trim §6 first, then Phase 1's clustering story.
      Never cut the Phase 3 → Phase 4 turn: it is the point of the video.
- [ ] Final render + upload once the leaderboard closes and §7 has the private rank.
