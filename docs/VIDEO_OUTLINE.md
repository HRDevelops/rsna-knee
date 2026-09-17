# Winners' Obligation Video — Outline

Draft outline for the competition presentation video (winners' obligation deadline: Nov 5, per
`docs/EXECUTION_PLAN.md` §6.4). Target length **8–10 minutes** — long enough to cover the label
engine (the actual differentiator) properly, short enough to stay watchable. Written 2026-09-15,
before the leaderboard result is known — §7's numbers are placeholders, fill in once probed.
Companion document: `docs/METHOD_WRITEUP.md` (full detail; this outline references its section
numbers rather than repeating content).

Recording notes: screen-record code/plots where noted; a talking-head segment is optional but not
required — plots and terminal/notebook output carry most sections better than a face on camera for
this kind of technical walkthrough.

---

## 1. Hook + problem framing (~45s)

- One sentence: 12-label knee MRI classification, macro-AUC, code competition.
- The one insight that shaped everything: ground truth is **image-reviewed**, training labels are
  mostly **report-derived**, and those two disagree ~18% of the time, systematically (not noise).
  Show the CLAUDE.md §2 bullet list of known gaps (meniscal degeneration ≠ tear, trace effusion ≠
  positive, etc.) as an on-screen callout — this single fact is why the project spent ~40% of its
  time on labels, not architecture.
- (Writeup source: §3)

## 2. Label engine (~2.5 min — the differentiator, give it the most time)

- Report clustering: the character-n-gram bug (48% of the dataset merged into one bogus cluster)
  and the word-n-gram + verification fix. Good visual: before/after cluster-size histogram if one
  exists, or just state the numbers (2,134-study bogus cluster → 3,732 real clusters, largest 4.7%).
- Two-LLM ensemble (Qwen2.5-72B, Llama-3.3-70B) via vLLM, structured JSON extraction, severity
  thresholds baked into the rubric. Show one example rubric snippet + one example extraction.
- Calibrated blend → `labels_v2_final`: strict LOO on gold-58, per-label strategy choice. **Macro
  LOO-AUC 0.823.**
- Noisy-student round → `labels_v3_candidate`: blend image OOF back in, pre-registered per-label
  direction. **Macro LOO-AUC 0.904.** One sentence on why iteration stopped there (round 2
  regressed, 0.904→0.892).
- (Writeup source: §4)

## 3. Image architecture (~1.5 min)

- 2.5D slot design diagram (draw or screen-share `src/rsna_knee/model.py`'s docstring/architecture
  comment): per-series DINOv2 ViT → slice-attention → 6 canonical slots → cross-slot transformer →
  per-target attention pooling.
- Laterality normalization in one sentence + why it matters (mirror-symmetric labels).
- One sentence on the fold-collapse bug and the per-parameter-group gradient-clip fix — good
  concrete "here's a real bug we found and fixed" moment, keeps it honest rather than a highlight
  reel.
- (Writeup source: §5, §6)

## 4. Scale ladder + ensemble (~1.5 min)

- ViT-S → ViT-B: tested twice, same answer both times (+0.0055, +0.0050). DINOv3: tried, dropped
  (−0.0252). One 3D architecture attempt (ResNet-50): killed at 40% trained — real diversity signal
  (correlation 0.444) but never cleared the accuracy bar to justify finishing it. Second diversity
  attempt (ConvNeXt-V2): cleared the accuracy bar, failed the diversity bar (correlation 0.93) —
  show both as examples of a **pre-registered accept/reject rule actually being applied**, not just
  results that happened to work.
- Greedy forward selection → 3-member ensemble, **macro 0.8744**. The rank-mean-has-no-population
  problem and its empirical-CDF fix — good "subtle correctness bug caught before it shipped" beat.
- TTA: measured, rejected (−0.0014). Another honest "we checked, it didn't help" moment.
- (Writeup source: §7)

## 5. Efficiency track (~1 min)

- Distillation setup (KL + BCE against the ensemble teacher) in one sentence.
- **0.8708 OOF, −0.0036 off the full ensemble, 7.5 minutes for ~1,300 studies** — the headline
  numbers. One sentence on the decode-dominates-runtime finding and the fast-decode path it
  motivated.
- (Writeup source: §8)

## 6. Infrastructure honesty beat (~1 min, optional but recommended)

- Pick ONE concrete story, not all of them: the quota-guard incident (a shared-cluster storage
  quota near its cap silently skipped checkpoint saves, cost ~35 GPU-hours once) is the most
  relatable "real engineering, not just ML" moment. Show the before/after fix.
- Purpose: judges/viewers value seeing that a solo entrant handled real infrastructure failure
  modes competently, not just that the model scored well.
- (Writeup source: §9)

## 7. Results + close (~1 min)

- OOF → public LB, both probes: ensemble **0.8744 → 0.893**, efficiency **0.8708 → 0.885**
  (−0.008 delta, inside the efficiency-track target). Both OOF↔LB gaps measured trustworthy
  (−0.0186, −0.0142, both under the pre-registered 0.02 bar) — a mild positive signal on the
  validation methodology, not just luck. **Private LB / final rank: fill in once the competition
  closes.**
- Gap to the public LB top (~0.95–0.96) is ~6–7 points and treated as structural: a diagnostic
  against a published external benchmark (Qiu et al. 2024, Nat Commun) showed our gap is *not*
  uniform across labels — ACL is dramatically behind the external reference while most other
  comparable labels are ahead of it — redirecting remaining effort toward MRNet auxiliary
  pretraining (ACL/meniscus-focused) and a resolution/slice-count ablation, rather than another
  label-engine round.
- One sentence: what's still running with more time — a genuinely diverse 3rd architecture family
  (exp007, 3D ResNet, relaunched to full convergence) and the MRNet pretraining route.
- Repo/weights/writeup pointer (Apache-2.0, `docs/METHOD_WRITEUP.md` for full detail).

---

## Recording checklist

- [ ] Screen-capture: report-cluster fix (before/after numbers), rubric snippet, model architecture
      diagram/docstring, gold-58 LOO-AUC table, ensemble greedy-selection log, final OOF/LB numbers
      once available.
- [ ] Keep code/terminal segments readable at video resolution (zoom terminal font before recording).
- [ ] Total runtime target 8–10 min — if a rough cut runs long, cut §6 first (labeled optional
      above), then trim §3's bug story to one sentence, before cutting anything from §2 or §4.
- [ ] Final render + upload once the competition's leaderboard closes and §7 has real numbers.
