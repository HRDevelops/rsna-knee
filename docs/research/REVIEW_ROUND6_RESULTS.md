Thank you for the last review. I did exactly what you recommended. Below are the results, then a request for your view
on what to do next. Nothing was re-tuned after seeing a number.

## 1. β-off diagnostic on the stopped Phase 1 models (final-stage tokens)

Method: same weights, every $\beta_h$ set to 0, dense, fold-0 validation. Bias = real cross-plane bias
$-\beta d^2/2$, in logits.

| Seed | AUC($G$) − AUC($\beta$=0) | Mean / max \|Δp\| | Bias p1 / p5 / p50 / p95 |
|---|---|---|---|
| 42 | −0.0000 | 3.7e-4 / 0.008 | −6.1 / −4.0 / −0.77 / −0.08 |
| 43 | −0.0000 | 4.2e-4 / 0.014 | −6.4 / −4.3 / −0.67 / −0.06 |

You were right that β alone was misleading: the biases are not small. Removing them still changes nothing, so the
pathway is functionally dormant.

## 2. Local-feature follow-up (pre-registered before it was built; `PHASE2_LOCAL_PREREG.md`)

- **Features:** `backbone.stages.1` of `coatnet_rmlp_2_rw_384` (the last MBConv map, stride 8, 48×48), taken
  immediately before the first `TransformerBlock2d`.
- **Runs:** fresh paired $B_{\text{local}}$ / $G_{\text{local}}$ from the same codebase; everything else as Phase 1.
- **Pre-launch:** tests passed; $\eta_\beta$ = 0.2065 from the same no-metric probe.

| Seed | $G_{\text{local}}$ | $B_{\text{local}}$ | $\Delta_{\text{geo}}$ | $\Delta_{\text{swap}}$ | AUC($G$) − AUC($\beta$=0) | Bias p5 / p50 |
|---|---|---|---|---|---|---|
| 42 | 0.9281 | 0.9237 | +0.0044 | +0.00002 | +0.0002 | −18.6 / −3.3 |
| 43 | 0.9257 | 0.9344 | −0.0087 | 0.0000 | +0.0001 | −18.7 / −3.6 |

**Outcome.** STOP by the pre-registered donor-swap rule, which **terminates the geometry research line**, as
pre-registered.

**Mechanism.**
- With local features the model did learn strong footprint locality: cross-plane attention is close to restricted to
  compatible footprints.
- But the *correct* geometry carries no information: donor geometry, which differs by a median 26 mm, gives the same
  AUC, and removing $\beta$ moves AUC by at most +0.0002.
- Local features are also about 0.025 weaker overall.

## 3. Blinded 40-pair review (descriptive only; completed after Phase 1, logged as a protocol deviation)

One rater, all 160 answers, A/B blinded.

| Phase 0A correction stratum | Raw DICOM preferred | MI correction preferred | Indistinguishable |
|---|---|---|---|
| Low (≤ 1.5 mm) | 1/10 | 7/10 | 2/10 |
| Moderate | 2/10 | 8/10 | 0 |
| High (> 3 mm, stable) | 1/10 | 9/10 | 0 |
| Unstable / gross | 9/10 | 1/10 | 0 |

**Reading.**
- Stable MI corrections were preferred in 24 of 30 pairs, mostly visible translations. This suggests real few-mm
  inter-sequence offsets in the raw geometry.
- The gross tail was rejected 9 of 10 times, which confirms it is optimizer failure.

**What it changes:** nothing. Phase 0A stays inconclusive, and both STOPs stand: few-mm offsets (d about 0.2) cannot
explain a null for a 26 mm donor swap.

**Caveats:** one non-radiologist rater; and confident "clear" preferences at ≤ 1.5 mm corrections were not checked for
rendering cues.

## 4. What I think we have

**A pre-registered negative result:** explicit acquisition-footprint correspondence between separately acquired planes
does not improve multi-planar knee-MRI classification. This holds with global-attention tokens (pathway dormant) and
with local tokens (pathway active but uninformative), under donor-swap and β-off controls.

**Plus reusable infrastructure and findings:**
- cache v2, with exact geometry and all series and slices;
- the preprocessing ladder: using all data changes macro-AUC by less than the noise, while shifting per-label strengths
  (ACL −0.019; MCL / lateral meniscus +0.012);
- a registration audit tool and its failure modes;
- the review.

**Deadlines.** The course report is due 2026-12-05; the competition is closed (private 0.912).

## Questions

1. Do you agree the geometry line is closed, and that nothing in §3 justifies reopening it?
2. Is this publishable as a short negative-results paper, for example a MICCAI or MIDL workshop or MELBA? If so, what
   is the minimum extra evidence a reviewer would demand: more folds for the $\Delta_{\text{geo}}$ estimate, a CI on
   the donor-swap null, or an external set?
3. What is the most valuable next research direction with the time and compute we have (H100s, about 5 weeks)? My
   candidates:
   - **(a) Labels.** The report→label engine was always the biggest lever: per-label rubric repair, and modelling the
     report-label error measured on gold-58.
   - **(b) Training on all data.** Cache v2's all-series / all-slices data with the strong window-MIL model across 5
     folds and seeds, plus a late submission if scoring works.
   - **(c) Investigate the ACL drop.**
   - **(d) Just write up.**
