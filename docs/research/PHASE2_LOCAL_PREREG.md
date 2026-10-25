# Pre-registration: local-feature geometry follow-up (B_local vs G_local)

Committed 2026-10-26, before any local-feature model has been built, trained or scored. This is the single follow-up
pre-listed in `PHASE1_PREREG.md` §5 ("local-stage tokens: only if G fails on final-stage tokens"). It is a different
mechanistic question, not a re-tuning of Phase 1:

> Does physical cross-plane correspondence help when the features are still spatially local?

The Phase 1 STOP (final-stage tokens) stands as logged and is not reopened.

## Frozen

| Item | Value |
|---|---|
| Feature source | `backbone.stages.1` of `coatnet_rmlp_2_rw_384` (timm `MaxxVit`): the output of the last MBConv stage (stride 8; 256 channels; 48×48 at 384 px), i.e. immediately before `backbone.stages.2`, the first `TransformerBlock2d` (global within-slice attention). Stages 2–3 and the final norm are removed. ImageNet weights as before |
| Tokens | 6×6 per window (adaptive average pool of the 48×48 map), 432 tokens per training pass, as in Phase 1 |
| Everything else | Identical to Phase 1: B5 data; dense inference; token transformer (d 384, 8 heads, 2 layers); embeddings; same-series bias; validity masking; readout; loss; schedule; budget; top-3 selection by bagged fold-validation; seeds and RNG streams |
| Arms | $B_{\text{local}}$ (no geometry) and $G_{\text{local}}$ = $B_{\text{local}}$ + $-\mathbb 1[\pi_j\neq\pi_k]\frac{\beta_h}{2}d^2_{jk}$. Fresh paired runs from the same codebase; the old $B_{\text{cap}}$ is not reused |
| $\beta$ | Exactly 0 at init; projected SGD. $\eta_\beta$ comes from the same no-metric 50-step gradient probe and formula as Phase 1, recorded before launch |
| Pre-launch tests | `tests/test_phase1_geometry.py` identity, pairing, padding and donor checks, repeated for the local feature source |

## Decision rules (fold 0, seeds 42 / 43; dense fold-validation macro-AUC; gold-58 never used)

- $\Delta_{\text{geo},s}=\mathrm{AUC}(G_{\text{local}})-\mathrm{AUC}(B_{\text{local}})$, same seed.
- $\Delta_{\text{swap},s}=\mathrm{AUC}(G_{\text{local}})-\mathrm{AUC}(G_{\text{local}}\text{ with donor geometry})$, with
  the same donor rule as Phase 1.
- **STOP** immediately if $\Delta_{\text{swap},s}<0.001$ in both seeds. That **terminates the geometry research line**:
  the mechanism will then have been tested both after global spatial mixing and with local features, under the same
  causal controls.
- **GO** only if both $\Delta_{\text{geo},s}>0$, their mean is $\ge+0.002$, and both $\Delta_{\text{swap},s}\ge0.001$.
- **Otherwise (ambiguous):** fold 1 with the same two seeds and fresh paired runs; continue only if the 4-run mean
  $\Delta_{\text{geo}}>0$ and the swap shows use. No third seed.
- **Not before the local mechanism passes:** no IPE, jitter, all-series geometry, k-NN or grid change.
- **Mechanism report** for both arms, including on a STOP:
  - $\beta$-off diagnostic: mean $|p_G-p_{G,\beta=0}|$ and AUC difference;
  - percentiles of the actual cross-plane bias $-\beta_h d^2/2$.

## Protocol deviation recorded with this pre-registration

- **What happened.** The ledger said the 40-pair blinded geometry review would finish before Phase 1. Phase 1 was
  launched before the ratings existed.
- **Why it does not change the result.** The review is descriptive only (`PHASE1_PREREG.md` §6) and cannot change any
  model choice or threshold. The Phase 1 STOP rests on the within-$G$ donor swap.
- **What happens now.** The review will be completed for the audit record only and will not be used to reinterpret
  Phase 1.
- Its export now refuses incomplete sheets (all 160 answers required). It was patched in place before any rating, so
  the blinded A/B assignment is unchanged.
