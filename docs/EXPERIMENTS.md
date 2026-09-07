# Experiments Ledger

One row per training/labeling run. Append, never edit past rows (add a note instead).
Per CLAUDE.md §11/§7: one experiment = one yaml in `configs/`; log OOF macro-AUC, per-label
AUC, gold-58 AUC, and public LB only when actually probed (budget ≤2 probes/week).

## Label engine runs

| Date | Config/rubric | Model(s) | Gold-58 macro agreement | Per-label agreement | Notes |
|---|---|---|---|---|---|
| | | | | | |

## Image model runs

| Date | Config | Backbone | Fold(s) | OOF macro-AUC | Gold-58 AUC | Public LB | Runtime | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

### Per-label AUC detail (append one block per run above)

<!--
### exp001_dinov2s_slot — fold avg — 2026-0X-XX
| ACL | MCL | MedMen | LatMen | MedOA | LatOA | PFOA | Effusion | Synovitis | Baker's | Contusion | Fracture | Macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |
-->

## Ablations

| Date | Base config | Change | Δ OOF macro-AUC | Kept? | Notes |
|---|---|---|---|---|---|
| | | | | | |

## Decisions log

- 2026-09-06: Repo scaffolded (src/scripts/configs/slurm/notebooks/docs/artifacts/logs) per
  CLAUDE.md §11. Phase 0 sync tooling (`scripts/push.sh`, `run.sh`, `pull.sh`) and
  `scripts/decode_audit.py` written. No data downloaded yet — pending manual Kaggle rules
  acceptance + HPC credential/quota setup.
- 2026-09-06: Data confirmed already on HPC. Real recon flipped several infra assumptions (no
  `/scratch`, no modules/conda, `/shared/home` BeeGFS pool, `--qos=issai-h100` mandatory) — see
  `docs/HPC_WORKFLOW.md` "CONFIRMED" section. Decode audit run on the full dataset (24371 series):
  Laterality tag covers exactly 50% of studies with 25 cross-series conflicts — hybrid side
  detection plan adopted (tag majority-vote + report-supervised classifier fallback), see
  `docs/DECODE_AUDIT.md`.
- 2026-09-06: **Report clustering bug found and fixed.** First pass used character 5-gram shingles
  for MinHash + raw LSH-band-match-implies-duplicate — produced one 2134-study (48% of the
  dataset) "cluster" that on inspection was completely unrelated Spanish/German/English reports.
  Root cause: character n-grams are too generic across any language's common short substrings, and
  LSH candidates were never verified against real similarity before merging (transitive chaining
  through spurious single-band collisions). Fixed by switching to WORD 5-grams and adding a
  post-LSH approximate-Jaccard verification threshold (0.5) before union-find merging — rerun
  produced 3732 clusters from 4407 studies, largest cluster now a genuine site template (209
  studies, "Exam Type: MRI KNEE ... WO CONTRAST" boilerplate) at 4.7% of the dataset. See
  `scripts/cluster_reports.py`, `docs/REPORT_CLUSTERS.md`. **Lesson: always spot-check clustering
  output against raw content before trusting a distribution number, especially for anything using
  hand-rolled LSH.**
- 2026-09-06: `folds.parquet` built group-only (GroupKFold on report_cluster_id, LPT-balanced,
  gold-58 quarantined) — **NOT YET label-stratified**, rerun `scripts/build_folds.py --labels
  artifacts/labels/labels_v1.parquet` once Track A produces it.
- 2026-09-06: Report-side extraction (`scripts/extract_side.py`) + tag/report cross-check
  (`scripts/resolve_laterality.py`): 98.5% agreement (784/796) where both sources exist and are
  unambiguous — validates both extraction paths. Final side coverage: 2178 tag, 712 report, 1
  conflict resolved by report, 1491 with no source (need the trained classifier), 24 unresolved
  tag-conflicts, 1 tag="B" (bilateral). **Important finding: the 24 unresolved conflicts are very
  likely genuinely bilateral studies** (both knees imaged in one study — a study-level L/R tag
  split is exactly what that looks like, and almost none of them have a report-stated side either)
  — this breaks the one-side-per-study assumption behind the flip-right-knees pipeline (CLAUDE.md
  §5) for these ~24 studies specifically. Recommendation logged in
  `docs/LATERALITY_RESOLUTION.md`: handle side per-SERIES for these studies, not per-study.
- 2026-09-06: `scripts/build_cache.py` (uint8/384px/0.3mm, per user decision — no 512px variant
  while the BeeGFS quota question is with admins) validated on a 3-study smoke test (~3.5MB/series
  → ~85GB projected for the full dataset, well within headroom) and launched as the full
  `slurm/cache_build.sbatch` array job (20 shards, ZEN4). Also caught real order-mismatches between
  InstanceNumber and ImagePositionPatient-projected order in the smoke sample — confirms the
  EXECUTION_PLAN 2.2 concern was correct, geometric ordering is doing real work here.
- 2026-09-06: `configs/rubric_v1.yaml` had a YAML syntax bug (unquoted list items mixing plain text
  with embedded double-quoted phrases, e.g. `- partial tear, "rotura del LCA", ...` — ambiguous
  plain-scalar parsing) caught by the first Phase-1 smoke test job failing instantly. Fixed by
  fully double-quoting every such list item. **Lesson: validate config YAML files with
  `yaml.safe_load` before submitting a job that depends on them, not after.**
- 2026-09-06: vLLM 0.11.0 (the version `pip install vllm` grabbed) requires Python 3.10+ syntax
  internally (`X | None` unions evaluated eagerly at import time) and crashes on this cluster's
  system Python 3.9 — pinned down to `vllm==0.6.3.post1` instead (pulls torch down to 2.4.0+cu121
  automatically), which imports and runs cleanly. **Pin vLLM version explicitly going forward,
  don't let `pip install vllm` grab latest on this cluster.**
- 2026-09-06/07: **Cache build — full run completed, all 20 ZEN4 shards.** 24371 series, 24369 ok
  (2 errors — both genuinely corrupt source DICOMs, pixel-data length mismatch, correctly caught
  and skipped, not a pipeline bug), all 4407 studies represented. **9295/24371 series (38%) had
  InstanceNumber order disagree with the geometric (ImagePositionPatient-projected) order** —
  strong confirmation that EXECUTION_PLAN 2.2's concern was real, not theoretical; naive
  InstanceNumber sorting would have mis-ordered over a third of series. 0 irregular-spacing flags.
  Total cache size: **113GB** (uint8/384px/0.3mm only, no 512px variant) — lean, well within any
  reasonable reading of the BeeGFS quota headroom regardless of the still-unresolved quota anomaly.
- 2026-09-06/07: **Label engine v1 — full run completed** (24384, 3h32m, all 4407 studies), but
  105/4407 (2.4%) failed JSON parsing even after the built-in retry. Root cause: 100% of failures
  had the model prepend an unrequested "Here is the JSON:" preamble, and 91% of those then got
  truncated mid-JSON before the closing brace — `max_tokens: 1400` wasn't enough headroom once a
  preamble + long non-English (Greek/Croatian seen) verbatim evidence quotes ate into the budget.
  A full parse failure meant ALL 12 targets for that study fell back to the imputed
  present="uncertain"/probability=0.275 default, discarding whatever the model got right before
  truncating — wasteful. Fixed: `rubric_v1.yaml` now explicitly instructs "start immediately with
  {, no preamble" and raises `max_tokens` 1400→2200 (retry gets 3000). Backfilled just the 105
  failed studies by dropping them from the checkpoint parquet and re-running label_gen.sbatch
  (resumability by design skipped the other 4302 already-good studies). **Lesson: a hard
  parse-or-discard-everything policy is expensive when failures cluster around token-budget
  truncation — worth a partial-JSON salvage path if this recurs at scale in later rounds.**
- 2026-09-07: **exp001 fold-collapse root cause found and fixed.** 3/5 folds (0,1,4) trained under
  the original config produced near-constant OOF predictions (std 0.005-0.05 per label) vs the
  2 healthy folds' real per-study variation (std 0.07-0.28), with nearly identical per-label means
  across all 5 — confirmed via `--eval-raw-too`/OOF-std check as genuine optimization collapse,
  not a data issue. Ruled out confidence-weighting (labels_v1.parquet has no confidence_weight
  columns, so weights were uniformly 1.0). Instrumented with `--debug-grad-norms`
  (scripts/train.py): backbone grad norm was wildly spiky (1-25+, no stable range) while the
  from-scratch slice_agg/study_head sat at 0.1-4 (10-50x smaller). Root cause: the single global
  `clip_grad_norm_(model.parameters(), 1.0)` computes ONE scale factor from the combined norm, so
  a backbone spike forces the same aggressive down-scaling onto the tiny, freshly-initialized
  head gradients too — intermittently starving their learning signal. **The fix that mattered
  most, with the clearest causal evidence: clipping backbone and head/slice-agg params
  SEPARATELY** (each to max_norm=1.0). Applied together with a broader stability bundle
  (norm_first=True pre-norm on both from-scratch transformer stacks, small-gain std=0.02 init on
  all from-scratch attention Q/K/V/out projections, warmup_epochs 2->5, lr 2.0e-4->1.0e-4,
  confidence-weight floor 0.1) since these are standard practice for training a from-scratch head
  downstream of a pretrained backbone and cheap to include — **not individually ablated**, so
  credit isn't cleanly separable between them, but the grad-norm mechanism directly explains the
  per-group-clipping fix in a way the diagnostic log makes hard to dispute; the others are
  reasonable insurance, not separately confirmed as load-bearing. **Acceptance test: 2 seeds
  (42, 123) x 5 folds = 10 runs, ~10 epochs each — 10/10 escaped the plateau**, all showing
  smooth, near-identical, monotonically decreasing loss curves (0.56->0.47) with no sign of
  collapse. Full 25-epoch exp002 training unblocked. See `slurm/acceptance_sweep.sbatch`.
- 2026-09-07: **Label engine finalized — labels_v2_final.parquet.** Part B of the strategy review:
  (1) reverted the Synovitis effusion-tiered conditional prior to a flat 0.35/confidence 0.1
  (`scripts/patch_synovitis_prior_revert.py`, patched the existing Qwen v2 parquet in place — no
  LLM rerun, the prior was applied post-hoc in code) since the tiering made gold-58 AUC worse, not
  better; kept the mild->negative Synovitis few-shot and the Contusion/Fracture hierarchy rule,
  both baked into the v2 prompt and already reflected in the existing Qwen/Llama outputs.
  (2) `scripts/build_labels_v2_final.py`: per-label choice among {qwen_only, llama_only, 50/50
  blend} selected by STRICT leave-one-out AUC on gold-58 (each held-out study's prediction comes
  from calibrator(s) refit on the other 57 only — genuinely out-of-sample, not a post-hoc
  rationalization). Blend wins 8/12 labels (often beating both individual models outright, e.g.
  MCL blend LOO-AUC 0.943 vs Qwen 0.807 / Llama 0.839); Llama-only wins Synovitis (0.711 vs
  Qwen's 0.341 — worse than random) and Baker's (0.862); Qwen-only wins Lateral Meniscus (0.812)
  and Fracture (0.747). **Macro LOO AUC of the chosen-per-label strategy: 0.823** — a properly
  honest number since it's strict LOO throughout, expected to read lower than the earlier
  non-LOO single-model AUCs (Qwen v1 0.851, Qwen v2 0.846, Llama 0.870), which had a small
  optimism bias from fitting and evaluating calibration on the same 58 points. Full table:
  `docs/label_blend_v2_final.md`. No rubric v3, no new labeler runs, per instruction. Backlog:
  guided/constrained JSON decoding (vLLM supports it via the outlines backend) would more
  robustly fix the residual ~1% parse-failure rate than further prompt tuning — only worth
  pursuing if/when we rerun either LLM.
