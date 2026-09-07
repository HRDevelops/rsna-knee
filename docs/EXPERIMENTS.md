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
- 2026-09-08: **exp002 completed — all 5 folds, 25/25 epochs, no crashes after the fragmentation
  fix.** OOF macro-AUC (vs report/soft labels_v2_final) **0.8489 +/- 0.0068** across folds
  [0.8444, 0.8605]; gold-58 tripwire macro-AUC **0.8675 +/- 0.0044** [0.8642, 0.8739]. Both clear
  the 0.80-territory bar (CLAUDE.md decision, Part C step 8) with a comfortable margin and very
  tight per-fold spread — strong evidence the stability-bundle fix (per-group grad clipping,
  see below) generalizes, not a lucky fold. Comparison against exp001 (0.6511/0.6193, huge spread
  0.49-0.83) is **not apples-to-apples** — exp001 ran on v1 labels under the *pre-fix* config with
  3/5 folds collapsed, so most of that gap is the collapse bug, not the label engine v1->v2
  upgrade; exp001 was always meant as pipeline-validation/expendable, not a real baseline. No
  raw-vs-EMA comparison available for exp002 (`--eval-raw-too` wasn't passed on this run — the
  flag exists in `scripts/train.py` from the exp001 diagnosis but wasn't threaded through
  `slurm/train.sbatch`'s default args).
  **Per-label: image model vs the label engine's own gold-58 LOO AUC** (`docs/label_blend_v2_final.md`)
  — 10/12 labels have image gold58 AUC > label engine LOO AUC, Phase 4 blend candidates per
  CLAUDE.md §6 step 5, confirming the pre-registered Synovitis/Contusion prediction and going
  well beyond it:
  | Label | image OOF | image gold58 | label engine LOO | image > engine |
  |---|---|---|---|---|
  | ACL | 0.790 | 0.925 | 0.978 | no |
  | MCL | 0.806 | 0.778 | 0.943 | no |
  | Medial Meniscus | 0.919 | 0.927 | 0.892 | **yes** |
  | Lateral Meniscus | 0.865 | 0.812 | 0.812 | **yes (tie)** |
  | Medial OA | 0.885 | 0.926 | 0.865 | **yes** |
  | Lateral OA | 0.851 | 0.815 | 0.727 | **yes** |
  | PF OA | 0.815 | 0.854 | 0.788 | **yes** |
  | Effusion | 0.890 | 0.934 | 0.815 | **yes** |
  | Synovitis | 0.803 | 0.794 | 0.711 | **yes** |
  | Baker's | 0.864 | 0.944 | 0.862 | **yes** |
  | Contusion | 0.839 | 0.880 | 0.731 | **yes** |
  | Fracture | 0.861 | 0.820 | 0.747 | **yes** |
  ACL/MCL are the only labels where reports still beat the image model — expected, reports state
  ligament tears explicitly and unambiguously most of the time. Full table:
  `artifacts/exp002_eval_report.csv` (`scripts/eval_exp002_report.py`). **Decision: proceed to
  submission-notebook prep (step 4) and launch the scale ladder (step 5) per the 2026-09-08
  housekeeping instructions; Phase 4 image-blend of the 10 flagged labels is a later round, not
  now.**
- 2026-09-08: **Scale ladder launched.** `configs/exp003_dinov2b_slot_v2labels.yaml` (DINOv2
  ViT-B/14, `vit_base_patch14_dinov2.lvd142m`, SLURM job 24503) and
  `configs/exp004_dinov3s_slot_v2labels.yaml` (DINOv3 ViT-S/16, `vit_small_patch16_dinov3.lvd1689m`
  — confirmed via `timm.list_models` that DINOv3 ships patch16 only, not patch14 like DINOv2, an
  unavoidable side effect of comparing model families rather than a deliberate size change; SLURM
  job 24504) — both otherwise identical to exp002 (same labels_v2_final, folds, stability bundle,
  slots, aug). `SlotModel` derives `embed_dim` from `backbone.num_features` at runtime, so no code
  changes were needed to swap backbones. Cluster is heavily shared right now (many other users'
  jobs across all 6 H100 nodes); exp003 got 4/5 folds running immediately, fold 4 and all of
  exp004 queued on `QOSMaxGRESPerUser` (per-user GPU cap) — expected to start as slots free, not
  an error. **Self-inflicted race condition caught and fixed**: `sbatch` was run before
  `scripts/push.sh` synced the new config YAMLs to HPC, so array tasks that started in that
  window (24503_0, 24503_2, and all of 24504_0-4) hit `FileNotFoundError` on
  `configs/exp00{3,4}_*.yaml` within ~1 minute and failed, while tasks that happened to start
  slightly later (after an unrelated later push landed) succeeded — a pure timing accident, not a
  cluster problem. Resubmitted the failed folds only (`sbatch --array=0,2` for exp003, full array
  for exp004, jobs 24514/24515) after confirming the configs were present on HPC; both confirmed
  running past the point of the original failure. **Lesson: always push before sbatch, no
  exceptions, even for "just a config file."**
- 2026-09-08: **Real bug found and fixed: exp004 (DINOv3-S) crashed instantly on all 5 folds**
  with `AssertionError: Input height (378) should be divisible by patch size (16)` inside timm's
  `patch_embed`. Root cause: `src/rsna_knee/dataset.py`'s `_nearest_patch_multiple` hardcoded
  `patch=14` regardless of the actual configured backbone — fine for DINOv2 (patch14, 384->378,
  divisible by 14) but silently wrong for DINOv3 (patch16, 378 is NOT divisible by 16). Fixed with
  `_patch_size_for_backbone()` (parses `patchNN` out of the timm model name) threaded through
  `RSNAKneeDataset.__init__`, `src/rsna_knee/infer_core.py`, and `scripts/infer.py` — DINOv3-S now
  correctly gets `out_size=384` (24*16), DINOv2 backbones unchanged at 378. Resubmitted exp004's
  full array (job 24522) after verifying the fix on both backbone name patterns directly.
  **This would have silently produced NOTHING (a crash, not a bad number) rather than a subtle
  metric regression — caught immediately, but worth remembering that "same everything else" scale
  ladders can still hide a config-shape assumption the original architecture never had to make.**
- 2026-09-08: **Kaggle submission notebook — multi-round debugging to first real run.**
  `notebooks/submit-exp002.ipynb` needed 7 kernel-push iterations before producing a real
  (non-prior-fallback) prediction, each a genuine distinct bug rather than flakiness:
  (1)/(2) `rsna-knee-code` dataset attached but not yet "ready" when the kernel first ran ->
  `/kaggle/input/rsna-knee-code` didn't exist. (3) Kaggle's CURRENT input-mount layout nests under
  `/kaggle/input/{datasets,competitions}/<owner>/<slug>/`, not the old flat
  `/kaggle/input/<slug>/` — fixed by searching for the directory instead of hardcoding the path.
  (4) The uploaded code-dataset zip's extraction did not preserve the `rsna_knee/` wrapper folder
  (files ended up flat), breaking `from rsna_knee.dataset import ...` even with the right
  directory resolved — fixed by finding `__init__.py` wherever it actually is and registering that
  directory as the `rsna_knee` module via `importlib.util.spec_from_file_location`, independent of
  on-disk layout. (5) The code dataset was stale (uploaded before `_patch_size_for_backbone` was
  added during the exp004 fix) — re-uploaded via `kaggle datasets version`. (6) **Real one, not a
  packaging bug**: Kaggle assigned a **Tesla P100** GPU (compute capability sm_60); the installed
  PyTorch build's minimum supported architecture is sm_70, so `torch.cuda.is_available()` returned
  True but every forward pass threw `CUDA error: no kernel image is available for execution on the
  device`, caught by the per-study defensive fallback and silently producing all-label-prior
  predictions for all 3 studies — a submission that would have measured the prior baseline's AUC,
  not the model's. Fixed with an explicit capability check (`torch.cuda.get_device_capability`
  against `torch.cuda.get_arch_list()`'s minimum, since CUDA/PTX compatibility is forward-only)
  that falls back to CPU when the visible GPU is too old, rather than trusting
  `is_available()` alone. **This GPU-generation variance is a real risk for the actual hidden-test
  scoring run too** (Kaggle can hand out P100 depending on quota/availability), not just this
  probe — CPU fallback is slower but the wall-clock guard already degrades gracefully if it runs
  long, so no separate handling was added beyond the capability check itself.
- 2026-09-08: **exp002 crash recovery had two confounded candidate fixes, applied together —
  flagging so neither gets over-credited.** The 2026-09-07 crash investigation applied (a)
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (CUDA fragmentation hypothesis, motivated by
  fold0 dying solo with no co-located job) and, in the same recovery window, (b) freed ~44GB of
  disk headroom (deleted diagnostic-run dirs + the unused Llama HF cache) after `scripts/push.sh`
  hit "Disk quota exceeded" mid-recovery. Both landed before the successful resubmit, so we cannot
  cleanly separate "fragmentation fix worked" from "quota pressure itself was contributing to
  instability" (e.g. failed/incomplete writes under near-100% quota). All 5 folds ran stable past
  1h+ afterward with no repeat of the exit 120:0 / 0:53 signatures, which is consistent with (a)
  but doesn't rule out (b) as a contributor. **Checkpoint retention + pre-flight quota guard added
  to `scripts/train.py`** (`quota_headroom_gb()`, `safe_torch_save()`, skips a checkpoint write and
  logs loudly rather than crashing if BeeGFS headroom drops below 15GB) so a repeat of the quota
  emergency degrades gracefully instead of corrupting a checkpoint or killing a job. Audited actual
  checkpoint footprint: each fold already writes exactly one `checkpoint.pt` (457MB, overwritten
  every epoch, not accumulated per-epoch) — so there was no retroactive per-epoch cleanup needed on
  exp002's artifacts (0 files deleted, 0GB freed; the earlier ~44GB recovered was diagnostic-run
  dirs + HF cache, not checkpoints). Added a lean `best.pt` (model+EMA only, no
  optimizer/scheduler state, ~3-4x smaller than `checkpoint.pt`) written once at end of training —
  this is what `notebooks/submit-*.ipynb` should load, not the full resume checkpoint.
