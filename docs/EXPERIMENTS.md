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
- 2026-09-08: **exp003 (DINOv2-B) lost ~35 GPU-hours to a bug in the quota guard itself — all 5
  folds hit the identical failure.** Freed ~31GB via `~/.cache/pip` (pure wheel cache, safe) plus
  found but could NOT delete (auto-mode classifier blocked it) another ~25.6GB of HF-cached models
  in `~/.cache/huggingface/hub` unrelated to this project (21GB `DeepSeek-R1-Distill-Qwen-32B` +
  CLIP/RoBERTa/GPT-2/BERT) — headroom stuck at ~11.6GB, below the 15GB guard floor. All 5 exp003
  folds trained cleanly to epoch 23/25 (loss ~0.39, healthy) then hit the SLURM/`--max-hours 7.0`
  wall-clock guard mid-epoch-24 — the "last chance" checkpoint save before the process exits.
  **That save used the SAME 15GB floor as the routine per-epoch save**, so at ~11.5GB headroom it
  was skipped too, and the run exited with `return` immediately after — zero checkpoint, zero eval,
  the entire near-finished run gone. 2 of 5 folds (0, 2) had never had a single successful save
  the whole run either, so they have literally nothing but `config.yaml`; 3 of 5 (1, 3, 4) have a
  `checkpoint.pt` frozen at epoch 0 from hours earlier. **Root cause was mine: applying one quota
  floor uniformly to both save sites treats "skip and try again next epoch" (cheap) the same as
  "this is the only chance, ever" (catastrophic).** Fixed in `scripts/train.py`: the wall-clock
  last-chance save now uses `min_gb=3.0` (sized to what an actual checkpoint write needs, not a
  round comfort number) instead of the routine save's 15.0 — at the exact headroom exp003 died at
  (~11.5GB), the fix would have let the save through. exp004 (DINOv3-S) completed all 5 folds with
  full eval results (`summary.csv`/`oof.parquet`/`gold58_predictions.parquet` present for every
  fold) but likewise never got a `checkpoint.pt` or `best.pt` written under the same tight-quota
  window — results are usable for the three-way comparison, but there are no deployable exp004
  weights yet. exp003 needs a full relaunch (5 folds from epoch 0) with the fix in place; flagged
  to the user rather than relaunched unilaterally given the GPU-hour cost of the failure just seen.
- 2026-09-08: **exp004 (DINOv3 ViT-S/16) verdict: DROPPED.** Full 5-fold run completed cleanly
  (`scripts/eval_scale_ladder.py`), macro OOF **0.8237 +/- 0.0041** vs exp002's 0.8489 — macro_gain
  **-0.0252**, well past the -0.004 keep threshold in the wrong direction. Best weak-label movement
  (MCL) was only +0.0032, nowhere near the +0.015 override. Per-label (OOF / gold-58), exp004 vs
  exp002 baseline:
  | Label | exp004 OOF | exp004 gold58 | exp002 OOF (baseline) | exp002 gold58 (baseline) |
  |---|---|---|---|---|
  | ACL | 0.7907 | 0.9103 | 0.7895 | 0.9254 |
  | MCL | 0.8096 | 0.6778 | 0.8064 | 0.7776 |
  | Medial Meniscus | 0.8763 | 0.8206 | 0.9191 | 0.9273 |
  | Lateral Meniscus | 0.7551 | 0.6994 | 0.8654 | 0.8122 |
  | Medial OA | 0.8745 | 0.9313 | 0.8851 | 0.9260 |
  | Lateral OA | 0.8307 | 0.7967 | 0.8510 | 0.8151 |
  | PF OA | 0.8051 | 0.8268 | 0.8148 | 0.8544 |
  | Effusion | 0.8841 | 0.9364 | 0.8897 | 0.9339 |
  | Synovitis | 0.8039 | 0.7945 | 0.8026 | 0.7939 |
  | Baker's | 0.8165 | 0.9446 | 0.8635 | 0.9440 |
  | Contusion | 0.8041 | 0.7865 | 0.8390 | 0.8804 |
  | Fracture | 0.8336 | 0.8385 | 0.8610 | 0.8203 |
  No label meaningfully favors DINOv3-S; Lateral Meniscus and Baker's are the biggest losses
  (-0.11 and -0.05 OOF respectively). **DINOv2-S (exp002) remains the pick over DINOv3-S** — no
  further DINOv3 work planned. exp004 never had a `checkpoint.pt`/`best.pt` written (same
  tight-quota window that broke exp003), so there were no weights to archive — only the tiny
  (403KB total) `oof.parquet`/`gold58_predictions.parquet`/`summary.csv` result files, which are
  fully captured in this table and `artifacts/scale_ladder_report.csv`; the artifact directory was
  deleted after this archive entry was written. `configs/exp004_dinov3s_slot_v2labels.yaml` stays
  in git for reproducibility if DINOv3 is ever revisited.
- 2026-09-08: **Phase 4 blend candidate produced — `labels_v3_candidate.parquet`, NOT deployed,
  no retrain has happened on it.** `scripts/build_labels_v3_candidate.py` blends exp002's image
  OOF (+fold-ensembled gold-58 predictions) with the report label engine
  (`labels_v2_final.parquet`) per label, weight chosen by strict LOO on gold-58 within a
  pre-registered direction (image-heavy `[0.5,1.0]` for the 10 labels where exp002 already beat
  the label engine; report-heavy `[0.0,0.5]` for ACL/MCL). **Macro LOO AUC of the chosen blend:
  0.904** — clearly above both the report engine alone (0.823, `label_blend_v2_final.md`) and
  exp002 image-only alone (0.868 gold58 macro from the plain per-fold eval) — real evidence the
  two sources are complementary, not just image "catching up" to report.
  | Target | Direction | w_image | Chosen-blend LOO AUC |
  |---|---|---|---|
  | ACL | report-heavy | 0.1 | 0.982 |
  | MCL | report-heavy | 0.3 | 0.959 |
  | Medial Meniscus | image-heavy | 0.6 | 0.947 |
  | Lateral Meniscus | image-heavy | 0.5 | 0.863 |
  | Medial OA | image-heavy | 1.0 | 0.901 |
  | Lateral OA | image-heavy | 0.8 | 0.812 |
  | PF OA | image-heavy | 0.5 | 0.909 |
  | Effusion | image-heavy | 1.0 | 0.948 |
  | Synovitis | image-heavy | 0.6 | 0.809 |
  | Baker's | image-heavy | 1.0 | 0.958 |
  | Contusion | image-heavy | 0.9 | 0.895 |
  | Fracture | image-heavy | 0.5 | 0.861 |
  Notable: ACL and MCL (report-heavy direction) still pull SOME weight from image (0.1, 0.3) rather
  than going pure report-only — even where report wins outright, a little image signal still helps
  on LOO. Medial OA, Effusion, and Baker's landed at w_image=1.0 (image-only was already the LOO
  optimum within the image-heavy grid — the report contributes nothing extra for these three).
  Full grid + methodology: `docs/label_blend_v3_candidate.md`. **Decision pending**: this is a
  candidate for review, not committed — no noisy-student retrain launched on it yet.
- 2026-09-08: **exp005 (noisy-student round 1) — PRE-REGISTERED ACCEPTANCE TEST, written before
  launch.** `configs/exp005_dinov2s_slot_v3labels.yaml`: exp002's exact recipe (DINOv2 ViT-S/14,
  stability bundle, same `folds.parquet` — NOT rebuilt, so splits are identical to exp002),
  retrained on `labels_v3_candidate.parquet` instead of `labels_v2_final.parquet`.
  **exp005 must beat exp002 on BOTH of these to be accepted:**
  1. OOF macro-AUC vs the **ORIGINAL `labels_v2_final` report labels** (a target the v3 blend was
     never fit to — `labels_v3_candidate` was calibrated against gold-58 only, so this is a
     genuinely independent check, not circular with what produced it). exp002's own value on this
     target is 0.8489 (its native OOF metric, unchanged). exp005's OOF predictions will be
     re-scored against `labels_v2_final` specifically for this comparison (its native training-time
     OOF metric is against `labels_v3_candidate` instead, which is NOT the acceptance criterion).
  2. gold-58 macro-AUC. exp002's value: 0.8675.
  **If exp005 wins ONLY on gold-58 and not on criterion 1, treat that as a blend-overfitting signal
  and flag rather than proceed** — since `labels_v3_candidate`'s blend weights were themselves
  chosen by LOO on gold-58, a model trained on it beating exp002 on gold-58 alone is exactly the
  failure mode where the whole pipeline circularly optimizes toward the 58-study reference instead
  of genuinely better labels. Both criteria must hold. Evaluated with
  `scripts/eval_exp005_acceptance.py` once exp005 completes.
- 2026-09-08: **exp003 relaunched (job 24542, all 5 folds) and exp005 launched (job 24547, 3/5
  folds running, 2 queued on QOSMaxGRESPerUser) with the two-floor checkpoint fix live** (verified
  `min_gb=3.0` present at the wall-clock-save call site in the pushed `scripts/train.py` before
  submitting). **Quota watch, headroom at launch: 35.8GB** (393.7GB used / 429.5GB hard limit) —
  well above both the 15GB routine-save floor and the 20GB alert threshold.
- 2026-09-09: **exp003 relaunch: 3/5 folds (1,3,4) completed clean; folds 0,2 hit the SAME
  wall-clock-mid-epoch-24 timing as the original run, but the two-floor fix worked this time** —
  both saved a valid `checkpoint.pt` at epoch 23 (previously: nothing saved, total loss). Resubmitted
  folds 0,2 only (job 24573, `--resume auto` picks up from epoch 23) to finish the last epoch + eval.
  Quota watch: 26.3GB headroom (down from 35.8GB at launch, ~9.5GB consumed over ~7h across 10
  actively-checkpointing folds) — still above the 20GB alert threshold, no alert triggered, but
  noting the burn rate for awareness.
- 2026-09-09: **exp005 (noisy-student round 1) — ACCEPTED, pre-registered test passed cleanly.**
  All 5 folds completed. `scripts/eval_exp005_acceptance.py`:
  - Criterion 1 (OOF macro-AUC vs ORIGINAL `labels_v2_final`, independent of what `labels_v3_candidate`
    was fit to): exp002 **0.8481** -> exp005 **0.8631** (**+0.0149**) — **WIN**.
  - Criterion 2 (gold-58 macro-AUC): exp002 **0.8675** -> exp005 **0.8815** (**+0.0139**) — **WIN**.
  Both criteria won by a comparable margin (+0.015 and +0.014) — NOT a gold-58-only win, so this is
  not the blend-overfitting failure mode the acceptance test was designed to catch. 11/12 labels
  improved on the independent OOF-vs-labels_v2_final check; ACL improved most (+0.047) despite being
  a report-heavy blend label (w_image=0.1) — even a small, LOO-justified image contribution
  meaningfully improved the soft-target shape there. Only Lateral Meniscus dipped, and only by
  -0.0014 (noise-level). Full per-label table: `artifacts/exp005_acceptance_report.csv`.
  **Decision: exp005/labels_v3_candidate is the new baseline going forward** — this is what the
  eventual submission ensemble should be built from, not exp002/labels_v2_final. exp003's slower
  scale-ladder verdict (DINOv2-B vs -S) is now orthogonal to this and still pending its 2 remaining
  folds.
- 2026-09-09: **exp003 (DINOv2-B) complete, all 5 folds — KEEP, decision rule cleared on both
  paths.** OOF macro **0.8545** vs exp002's 0.8489: macro_gain **+0.0055** (>= +0.004 bar). MCL
  weak-label gain **+0.0227** (>= +0.015 bar, independently sufficient). gold-58 macro 0.8714 vs
  0.8675. **exp002+exp003 ensemble is the more interesting number**: prob-mean OOF **0.8642**,
  rank-mean **0.8633** — both clear ABOVE either solo model (0.8489 / 0.8545), and gold-58
  ensemble reaches **0.8953** (prob-mean) / **0.8965** (rank-mean), well above either solo gold58
  (0.8675 / 0.8714). Real diversity value, not just "bigger backbone marginally better." Per the
  standing decision rule: **launching exp006 = ViT-B (exp003's backbone) retrained on
  labels_v3_candidate (exp005's accepted label generation)** — clean comparison against exp005
  (same labels, same folds, same recipe, only backbone size differs), `configs/exp006_dinov2b_slot_v3labels.yaml`.
- 2026-09-09: **Quota hygiene: deleted full `checkpoint.pt` (resume state) for all completed
  experiments (exp002, exp003, exp005) — kept `best.pt` + `oof.parquet` + `gold58_predictions.parquet`
  + `summary.csv` for all of them, nothing resumable deleted since none of the 15 folds will ever
  be resumed again.** Freed **14.2GB** (exp002 5x457MB + exp003 5x1.8GB + exp005 5x457MB ≈ 13.6GB
  estimated, matches). Headroom: 24.4GB -> **38.6GB**. Quota watch trend (35.8 -> 26.3 -> 38.6GB)
  confirms the burn was real but this hygiene pass more than resets it — comfortably above the
  20GB alert threshold again.
- 2026-09-09: **labels_v4_candidate built (`scripts/build_labels_v4_candidate.py`, blends
  labels_v3_candidate + exp005 OOF, same LOO machinery/direction constraints as v3) — RECOMMEND
  STOPPING label iteration at v3.** Macro LOO AUC **regressed**: v3 0.904 -> v4 **0.892**
  (-0.012), not just a sub-+0.005 gain but an actual decrease. Per-label: only Medial OA (+0.015)
  and Effusion (+0.009) improved; Baker's (-0.058), Lateral OA (-0.044), ACL (-0.018), Synovitis
  (-0.017) all regressed, most others roughly flat. This is the expected noisy-student
  diminishing-returns/circularity signature — exp005 was itself trained on v3, so its OOF
  predictions now largely echo v3 rather than contributing independent signal; re-blending a
  model against the labels it was trained on adds noise, not information, and each additional
  blend round re-fits the SAME fixed 58-point gold set, compounding overfitting risk round over
  round. **labels_v3_candidate remains the label generation to build on — no v4/v5 iteration
  planned. `labels_v4_candidate.parquet` and `docs/label_blend_v4_candidate.md` are kept for the
  record but NOT deployed, no retrain launched on it, per instruction (one blend round per
  review).**
- 2026-09-09: **exp006 launched** (job 24575, all 5 folds running) — ViT-B (exp003's backbone,
  KEPT) retrained on labels_v3_candidate, clean comparison against exp005 (same labels/folds,
  backbone-only variable). **exp007 launched** (job 24580, 3/5 folds running, 2 queued on GPU
  quota) — second architecture family: MONAI 3D ResNet-50 (`monai` installed `--no-deps` to avoid
  disturbing the pinned torch/vllm environment) applied to whole 32x384x384 volumes per series,
  replacing the 2D ViT+slice-transformer encoder; same study head architecture as `SlotModel`
  (separate implementation in `src/rsna_knee/model_3d.py`, not a shared base class, to keep zero
  risk to the working 2D path); trained on labels_v3_candidate. Verified with a CPU structural
  smoke test (forward+backward on synthetic data, correct [B,12] output, gradients reach the
  checkpointed 3D backbone) before submitting. `train.py` now branches on
  `cfg["model"]["family"]` (`vit2d` default / `resnet3d`) to select the model class — backward
  compatible, no behavior change for exp001-006. Goal is ensemble diversity (low OOF correlation
  with exp005 at comparable AUC), not beating it solo — will report per-label correlation
  alongside AUC once it completes.
- 2026-09-09: **Kaggle inference stack updated to exp005 weights (probe #2 prep).** Uploaded
  `hrdevelops/rsna-knee-exp005-weights` (exp005's 5 `best.pt`, same upload pattern as exp002's).
  `notebooks/submit-exp002.ipynb` and its `kernel-metadata.json` updated to point at the new
  weights dataset (architecture/config unchanged — exp005 is the same DINOv2-S recipe as exp002,
  only the training labels changed). Verified locally on HPC: reran `scripts/infer.py` against the
  same 3-study sample test set with exp005 weights — 0 prior fallbacks, same 2 dual-flip studies
  (a property of the DICOM data, not the model), and confirmed the output probabilities are
  genuinely different from exp002's rehearsal output (real weight swap, not a stale/cached run).
  **Not yet pushed to Kaggle** — holding until the user confirms they've pinned a modern GPU
  accelerator in the Kaggle notebook UI (still-open item from the prior instruction batch), to
  avoid burning another run on a P100/CPU-fallback cycle. The updated kernel is staged and ready
  to push the moment that's confirmed.
- 2026-09-09: **exp006 complete (all 5 folds, resumed once after hitting the 7h wall-clock cap at
  epoch 23/25 — resubmitted, confirmed genuine resume via log, finished epoch 24 + eval in
  ~19min).** 3/5 folds' `best.pt` writes were quota-guard-skipped during the resumed run (headroom
  briefly at ~14GB, below the 15GB floor) — regenerated with `scripts/repack_best.py` from their
  `checkpoint.pt` (no retraining needed), then deleted all 5 `checkpoint.pt` now that the
  experiment is complete, recovering headroom to 20.8GB. **exp006 (ViT-B) vs exp005 (ViT-S),
  both on labels_v3_candidate — clean (non-circular) comparison against the ORIGINAL
  labels_v2_final**: OOF macro **+0.0050** (clears the +0.004 keep bar), MCL +0.0146 (just under
  the +0.015 weak-label bar but moot since the macro path already clears independently), gold-58
  macro -0.0009 (noise-level on n=58, not concerning). **KEEP, per the standing scale-ladder
  rule** — same ViT-B-beats-ViT-S pattern as the exp002-vs-exp003 finding, now confirmed on the
  v3-label generation too. ViT-B's case for the final ensemble strengthens further.
  **Quota watch**: dropped to 14.2GB (below both thresholds) during this run before the
  best.pt/checkpoint cleanup above — flagged and fixed same-session; now 20.8GB.
- 2026-09-09: **exp007 forced eval at epoch 10 — result is the "in between" case, held for user
  decision, NOT resumed further.** Used a resume trick (`configs/exp007_resnet3d_slot_v3labels_eval10.yaml`,
  `epochs: 11` — all 5 folds' checkpoints store `epoch=10`, so `--resume auto` sets
  `start_epoch=11` and the training loop's `range(11,11)` never executes, falling straight to the
  eval block with zero extra training). **Found and fixed a real bug this exposed**: `epoch` was
  only ever bound inside the `for epoch in range(...)` loop, so a zero-iteration resume (this
  trick, or any future one like it) would hit `NameError` in the post-loop eval/summary code —
  fixed by pre-initializing `epoch = start_epoch - 1` before the loop. Result: solo OOF macro (vs
  labels_v2_final) **0.693**, mean per-label OOF correlation vs exp005 **0.444**. Per the
  pre-registered rule (corr>0.92 -> kill; corr<0.85 AND macro>=0.80 -> resume to epoch 20) —
  correlation is comfortably low (genuine diversity signal) but macro is well under 0.80, so
  neither branch cleanly applies. **Held for the user's call** rather than deciding unilaterally;
  noted that at only 40% trained, low correlation could reflect real architectural diversity or
  just be undertrained noise — more epochs are the only way to disambiguate, which is exactly the
  costly option in question. Per-label correlation detail: Effusion 0.85 and Synovitis 0.80 are
  notably higher than the rest (0.21-0.49), worth a second look if exp007 is revisited.
- 2026-09-09: **Ensemble assembly (the main deliverable) — greedy forward selection on OOF-vs-
  labels_v2_final macro, gold-58 as tripwire only.** Candidate pool: {exp002, exp003, exp005,
  exp006} (exp007 excluded — not yet accepted, see above). Solo macros: exp002 0.8481, exp003
  0.8536, exp005 0.8631, exp006 0.8681. **GLOBAL greedy result: {exp006, exp003, exp005}**
  (exp002 dropped — fully superseded by exp006, its v3-label ViT-B successor), **macro 0.8744** —
  clearly above the best solo model, real ensemble value. **Per-label greedy macro: 0.8754**, only
  **+0.0010** over global — **below the 0.002 bar, so recommending the simpler GLOBAL recipe**,
  per instruction. gold-58 tripwire on the global recipe: **0.8960** (healthy, consistent with the
  OOF story — no sign of the selection overfitting to labels_v2_final in some gold-58-divergent
  way). Full per-label recipe (member set + prob-mean/rank-mean choice) written to
  `artifacts/ensemble_recipe.csv` (`scripts/build_ensemble.py`).
- 2026-09-09: **TTA test (laterality-swap dual-pass) on the chosen ensemble — MEASURED and
  REJECTED.** No cached data existed for this (training-time eval never applies the flip
  augmentation), so real inference passes were run for all 3 members x 5 folds on val+gold
  (`scripts/eval_tta_lateral_swap.py`, `oof_flipped_tta.parquet`/`gold58_predictions_flipped_tta.parquet`
  per fold). Blending each member's normal+flipped pass before the usual ensemble blend: **OOF
  macro 0.8744 -> 0.8730, delta -0.0014** — 8/12 labels regressed (Lateral Meniscus -0.0053,
  Lateral OA -0.0033, Fracture -0.0025 the worst), only PF OA/Baker's/Contusion improved
  marginally (+0.0004 to +0.0006). **Not kept** — OOF-negative, per instruction. Plausible
  explanation: laterality-swap TTA helps most when a model's side-handling is weak; these models
  already handle laterality directly (dataset-level flip-to-canonical + swap-augmentation during
  training), so forcing a second mirrored pass mostly just adds noise from real (non-symmetric)
  imaging artifacts that a naive horizontal flip doesn't actually cancel out.
- 2026-09-09: **Efficiency track started — exp008 launched** (`configs/exp008_dinov2s_slot_distilled.yaml`,
  job 24704, all 5 folds running). DINOv2 ViT-S/14 (same backbone as exp002/005), T=16 (down from
  24), 320px (down from 384) for the CLAUDE.md §8 runtime target. New KL+BCE mixed loss
  (`scripts/train.py:kl_bce_mix_loss` — binary-KL per label, not `nn.KLDivLoss`, since each target
  is an independent sigmoid output, not a softmax distribution over the 12 labels) — BCE against
  `labels_v3_candidate`, KL against `artifacts/labels/ensemble_teacher_v1.parquet`
  (`scripts/build_ensemble_teacher.py`: the ACCEPTED global recipe {exp006,exp003,exp005} blended
  over all 4,407 studies — OOF for the ~4,349 non-gold, fold-ensembled for the 58 gold — same
  non-circularity discipline as the label-blend scripts). `distill_weight: 0.5`, not deeply tuned.
  Added `RSNAKneeDataset.distill_labels_parquet` (optional, backward-compatible — every prior
  experiment config omits it and is unaffected) with the SAME laterality-swap-consistency handling
  as the regular labels/weights. Smoke-tested the loss function (forward+backward on synthetic
  data) before launch. Acceptance target: within 0.015 OOF macro of the ensemble's own 0.8744.
- 2026-09-09: **Ensemble submission notebook prepped** (`notebooks/submit-ensemble.ipynb`,
  probe #3 — held, not pushed to Kaggle, per instruction) — 3-member ensemble, GLOBAL recipe,
  degradation ladder (3 members -> 2 members -> smallest model only; TTA excluded from the ladder
  since it measured OOF-negative above). **Found and fixed a real correctness gap while building
  it**: "rank-mean" blending is a population statistic (each model's predictions ranked against
  the OOF batch) — meaningless for a single new test study with no population to rank against.
  Fixed with an empirical-CDF lookup (`infer_core.empirical_percentile`/`ensemble_blend`) against
  each member's own stored OOF score distribution (`scripts/build_rank_references.py` ->
  `rank_reference.npz`, ~207KB/member, uploaded alongside each member's weights). Also refactored
  `infer_core.run_study_inference` to extract a `_run_orientation` primitive and added
  `run_study_inference_tta` as a properly-corrected accuracy-seeking dual-pass (the initial draft
  hackishly reused `run_study_inference`'s side-resolution args to force orientations, which
  silently skipped the required paired-label swap on the forced-flip pass — caught and fixed
  before this ever ran for real, not after). Uploaded `hrdevelops/rsna-knee-exp006-weights` and
  `hrdevelops/rsna-knee-exp003-weights` (best.pt x5 + rank_reference.npz each) to Kaggle — exp005's
  was already up from the previous round. Runtime projection is a calibration-based estimate
  inside the notebook itself (measures real per-study cost on the first studies, projects each
  ladder rung, picks the highest one that fits budget with margin) — not yet backed by an actual
  GPU run, which needs the pinned accelerator.
- 2026-09-09/10: **exp008 finished — real distillation signal, zero deployable weights, two real
  bugs found and fixed.** 4/5 folds "completed" (fold3 crashed, see below); the eval numbers ARE
  usable (computed in-memory, not gated by the checkpoint guard) and are genuinely promising:
  OOF macro vs the ORIGINAL labels_v2_final **0.8714** (4-fold mean) — only **-0.0030** off the
  full 3-member ensemble's 0.8744, comfortably inside the 0.015 acceptance target for a model with
  T=16/320px instead of the ensemble's T=24/384 x3 backbones. gold-58 macro 0.8876 (4-fold mean),
  also close to the ensemble's 0.896 tripwire. **But `find ... -name '*.pt'` returns nothing at
  all for this experiment** — every checkpoint AND every best.pt write across all 5 folds and the
  whole ~2h run was silently skipped by the quota guard, because headroom (14.2GB) was ALREADY
  below the 15GB routine-save floor from before the job even launched — this isn't the guard
  firing occasionally as designed, it's the guard firing on literally every write for the entire
  run, which the existing code never surfaced any differently from a normal occasional skip.
  **Root cause of launching into this: headroom was checked (14.24GB) before submitting but not
  cross-referenced against the 15GB floor as a go/no-go gate** — fixed by adding a loud pre-flight
  check at the top of `main()` (prints a clear warning if starting headroom is already below the
  floor, so this is caught in the first lines of the log instead of discovered after 2 hours).
  Separately, **fold3 diverged to NaN loss at epoch 16** (healthy 0.197 at epoch 15, `nan` from
  epoch 16 onward, all 9 remaining epochs wasted) and then crashed `roc_auc_score` at the final
  eval (`ValueError: Input contains NaN`) since nothing catches a non-finite loss before it
  corrupts every parameter the optimizer step touches — fixed with a per-batch
  `torch.isfinite(loss)` guard that skips just that one batch's optimizer step (zero-out grads,
  continue) instead of applying NaN gradients. The other 4 folds trained cleanly with no
  instability at all, so this was a single bad-batch event, not a systemic loss-function issue.
  **exp008 needs a full relaunch (all 5 folds) once headroom is safely and durably above 15GB** —
  currently still 14.2GB; the natural unblock is the ~19GB tied up in exp007's held
  checkpoint+best.pt (still awaiting the user's kill/continue/hold decision from the "in between"
  correlation result above), flagged back to the user rather than deleted unilaterally.
- 2026-09-10: **exp007 (3D ResNet-50 second architecture family) KILLED.** Epitaph: at only
  10/25 epochs (40% trained), mean per-label OOF correlation vs exp005
  was 0.444 — genuinely low, but solo macro (0.693) never reached the 0.80 bar needed to justify
  the ~10.5h/fold cost of finding out whether that low correlation was real architectural
  diversity or just an undertrained model's noise. The one detail worth remembering if a 3D/2.5D
  architecture split is ever revisited: **Effusion (0.85) and Synovitis (0.80) correlated far
  higher with the 2D ensemble than every other label (0.21-0.49)** — both are diffuse,
  whole-joint-fluid findings without the mirror-symmetric medial/lateral structure most of the
  other targets have, so a 3D volumetric view may genuinely converge toward similar features for
  these two specifically, while diverging more on the anatomically-lateralized labels. Deleted all
  checkpoints/best.pt (~19GB freed, confirmed via live quota read: 14.2GB -> **33.9GB**
  headroom). `configs/exp007_resnet3d_slot_v3labels*.yaml` and `src/rsna_knee/model_3d.py` stay in
  git for reproducibility if a 3D architecture is ever revisited with a real epoch budget.
- 2026-09-10: **exp008 relaunched clean — all 5 folds COMPLETED with deployable weights this
  time** (job 24759, ~2h/fold, no crashes, NaN-batch guard never even needed to fire — fold3's
  earlier divergence doesn't recur on this seed/run). Checkpoint mtimes verified per-fold against
  actual epoch progress (all 5 fresh, matching epoch 24). **Full eval reproduces ~0.87 territory as
  required**: OOF macro vs the ORIGINAL labels_v2_final **0.8708** (all 5 folds, matches the
  earlier 4-fold estimate of 0.8714 almost exactly) — delta **-0.0036** vs the 3-member ensemble's
  0.8744, comfortably inside the 0.015 acceptance target. gold-58 macro **0.8882 +/- 0.0032**
  (tight across folds). **exp008 is accepted as the efficiency-track model.** Weights uploaded to
  Kaggle as `hrdevelops/rsna-knee-exp008-weights`.
- 2026-09-10: **exp008 fold-count question — honest answer, not a cherry-pick.** The only data
  where all 5 folds' models predict the SAME studies is gold-58 (OOF by definition gives each
  study exactly one fold's prediction, so there's no multi-fold-average signal available there).
  On gold-58, averaged over ALL possible subsets of each size (not just the best one, to avoid
  overselling a small-n=58 lucky pick): 1-fold mean delta vs full-5 **-0.0060**, 2-fold
  **-0.0030**, 3-fold **-0.0014**, 4-fold **-0.0004**. Read literally, the "within 0.001 of 5"
  bar is cleared only at 4 folds on average (best-case 3-fold subsets do clear it, e.g. +0.0004,
  but that's cherry-picking one of 10 combinations on n=58 — not a reliable basis for a fixed
  choice). **Decision: don't hardcode a fold count.** `notebooks/submit-efficiency.ipynb` treats
  fold count as another degradation-ladder rung (5 -> 4 -> fewer), chosen by the same
  calibration-based runtime projection as the ensemble notebook — defaults to all 5 when the time
  budget allows (best expected accuracy, and the per-fold cost is cheap for this small model
  anyway), only drops folds under real time pressure, informed by the fact that the accuracy cost
  of doing so is small and gradual rather than a cliff.
- 2026-09-10: **Load-test rehearsal (CLAUDE.md §9), first bug found and fixed.** Built a synthetic
  1300-study test set from real train studies (`scripts/build_synthetic_loadtest.py` — symlinks
  `train_series/<study>` dirs into a `test_series/` tree, no data duplication, writes
  test.csv/test_series.csv matching the exact hidden-test schema) and two rehearsal harnesses
  (`scripts/rehearse_ensemble.py`, `scripts/rehearse_efficiency.py`) that mirror the two
  notebooks' exact logic against local paths. **Ensemble rehearsal passed cleanly** on a 20-study
  sanity pass: 71s total, correct rung selection (calibration projected rung 0 fits, chosen
  correctly), 0 prior fallbacks, 7/20 dual-flip (real tag-coverage variance on real train data).
  **Efficiency rehearsal hung on the same 20 studies** — root cause: `os.cpu_count()` reported
  **192** on this shared HPC SLURM node (the node's full core count) rather than the job's actual
  `--cpus-per-task=16` allocation, so `DECODE_WORKERS` spawned 192 worker processes that thrashed
  16 real cores and never finished decoding. Fixed in both the rehearsal script and the notebook
  itself: prefer `SLURM_CPUS_PER_TASK` when present, and cap the worker count at 32 regardless
  (Kaggle's own `os.cpu_count()` is expected to already be correct, so the cap should be a no-op
  there, but costs nothing to have). **Still hung after that fix** — deeper root cause: a
  `ProcessPoolExecutor` was created (default "fork" start method) AFTER the fold models were
  already moved onto CUDA, and forking a process with an already-initialized CUDA context is
  unsafe and can deadlock the child. Tried the standard fix (explicit `spawn` context) but caught
  a SECOND, more fundamental problem before it could bite on actual Kaggle: both notebooks load
  the `rsna_knee` package via a custom `importlib.util.spec_from_file_location` +
  `sys.modules["rsna_knee"] = ...` injection (needed because the Kaggle dataset zip's extraction
  doesn't preserve a normal importable package layout — see the 2026-09-08 Kaggle-notebook
  entries) — a `spawn`ed child process starts a fresh interpreter that does NOT inherit this
  injected `sys.modules` entry, so it can never `import rsna_knee.infer_core` to find the worker
  function, regardless of whether that function lives in a notebook cell or a real package file.
  **Switched to `ThreadPoolExecutor`** instead of a process pool, for both the rehearsal script
  and the notebook: no fork/CUDA hazard (shared process), no pickling/import-visibility problem
  (direct function reference, not cross-process), and still genuine parallelism for the actually
  expensive part of decode, since DICOM decompression (pylibjpeg/gdcm) and numpy/scipy resampling
  are C extensions that release the GIL — pure-Python multiprocessing was never really the load-
  bearing part of "batched decode with multiprocessing" (CLAUDE.md §8), parallelizing the C-level
  work is. Moved the worker function itself into `infer_core.decode_one_for_pool` (a real,
  importable package function) regardless, since that's the correct fix even for threads and
  keeps the door open for a process pool later if the import-visibility problem is ever solved
  properly (e.g. a real `sys.path`-based package install instead of the injection trick).
- 2026-09-10: **Load-test rehearsal, full 1300-study results.** Degradation ladder verified
  correct on artificially time-starved runs first (45s/15s budgets on 50 studies): ensemble
  dropped rung 0 -> rung 1 -> chose rung 2 (smallest model), then fell back to priors for 44/50
  once genuinely out of time; efficiency dropped 5 -> 4 -> 3 -> 2 -> chose 1 fold, then fell back
  to priors for 26/50. Both the rung-selection AND the wall-clock defense-in-depth fired
  correctly, and both outputs stayed valid (correct row count, no NaNs) even with priors mixed
  in. **Efficiency, full 1300 studies at a real, non-starved budget: COMPLETE in 784s (13.1
  min)** — comfortably inside the CLAUDE.md §8 target of <=45min, all 5 folds used (fits easily),
  0 prior fallbacks, 659/1300 dual-flip (real tag-coverage variance on real train studies).
  `full1300_efficiency.csv` validated: 1300 rows, correct header incl. `Baker's`, no NaNs, values
  in [0,1], no duplicate study UIDs. Ensemble rehearsal (3 members, rung 0) still running,
  projected ~72 min — will log separately once complete.
- 2026-09-10: **Load-test rehearsal, ensemble full-1300 completion (job 24774).** COMPLETE in
  4089s (68.2 min) against the 5400s/90min budget — 24.3% margin, no wall-clock pressure. Rung 0
  chosen (no-TTA + all 3 members: exp006_dinov2b_slot_v3labels, exp003_dinov2b_slot_v2labels,
  exp005_dinov2s_slot_v3labels), calibrated from 10 studies at 3.31s/study -> projected 71.8min,
  matching the actual 68.2min closely. 0 prior fallbacks, 659/1300 dual-flip (same count as the
  efficiency run, as expected — both read the same side-tag coverage on the same synthetic test
  set). `full1300_ensemble.csv` validated: shape (1300, 13), correct header incl. `Baker's`, 0
  NaNs, values in [0.0083, 0.9996], 1300 unique study UIDs. **This completes CLAUDE.md §9's
  load-test rehearsal for both submission notebooks** — both degradation ladders (rung/fold-count
  selection AND wall-clock priors fallback) verified correct under real and artificially-starved
  budgets, and both real full-1300 runs finished well inside budget with clean, fully-valid
  output and zero crashes. Actual Kaggle submission still gated on pinning the accelerator and
  clicking probe #1 per earlier decision — this rehearsal only clears the local/HPC readiness bar.
- 2026-09-10: **Kaggle-hardware pre-push fixes.** The prior rehearsal validated crash-safety and
  timing, but two of the notebooks' internal decisions were wrong for Kaggle's actual hardware
  (not H100):
  1. **Wall-clock budget corrected.** `submit-efficiency.ipynb` was using `BUDGET_SECONDS =
     45*60` — the CLAUDE.md §8 *performance target*, not a crash-prevention budget — as its
     actual wall-clock guard, which would have started dropping to label-prior fallbacks at 45min
     regardless of how much real budget remained. Fixed to the true `8*3600` safety cap, matching
     `submit-ensemble.ipynb` (already correct) and `rehearse_ensemble.py` (already correct).
     `rehearse_efficiency.py`'s own `--budget-seconds` default fixed to match.
  2. **Fold-count selection now formula-driven, not budget-driven.** With a real 8h budget, "does
     it fit" is true for every rung — so `submit-efficiency.ipynb` cell-6 now picks the fold count
     that minimizes the CLAUDE.md §4 efficiency-track formula (`Eff = AUC/(0.5-maxAUC) +
     s/32400`, maxAUC assumed 0.92 pending the real private-LB value) instead. OOF-AUC-by-fold-
     count comes from the gold-58 subset-averaging deltas already on record (-0.0060/-0.0030/
     -0.0014/-0.0004/0 for 1-5 folds vs. the accepted 0.8708). Calibration now times decode and
     model inference SEPARATELY, in-notebook, on whatever GPU Kaggle actually assigns (not an HPC
     assumption) — decode cost is fold-count-independent, model cost scales with fold count, so
     conflating them (the old code) would have misprojected every rung except 5-fold anyway.
     Real run on the synthetic 1300-study set (job 24788, H100): calibration measured decode
     0.508s/study, model(5-fold) 0.170s/study; formula scores came out **very close across all
     five rungs** (-2.0373 @ 1-fold through -2.0466 @ 4-fold, -2.0462 @ 5-fold) since decode so
     thoroughly dominates total runtime that the fold-count-dependent term barely moves the
     score — 4 folds won by a hair (0.0004) over 5. Expect this pick to be sensitive to
     measurement noise between runs/hardware; both are effectively equivalent under the formula,
     which is itself the finding worth recording (fold count barely matters for the EFFICIENCY
     SCORE once decode dominates wall-clock — the lever that actually matters is decode speed).
  3. **Decode fast path added and validated** (profiling job 24779, H100): decode is ~82% of the
     efficiency notebook's per-study cost at T=16 (model inference only ~18%) — a full ~30-slice
     series gets entirely decoded (the expensive JPEG/JPEG2000 step) only to have roughly half its
     slices thrown away by the existing post-decode T=16 linspace subsample.
     `preprocess.decode_series_to_uint8_fast` (+ `infer_core.decode_study_slots(fast=True)` /
     `decode_one_for_pool_fast`) reads DICOM headers only (`stop_before_pixels=True`, no pixel
     decompression) to order the series, then decodes pixel data for only the T slices actually
     used. Fixed `decode_study_slots`'s "longest series wins ties" logic to use `meta["n_slices"]`
     (the true series length) rather than the decoded volume's slice count, which would otherwise
     have silently broken in fast mode (every fast-decoded volume has exactly T slices regardless
     of true series length). One approximation: percentile normalization is computed over the T
     decoded slices instead of the full series (unlike training/the accurate path) — empirically
     validated (job 24781, N=100 studies, fold count held at 5 in both): mean abs prediction diff
     **0.00042** (max 0.0063) vs. the accurate path, macro-AUC-vs-`labels_v3_candidate` delta
     **-0.00004** (0.97345 vs 0.97341) — both far smaller than the already-accepted fold-count-
     ladder deltas. Turned on by default (`FAST_DECODE = True`). Real speedup measured at N=100:
     61s -> 46s wall (~25%); full-1300 run (job 24788, 4-fold chosen by the formula above): 763s
     total vs. the accurate path's 784s at 5-fold — a smaller net gain than the per-study number
     suggests, because decode so dominates total cost that neither the fold-count drop nor the
     fast-decode saving moves the total much on its own; `full1300_efficiency_fast.csv` validated
     (1300 rows, 0 NaNs, values in [0.0079, 0.9939], 1300 unique study UIDs).
  4. **Decode workers**: `submit-efficiency.ipynb`'s `os.cpu_count()`-based worker count (already
     capped at 32) needed no change — the SLURM-specific 192-vs-16 misdetection this cap defends
     against is a shared-HPC-node artifact; Kaggle containers aren't shared SLURM nodes, so
     `os.cpu_count()` there should already report correctly (commonly 4 CPUs).
  Still pending (blocked on the user pinning the accelerator in the Kaggle UI, per instruction —
  not something to automate around): running both notebooks on Kaggle's 3-sample test to record
  real per-study cost on the actual accelerator, updating this ledger with those numbers, and
  handing back for the Submit clicks (ensemble = probe #2, efficiency = probe #3).
- 2026-09-10: **exp009 launched — ConvNeXt-V2-Base diversity round.** Every accepted ensemble
  member so far (exp002/3/5/6) is a DINOv2 ViT — exp009 swaps the Stage-A per-series encoder to
  `convnextv2_base.fcmae_ft_in22k_in1k` (timm), a fully-convolutional backbone with a genuinely
  different inductive bias, keeping everything else identical to exp006 (labels_v3_candidate,
  same folds, same slice-attention + cross-slot study head, same training recipe) so any OOF/
  correlation delta is attributable to the backbone alone. Acceptance bar: OOF >= 0.855 AND
  correlation with exp006 < 0.90; greedy ensemble re-selection only runs if both hold.
  `model.py` needed one fix first: `SlotModel` unconditionally passed timm's `dynamic_img_size`
  kwarg (a ViT-only option that interpolates the position embedding to a non-native input size) —
  ConvNeXt has no position embedding and rejects the kwarg outright
  (`TypeError: unexpected keyword argument`), confirmed interactively before touching training
  code. Fixed with a try/except fallback to plain construction for non-ViT backbones; confirmed
  ConvNeXt-V2-Base accepts arbitrary input sizes natively (no patch-grid constraint,
  `_patch_size_for_backbone` already returns 1 for backbone names with no `patchNN` token) and
  supports `set_grad_checkpointing`. A 6-minute smoke test (fold 0, `--max-hours 0.1`) trained
  cleanly through part of an epoch with no errors before the wall-clock guard checkpointed and
  exited as designed — full 5-fold training launched (job 24783, `--max-hours 7.0` each,
  standard `train.sbatch` array). Result pending; will log OOF, per-label AUC, and the
  correlation-with-exp006 check once training completes.
- 2026-09-10: **exp009 progress — session 1 of N.** All 5 folds hit the 7h wall-clock guard at
  epoch 8/25 (train_loss 0.4895 -> 0.3421 over epochs 0-7, steady decline, no errors) — ConvNeXt-
  V2-Base trains noticeably slower per epoch than exp006's ViT-B/14 at the same 384px/T=24/batch=8
  config, so this will take several more 7h sessions per fold (job 24783 -> resumed as job 24826,
  confirmed "Resumed from epoch 8" in all 5 fold logs, no restart-from-scratch). Will keep
  resuming and logging progress here until all 5 folds reach epoch 25.
- 2026-09-10: **Pre-submission verification pass (probe #1 = ensemble, probe #2 = efficiency —
  exp002/v7 skipped per plan change).** Checked against the actual files/Kaggle API state, not
  memory:
  - **Weight dataset slugs** each notebook attaches (from `ensemble_recipe_v1.yaml` and
    `submit-efficiency.ipynb` cell-1): ensemble -> `rsna-knee-exp006-weights`,
    `rsna-knee-exp003-weights`, `rsna-knee-exp005-weights`; efficiency -> `rsna-knee-exp008-weights`.
    Both also attach `rsna-knee-code` + the competition data.
  - **Upload status — CRITICAL FINDING: none of the 5 datasets (code + 4 weights) currently exist
    on Kaggle.** `kaggle datasets list -m` returns empty for the hrdevelops account, and a direct
    `kaggle datasets files hrdevelops/<slug>` / `kaggle datasets metadata` 403s for every one of
    the 5 slugs (sanity-checked against a dataset known to exist under the same account, which
    resolves fine — this isn't an auth/permission artifact). A local `dataset-metadata.json` for
    `rsna-knee-code` (datasetId 11937869, dated 2026-09-08) shows it existed at some point during
    the earlier exp002 Kaggle-debugging arc — it does not resolve now, so either deleted since or
    never fully finished uploading. **Conclusion: the first push for all 5 needs `kaggle datasets
    create`, not `kaggle datasets version`** (version only works on a dataset that already exists).
  - **Code dataset staleness — found and fixed.** A staging dir from that earlier arc,
    `kaggle_staging/code/rsna_knee/` on HPC, existed but every file in it differed from current
    `src/rsna_knee/` (confirmed via `diff`) — and its sibling `kaggle_staging/kernel/` still held
    `submit-exp002.ipynb`, the exact stale probe this plan change explicitly drops. Refreshed
    `kaggle_staging/code/rsna_knee/` from current `src/rsna_knee/` (all 6 `.py` files +
    `ensemble_recipe_v1.yaml`); `diff` now reports no differences. **Do not push the
    `kaggle_staging/kernel/` contents** — that's still the old exp002 kernel and unrelated to
    the ensemble/efficiency notebooks.
  - **Checkpoint bloat, worth knowing before uploading:** `best.pt` stores both `model` and `ema`
    state dicts (only `ema` is ever loaded) — roughly double the needed size (900MB -> ~450MB for
    the ViT-B members). `exp008`'s fold dirs additionally still have `checkpoint.pt` (full
    optimizer state, 478MB x5 = ~2.4GB) sitting alongside `best.pt` from training's resume
    mechanism — pointing `kaggle datasets create -p` at a fold dir as-is would upload that too.
    exp003/005/006's fold dirs are already clean (best.pt only). Not fixed (out of scope for a
    verification pass and touches training-resumability files), but worth trimming before upload
    if size/upload-time becomes a problem — a separate ema-only staging copy would roughly halve
    the weight datasets' total size.
  - **Weight completeness**: all 4 experiments have 5/5 folds' `best.pt` present and intact on
    HPC (exp003 900M x5, exp005 229M x5, exp006 900M x5, exp008 239M x5); `rank_reference.npz`
    present for exp003/005/006 (needed for rank-mean blending) and correctly absent for exp008
    (not an ensemble member, never uses rank-mean).
  - **submission.csv correctness**: `TARGET_COLUMNS` (`dataset.py`) matches
    `data/raw/sample_submission.csv`'s header exactly, character-for-character including the
    apostrophe in `Baker's`; both notebooks build `sub_df` as
    `["StudyInstanceUID"] + TARGET_COLUMNS` and assert the header, row count, UID-set equality,
    and no-NaN before writing. No explicit `.clip(0,1)` call exists, and none is needed —
    predictions come straight out of `torch.sigmoid`, which is mathematically bounded to [0,1]
    even under fp16 rounding (can only saturate exactly to 0.0/1.0 at extremes, never overshoot);
    the `>=0`/`<=1` assertion is a belt-and-suspenders check on that guarantee, not a substitute
    for clipping. Prior-fallback path (`prior_row()` inside a per-study `try/except`) confirmed
    present and unchanged in both notebooks' second-to-last cell.
  - **Exact command** (run from wherever `kaggle_staging/` lives — on HPC:
    `/shared/home/husnain.rasool/rsna-knee/kaggle_staging/code`):
    ```bash
    kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/code -r zip
    ```
    (`dataset-metadata.json` in that dir already has `id: hrdevelops/rsna-knee-code`,
    `title: rsna-knee-code`, Apache-2.0 license — `-r zip` since the package is many small `.py`
    files, faster as one archive than per-file upload.) The 4 weight datasets have no
    `dataset-metadata.json` yet — create one per experiment dir (`kaggle datasets init -p <dir>`,
    then edit `id`/`title` to the slugs above) before their first `kaggle datasets create -p <dir>`.
- 2026-09-10: **Kaggle-CPU sanity, ensemble notebook.** Correction to the initial framing: the
  ensemble notebook/`rehearse_ensemble.py` has **no `ThreadPoolExecutor` at all** — decode is
  fully serial, one study at a time — so there's no "15 decode threads" setting on it to cap
  (that figure is the efficiency notebook's `DECODE_WORKERS`). The real HPC-vs-Kaggle CPU risk
  for a serial decode loop is numpy/scipy/torch's own BLAS/OpenMP thread pools implicitly using
  every core the job's cgroup exposes, not anything the Python code spawns itself — simulated by
  constraining both the SLURM allocation (`--cpus-per-task=4`) and `OMP_NUM_THREADS` /
  `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `NUMEXPR_NUM_THREADS` to 4 (job 24831, full 1300
  studies). Result: **4084s (68.1 min), matching the 16-CPU run's 4089s within 5 seconds** — the
  ensemble pipeline is GPU-bound (3 members x 5 folds, mostly ViT-B), decode is serial anyway, so
  Kaggle's ~4 CPUs vs. HPC's 16 makes no measurable difference. No trims needed.
- 2026-09-11: **exp009 made self-resuming.** The prior single-session-per-check approach stalled
  for 3 days (nobody manually resumed it between sessions) — fixed by chaining SLURM
  `--dependency=afterany:<jobid>` array jobs so it drives itself to epoch 25 without a live
  session: job 25033 (running, resumed from epoch 16) -> 25038 -> 25039 -> 25040, each depending
  on the previous. `afterany` fires regardless of exit status, and `train.py`'s resume logic is
  itself idempotent past completion (an already-finished fold's `for epoch in range(start_epoch,
  epochs)` loop is empty, so it just re-runs the cheap final eval and re-writes an identical
  `best.pt`) — so over-chaining costs a little wasted eval time per extra session, never
  correctness. 3 extra sessions queued is generous margin over the ~1 more session actually
  needed at the observed ~8 epochs/7h rate. Verified queued via `squeue`
  (`25040_[0-4] ... DEPENDENCY=afterany:25039_*`, etc., all the way back to the running 25033).
- 2026-09-11: **Kaggle-hardware pre-push fixes, round 2 — T4 x2 adaptation.** The pinned Kaggle
  accelerator turned out to be T4 x2, not H100 — three real issues found and fixed:
  1. **bf16 bug, ensemble notebook only.** `infer_core.run_study_inference[_tta]` default to
     `autocast_dtype=torch.bfloat16` (correct for H100 HPC rehearsal, left unchanged) — the
     ensemble notebook never overrode this, so it would have run bf16 autocast on a T4 (sm_75,
     Turing, no native bf16 tensor cores) silently. The efficiency notebook already passed fp16
     explicitly, so it was unaffected. Fixed: ensemble notebook now passes `AUTOCAST_DTYPE =
     torch.float16` explicitly at every call site, matching the efficiency notebook.
  2. **Multi-GPU member sharding, ensemble notebook.** Profiled H100 decode:model split per
     notebook (both via a clean serial N=20/N=30-study measurement, `scripts/_profile_ensemble_dm.py`
     and `scripts/profile_decode_vs_model.py`):
     | notebook | decode | model | conclusion |
     |---|---|---|---|
     | ensemble (T=24, 3 members x 5 folds) | 67.9% (2.838s/study) | 32.1% (1.343s/study) | model-bound enough to be worth sharding |
     | efficiency (T=16, 1 model x 5 folds) | 82.4% (0.715s/study) | 17.6% (0.153s/study) | decode dominates too much for a 2nd GPU to matter |

     Ensemble members are now greedy-balanced across however many CUDA devices are visible
     (`{exp006, exp005}` on one, `exp003` alone on the other — a ~4:3 relative-cost split),
     driven from separate Python threads per member so different-device members' CUDA kernels
     overlap (PyTorch releases the GIL during a CUDA call). Falls back to one device automatically
     on a normal single-GPU HPC job. Efficiency notebook deliberately does NOT get this — the
     model-inference slice it would speed up is too small relative to decode to be worth the
     complexity (see cell-0 markdown in both notebooks for the full writeup).
  3. **Decode/inference overlap, both notebooks.** Both previously decoded-then-inferred in
     lockstep (efficiency: a whole chunk decoded, then the whole chunk inferred; ensemble: fully
     serial per-study) — CPU idle during every inference, GPU idle during every decode. Added
     `infer_core.prefetch_decode`, a generator that decodes study N+1 in a background thread pool
     while study N runs on the GPU. **First version had a real bug**: it submitted all 1300 tasks
     to the thread pool up front with only a bounded *hand-off queue* between the pool and the
     consumer — that bounds how many completed-but-unconsumed items wait for the consumer, but
     nothing bounded how far the pool itself could race ahead completing (and holding in memory)
     tasks the consumer hadn't reached yet. Both full-1300 rehearsals (jobs 25046/25047) got
     OOM-killed at the 32GB job memory limit as a result. Fixed by keeping a bounded *sliding
     window* of in-flight futures (`window_size`, default 2x num_workers) instead — submit one
     more only once the oldest is consumed — which caps buffered memory to a small fixed multiple
     of the worker count regardless of test-set size. Re-ran clean (jobs 25048/25049, see below).
     Verified bit-for-bit: the same fold count (4/5) run through the old serial efficiency path
     and the new overlapped path produced **identical predictions, 0.0 mean/max abs diff** — the
     overlap is a pure scheduling change, no math touched.
  4. **Real H100 numbers, full 1300 studies, post-fix** (also confirms dual-GPU sharding and the
     flat weight-dataset layout below both work correctly end-to-end, not just in isolation):
     - Ensemble (2 GPUs, rung 0/3 members): **5200s (86.7 min)**, 0 prior fallbacks, 659 dual-flip,
       output validated (1300 rows, 0 NaNs, values in [0.0083, 0.9996], 1300 unique UIDs).
     - Efficiency (1 GPU, formula picked 4/5 folds): **447s (7.5 min)** — down from the prior
       serial run's 763s at the same fold count, confirming the overlap's real throughput gain.
       0 prior fallbacks, 663 dual-flip, output validated (1300 rows, 0 NaNs, values in
       [0.0079, 0.9939], 1300 unique UIDs).
  5. **T4 projection** (3.5x model-time multiplier per instruction, decode unchanged — CPU-bound,
     already shown CPU-count-insensitive above), from the clean N=20/30 H100 decode:model split,
     with decode/inference overlap meaning per-study throughput is bound by whichever of
     decode/model is slower (not their sum):
     - Ensemble, single T4: max(2.838, 1.343x3.5=4.700) = 4.700s/study -> **~102 min** for 1300.
     - Ensemble, T4 x2 (~4:3 split): max(2.838, 4.700x4/7=2.686) = 2.838s/study -> **~62 min**
       — dual-GPU flips the bottleneck from model to decode, the real reason it's worth doing here.
     - Efficiency, single T4 (no dual-GPU, per the decision above): max(0.508, 0.153x3.5=0.535) =
       0.535s/study -> **~11.6 min** — already close to its own achievable floor.
     Caveat, stated plainly: the real H100 dual-GPU run (86.7 min) came in ~40% above its own
     theoretical max(decode,model) floor (61.5 min) — real thread-scheduling/orchestration
     overhead the idealized formula doesn't capture. Treat the T4 numbers above as optimistic
     floors, not guarantees; the pending 3-sample Kaggle test is the real check. Either way, both
     notebooks land far under the 8h safety cap regardless of which estimate is closer to truth.
- 2026-09-11: **Kaggle dataset staging finalized.** Stripped `best.pt` to `ema`-only (the `model`
  key is never loaded by any notebook) for all 4 weight experiments — exactly 50% smaller each,
  as the model/ema-doubling finding predicted: exp003 4.72GB->2.36GB, exp005 1.20GB->0.60GB,
  exp006 4.72GB->2.36GB, exp008 1.20GB->0.60GB (11.8GB -> 5.9GB total). Excluded exp008's leftover
  `checkpoint.pt` files (478MB x5, resumability-only, dead weight for a finished experiment).
  **Restaged weight datasets FLAT** (`fold0_best.pt` … `fold4_best.pt` at the dataset root, no
  subdirectories) instead of `foldN/best.pt` — `kaggle datasets create`'s default `--dir-mode
  skip` silently DROPS subdirectories entirely (confirmed via `kaggle datasets create --help`),
  so uploading the old per-fold-subdirectory layout would need `--dir-mode zip`, and the
  2026-09-08 Kaggle debugging arc already found that zip-based subdirectory upload does NOT
  reliably preserve folder nesting on extraction (that's why the code package is found via
  `rglob("__init__.py")` instead of a hardcoded path) — harmless for the code package's uniquely-
  named `.py` files, but every fold's checkpoint is identically named `best.pt`, so a flattening
  extraction could silently collide 5 files into 1. Both notebooks' loaders now check the flat
  layout first, falling back to the subdirectory layout for robustness either way. All 5
  `dataset-metadata.json` files staged (code already existed from the 2026-09-08 arc, `datasetId
  11937869` — no longer resolves, confirmed via the API 403s logged in the prior entry; the 4
  weight ones newly written, matching the same `id`/`title`/Apache-2.0 format).
  **Correction to the prior entry's suggested command**: `kaggle datasets create -p <dir>` alone
  (no `-r`) would silently upload NOTHING for any dataset with subdirectories — `--dir-mode skip`
  ignores them, it doesn't error. The code dataset (has one `rsna_knee/` subdirectory) needs
  `-r zip`; the weight datasets, now flat, don't need `-r` at all. Copy-pasteable block, in order:
  ```bash
  kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/code -r zip
  kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/rsna-knee-exp003-weights
  kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/rsna-knee-exp005-weights
  kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/rsna-knee-exp006-weights
  kaggle datasets create -p /shared/home/husnain.rasool/rsna-knee/kaggle_staging/rsna-knee-exp008-weights
  ```
- 2026-09-11: **Storage quota near its cap — found and partly cleared.** The per-user BeeGFS quota
  is 400GB hard; usage was at 390.45GB (9.55GB headroom) when exp009's training log showed
  `QUOTA GUARD: headroom 10.3GB < 15.0GB floor — SKIPPING checkpoint write` — the existing
  quota-guard protection working as designed (skip, don't corrupt), but still worth clearing
  since a skipped checkpoint means a crash would lose more recent progress than necessary. Found
  and removed two safe, confirmed-redundant items: `artifacts/smoke_exp009/` (2.2GB, a leftover
  from the very first exp009 smoke test, fully superseded) and `exp008`'s 5 `checkpoint.pt` files
  (2.28GB, training confirmed complete at `epochs_completed=24`, `best.pt` already the record of
  record). Freed 4.4GB (390.45GB -> 386.08GB used, ~13.9GB headroom) — did not chase further
  since exp009 only needs ~4 more epochs to finish and this is enough margin for that. Not a
  permanent fix: `data/` alone is 644GB (`du`, though this doesn't reconcile cleanly with the
  391GB quota figure — resolved below: `data/raw` (531GB) + `data/cache` (113GB) sit outside this
  user's personal quota accounting entirely, evidently a shared/group-mounted pool — nothing
  reclaimable there for this specific problem).
- 2026-09-14/15: **exp009 verdict — DOES NOT CLEAR THE BAR.** All 5 folds completed (chain ran
  25033 -> 25038 -> 25039 -> 25040 exactly as designed, no live session needed). OOF macro-AUC vs
  `labels_v2_final` **0.8652** (bar >=0.855 — **passes**), gold-58 tripwire 0.8809, native
  training-time metric (vs `labels_v3_candidate`) 0.9446. Mean per-label OOF-prediction
  correlation with exp006 **0.9296** (bar <0.90 — **fails**, per-label range 0.86-0.96, nothing
  meaningfully decorrelated). **Verdict: no greedy re-selection** — ConvNeXt-V2-Base clears the
  accuracy bar but isn't diverse enough from the DINOv2 ViT members to be worth adding; same
  study-level errors, not independent ones. Full per-label breakdown:
  `artifacts/exp009_acceptance_report.csv` (`scripts/eval_exp009_acceptance.py`, mirrors
  `eval_exp005_acceptance.py`'s pattern).
  **Infrastructure problem found in the process, disclosed even though it turned out not to
  matter for this verdict**: the storage-quota headroom freed in the entry above (13.9GB) was
  still under the 15GB *routine-save* floor `safe_torch_save` gates on (separate from the lower
  3GB emergency-save floor) — every fold's FINAL `best.pt` write (and every `checkpoint.pt` write
  in sessions 25038-25040) was silently skipped, logged but easy to miss
  (`QUOTA GUARD: starting with headroom 14.674GB, already below the 15GB routine-save floor ...
  no weights will be deployable unless quota is freed before this run finishes`). The eval
  artifacts (`oof.parquet`, `gold58_predictions.parquet`, `summary.csv`) still saved (smaller,
  under the emergency floor) — that's how this verdict was computable at all — but **exp009 has
  no usable checkpoint on disk**, only a stale `checkpoint.pt` per fold at epoch 23/25 (one epoch
  short). Moot for the ensemble decision (rejected on correlation regardless), but real: freeing
  more headroom and resuming once more (only 1 epoch left to redo per fold, cheap) is what it'd
  take to get a real `best.pt` if this backbone is ever revisited. Found one more safe-looking
  reclaim candidate — `exp002_dinov2s_slot_v2labels`'s 5 `best.pt` files (1.145GB): not an
  ensemble-recipe member, and explicitly dropped as a submission target ("skipping the stale
  exp002/v7 probe entirely") — but deleting model weights tripped this session's own irreversible-
  action guard, so it's not done. Flagging rather than working around it: say the word and I'll
  delete those 5 files (or free space a different way) and re-run exp009 to actually get its
  weights saved, if you want them preserved despite the reject verdict.
- 2026-09-15: **Disk cleanup, explicitly authorized this time.** Deleted `exp002`'s 5 `best.pt`
  files (1.1GB — rejected as a submission target, not an ensemble-recipe member) and `exp009`'s 5
  `checkpoint.pt` files (11GB — the only weight artifact it had, since `best.pt` never saved per
  the entry above; `exp009` is rejected on correlation anyway, so no weights are needed). Eval
  outputs (`oof.parquet`, `gold58_predictions.parquet`, `summary.csv`, `config.yaml`) kept for
  both. Quota: 386.08GB -> **374.29GB used, 25.71GB headroom** (target was >20GB) — comfortably
  clear of both the 15GB routine-save floor and the 3GB emergency floor for any future training.
- 2026-09-17: **Probe #2 (efficiency, `submit-efficiency.ipynb`, exp008) scored: 0.885 public LB.**
  Delta vs. the ensemble's 0.893 is **-0.008** — inside the CLAUDE.md §4 "AUC within ~2 pts of the
  big ensemble" efficiency-track target, and better than the -0.0036 OOF delta predicted (0.8708 vs
  0.8744) would suggest, i.e. the distilled student held up on the real hidden test set at least as
  well as the OOF comparison implied. Ran on a single Tesla T4 (per plan, §2 of
  `docs/SUBMISSION_RUNBOOK.md` — pin T4 x1 for this notebook), `fast_decode: True`, formula-chosen
  fold count 5/5 (matches the "4 or 5" healthy-range expectation), zero prior fallbacks.
  **Both submission notebooks are now live and scored — Phase 6.3's two final selections are
  complete.**
  **Correction to a 2026-09-16 note above**: the earlier "shared local submission.csv has only 3
  rows, likely a truncated download" note was WRONG about the mechanism (though right to flag it).
  The notebook file shared this turn (`rsna-knee-submit-efficiency.ipynb`) makes it unambiguous:
  its own cell output prints `3 test studies, 15 series` — Kaggle's code-competition interactive/
  commit environment mounts a tiny placeholder `test.csv` (here, literally 3 rows), NOT the ~1,300-
  study hidden set; the real hidden set is only substituted in during Kaggle's own backend rerun
  triggered by clicking Submit, whose intermediate cell outputs are never shown back to us — only
  the final LB score is. So every `submission.csv`/notebook-output artifact either of us can see
  locally is necessarily the 3-sample interactive run, by design, not a download problem. The
  actual submitted runs are opaque to us beyond the LB score — which is exactly why the OOF↔LB gap
  check below is the only real signal we get on the full-scale run's health.

## OOF↔LB gap analysis (template — fill in once probe #1 is scored)

Prepared 2026-09-10, not yet run — CLAUDE.md §7: "Expect an OOF↔LB gap; measure it once early,
then trust deltas not absolutes." This is that one-time measurement, for probe #1 (the ensemble,
first real submission under the plan-change decision skipping the stale exp002/v7 probe).

**Reference point**: ensemble OOF macro-AUC (vs. `labels_v2_final`, gold-58 held out) = **0.8744**
(greedy-forward-selected 3-member recipe, `ensemble_recipe_v1.yaml` — see the 2026-09-09 ensemble
selection entries above). This is the number the LB score gets compared against.

**Once the public LB score for probe #1 is in, fill in:**

| | value |
|---|---|
| OOF macro-AUC (reference) | 0.8744 |
| Public LB macro-AUC | *(fill in)* |
| Gap (OOF − LB) | *(fill in)* |
| Gold-58 macro-AUC (tripwire, for context only) | *(fill in from the run that produced 0.8744)* |

**Interpretation rule (pre-registered, apply mechanically — don't rationalize a bad gap after
the fact):**
- **`|gap| < 0.02`**: validation is trustworthy. Proceed as planned — efficiency notebook is
  probe #2, no changes to the label engine or fold construction triggered by this alone.
- **`|gap| >= 0.02`**: **stop.** Do not burn probe #2 (or any further submission) until the gap is
  understood. Investigate, in order of likelihood given CLAUDE.md §12's risk register:
  1. **Site-level/report-template leakage** — check whether GroupKFold's report-template/site
     clustering (§7) actually kept near-duplicate reports and same-site studies together; a leak
     here inflates OOF relative to a genuinely held-out LB population.
  2. **Prevalence shift** — CLAUDE.md §3 warns train/public-LB/private-LB prevalences may differ;
     since AUC is prevalence-invariant per label this shouldn't by itself cause a gap, but confirm
     no per-label AUC computation accidentally became threshold-sensitive.
  3. **Label-engine ceiling** — the gold-58 tripwire AUC is the sanity check here: if gold-58 also
     looks off relative to what individual experiment entries recorded, the label engine (not the
     fold construction) is the more likely culprit.
  - A negative gap (LB > OOF) is not automatically "good news" — re-run the same checklist; it can
    indicate a fold-construction issue in the other direction (OOF harder than it should be) just
    as easily as a real generalization win.

## OOF↔LB gap analysis — probe #1 RESULT (2026-09-17)

Filling in the template above per its own instruction. Probe #1 (`submit-ensemble.ipynb`, T4 x2)
completed on real Kaggle hardware with zero OOM/timeout issues (the 2026-09-16 backpressure/
`inference_mode`/shallow-`_find_dir` fixes held).

| | value |
|---|---|
| OOF macro-AUC (reference, pre-registered) | 0.8744 |
| Public LB macro-AUC | 0.8930 |
| Gap (OOF − LB) | **−0.0186** |
| Trust threshold | < 0.02 |
| Gold-58 macro-AUC for this run | not available — no gold labels at Kaggle test time (see below) |

Produced via `python scripts/eval_oof_lb_gap.py --lb-score 0.893`
(`artifacts/oof_lb_gap_report.md`).

**Verdict: TRUSTWORTHY — |gap| = 0.0186 < 0.02.** Per the pre-registered rule, this does NOT
trigger the investigation gate. Proceeding to probe #2 (efficiency notebook) — no label-engine or
fold-construction changes triggered by this result alone.

**Interpretation, since the user asked for one despite the gap clearing the bar** (direction is
negative — LB beats OOF — which the rule above explicitly says isn't automatically good news):
1. **Site/template leakage** would inflate OOF relative to LB (push the gap *positive*) — the
   opposite of what we observed. Argues against a fold-construction leak being the explanation.
2. **Prevalence shift** (CLAUDE.md §3): AUC is prevalence-invariant per label in theory; no
   mechanism identified that would specifically explain a negative-direction shift.
3. **Label-engine ceiling**: can't be checked directly for this run — gold-58 has no test-time
   equivalent (test studies aren't the gold-58 studies, and true test labels are never exposed to
   us). The gold-58 AUC from HPC eval measures a disjoint population, not this submission.
4. **Most parsimonious reading**: our OOF proxy target (`labels_v2_final`, LLM/report-derived,
   ~18% systematic disagreement with true image-reviewed criteria per CLAUDE.md §2) is a noisier
   target than the true hidden-test labels. A model that learned real image pathology features can
   plausibly score higher against the cleaner true labels than against our own noisier
   training/validation proxy — consistent with (not proof of) the weak-supervision approach working
   as intended, not a red flag.
5. **Small-sample noise**: the public LB is a subset of the ~1,300 test studies (split size not
   disclosed by Kaggle); some of the 0.0186 swing is plausibly just per-label AUC variance on a
   smaller population, either direction.

**Note on the shared local `submission.csv`**: the copy shared alongside this result
(`C:\Users\husna\Downloads\submission.csv`) contains only 3 data rows, not ~1,300. Since
`submit-ensemble.ipynb` cell 7 asserts `len(sub_df) == len(test_df)` before ever writing the file,
and Kaggle's scorer would not produce a meaningful macro-AUC off a handful of rows, the file that
was actually submitted almost certainly had the full row count — this local copy is most likely a
truncated download or a manually-saved excerpt, not the submitted file itself. Flagged for
awareness; does not affect the gap analysis above, which uses the reported LB score directly.
**Correction, 2026-09-17**: the mechanism guessed above was wrong — see the probe #2 decisions-log
entry above this section for the actual explanation (Kaggle's interactive/commit environment
mounts a 3-row placeholder `test.csv`, not a download issue). The conclusion (doesn't affect the
gap analysis) still holds.

## OOF↔LB gap analysis — probe #2 RESULT (2026-09-17)

`submit-efficiency.ipynb` (exp008, DINOv2 ViT-S/14 distilled student, single Tesla T4) scored on
real Kaggle hardware with zero OOM/timeout issues, formula-chosen fold count 5/5.

| | value |
|---|---|
| OOF macro-AUC (reference — exp008's own accepted 5-fold OOF vs `labels_v2_final`) | 0.8708 |
| Public LB macro-AUC | 0.8850 |
| Gap (OOF − LB) | **−0.0142** |
| Trust threshold | < 0.02 |

Produced via `python scripts/eval_oof_lb_gap.py --lb-score 0.885 --oof-score 0.8708`
(`artifacts/oof_lb_gap_report_probe2.md`).

**Verdict: TRUSTWORTHY — |gap| = 0.0142 < 0.02.** Smaller in magnitude than probe #1's −0.0186 and
the same direction (LB beats OOF both times) — consistent with, and reinforces, the probe #1
interpretation: the true hidden-test labels are plausibly a cleaner target than our report-derived
OOF proxy, so a model that generalizes on real image pathology scores slightly better against them
than our own internal validation predicts. Two independent probes landing on the same sign and a
comparable (sub-0.02) magnitude is itself a mild positive signal about validation-methodology
soundness — not just luck on one run.

**Efficiency-track delta**: LB 0.885 vs. the ensemble's LB 0.893 is a **−0.008** delta, comfortably
inside the "AUC within ~2 pts (0.02) of the big ensemble" target (CLAUDE.md §4) — the distillation
held up on real hidden data essentially as well as OOF predicted (OOF delta was −0.0036; LB delta
is −0.008, i.e. slightly larger but still well inside tolerance).

**Both probes now submitted and scored** — Phase 6.3 (final two selections: max-AUC ensemble +
efficiency student) is complete. Next: see the priorities review below.

## Diagnostic — gap to leaderboard top, and priority shift (2026-09-17)

With both probes trustworthy but ~6-7 points behind the public LB top (~0.95-0.96), the user
asked whether the gap is uniform across labels or concentrated, before launching anything further.

**Ensemble per-label OOF-AUC (vs. `labels_v2_final`), worst to best** — recomputed exactly from
`artifacts/ensemble_recipe.csv` (the actual greedy-selection output, per-label global method,
mean = 0.87439, matches the documented 0.8744 macro to 4 decimals):

| Label | OOF AUC |
|---|---|
| Synovitis | 0.8258 |
| MCL | 0.8365 |
| PF OA | 0.8442 |
| ACL | 0.8522 |
| Contusion | 0.8593 |
| Lateral OA | 0.8664 |
| Lateral Meniscus | 0.8821 |
| Baker's | 0.8928 |
| Medial OA | 0.8978 |
| Fracture | 0.9018 |
| Effusion | 0.9062 |
| Medial Meniscus | 0.9277 |

**External reference: Qiu et al. 2024, "Learning co-plane attention across MRI sequences for
diagnosing twelve types of knee abnormalities," Nature Communications 15:7637**
(https://www.nature.com/articles/s41467-024-51888-4, PMC11368947). Important caveat found while
pulling this: **the paper's "twelve types" are NOT the same twelve labels as this competition** —
despite the superficially identical count, the label sets only partially overlap:

| Our label | Qiu et al. category | AUC (Qiu et al., internal test) | Valid comparison? |
|---|---|---|---|
| ACL | ACL | 0.949 | Yes — same pathology |
| MCL | MCL | 0.782 | Yes — same pathology |
| Effusion | EFFU | 0.787 | Yes — same pathology |
| Contusion | CONT | 0.821 | Yes — same pathology |
| Medial Meniscus + Lateral Meniscus | MENI (combined, not split by side) | 0.763 | Partial — they don't split medial/lateral |
| Baker's | CYST (popliteal + ganglion + meniscal + synovial cysts, combined) | 0.862 | Partial — broader category than just Baker's |
| Synovitis | PLICA (synovial plica — an anatomic fold, not inflammation) | 0.877 | **No — different pathology, not a real comparison** |
| Medial OA / Lateral OA / PF OA | CART (one combined cartilage-damage category, not compartment-split) | 0.790 | Partial — different task granularity |
| Fracture | *(not evaluated in their 12)* | — | No comparison exists |
| *(n/a)* | PCL, LCL, IFP, PR | — | We don't evaluate these at all |

**Answer: the gap is NOT uniform — it's concentrated, and the diagnostic is unambiguous.** Where a
real same-pathology comparison exists:
- **ACL: ours 0.8522 vs. 0.949 external → gap −0.097.** By far the worst and only label exceeding
  the >0.05-behind flag by a wide margin.
- MCL: ours 0.8365 vs. 0.782 → we're AHEAD by +0.055.
- Effusion: ours 0.9062 vs. 0.787 → AHEAD by +0.119.
- Contusion: ours 0.8593 vs. 0.821 → AHEAD by +0.038.
- Meniscus (crude avg of our split vs. their combined): ours ~0.905 vs. 0.763 → AHEAD by +0.14.
- OA compartments (crude avg of our 3 vs. their 1 combined CART): ours ~0.870 vs. 0.790 → AHEAD by
  ~+0.08, though this is the least apples-to-apples comparison in the table.
- Synovitis has NO valid external comparator (Plica ≠ Synovitis) despite being our lowest
  absolute OOF-AUC — genuinely inconclusive, not evidence of a labeler ceiling.

**This directly cross-checks against the noisy-student blend weights already on record** (2026-09-08
entry above, `label_blend_v3_candidate.md`): ACL and MCL are the only two labels where the accepted
blend went report-heavy (w_image 0.1 and 0.3 respectively) — i.e. the report/label engine already
beats our image models specifically on these two, and MCL is NOT where the external gap is (we're
ahead externally on MCL) while ACL is both internally report-favored AND externally the single
biggest published-benchmark gap. Two independent signals point at the same label.

**Decisions made on this evidence:**
1. **exp010 launched** (T=32 slice-count half only — see below for why 448px is deferred).
2. **exp011 (MRNet) prep written, NOT launched** — see below, license eligibility is a real open
   question, not confirmed.
3. **exp007 (3D ResNet) relaunched to convergence**, low priority, background.
4. **Label engine round 3 stays deprioritized**, per the user's own pre-stated conditional: it
   would only be revisited if the gap concentrated in Synovitis/Contusion specifically — Contusion
   is *ahead* of the external reference (not a gap) and Synovitis has no valid external comparator
   (inconclusive, not confirmed as a gap). Neither condition is met. ACL is the real finding, and
   ACL is already established (via the blend weights) as an *image-model* weakness relative to the
   label engine, not a label-engine weakness — consistent with prioritizing MRNet (ACL-focused
   auxiliary data) over another label round.

### exp010 (resolution + slice-count ablation) — launched T=32 only, 448px BLOCKED

Requested as the single biggest untested axis. Found a real infrastructure gotcha before
launching (exactly what "diagnostic first" was for): `scripts/build_cache.py` resamples DICOM to a
**fixed** 0.3mm/px, 384×384 grid at cache-BUILD time (`data/cache/<study>/<series>.npy` is a
hardcoded `[n_slices,384,384]` uint8 array) — `dataset.py`'s `__getitem__` only bilinear-upsamples
this cached array to whatever `image_size` a config requests (`F.interpolate`, line ~276). Setting
`image_size: 448` without a cache rebuild would therefore test bilinear-upsampling artifacts, not
real resolution — an invalid experiment that would burn ~10+ H100-hours to answer the wrong
question.

`slices_per_series` (T), by contrast, IS a pure load-time parameter (`_load_slot_volume`
linspace-subsamples from the cache's full native slice count) — free to change, no cache rebuild.

**Split the ablation accordingly:**
- **`configs/exp010_dinov2b_slot_v3labels_t32.yaml`** — T=32 (up from 24), 384px unchanged,
  otherwise identical to exp006 (same backbone/labels/folds). Launched as a 5-fold array job
  (`sbatch slurm/train.sbatch configs/exp010_dinov2b_slot_v3labels_t32.yaml`, job 25325). All 5
  folds confirmed running cleanly (train=3470 val=879 gold=58 loaded, no immediate OOM).
- **448px is BLOCKED on quota, not launched.** A genuine test needs a NEW cache at ~0.257mm/px
  (same ~115mm FOV, finer spacing — NOT just a wider-FOV cache at the same spacing, since the
  hypothesis is about small/fine structures like meniscus and cartilage). Measured cost: current
  384px cache is 113GB; (448/384)^2 ≈ 1.361× → **~154GB new storage needed**, against only
  **25.71GB free** quota as of this check. Not close — needs either a quota increase or freeing
  >130GB from somewhere (nothing safe identified; the easy candidates were already reclaimed in
  the 2026-09-15 cleanup). **Left for the user to decide**: request a quota increase, accept a
  much smaller subsampled finer-spacing test first (~500 studies ≈ 17.4GB, fits but leaves very
  little margin — this project has hit the 15GB/3GB quota-guard floors three separate times this
  session, so cutting it that close again is not recommended without deliberate headroom
  management), or drop the 448px half entirely and treat T=32 as the full ablation.
- **Added real `grad_accum_steps` support to `scripts/train.py`** while building this config (the
  user's own fallback instruction: "if VRAM forces it, grad-accum rather than reducing batch") —
  defaults to 1 (a no-op; every existing config's training dynamics are byte-for-byte unchanged),
  accumulation-window-aware NaN/Inf guard (a poisoned micro-batch discards the WHOLE window's
  accumulated gradient, not just its own contribution, since a summed accumulator can't un-mix a
  bad batch from good ones after the fact). Not actually needed for T=32 alone (a ~1.33× memory
  increase, expected to fit in H100 80GB headroom without accumulation) but available if a future
  448px config needs it.

### exp007 (3D ResNet-50) — relaunched from scratch, chained to convergence

**Not a resume** — the epoch-10 checkpoint was deleted 2026-09-10 to free quota for exp008's
relaunch (see that entry above), so there was nothing to resume from. This is a full relaunch
(`train.py --resume auto` finds no `checkpoint.pt`, starts epoch 0 as designed). Launched as a
5-fold array job (job 25331) plus a 4-deep `sbatch --dependency=afterany:<jobid>` self-resuming
chain (25336→25337→25338→25339, matching the exp009 self-resuming lesson from earlier this
session — must not require a live monitoring session) — **5 total sessions × 7h max-hours ≈ 35h
wall-clock budget**. Real cost estimate, flagged since it's well above the original EXECUTION_PLAN
budget: the killed run measured ~10.5h/fold to reach 10/25 epochs (40%), implying **~26h/fold full
cost ≈ 131 GPU-hours total for all 5 folds** — more than double the plan's original ~60 GPU-hour
estimate for the whole diversity track. Confirmed queued alongside exp010 without a quota/GPU
conflict (both running concurrently as of this check; per-user GPU cap did queue 2 of exp007's
folds behind exp010's, resolves naturally as folds complete).

### exp011 (MRNet auxiliary pretraining) — prep script written, NOT launched, license NOT confirmed

`scripts/prep_mrnet.py` written (slot-mapping plan: MRNet's single sequence per plane duplicated
into both FS/nonFS halves of the corresponding slot pair; pretrain `SlotModel` with `num_targets=3`
[abnormal/ACL/meniscus] on MRNet's 1,130/120 split, then transfer the backbone weights, dropping
the study-head, into a normal 12-target fine-tune on our data).

**Did NOT confirm eligibility — found a real problem instead.** MRNet's terms of use (checked
2026-09-17): "Permission is granted to view and use the MRNet Dataset without charge for
**PERSONAL, NON-COMMERCIAL RESEARCH PURPOSES ONLY**... you may **NOT share the download link**
with others," gated behind a signed Research Use Agreement. Two separate problems for this
project: (1) whether training for a cash-prize Kaggle competition counts as "non-commercial
research" is a real legal question this script doesn't and shouldn't resolve; (2) "may not share
the link" + "personal use only" is in direct tension with CLAUDE.md's winners'-obligation
requirement to PUBLICLY release weights — baking an MRNet-pretrained encoder into a publicly
released model is arguably redistributing a derivative of gated, non-shareable data, and plausibly
fails EXECUTION_PLAN.md Phase 4.5's own "freely & publicly available" bar regardless of the
commercial-use question. **Recommendation, not a decision made here: the user should personally
verify against the actual RSNA competition's external-data rules and the full MRNet Research Use
Agreement text before any download happens under this project.** No data downloaded, no
eligibility claimed.

## exp007 killed, ACL dedicated workstream launched (2026-09-17)

Three decisions on the diagnostic above: kill exp007 (131 GPU-hours not defensible against a
10-point ACL hole), drop exp011/MRNet, make ACL a dedicated workstream.

- **exp007 killed.** `scancel --name=rsna_train --user=husnain.rasool` — **this also killed
  exp010 by accident** (name-matches every job from `train.sbatch`, not just exp007's). Caught
  immediately: neither exp010 nor exp007 had written a `checkpoint.pt` yet (both only minutes into
  epoch 0), so nothing was actually lost — exp010 relaunched clean (job 25344, was 25325).
  exp007's chain (25331/25336-25339) confirmed fully cancelled, quota unaffected (it hadn't
  written anything either).
- **exp011/MRNet marked dead**, not deleted (kept for the record — the license research took real
  effort and is the reason KneeMRI is now the thing being checked instead). `scripts/prep_mrnet.py`
  now `raise SystemExit`s immediately if run, with the docstring's status line changed to DEAD.

### ACL error analysis (`scripts/analyze_acl_errors.py`, `artifacts/acl_error_report.md`)

First pass had a bug worth recording: gold-58 is excluded from every fold's train/val split by
design (CLAUDE.md §7), so it can never appear in `oof.parquet` — an initial version of this script
joined against `oof.parquet` and silently got 0 gold-58 rows, producing NaN for the "true
out-of-sample" correlation check. Fixed by loading `gold58_predictions.parquet` (every fold's
model predicts all 58 gold studies) separately.

**The corrected gold-58 comparison changes the read on ACL, and matters for what to do next**:

| | AUC vs. true gold-58 label | corr vs. true gold-58 label |
|---|---|---|
| Report label (`labels_v2_final`, LOO-calibrated) | 0.9890 | 0.9024 |
| Image ensemble (fold-ensembled, out-of-sample) | 0.9877 | 0.8949 |

**Essentially tied, both near-ceiling.** The v3 blend's report-heavy weight for ACL (w_image=0.1)
was optimizing a thin, likely noise-level margin on n=58 (CLAUDE.md's own "wide CIs" caveat) — not
evidence of a large image-model blind spot. This means the −0.097 gap vs. Qiu et al. (measured
against our noisy 4349-study soft-label OOF target, 0.8522, not the clean gold-58 signal) is
plausibly INFLATED by proxy-label noise in the eval target itself, not purely a real
model/architecture deficit vs. their large, cleanly-labeled test set. **Doesn't deprioritize ACL**
— 0.8522 remains our lowest comparable label and the true number is unknowable without more clean
labels — but it argues against expecting an architecture change alone to close most of the gap,
and for treating ACL-specific label/target quality on the broader population as a live hypothesis,
not something gold-58 alone can rule out.

**Top-40 highest-error non-gold ACL cases, aggregate pattern vs. the 4349-study population:**
sagittal is present in 100% of BOTH groups (not a differentiator — ACL is reliably imaged, not a
data-availability problem); side-unresolved is meaningfully overrepresented (50-60% of top errors
vs. 34.4% baseline, ~1.5-1.75x) — plausible mechanism: the "run both orientations and average"
fallback for unresolved-side studies may specifically hurt a spatially-localized, non-paired
finding like ACL more than it hurts labels this project already treats as mirror-symmetric, though
this is a hypothesis flagged here, not cross-checked against another label's error list this
session; language distribution shows NO clear driver (English is if anything slightly
OVERrepresented in errors, 52% vs. 39% baseline — the opposite of what a translation-quality theory
would predict).

### exp012 — anatomy-aware ACL head, launched

`src/rsna_knee/model.py`: added an optional `target_slot_prior: dict[int, list[int]] | None`
constructor arg to `SlotModel` — NOT a new mechanism (the existing per-target
`nn.MultiheadAttention` pooling can already learn arbitrary slot-weighting per target purely from
gradient descent, confirmed by reading the forward pass before touching it) — only an
INITIALIZATION bias, nudging a target's query toward the mean of specified slot identity
embeddings instead of unbiased small-random noise, scaled to match the query row's own initial
norm (a nudge, not an override). Defaults to `None`; every existing config is unaffected.
Smoke-tested locally (construction + forward pass, both with and without the prior): confirmed
only the biased target's query row differs at all from an identical-seed unbiased model, every
other row byte-identical, forward pass produces finite output either way.

`configs/exp012_dinov2b_slot_v3labels_acl_prior.yaml`: identical to exp006 except
`model.target_slot_prior: {0: [0, 1]}` (ACL -> Sagittal_FS, Sagittal_nonFS). Launched as a 5-fold
array job (job 25353). Report per-label deltas vs. exp006 when it lands — ACL gain must not cost
macro.

### exp011/KneeMRI — license NOT confirmed, stopped per the user's own pre-registered condition

Checked (2026-09-17): the dataset's original host page
(riteh.uniri.hr/~istajduh/projects/kneeMRI/) 404s, the paper's own PDF link 404s, and no
authoritative Creative Commons license text was found via search despite several query angles — a
Kaggle mirror exists (`sohaibanwaar1203/kneemridataset`) but a third-party mirror existing doesn't
establish the original license terms, and its page wasn't renderable to check. **This is exactly
the "ambiguous — stop and tell me, don't download" case the user pre-registered.** No prep script
written, no download attempted. If the user finds the actual license text (or the host page comes
back), it's worth rechecking — 917 sagittal PD-weighted exams with ACL grades would be a strong
match for ACL-focused auxiliary pretraining if the licensing genuinely is clean.

### exp013/exp014 — quota-cheap 448px test, launched (corrected scope from the original ask)

The original ask ("~880 studies, ~30GB") doesn't quite work as specified: (1) 30GB exceeds the
25.71GB free quota at the time, and (2) "fold 0's studies" (~880) is fold 0's VALIDATION set size
— training a model on only its own validation studies isn't a meaningful comparison to exp006's
real fold-0 baseline (which trains on the ~3470 OTHER studies). Reinterpreted as a smaller,
self-contained sanity check instead: a fresh 400-study subset (320 "train" from folds 1-4, 80
"val" from fold 0, sampled from the existing `folds.parquet` so report-cluster grouping is
preserved) — a valid, isolated 384-vs-448 comparison on ITS OWN scale, not a number directly
comparable to exp006's real fold-0 OOF (different, much smaller training set).

- `scripts/build_cache.py`: added `--study-list <file>` (default None, unchanged behavior) to
  restrict a cache build to a study subset instead of the full corpus.
- `slurm/acl_448_cache_build.sbatch`: builds `data/cache_448_subset/` at ~0.2571mm/px, 448px (same
  ~115mm FOV as the main 0.3mm/px/384px cache, genuinely finer spacing — not an upsample) for just
  the 400-study subset. ~14GB, comfortably inside the 25.59GB free quota at launch time (checked
  immediately before submitting). Job 25357, 4 shards, running on the ZEN4 CPU partition.
- `configs/exp013_..._subset384_baseline.yaml`: 384px, using the EXISTING main cache (no rebuild
  needed for this half) filtered to the subset. Job 25358.
- `configs/exp014_..._subset448_test.yaml`: 448px, using the new subset cache, `grad_accum_steps:
  2` (batch_size_studies halved to 4, matching exp010's fallback mechanism) to keep the effective
  batch at 8. Job 25359, `--dependency=afterok:25357` so it only starts once the cache build
  actually finishes successfully.
- **Delete `data/cache_448_subset/` once this comparison is read out** — it's a one-off diagnostic
  cache, not a permanent second variant, and 14GB is real headroom this project can't spare
  indefinitely.

### exp010 (T=32) — still running, no result yet

Job 25344 (relaunched after the accidental kill above), all 5 folds active. Per-label deltas vs.
exp006 — ACL especially — to be reported once it lands.

## Quota crisis recurrence (2026-09-18) — same failure mode as exp003/exp008/exp009, caused this time by running too much concurrently

Checked all four 2026-09-17 jobs the next day. **None finished cleanly.** Headroom crossed the
15GB routine-save floor while exp010, exp012, and the 448px cache build were all running at once
(14GB cache + growing checkpoints from two simultaneous 5-fold trainings), and kept dropping —
exp013/exp014 (the 400-study subset test) actually ran into the 3GB *emergency* floor too
(1.6GB headroom at the time). This is exactly the failure mode `docs/EXPERIMENTS.md` has logged
three times before (exp003, exp008, exp009) — this time self-inflicted by launching 4-5 jobs at
once without re-checking quota as they consumed it, not by a single big write.

**exp010 (T=32)**: all 5 folds hit the wall-clock guard mid-epoch (18/25) with checkpoint writes
silently skipped from ~epoch 15-17 onward. Resumable — every fold has a real checkpoint at
**epoch 17** (18 epochs done, 7 remaining), no progress actually lost beyond what quota-skipping
cost between the last successful save and the wall-clock cutoff. 8.8GB on disk.

**exp012 (ACL anatomy-aware head)**: only fold0 has a checkpoint (**epoch 23**, 1 epoch short of
the full 25 — very cheap to finish). Folds 1-4 have **no checkpoint at all** — they started later
(queued behind exp010 for GPU slots) and headroom was already below the routine floor by the time
they got a GPU, so they never wrote a first save. Those 4 folds need a full restart, not a resume.

**exp013/exp014 (400-study subset, 384 vs 448px)**: training itself actually completed 25 epochs
and printed real OOF results before crashing — the crash was a bug in this session's own subset
construction (the 400-study `folds.parquet` subset excluded gold-58 entirely, but `train.py`
unconditionally evaluates a gold split; an empty gold set produced an empty DataFrame with no
`study_uid` column, `KeyError: 'study_uid'` inside the gold-58 eval, after the OOF eval had already
run and printed). **The numbers we actually needed survived the crash**, in the logs:

| | macro | ACL | MCL | MedMen | LatMen | MedOA | LatOA | PFOA | Effusion | Synovitis | Baker's | Contusion | Fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp013 (384px) | 0.7482 | 0.6311 | 0.5865 | 0.7697 | 0.6071 | 0.9507 | 0.7083 | 0.8054 | 0.9462 | 0.9052 | 0.6236 | 0.6813 | 0.7628 |
| exp014 (448px) | 0.7447 | 0.6281 | 0.5513 | 0.7729 | 0.6014 | 0.9326 | 0.7244 | 0.8162 | 0.9473 | 0.9136 | 0.5879 | 0.6787 | 0.7821 |
| Δ (448-384) | −0.0035 | −0.0030 | −0.0352 | +0.0032 | −0.0057 | −0.0181 | +0.0161 | +0.0108 | +0.0011 | +0.0084 | −0.0357 | −0.0026 | +0.0193 |

**Read with real caution**: n=80 validation studies, mixed-sign deltas of similar magnitude to
gold-58's own noise band, and both numbers sit far below exp006's real ~0.87 macro simply because
the training set here is 320 studies vs. exp006's ~3470 — not comparable to the real baseline, and
not powered to detect a resolution effect at this scale. **Net read: no clear resolution benefit
on this cheap test, but "inconclusive" is the honest label, not "resolution doesn't help."** ACL
specifically went slightly negative (−0.0030) — doesn't support the resolution hypothesis for ACL
on this evidence, though again, not a well-powered test of it. Neither run left a usable checkpoint
(quota was already past the emergency floor by the time they'd have written one) — not resumable,
would need a full relaunch to get weights, and the 448px cache that fed exp014 has already been
deleted (see below) since its purpose — this comparison — is served.

**Remediation taken**: deleted `data/cache_448_subset/` (14GB, its comparison already extracted
above) — headroom 1.56GB -> **15.17GB**, just barely back above the routine-save floor, no real
margin. **Did not relaunch anything** — resuming exp010 (7 epochs x 5 folds) and restarting
exp012's 4 missing folds at the same time would very plausibly repeat this exact crisis again
immediately. Left for the user: whether to resume sequentially with active quota monitoring,
request a durable quota increase given this is the fourth recurrence of the same failure mode, or
free more headroom first (nothing else obviously safe to delete was found in this pass).

## Quota raised to 650GB; guardrails; resumes; exp015 (2026-09-18)

Quota confirmed active: 650GB hard limit, **265.2GB free** (384.8GB used). Storage is no longer the
binding constraint — but the process failures behind the last two incidents are fixed separately
below, not left to headroom.

### Process guardrails (two self-inflicted incidents: name-matched scancel, concurrent-launch quota crash)
- **`scripts/safe_scancel.py`**: cancels by explicit job ID only, lists exactly what would be
  cancelled first (list-only by default, `--confirm` to act); rejects names, patterns, `--user`.
  Verified: `safe_scancel.py rsna_train` -> REFUSING, exit 2.
- **`scripts/safe_sbatch.py`**: pre-launch disk projection. Sums projected disk of every queued+
  running job (ledger in `artifacts/disk_ledger.json`; per-fold footprint by backbone x folds, or a
  `# DISK_PROJECTION_GB` header for cache builds; live jobs launched outside the wrapper counted at a
  conservative 5GB) and refuses if the total exceeds free quota minus 30GB. Verified both ways:
  passes at 173.5GB projected vs. a 234.7GB budget; refuses (exit 2) when headroom is forced to
  260GB. `--reuses JOBID` marks chain/resume links so the same checkpoint files aren't counted
  twice. Also names train jobs after their experiment (previously every job was `rsna_train`, which
  is exactly what made the name-based cancel so destructive). Known conservatism: a running job's
  already-written files are counted again, so it can refuse too often but not too rarely.
- Every launch below went through `safe_sbatch.py`.
- **Orphan checkpoints: there were none to delete.** Inventory of every `*.pt` under `artifacts/`:
  only the deployed models' `best.pt` (exp003/005/006/008), plus exp010's five and exp012 fold0's
  `checkpoint.pt` — which are the *resumable* ones, not orphans. The killed/failed runs (exp007,
  exp013, exp014) wrote no checkpoints at all, only ~50KB of `config.yaml`/`oof.parquet` stubs, left
  in place. Free space after cleanup: unchanged, 265.2GB.

### Bug found in my own grad-accumulation code (2026-09-17), fixed
`scheduler` was built from `len(train_loader)` steps/epoch but `scheduler.step()` fires once per
accumulation *window*. With `grad_accum_steps=2` that doubled warmup (5 -> 10 real epochs) and
stopped the cosine decay at its midpoint. Only **exp014** ran with accumulation, so only exp014 was
affected (exp010/exp012 use `grad_accum_steps=1`, where the fix is a no-op) — one more reason the
400-study test is void. Fixed in `train.py` (`-(-len(train_loader)//grad_accum_steps)`); found while
planning exp015, before it could bite a real run.

### 400-study 384-vs-448 test (exp013/exp014): VOID — underpowered, do not cite
Only 80 validation studies, mixed-sign per-label deltas inside the noise band, macros ~0.75 vs
exp006's ~0.87 purely from a 10x smaller training set, and exp014 additionally ran with the
scheduler bug above. The numbers logged in the 2026-09-18 quota-crisis entry are retained as a
record of what happened, **not as evidence for or against resolution**. The question is being
answered properly by exp015.

### ACL conclusion — updated (supersedes the 2026-09-17 "ACL dedicated workstream" framing)
On the clean gold-58 labels, ACL AUC is **~0.988 for both the report labels (0.9890) and the image
ensemble (0.9877)**. The 0.8522 ensemble OOF ACL AUC is measured against noisy report-derived soft
labels over 4,349 studies. So the "−0.097 gap vs. Qiu et al." was mostly **proxy-label noise in our
evaluation target, not a vision deficit**. Consequence: **OOF systematically understates true
performance** — consistent with both LB probes beating their OOF (+0.0186 ensemble, +0.0142
efficiency). **ACL is no longer a dedicated workstream**: exp012 (anatomy-aware ACL init) finishes
since it's already mostly trained, and nothing further ACL-specific is pursued unless it wins.
Caveats kept honest: gold-58 is n=58 (wide CIs), and the side-unresolved overrepresentation among
ACL's highest-error studies (60% vs. 34%) was a hypothesis never cross-checked against other labels
— left unresolved, not refuted. Also: because OOF is a noisy proxy, small OOF deltas between runs
(exp010/012/015 vs. exp006) should be read with that in mind; gold-58 is the clean-but-tiny
tripwire, both reported.

### External data: none
KneeMRI stays dropped (license unresolved: host page and paper PDF 404, no authoritative CC text
found). MRNet was dropped earlier (non-commercial-only + no-link-sharing agreement vs. prize
eligibility and the public-weights obligation). No external data will be used.

### Launches (staggered, guarded)
- **exp010 (T=32) resumed**, job **25543**, all 5 folds, `Resumed from epoch 18` confirmed in every
  fold's log (checkpoints were at epoch 17; 7 epochs to go, ~2.7h at the observed ~23 min/epoch).
- **448px cache rebuild**, job **25548** (20 shards, ZEN4 CPU partition), `data/cache_448`, ~0.2571
  mm/px on a 448px grid (same ~115mm FOV as the main cache — a real resolution increase, not an
  upsample), projected ~154GB (main cache 113GB x (448/384)^2). Launched while waiting on exp010's
  first post-resume checkpoint since it needs no GPU.

### exp010 resume: folds 0-3 OOM'd on g006, resubmitted (2026-09-18)
Job 25543 (5 folds): folds 0-3 died in ~43s with CUDA OOM — a co-tenant process held 72GiB on their
GPU (node g006's only other job, `sergiy.bu` 25482, requests no GRES; see HPC_WORKFLOW.md §3.5a).
Not a code/checkpoint problem: every checkpoint was verified intact at epoch 17 afterwards. Fold 4
landed on g002 and kept training. **Folds 0-3 resubmitted as job 25568** (`--array=0-3
--exclude=g006`, `--reuses 25543` since it rewrites the same checkpoints), resuming from epoch 17 ->
18. The new `[gpu-binding]` diagnostics show all four got their own GPU (distinct UUIDs, 0 MiB used)
on g002/g003.
- **`safe_sbatch.py` fix**: this resubmission was first REFUSED — a false positive, the guard was
  counting the 448px cache build's full 160GB projection although ~148GB was already written (free
  quota had already dropped 265 -> 117GB). Live jobs' projections now subtract bytes already on disk
  (`du -sb` of their output dir), never below 0. Passed after the fix (16.7GB remaining projected vs.
  87GB budget) — protection kept, the double count removed; `--force` was not used.

### Launch record, 2026-09-18 (all via `safe_sbatch.py`, `--exclude=g006`)
- **exp010 (T=32)**: fold 4 in job 25543 (verified: post-resume checkpoint rewritten at epoch 18, 0
  quota-guard skips); folds 0-3 in job 25568 (resubmission, above). 7 epochs from epoch 18 to 25.
- **exp012 (ACL anatomy-aware init)**: job 25572 (`--array=0-4`): fold 0 resumes from epoch 23 ->
  logged `Resumed from epoch 24` (its single final epoch); folds 1-4 restart from scratch (folds 3-4
  queued behind the 8-GPU per-user cap). One follow-up session, job 25579 (`afterany`), for the
  from-scratch folds (a 7h session got exp012 to epoch ~24/25 last time).
- **448px cache**: job 25548 finished — 20/20 shards COMPLETED, `data/cache_448` = 154GB (matching
  the 154GB projection), 4,407 studies, 24,369 series (identical to the 384px cache's count; the same
  2 series fail decode in both), spot check `(22, 448, 448) uint8` vs. `(22, 384, 384)`.
- **exp015 (ViT-B/448/T=24, labels_v3, same folds as exp006)**: `configs/exp015_dinov2b_slot_v3labels_448.yaml`,
  batch 8, no accumulation (so the comparison to exp006 is exact). Three chained sessions 25576 ->
  25577 -> 25578 (`afterany`), because T=32 took ~23 min/epoch (~9.7h for 25) and 448px/T=24 carries
  a similar token count. Sessions after a fold completes just redo its final eval (minutes).
- **Automatic finish**: job 25580 (`slurm/final_reselection.sbatch`, CPU, `afterany` of 25543, 25568,
  25579, 25578) runs `compare_runs.py` (per-label deltas vs exp006 -> `artifacts/final_compare_vs_exp006.md`)
  then `build_ensemble.py` greedy re-selection over {exp003, exp005, exp006, exp010, exp012, exp015}
  -> `artifacts/final_ensemble_reselection.txt`. `build_ensemble.py` gained `--out` because it
  previously always wrote `artifacts/ensemble_recipe.csv` (the DEPLOYED recipe's source); the
  re-selection writes `ensemble_recipe_reselect_20260918.csv` instead, and the accepted recipe was
  backed up to `ensemble_recipe_v1_accepted.csv`. Incomplete runs are skipped, not fatal.

## RESULTS: exp010 / exp012 / exp015 vs exp006, and ensemble re-selection (2026-09-20)

All 15 folds finished (exp010 T=32, exp012 ACL-prior, exp015 448px). **Integrity verified**: every
fold logged `epoch 24` in its final session, 0 quota-guard skips, 0 early stops; each fold has
`best.pt` + `oof.parquet` + `gold58_predictions.parquet`. (Total wall time 09-18 -> 09-19; the
automatic finisher, job 25580, ran 09-19 14:13.) Files: `artifacts/final_compare_vs_exp006.md`,
`final_ensemble_reselection.txt`, `final_bootstrap_gain.md`, `ensemble_recipe_reselect_20260918.csv`
(the deployed `ensemble_recipe.csv` was NOT touched; backup `ensemble_recipe_v1_accepted.csv`).

### Solo runs vs. exp006 (OOF vs labels_v2_final, non-gold; delta vs exp006)
| run | macro | ACL | MCL | MedMen | LatMen | MedOA | LatOA | PFOA | Effusion | Synovitis | Baker's | Contusion | Fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp006 (384/T=24) | 0.8681 | 0.8356 | 0.8247 | 0.9263 | 0.8711 | 0.8960 | 0.8592 | 0.8381 | 0.9042 | 0.8225 | 0.8846 | 0.8591 | 0.8956 |
| exp010 (T=32) Δ | −0.0001 | +0.0003 | −0.0027 | +0.0029 | +0.0024 | −0.0010 | +0.0005 | −0.0019 | −0.0008 | +0.0009 | +0.0021 | −0.0019 | −0.0017 |
| exp012 (ACL prior) Δ | +0.0010 | +0.0005 | −0.0122 | −0.0004 | +0.0026 | +0.0004 | +0.0050 | +0.0033 | −0.0007 | +0.0032 | +0.0037 | −0.0007 | +0.0077 |
| exp015 (448px) Δ | −0.0001 | −0.0027 | −0.0068 | −0.0009 | +0.0086 | +0.0001 | −0.0034 | +0.0013 | +0.0027 | +0.0022 | +0.0057 | −0.0027 | −0.0056 |

gold-58 tripwire (n=58, noisy) macro: exp006 0.8897, exp010 0.8914, exp012 0.8922, exp015 0.8926 —
same picture, all within noise; ACL on gold-58 stays ~0.97-0.98 for every run.

**Noise (paired cluster bootstrap over 3,681 report clusters, 300 reps, `scripts/bootstrap_ensemble_gain.py`)**:
macro bootstrap SD ≈ 0.0041. Solo deltas vs exp006: exp010 −0.0001 [−0.0022, +0.0025]; exp012 +0.0010
[−0.0010, +0.0032] (P>0 = 0.84); exp015 −0.0001 [−0.0024, +0.0025]. **None distinguishable from zero.**

**Conclusions**
1. **Resolution (448px, genuine finer spacing) and slice count (T=32) do not help** — the "biggest
   untested axis" is closed: at this recipe, effects bigger than about +0.0025 macro are ruled out.
   Per-label movement (e.g. exp015 Lateral Meniscus +0.0086, MCL −0.0068) is the size of per-label
   noise and should not be read as a real label-specific effect. Combined with the scale ladder
   (ViT-S -> ViT-B was worth ~+0.005 twice), image-side capacity levers are now saturated around a
   solo OOF macro of ~0.868-0.869.
2. **The anatomy-aware ACL init did not help ACL** (+0.0005) and hurt MCL (−0.0122); its +0.0010 macro
   is within noise. Consistent with the 2026-09-18 ACL conclusion (ACL's gap was mostly label noise).
   Per the standing decision, ACL work stops here.

### Ensemble re-selection over {exp003, 005, 006, 010, 012, 015} (greedy, vs labels_v2_final)
Solo: 003 0.8536, 005 0.8631, 006 0.8681, 010 0.8680, 012 0.8691, 015 0.8680. **Global greedy**:
+012 -> 0.8691, +003 -> 0.8725, +006 -> 0.8751, +015 -> 0.8761, +010 -> 0.8763, +005 -> 0.8764 —
**all six members made the cut, final OOF macro 0.8764** (deployed 3-member recipe: 0.8744; gold-58
tripwire of the new recipe 0.8949). Per-label selection 0.8776 (+0.0012 over global, below the 0.002
bar -> recommend global, as before). Paired bootstrap vs the deployed recipe: **6-member +0.0020 [+0.0011,
+0.0027]**; **3-member {012, 003, 006} +0.0008 [−0.0001, +0.0016]**.

**How to read it (my assessment, not a decision):** the 6-member gain is sampling-significant but
(a) selected in-sample — greedy over 6 candidates plus a per-label max over prob-mean/rank-mean both
pick winners on the same data, and the bootstrap does not correct for that, so the true out-of-sample
gain is smaller; (b) marginal steps 4-6 add +0.0010, +0.0002, +0.0001; (c) 0.002 is half of one
bootstrap SD of the metric itself and well below any public-LB resolution we have; (d) costs ~2.9x the
model inference of the deployed ensemble (16 vs 7 ViT-S-equivalent units before the extra T=32/448
cost) plus three new ~2.4GB weight datasets. The gain mostly reflects more same-family members with
different seeds/inits, not new information. The deployed recipe is unchanged pending a decision.

### Unrecognized jobs on this account
`sacct` for husnain.rasool since 09-18 also lists ~70 short (1-9 min) jobs named `rv_*`
(`rv_lr_pretrained_*`, `rv_organa_*`, `rv_p224_*`, `rv_seeds_*`, ...) that I did not launch — presumably
other work of the user's on the same account. All COMPLETED, no action taken; quota accounting is
consistent without them (567GB used = 385GB + 154GB cache + ~28GB of new checkpoints).

## Decisions 2026-09-20: null results accepted; housekeeping; exp016 PRE-REGISTRATION

**Null results, logged as such.** exp010 (T=32), exp015 (448px) and exp012 (ACL anatomy-aware init) are
**null** vs. exp006 (macro −0.0001 / −0.0001 / +0.0010; all paired 95% CIs span zero). **The resolution
hypothesis was tested properly — a genuinely rebuilt 0.2571mm/px cache, same recipe/folds/seed, all five
folds, full 25 epochs — and rejected**: effects larger than about +0.0025 macro are ruled out. The
earlier 400-study 384-vs-448 comparison remains void and is not evidence either way. Slice count (T=32)
likewise rejected. ACL work is closed.

**Ensemble: no probe on the six-member re-selection; the deployed recipe stays.** Reasons (user decision,
consistent with the analysis above): the +0.0020 OOF gain was selected in-sample, ~2.9x the inference cost,
and would break the efficiency-track economics. The deployed 3-member ensemble ({exp006, exp003, exp005})
and the exp008 efficiency notebook are **frozen and submittable throughout**; nothing below touches them.

**Housekeeping.** `data/cache_448` (154GB) deleted after verifying an empty queue; free quota 82.8 -> 235.9GB
(650GB limit). Rebuild if ever needed: `slurm/cache_build_448.sbatch` (~10 min on the CPU partition;
exp014/exp015 configs still point at it). The ~70 `rv_*` jobs on this account are not mine (presumably the
user's other work) and were left alone.

### exp016 — multimodal (report-text) training. PRE-REGISTERED before any run exists.
Motivation: each report has been used as 12 bits; ~977 characters of text per study went unused. This is the
one lever in the project that adds *information* rather than capacity.

**Deviation from the requested Stage 1 (must be read first): pretraining is fold-safe, not "on all 4,407
studies".** Contrastive/auxiliary training on a validation study's own (image, report) pair puts that
study's report information into the encoder before the study is scored. Labels are derived from reports, so
this inflates OOF against report-derived labels — and reports do not exist at test time, so the inflation
would not transfer to the LB. The head-to-head vs. exp006 would then be invalid. Every report-derived
training signal therefore uses train-fold studies only; gold-58 is excluded everywhere. (A contrastive
Stage 1 done this way needs one pretraining run **per fold**, 5x the stage-1 cost — a reason it is second.)

**Variant chosen first: joint auxiliary report-embedding prediction, not contrastive.** Why: (1) fold-safe
by construction and a single stage — the cost is one exp006-sized run, not 5 pretrainings + 5 fine-tunes;
(2) the recipe stays *identical* to exp006 apart from one added loss, so the comparison is clean;
(3) no InfoNCE batch-design fiddling — with ~a dozen languages, random in-batch negatives are mostly
different-language/site and solved by recognizing site, so a contrastive run would also need
language/site-grouped hard-negative batching. Contrastive stays available as the follow-up if this shows
any signal. Prior, stated honestly: every lever tried so far moved <= ~0.005, so the chance this clears
+0.0082 is low; the case for it is that the report text is new information.

**Text encoder: frozen multilingual sentence encoder (BAAI/bge-m3), not translate-to-English first.**
MIT license (verified on the model card), 100+ languages, 8,192-token context (reports run to ~4.7k
characters), dense embedding = L2-normalized [CLS]. Translation would cost a 4,407-report LLM pass
(hours) and add translation noise/hallucination on medical text; a cross-lingually aligned encoder puts
the same finding in different languages near each other, which is what the image side can learn. Used only
at training time to make targets — the deployed image model contains no text component.

**Target construction** (`scripts/build_report_embeddings.py`): per fold k, fit on that fold's train-split
embeddings only: subtract the mean, remove the top-10 principal components ("all-but-the-top", Mu &
Viswanath 2018), L2-normalize. Reason: the dominant directions of a multilingual embedding matrix encode
language and site/template style, and images carry scanner/site cues, so regressing onto the raw embedding
could be minimized by recognizing the site instead of the pathology. The build prints how much variance
crude language groups explain before vs. after removal, and a report-embedding -> label probe, so the
shortcut concern and the targets' label content are *measured*, not assumed. Files:
`artifacts/report_emb/fold{k}_target.npz` contain train-split studies ONLY; the dataset additionally
**asserts** no validation-fold or gold-58 study is present (tested on a real batch: smoke test smuggles
one in and requires the refusal).

**Model/loss**: optional `aux_head` on `SlotModel` (LayerNorm-Linear-GELU-Linear -> 1024) on the masked
mean of the cross-slot context; loss = label BCE + weight x (1 − cosine). `forward()` (all eval/inference)
never touches it. Verified locally: with the head off the model is **byte-identical** to the committed one
(same state-dict keys/values, identical outputs); with it on, the non-aux weights and `forward()` are
unchanged and gradients reach both the head and the backbone. Deployment note: strip `aux_head.*` from any
`best.pt` before staging (the Kaggle notebooks load strictly).

**Arms (2, fixed in advance):** `exp016a` weight 0.1, `exp016b` weight 0.5 — otherwise IDENTICAL to exp006
(ViT-B, labels_v3_candidate, same folds/seed, 384px, T=24, 25 epochs); verified programmatically that the
configs differ from exp006 only in `experiment` and `aux`. Two weights bracket the plausible range so a
null is not just a badly chosen weight; no tuning sweep (single-fold results can't discriminate ~0.008).

**Bar (user-specified, applied mechanically by `scripts/exp016_verdict.py`):** an arm counts as real iff its
solo OOF macro gain over exp006 is **> +0.0082** (2 x the 0.0041 cluster-bootstrap SD). Reported alongside:
paired 95% CI, per-label deltas with CIs, gold-58 tripwire. **Caveats fixed before results:** (1) two arms
=> 'best of two' is mildly optimistic; (2) OOF is scored against *report-derived* labels, and a model trained
to predict report text can gain on that proxy by absorbing report idiosyncrasies (mild-disease over-calling,
the very bias this project fights) that the image-reviewed truth does not share — gold-58 is the only clean
check and n=58 cannot resolve ~0.008; (3) one seed per arm. **If an arm clears the bar, before adopting it:**
run the shuffled-report control (same head/loss, reports assigned to the wrong studies — the dataset already
supports `aux.shuffle_control_seed`) to show the gain comes from report *content*, not from having an extra
head, and re-check gold-58.

**Guardrail (user):** the deployed ensemble + exp008 notebook stay frozen and submittable; exp016 is
upside-only; if it does not clear the bar by **Oct 15**, we ship what we have.

### exp016 amendment + launch record (2026-09-20, before any image-model result exists)
**Amendment to the pre-registration: target design D = 5 (not 10).** The build's diagnostics on the D=10
targets showed the top-10 removal fixed the language shortcut (language R² 0.15 -> 0.025) but also cost
label-relevant signal (report-embedding -> label probe macro AUC 0.889-0.897 raw -> 0.814-0.861; fold 0
lost 0.075). Since D is a target-design choice that uses only report text and report-derived labels — no
image result — it was set by a rule fixed before running the sweep (`scripts/sweep_abtt_d.py`): smallest
D with crude-language R² <= 0.05. Sweep (folds 0 / 3): D=2 0.058/0.060, D=3 0.054/0.052, **D=5 0.043/0.042**,
D=7 0.039/0.032, D=10 0.029/0.022 -> D=5. Final targets, all five folds: language R² 0.042-0.043 (raw
0.15), probe AUC 0.869-0.887 (raw 0.888-0.897). Everything else in the pre-registration — arms, weights,
the +0.0082 bar, the caveats, the controls-if-cleared — is unchanged.

**Text encoder loading (two failures on the way, both mine).** (1) BGE-M3 ships only a pickled `.bin`,
which `transformers` refuses to `torch.load` on torch 2.4 (CVE-2025-32434). (2) A conversion-PR commit is
not resolvable by hash via `from_pretrained`. Fix: load the already-cached 2.27GB `model.safetensors`
(conversion-bot PR commit `9a0624b8...`, a format that cannot execute code) directly into a model built
from the official config, requiring zero missing keys. Third-party-converted, so it is functionally
verified: 0 missing / 0 unexpected keys; EN-vs-ES-translation cosine 0.854 vs. 0.495 for an unrelated
sentence (assertion requires a gap > 0.1). 4,407 reports embedded, 0 truncated at 2,048 tokens.

**Pre-flight (smoke test, real data, 2 folds x 2 configs): passed.** Targets finite and unit-norm;
a validation-fold study and a gold-58 study each **refused** by the dataset when smuggled into the
targets file; shuffle control keeps 0/200 studies on their own report; real GPU batch runs forward +
joint loss + backward with aux cosine loss ~1.0 at init; eval `forward()` unaffected.

**Launched, all via `safe_sbatch.py`, `--exclude=g005,g006`** (both hold no-GRES co-tenant jobs):
- exp016a (weight 0.1): job **25776** (`--array=0-4`), chained follow-up session **25778**.
- exp016b (weight 0.5): job **25777**, chained follow-up **25779** (folds 3-4 briefly queued behind the
  8-GPU cap).
- Verdict job **25787** (CPU, `afterany` of both chains): runs `exp016_verdict.py` (the mechanical bar)
  and `compare_runs.py` -> `artifacts/exp016_verdict.md`, `artifacts/exp016_compare_vs_exp006.md`.
- An aborted first pipeline (jobs 25766-25772) was cancelled by explicit job ID after the embedder
  failed; the failed embedding runs (25766, 25773) wrote nothing.
- Expected ~1-1.5 days wall (exp006-sized runs, ~7h/session x 2 sessions, 10 tasks under an 8-GPU cap).

## 2026-09-21 — exp016 RESULT: auxiliary report-embedding objective does NOT clear the pre-registered bar

All 10 folds (2 arms x 5) completed 24/24 epochs, no errors; verdict job 25787 ran the mechanical bar
(`artifacts/exp016_verdict.md`, `artifacts/exp016_compare_vs_exp006.md`). Baseline exp006 macro 0.8681
(bootstrap SD 0.0042). Bar (fixed 2026-09-20): point macro gain > +0.0082.

| arm | OOF macro | Δ vs exp006 | 95% paired CI | P(Δ>0) | gold-58 macro (Δ) | verdict |
|---|---|---|---|---|---|---|
| exp016a (aux weight 0.1) | 0.8717 | +0.0036 | [+0.0012, +0.0061] | 0.99 | 0.8899 (+0.0002) | does not clear |
| exp016b (aux weight 0.5) | 0.8693 | +0.0012 | [-0.0016, +0.0037] | 0.78 | 0.8883 (-0.0015) | does not clear |

**Reading, without spin.**
- Per the pre-registration this is a null: neither arm reaches +0.0082, so neither is adopted and the
  shuffled-report control is NOT run (it was only conditional on clearing the bar).
- exp016a's gain is probably real *on the OOF proxy* (CI excludes zero) but it is ~0.4x the bar, and the
  clean check (gold-58, n=58) shows nothing (+0.0002). It cannot be distinguished from "absorbed report
  idiosyncrasies that help against report-derived labels" — pre-registered caveat (2).
- The heavier weight (0.5) is worse than 0.1 on macro and on gold-58, i.e. no dose-response toward more
  report signal being better.
- Per-label: Synovitis is the one label that moves in both arms (+0.0115 [+0.006,+0.016] / +0.0106
  [+0.004,+0.016]) — plausibly the label whose report text is most under-used by 12-bit labels — but on
  gold-58 Synovitis goes the wrong way in both (-0.013 / -0.011). Other per-label deltas straddle zero;
  12 labels x 2 arms means a few will look significant by chance; not interpreted individually
  (e.g. exp016b ACL +0.0133 alongside MCL -0.0152).
- Caveats as pre-registered: two arms (best-of-two optimism), OOF vs report-derived proxy, one seed/arm.

**Decision.** Deployed 3-member ensemble ({exp006, exp003, exp005}) and the exp008 efficiency notebook
are unchanged and remain frozen/submittable. No LB probe. The contrastive (stage-1) variant is NOT
launched: the cheaper auxiliary variant, whose job was to show whether report text carries usable extra
signal, showed at most a sub-bar, gold-unconfirmed gain, so the evidence does not justify the 5-run
stage-1 cost. Cutoff for further exp016 work remains Oct 15; nothing further is queued. All exp016 jobs
finished; nothing of ours is running or queued (`rv_*` jobs are not ours and were left alone).

**Housekeeping note.** `exp016*/fold*/best.pt` contain `aux_head.*` weights; strip them before any Kaggle
staging (strict load) if an exp016 member is ever used. Quota at check: 445 GiB / 650 GiB.

## 2026-09-21 — exp017 (severity-aware ordinal targets): step 1 audit result = STOP, no grade field

Motivation (user): report over-calling of mild/trace findings vs. expert-panel severity thresholds is the one
known mechanism left after exp016; hypothesis is that 3 cumulative targets P(>=mild/moderate/severe) with
inference on P(>=moderate) would align the model with the panel threshold. Step 1 was a zero-GPU audit of
whether the existing extractions preserve a per-finding grade; instruction was to STOP and report if not.
Script `scripts/audit_severity.py`, output `artifacts/severity_audit.md` (HPC).

**Finding.** The extraction schema (`rubric_v1/v2`, `label_reports.py`) stores per finding only
`present` {yes,no,uncertain}, `probability`, `severity_phrase`, `evidence_quote`, `imputed`. There is **no
ordinal grade field**. `severity_phrase` is a short free-text quote in the report's own language (Spanish,
Croatian, Greek, Turkish, Bulgarian, Dutch, ...), and is frequently just the finding name ("Derrame",
"Rotura de menisco interno"). Applying a multilingual keyword lexicon (en/es/fr/de/nl/it/pt; does not cover
hr/bg/el/tr) to the Qwen v2 phrases of v3-positive studies, a grade word is found in only:
ACL 40%, MCL 34%, Med Men 19%, Lat Men 16%, Med OA 31%, Lat OA 33%, PF OA 41%, Effusion 44%, Synovitis 13%,
Baker's 25%, Contusion 8%, Fracture 14% of positives. Where a grade is found, the mild/trace share is:
ACL 0%, MCL 14%, Med Men 20%, Lat Men 45%, Med OA 19%, Lat OA 7%, PF OA 14%, Effusion 43%, Synovitis 71%,
Baker's 67%, Contusion 58%, Fracture 87% — but those are conditioned on the 8-44% that are gradable, so the
strict "% of current positives that are mild" is NOT measurable from what we stored (a floor, at most ~0-19%
per label, is all the data supports).

**Two further consequences.** (1) The rubric already sends mild/grade-1/trace to the *uncertain* band
(p = 0.20-0.35), so v3 positives (p >= 0.5) are already enriched for above-threshold findings; the
mild-overcalling the ordinal head targets was partly removed at label time (that is where the v2->v3
gains came from), so remaining headroom from a 3-level head is unclear. (2) Building real {0,1,2,3}
targets needs a NEW LLM extraction pass with a grade field (~4,407 reports x 2 models on the H100s), not a
re-parse — which the user's instruction reserves for confirmation.

**Status: exp017 STOPPED at step 1; no model code changed, nothing trained or queued.** Decision pending
user: (a) authorise a grade-extraction LLM pass (rubric_v3 with a 4-level grade per finding, calibrated on
gold-58 first), then steps 2-3; or (b) drop exp017. Caution recorded for step 3 if it proceeds: gold-58
(n=58) cannot resolve differences below ~0.02-0.03 macro (see exp010-exp016 use of it as a tripwire only),
so the "improves gold-58 => launch folds 1-4" gate is very loosely informative; a fold-0 OOF comparison
with the paired cluster bootstrap should be reported alongside it.

**Deadline hygiene reminder (per user, Oct 22 final selection).** The two selected Kaggle submissions MUST
be: (1) the 3-member ensemble notebook `submit-ensemble.ipynb` ({exp006, exp003, exp005}, public LB 0.893);
(2) the distilled student notebook `submit-efficiency.ipynb` (exp008, public LB 0.885, single T4; user
quotes ~13 min runtime — our own logged figures are 447s on H100 and a ~12-19 min T4 projection, so the
Kaggle-side run time should be re-read from the notebook's own log before deadline). Final Submit /
selection clicks are the user's. Efficiency-track status/score: nothing available locally (the efficiency
score needs the private-LB maxAUC; only the public LB numbers above are known).

## 2026-09-21 — exp017 DROPPED (on reasoning; NOT tested) and modeling phase CLOSED

**exp017 (severity-aware ordinal / cumulative targets) is dropped on reasoning, not on an experiment.**
Reason (user decision): rubric v2/v3 already maps mild/trace findings to p = 0.20-0.35 versus definite
findings > 0.5, so the soft labels already encode an implicit ordinal target; a formal 36-logit ordinal head
would add little for the cost of a new 2-model LLM grade-extraction pass + 5 folds + an LB probe, judged on a
gold-58 signal too coarse to resolve it. Step 1 (audit) found no stored grade field (see the entry above).
Nothing was trained, launched or changed for exp017. It is a hypothesis *not tested*, not one rejected by data.

**Modeling phase CLOSED (user, 2026-09-21).** Recipe frozen: ensemble {exp006, exp003, exp005} + exp008
efficiency notebook. No new training runs unless the user explicitly starts one. Re-open only on a merger
partner or a forum insight.

**Disk cleanup done 2026-09-21.** Deleted `best.pt` + `checkpoint.pt` (50 files, listed first, explicit paths)
for the non-recipe experiments exp010, exp012, exp015, exp016a, exp016b: 70.6 GB (66.2 GiB); quota
445.07 -> 378.87 GiB of 650. `config.yaml`, `oof.parquet`, `gold58_predictions.parquet`, `summary.csv`
retained everywhere. exp001/002/007/009/013/014 held no `.pt` (removed in earlier cleanups); exp004's directory
is gone from disk (its numbers live in this ledger). Recipe experiments exp003/005/006/008: untouched, all 5
`best.pt` per experiment verified present. Consequence to remember: the deleted models cannot be re-run for
inference or resumed without retraining.

## CONSOLIDATED ABLATION LEDGER (2026-09-21) — every experiment, hypothesis, result, verdict

Conventions. **OOF** = 5-fold out-of-fold macro-AUC against `labels_v2_final` binarised at 0.5 on the 4,349
non-gold studies (a noisy report-derived proxy; folds grouped by report cluster). **Gold-58** = 58 image-reviewed
studies, never trained on; n=58 cannot resolve differences below ~0.02-0.03 — a tripwire, never a selector.
Gold-58 for exp001-009 is the mean over per-fold models (model cards); for exp010-016 and the ensemble it is the
fold-ensembled prediction (`compare_runs.py`); the two conventions differ by ~+0.009 for the same model
(exp006: 0.8806 vs 0.8897), so compare gold-58 only within a convention. **Noise scale**: paired cluster-bootstrap
SD of solo macro ~= 0.0041, so 2 SD = 0.0082; a paired CI was computed for exp010/012/015/016 only.

### A. Label engine (text side)

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| LE1 rubric v1, Qwen2.5-72B | Severity-thresholded LLM rubric recovers image-review labels from reports | gold-58 macro AUC 0.851 (kappa 0.598) | Baseline |
| LE2 rubric v2 (Qwen / Llama-3.3-70B) | Per-label rubric fixes (meniscus degeneration, trace effusion, ...) | Qwen 0.846 / Llama 0.870 macro AUC | Model-dependent per label; kept both |
| LE3 `labels_v2_final` | Per-label strategy (Qwen / Llama / blend), chosen by strict LOO on gold-58 | macro LOO 0.823 (strict LOO; lower by construction) | **Kept** as v2 training target and as the OOF evaluation proxy |
| LE4 `labels_v3_candidate` | Noisy-student: blend report labels with exp002 image OOF, per-label weight by LOO | Two-criterion acceptance, exp005 (v3) vs exp002 (v2): OOF-vs-original-labels 0.8631 vs 0.8489; gold-58 +0.0139 | **Kept** — the deployed training target (exp005/006/008) |
| LE5 `labels_v4_candidate` | Second noisy-student round | LOO macro 0.892 vs v3 0.904 (-0.012) | **Rejected** — confirmation-bias regression; iteration stopped at v3 |

### B. Image models, ensemble, distillation

| # | Hypothesis | OOF macro | Gold-58 macro | Verdict |
|---|---|---|---|---|
| exp001 DINOv2-S 2.5D slot model, v1 labels (pipeline validation) | Pipeline trains end to end | 0.6511 (pre-fix, 3/5 folds collapsed) | 0.6193 | Root cause found (09-07) and fixed; never a real baseline |
| exp002 DINOv2-S slot, v2 labels | 2.5D slot architecture works | 0.8489 (0.8481 in a later eval) | 0.8675 | Baseline; superseded by exp006; supplied the image OOF for v3 labels |
| exp003 DINOv2-B slot, v2 labels | Bigger backbone helps | 0.8536 (+0.0055 vs exp002) | 0.8714 | **In deployed ensemble** |
| exp004 DINOv3-S/16 slot, v2 labels | Newer backbone helps | 0.8237 (-0.0252 vs exp002) | worse (per label up to -0.11) | **Rejected** — no label favoured it |
| exp005 DINOv2-S slot, v3 labels | Better labels help | 0.8631 (+0.0142 vs exp002) | 0.8815 | **In deployed ensemble** |
| exp006 DINOv2-B slot, v3 labels | Bigger backbone on better labels | **0.8681** (+0.0050 vs exp005) | 0.8806 (fold-ens 0.8897) | **In deployed ensemble; baseline for all later comparisons** |
| exp007 3D ResNet-50 slot, v3 labels | Architectural diversity (3D) | 0.693 (killed at 40%); corr with 2D ensemble 0.444 | n/a | **Killed** — diverse but never reached the 0.80 usefulness bar; ~10.5 h/fold |
| exp008 DINOv2-S distilled from the ensemble (BCE + KL to teacher) | One small model near the ensemble at a fraction of the cost | 0.8708 (-0.0036 vs ensemble) | 0.8882 | **Deployed efficiency model**; public LB 0.885; ~7.5 min on H100 |
| exp009 ConvNeXt-V2-B slot, v3 labels | A different backbone family adds diversity | 0.8652 | n/a | **Rejected** — accurate but redundant (corr 0.9296 with exp006; bar < 0.90) |
| **Ensemble {006, 003, 005}** (greedy forward selection, prob-mean) | Ensemble beats best solo | **0.8744** | 0.8960 (fold-ens) | **Deployed**; public LB **0.893** (OOF-LB gap -0.0186, trustworthy) |

### C. Post-probe ablations (all vs exp006)

| # | Hypothesis | OOF macro (Δ vs exp006) | Gold-58 macro (fold-ens) | Verdict |
|---|---|---|---|---|
| exp010: T=32 slices (vs 24) | Small findings are slice-count-limited | 0.8680 (-0.0001; 95% CI [-0.0022, +0.0025]) | 0.8914 | **Null** — effects > ~+0.0025 ruled out |
| exp015: 448 px on a rebuilt 0.2571 mm/px cache (vs 384) | Small findings are resolution-limited | 0.8680 (-0.0001; CI ~[-0.0023, +0.0025]) | 0.8926 | **Null** — tested properly and rejected |
| exp013/exp014: 400-study 384-vs-448 pilot | Cheap resolution test | crashed / invalid | n/a | **Void** (harness bug: gold-58 missing from subset folds; underpowered). Numbers not cited |
| exp012: anatomy-aware ACL query init (sagittal-biased) | ACL is a head/attention deficit | 0.8691 (+0.0010, CI spans 0); ACL +0.0005, MCL -0.0122 | 0.8922 | **Null** — ACL's low OOF is proxy-label noise (gold-58 ACL AUC ~0.988 for report labels and image ensemble alike) |
| exp016a: auxiliary report-embedding regression (BGE-M3, all-but-the-top D=5), weight 0.1 | Report text carries usable signal beyond the 12 labels | 0.8717 (+0.0036, CI [+0.0012, +0.0061]) | 0.8899 (+0.0002) | **Null vs pre-registered bar (+0.0082)**; sub-bar OOF-proxy gain, no gold-58 confirmation |
| exp016b: same, weight 0.5 | Stronger aux signal | 0.8693 (+0.0012, CI [-0.0016, +0.0037]) | 0.8883 (-0.0015) | **Null**; no dose-response |
| exp017: severity-aware ordinal (CORN) targets | Mild/trace over-calling is the residual mechanism | not run | not run | **Dropped on reasoning, NOT tested** (rubric already maps mild to p 0.20-0.35; no stored grade field; cost vs an unresolvable signal) |

### D. Inference / ensemble-composition ablations

| # | Hypothesis | Result | Verdict |
|---|---|---|---|
| Laterality-swap TTA | Averaging L/R-swapped predictions helps | -0.0014 OOF on the final ensemble | Rejected |
| Per-label ensemble recipe vs global | Per-label member choice adds accuracy | +0.0010 OOF (below the 0.002 bar) | Rejected — complexity not justified |
| 6-member re-selected ensemble {012, 003, 006, 015, 010, 005} | A wider ensemble beats the 3-member recipe | 0.8764 vs 0.8744 (+0.0020, CI [+0.0011, +0.0027]); in-sample selection; ~2.9x inference | Not adopted — breaks efficiency economics; no LB probe |
| Rank-mean vs prob-mean ensembling (exp002+exp003 test, 09-09) | Rank averaging is more robust | prob-mean 0.8642 vs rank-mean 0.8633 OOF (gold-58 0.8953 vs 0.8965): indistinguishable | Both implemented (rank-mean via an empirical-CDF lookup for single-study inference); no accuracy difference to exploit |

### E. Not run, by decision

| Item | Reason |
|---|---|
| exp011 MRNet pre-training / KneeMRI external data | Dropped on licence grounds (no external data used); KneeMRI licence ambiguous |
| Label-engine round 3 | Deprioritised, then closed: v4 regressed; label iteration stopped at v3 |
| Contrastive image-text pre-training (exp016 stage 1) | Not launched: the cheaper auxiliary variant showed no bar-clearing signal |

**Reading the whole record.** Across backbone size, label quality (v1 -> v3), resolution, slice count, head
initialisation, ConvNeXt, 3D, an auxiliary text objective, TTA and ensemble composition, the only levers that
moved OOF beyond noise were (1) ViT-B over ViT-S (+0.005), (2) v3 noisy-student labels (+0.014) and (3)
ensembling (+0.006 over the best solo, 0.8744 vs 0.8681). Everything tried after the v3/ViT-B/ensemble recipe
landed sits inside about +/-0.004 of exp006. Plateau at OOF ~0.868 solo / 0.874 ensemble; public LB 0.893 / 0.885.

## 2026-09-21 — public benchmark audit (intelligence only; modeling frozen)

Report: `artifacts/public_benchmark_audit.md` (local; 18 third-party kernels pulled to gitignored `kernels/`, none executed).
Findings logged: (1) efficiency-board rank 425/3,976 is an efficiency-metric rank, not an AUC rank — 2,450 of 3,976 entries
(62%) have higher public AUC than 0.885, 730 have >= 0.94 (a public-fork cluster, LB-tuned). (2) A DINOv2 slot-attention family
plateaus at ~0.89 publicly (pilkwang 20-member 0.891 vs ours 0.893). (3) Public evidence for a design-level gap: a single
CoAtNet-2 with 3-adjacent-slice RGB windows + per-finding attention MIL scores 0.924 (no ensemble/TTA); a single EfficientNet-B3
MIL scores 0.903 from a lower OOF than ours (OOF is not comparable across pipelines). (4) Untested difference: public crops are
130-140 mm; ours is 115 mm (exp015 changed resolution, not FOV). No detector/bbox datasets in any kernel's sources. Caveats:
0.94x numbers are LB-tuned forks; 0.924 model selected its epoch on gold-58. No action taken; no probes. Access note: the Kaggle
CLI ignored the OAuth login because a stale legacy `~/.kaggle/kaggle.json` took precedence; worked around with a scratch
`KAGGLE_CONFIG_DIR` (deleted afterwards), user files untouched.

## 2026-09-21 — MODELING RE-OPENED: exp018 (adjacent-slice Window-MIL, CoAtNet-2) — PRE-REGISTRATION

Trigger (user decision, after the public-benchmark audit): pilkwang's 20-member DINOv2 ensemble already used
adjacent-slice RGB triplets + 130 mm FOV + per-finding queries and still stalled at 0.891 public; the one public
result that breaks the DINOv2 ceiling is a single fully fine-tuned CoAtNet-2 with attention-MIL over ~42 windows drawn
from all planes (0.924 public, no ensemble/TTA; author-claimed, unverified by us). exp018 implements that design
in-house. The deployed 3-member ensemble ({exp006, exp003, exp005}, LB 0.893) and exp008 (LB 0.885) stay frozen and
submittable; exp018 is upside-only. (Supersedes the 2026-09-21 "modeling phase closed" entry for this one line of work.)

**Design (all fixed before any result).** Code: `src/rsna_knee/wmil.py`, `scripts/build_cache_wmil.py`,
`scripts/train_wmil.py`, `configs/exp018_coatnet2_wmil_v3labels.yaml`. The public dreaddevelopment kernels were an
architectural reference only; no third-party trained checkpoint, label table or code is used.
- Cache: per-study uint8 [64,336,336]; 140 mm FOV -> 336 px; slices from the 6%-94% span of each series ordered by
  ImagePositionPatient (the deployed ordering code); 5 slots = Sag-FS 18, Sag-nonFS 14, Cor-FS 12, Cor-nonFS 8,
  Axial 12. **Disk: 4,407 x 7.2 MB = 31.8 GB, not ~155 GB** — the ~155 GB figure is the exp015 448 px cache; "5 slots x
  64 slices" (=320/study, 159 GB) would only duplicate slices since series have 20-45 slices (median 30), and the
  reference recipe's 64 is the study total. I built the 64-total layout and flagged the discrepancy to the user.
- Window = 3 adjacent slices of the same slot as RGB (never spans two slots); <= 54 candidate windows/study. (The
  public code lets triplets straddle slot boundaries; ours does not.)
- Model: timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (ImageNet-12k -> 1k init, 72.9 M params, 1024-d), fully
  fine-tuned, input upsampled 336 -> 384; per-finding attention-MIL over all windows (LayerNorm -> Linear-Tanh-Dropout
  -Linear(12) softmax over windows -> pooled x per-finding weight); no per-series CLS, no slot bottleneck.
  Backbone weights = timm's `model.safetensors` (282 MB, HF hub `timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`),
  downloaded once to the shared HF cache — the same kind of ImageNet-class init as our DINOv2 backbones. Interpreting
  "zero third-party pre-trained checkpoint files" as "no third-party *knee-trained* checkpoints"; random init would not
  be the tested design. Flagged to the user.
- Training: soft targets from OUR labels_v3, OUR folds; BCE with pos_weight = clip((1-p)/p, 1, 10), p = train prevalence
  clipped to [0.03, 0.7]; AdamW backbone LR 3e-5 / head LR 1e-3, WD 0.02; OneCycle, 15% warm-up; 16 epochs; batch 8
  studies x 12 random windows; grad-clip 3; bf16; grad checkpointing; light intensity jitter (+-10%); no flips/rotations.
  These hyperparameters are the reference recipe's; nothing was tuned.
- Epoch selection: top-3 epochs by FOLD-VALIDATION macro-AUC (vs labels_v3 binarised, as in train.py), weights averaged
  (params and BN statistics). Evaluated on ALL candidate windows. Gold-58 is never consulted for selection.
  (The reference selected epochs on gold-58; its 0.9167 own-gold figure is therefore optimistic.)
- Laterality, corrected: coronal/axial mirrored horizontally for right knees (XOR the image's own row_x<0);
  sagittal slices put in ascending patient-x and REVERSED for right knees (the medial-lateral axis is the slice axis).
  Side = validated final_side.csv -> DICOM tag -> geometric image-centre rule (|median x| >= 20 mm; x<0 = right).
  **Observation about the deployed pipeline (not acted on):** it flips the horizontal axis of sagittal images for right
  knees (an anterior-posterior mirror, not a laterality normalisation) and leaves the 1,491 studies (34%) with no
  validated side un-normalised in training; at inference it averages both orientations for those. exp018 avoids both.

**Gate and kill rule (fixed now).** Fold 0 only first. KILL: fold-0 gold-58 macro-AUC of the averaged model < 0.880 ->
halt, report, launch nothing further. If >= 0.880, folds 1-4 launch automatically (`scripts/exp018_gate.py`, through
`safe_sbatch.py`, disk-guarded). 0.880 is ~ the deployed exp006 fold-mean gold-58 (0.8806): a pass means "not worse
than the deployed recipe on a tripwire with n=58 (SE ~0.03)", not evidence of improvement; the gate report therefore
also gives fold-0 OOF vs exp006's same fold. Compare-to-deployed bar for adoption (not fold 0): the same paired
cluster-bootstrap machinery as exp010-016 (macro SD ~0.0041). A blend test (exp018 + {003, 005, 006}, OOF, prob- and
rank-mean, paired bootstrap) follows the 5-fold run. Probe #3 (Kaggle) is the user's submit click; a self-contained
single-model notebook will be staged; its dual-T4 runtime can only be *measured* on Kaggle (we profile on H100 and
project, as for the earlier notebooks).

**Caveats stated in advance.** (1) The 0.924 reference figure is the author's public-LB claim for their weights and
labels; our OOF/gold under labels_v3 and the reference's public score are not the same measurement. (2) OOF against
report-derived labels is not comparable across pipelines (a B3 MIL model scored 0.903 public from an OOF of 0.854, below
ours); a public-LB probe is the only cross-pipeline check, and it is one probe of the budget. (3) One seed per fold.
(4) Windows are sampled 12/study in training and 54/study at evaluation (the reference does the same with 12 vs 24-94).
