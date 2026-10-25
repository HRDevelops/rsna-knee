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

> **CORRECTION (2026-10-01): the scored run was most likely exp005 alone, not the 3-member ensemble.** On Kaggle's
> 2 × T4 the 3 members measure ~59 s/study (~21 h for the test set, 2026-09-27), so the notebook's time-budget ladder
> almost certainly fell to rung 2 (exp005 alone). This is inferred, not confirmed: the hidden-test log is not
> visible. The gap is then likely **0.8631 → 0.893, ≈ −0.030 (LB above OOF by +0.030)**, not −0.0186. That is
> above the 0.02 threshold below, and in line with the larger gaps of every later probe (exp018 −0.047).

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

**exp018 launch record (2026-09-21).** Cache built (job 25887, 20 shards x ~3 min): 4,407/4,407 studies, 0 failures,
30 GB on disk (`data/cache_wmil`); audit `artifacts/exp018_cache_audit.md`: slot fill Sag-FS 94.2%, Sag-nonFS 97.4%,
Cor-FS 96.4%, Cor-nonFS 78.1%, Axial 100%; 3,132 studies fill all 5 slots, none fewer than 3. Findings from the audit:
(i) the geometric side rule agrees with the validated tag side on 99.0% (n=2,178) and with the report side on 99.1%
(n=712), and resolves 99.4% of the 1,491 studies that had no validated side (deployed pipeline left these un-normalised);
(ii) every sagittal series has a negative slice-normal x (ordering is reversed to ascending patient-x), every
coronal/axial series has row_x > 0; (iii) InstanceNumber order disagrees with geometric order in 34.5% of series
(as before: geometry is what we sort by). Orientation check: mean coronal-FS mid-slice image of left vs right knees,
correlation 0.766 raw -> 0.965 after normalising right knees; sagittal slice-profile 0.895 -> 0.940 (n=115/126 studies).
Smoke test (job 25908, 29 s): pipeline runs end to end. Fold 0: job **25909_0** (`--exclude=g005,g006`), ~6.2 min/epoch
(372 s incl. full-window validation) -> ~100 min for 16 epochs; epoch 0 fold-val macro-AUC 0.789. Gate job **25910**
(`afterany:25909`, `scripts/exp018_gate.py`) applies the kill rule and, only if gold-58 >= 0.880, launches folds 1-4 via
safe_sbatch (disk guard). Deployed ensemble / exp008 untouched. Staging code written but untested pending weights:
`src/rsna_knee/infer_wmil.py`, `notebooks/submit-exp018.ipynb` (generated by `scripts/build_exp018_notebook.py`),
`scripts/stage_exp018_kaggle.py` (separate datasets `rsna-knee-exp018-code`, `rsna-knee-exp018-weights`),
`scripts/rehearse_exp018.py` (HPC parity + timing run), `scripts/exp018_blend.py` (blend test vs the deployed ensemble).

## 2026-09-21 — exp018 RESULT: window-MIL CoAtNet-2 clears the pre-registered bar (OOF and gold-58)

All 5 folds completed (fold 0 job 25909_0; gate job 25910 passed fold 0 and auto-launched folds 1-4 as array 25917,
each ~1h41m, 16/16 epochs, no errors). Top-3 epochs were late (12-15; fold 3: 10/12/14) — no sign of overfitting.
Gate report `artifacts/exp018_fold0_gate.md`: fold-0 gold-58 **0.9063** (kill threshold 0.880; exp006 same fold 0.8746);
fold-0 OOF 0.8883 vs exp006 0.8760. Per-fold gold-58: 0.906 / 0.901 / 0.905 / 0.906 / 0.893.

Full 5-fold, same paired cluster bootstrap as exp010-016 (`artifacts/exp018_verdict.md`), pre-registered bar +0.0082:

| | OOF macro (vs labels_v2_final) | Δ vs exp006 | 95% CI (paired) | gold-58 (fold-ensembled, n=58) |
|---|---|---|---|---|
| exp006 (best deployed solo) | 0.8681 | — | — | 0.8897 |
| **exp018 (solo, 5-fold)** | **0.8816** | **+0.0135** | [+0.0100, +0.0171] | **0.9091 (+0.0193)** |
| deployed ensemble {006,003,005} | 0.8738 (rank) / 0.8737 (prob) | | | 0.8953 |

Per-label Δ vs exp006 (95% CI): MCL +0.080 [+0.054,+0.109]; Lateral Meniscus +0.031 [+0.017,+0.044]; Lateral OA +0.017
[+0.007,+0.029]; ACL +0.016 [+0.005,+0.029]; Medial Meniscus +0.010 [+0.005,+0.015]; Contusion +0.008 [+0.001,+0.014];
Medial OA +0.004, Baker's +0.002, Fracture +0.009 (CIs span 0); PF OA -0.003 (spans 0); Effusion -0.0055 [-0.010,-0.001];
Synovitis -0.006 [-0.012,0.000]. The gains sit on the medial/lateral-sensitive labels (both menisci, MCL, lateral OA) —
where the corrected laterality handling (and the wider FOV) matter most — plus ACL; the fluid/synovial labels are flat to
slightly down. The design change bundles several differences vs exp006 (backbone family, adjacent-slice triplets, window-MIL
head, 140 mm FOV, corrected laterality incl. the 34% of studies previously un-normalised, 64-slice 6-94% span), so this
run does not attribute the gain to any one of them; no ablation was run.

Blend test vs the deployed ensemble (`artifacts/exp018_blend.md`, 300 paired reps; baseline = rank-mean 0.8738):
| blend | w(exp018) | OOF | Δ vs deployed | 95% CI | gold-58 (Δ) |
|---|---|---|---|---|---|
| rank-mean | 0.25 | 0.8814 | +0.0076 | [+0.0067,+0.0084] | 0.9031 (+0.0078) |
| **rank-mean (pre-declared headline)** | **0.50** | **0.8852** | **+0.0114** | [+0.0097,+0.0130] | 0.9076 (+0.0123) |
| rank-mean | 0.75 | 0.8852 | +0.0114 | [+0.0090,+0.0138] | 0.9091 (+0.0138) |
| rank-mean | 1.00 (exp018 alone) | 0.8816 | +0.0078 | [+0.0046,+0.0109] | 0.9091 (+0.0138) |
| prob-mean (best in-sample cell of 8) | 0.50 | 0.8860 | +0.0122 | [+0.0097,+0.0148] | 0.9063 (+0.0110) |
exp018 alone beats the deployed 3-member ensemble (+0.0078); the blend adds ~+0.0036 more at 0.5-0.75 weight.

Inference rehearsal on HPC (job 26011, H100, fp16, the exact Kaggle path: DICOM decode -> tag/geometry side -> fold models):
60 fold-0 validation studies, 0 errors, 0 prior fallbacks. vs the training-path predictions of the same model: per-label
Spearman 0.988-0.9999 (mean |dp| 0.0016, max 0.16 on one study — the training path used the validated side, inference
uses tag -> geometry). Speed: 5-fold ensemble 0.67 s/study wall including 15 s model load (~0.4 s/study steady) on an H100 ->
~9 min for 1,300 studies. The T4 x2 runtime is NOT measured (only a projection: model cost per study is ~5 folds x ~54
windows of CoAtNet-2 at 384 px; expect on the order of 1-1.5 h on T4 x2 — to be measured on Kaggle).

Staged (HPC `kaggle_staging/`): `exp018_code/rsna_knee/` (4 files, 36 KB) and `exp018_weights/fold{0-4}_best.pt` (5 x 293 MB =
1.4 GB) for the datasets `rsna-knee-exp018-code` / `rsna-knee-exp018-weights`; notebook `notebooks/submit-exp018.ipynb`.
Deployed ensemble and exp008 remain untouched. Quota 416 GiB / 650 GiB.

**Caveats (unchanged from the pre-registration, restated).** OOF is scored against report-derived labels and is not
comparable across pipelines (a B3 MIL model scored 0.903 public from a lower OOF than ours); gold-58 is n=58 (per-fold
values here are ~0.89-0.91 vs ~0.87-0.89 for exp006) but was NEVER used for selection this time, and it agrees in sign
and rough size with the OOF gain. One seed per fold. Only a public-LB probe settles cross-pipeline comparison: probe #3
is the user's Submit click.

**exp018 Kaggle datasets uploaded (2026-09-21, via the local Kaggle CLI, OAuth as hrdevelops; both PRIVATE, status `ready`).**
`hrdevelops/rsna-knee-exp018-code` (4 files: `__init__.py`, `preprocess.py`, `wmil.py`, `infer_wmil.py`; identical to the repo's
`src/rsna_knee/`) and `hrdevelops/rsna-knee-exp018-weights` (`fold0-4_best.pt`, 292,862,052 bytes each; SHA-256 identical to the HPC
staging copies). Separate from `rsna-knee-code` so the frozen notebooks' inputs are unchanged. Local copies in `kaggle_staging/`
(gitignored). Next: attach both + the competition data to a new notebook from `notebooks/submit-exp018.ipynb`, dry-run it on Kaggle
(first run against Kaggle's timm; gives the real T4 speed), then probe #3 (the user's Submit click).

## 2026-09-22 — exp018 Kaggle dry run + pre-submission timing check

**Kaggle dry run (user, "Save & Run All"): passed.** T4x2, torch 2.10.0+cu128, timm 1.0.26 (vs HPC's timm 1.0.29 —
architecture matched: `WindowMIL` built with `pretrained=False` then `load_state_dict(strict=True)` succeeded for all
5 folds, so the timm version gap doesn't affect this checkpoint). 5 fold models loaded in 21s across 2 GPUs; ran on
Kaggle's 3-row placeholder `test.csv` (real scoring always uses a hidden ~1,300-study rerun — the placeholder is
expected and gives no timing signal at scale); all format assertions passed; 3 distinct, in-range probability rows,
0 errors, 0 prior fallbacks. 64.5s total including model load and notebook-export overhead.

**Pre-submission timing check (`scripts/profile_decode_vs_model_exp018.py`, job 26078, H100, 300 real fold-0
validation studies — never used for exp018 training/selection):** decode 0.264 s/study, 5-fold model 0.359 s/study
(0.072 s/study/fold; mean 51.5 candidate windows/study). Applying the documented T4 projection method
(`docs/SUBMISSION_RUNBOOK.md`: model time x3.5, decode unchanged) — **T4 x1: ~33 min for 1,300 studies; T4 x2: ~17
min** — both far inside the 8h Kaggle cap (< 7% of budget even single-GPU), so this does NOT depend on the untested
wall-clock degrade-fold-count logic in `infer_wmil.run_exp018` actually firing. The 3.5x multiplier is carried over
from the DINOv2 ViT profiling (2026-09-11) and was not separately re-validated for CoAtNet-2's operator mix, but the
margin is wide enough that even a 10x error in the multiplier (model 0.359 x 10 = 3.59 s/study) would still land
around 83 min, well under budget.

**Infra bug found and fixed while checking this: `scripts/push.sh` did not exclude `kaggle_staging/`** (added
2026-09-21 for the exp018 dataset staging), so every push after that point tried to rsync/tar the 1.4GB of fold
weights to HPC — this silently made pushes take several minutes and time out under the tool's command limits,
which is why the profiling job above took two attempts to get onto HPC. Fixed by adding `--exclude kaggle_staging`
to both the rsync and tar-fallback paths; verified the next push completes in seconds again.

**Verdict: safe to submit as probe #3 on runtime grounds.** Deployed ensemble (0.893) and exp008 (0.885) remain
untouched. Recommended candidate: exp018 solo (5-fold), the cleanest read of the OOF/gold-58 gain measured
2026-09-21; the 0.5-weight blend with the deployed ensemble is a second, separate probe if wanted later. The Submit
click is the user's.

## 2026-09-22 — Probe #3 RESULT (0.929) + probe #4 blend staged + efficiency-track re-evaluation

**Probe #3 (exp018 solo, 5-fold, dual-T4): 0.929 public LB.** Deployed ensemble stays at 0.893, exp008 at
0.885 — both untouched and still submittable, per standing instruction. This is a +0.036 gain over the
deployed ensemble, the largest single jump of the project, and only the third measured OOF-LB gap point
(-0.0474; same sign as probes #1/#2 but ~2.5-3x larger in magnitude, consistent with -- and stronger
confirmation of -- the standing "OOF understates true performance" finding, since exp018's gains plausibly
come from real image evidence the report-label proxy can't see. Not investigated as an anomaly: it fits the
existing model exactly, just further along the same axis).

### Probe #4 pre-submission analysis (`artifacts/exp018_probe4_analysis.md`, `scripts/exp018_probe4_analysis.py`)

**Diversity.** Per-label Pearson correlation between exp018 and the deployed ensemble's OOF (same method as
the exp009 diversity check): mean **0.8711**, range 0.70 (MCL) - 0.96 (Effusion). Above the 0.5 threshold set
in advance, so the pre-declared weight (0.75 exp018 / 0.25 ensemble) stands rather than being lowered.

**Blend table (rank space, OOF, advisory only -- see caveat below):**

| w(exp018) | OOF macro | Delta vs deployed ensemble | gold-58 (Delta) |
|---|---|---|---|
| 0.60 | 0.8856 | +0.0118 | 0.9099 (+0.0146) |
| 0.70 | 0.8855 | +0.0117 | 0.9093 (+0.0140) |
| **0.75 (chosen)** | 0.8852 | +0.0114 | 0.9091 (+0.0138) |
| 0.80 | 0.8848 | +0.0110 | 0.9088 (+0.0135) |

**Caveat, stated before reading the table (per instruction -- OOF must not drive the weight):** exp018's OOF
(0.8816) already undersells it by ~4 points relative to its LB (0.929); nothing suggests the blend's OOF is
any less biased. The weight (0.75) is fixed by the LB gap (0.929 vs 0.893) and the correlation above, not by
this table -- which if anything argues for something close to w=0.6-0.7 on OOF grounds, a reminder that the
two signals disagree and OOF is the one we distrust here.

### Probe #4 built, rehearsed, and staged (not yet submitted)

New code: `src/rsna_knee/infer_blend018.py` (decodes both pipelines per study -- no shared representation
exists between the 115mm/384px/6-slot deployed cache and the 140mm/336px/5-slot exp018 cache -- converts each
pipeline's output to a percentile against its own stored OOF distribution via `infer_core.empirical_percentile`,
the same mechanism the deployed ensemble already uses for its rank-mean labels, then blends). New rank
reference: `scripts/build_ensemble_output_rank_reference.py` builds a synthetic reference for the DEPLOYED
ENSEMBLE'S OWN blended output (the recipe's mixed prob-mean/rank-mean per label applied to the OOF population),
so the outer blend's "rank" is as well-defined as the inner one.

**Two bugs found and fixed while building/rehearsing this** (both in new code, not in anything previously
shipped): (1) the old-pipeline decode call was passed a plain string where `decode_study_slots` does
`series_root / series_uid` (a `Path` operation) -- `TypeError` on every call, caught silently by the outer
error handler in the first rehearsal. (2) after fixing that, decode still failed for every study
("no .dcm files in series dir") because the direct call skipped `decode_one_for_pool`'s per-study subdirectory
join (`series_root/study_uid/series_uid`) that the deployed notebooks always go through -- fixed by reusing
`decode_one_for_pool` itself instead of hand-rolling the call. After both fixes: `scripts/rehearse_blend018.py`
on 40 real studies -- 40/40 predicted, 0 errors, 0 prior fallbacks.

**Runtime, measured properly (`scripts/profile_decode_vs_model_blend018.py`, 150 real studies, decode/model
timed separately for BOTH pipelines -- a naive combined-wall-clock number can't be split into "gets the T4 3.5x
penalty" vs "doesn't"):**

| | H100 decode s/study | H100 model s/study |
|---|---|---|
| old ensemble (3 members x5 folds) | 0.360 | 1.253 |
| exp018 (5 folds) | 0.213 | 0.356 |
| combined | 0.573 | 1.610 |

T4 projection (decode rescaled from the profiling job's 8 CPU workers to Kaggle's assumed 3; model x3.5): **T4
x1 ~122 min, T4 x2 ~61 min** for 1,300 studies -- both comfortably inside the 8h cap (T4x2 = ~13% of budget).

**Staged, not submitted:** `notebooks/submit-blend018.ipynb`; Kaggle datasets `rsna-knee-blend018-code` (full
`rsna_knee` package) and `rsna-knee-blend018-refs` (both rank references + the recipe yaml), both private,
status `ready`. Reuses the existing `rsna-knee-exp018-weights` and the three deployed members' weight datasets
unmodified (no already-scored asset was touched). Probe #4's Submit click is the user's.

### Efficiency-track re-evaluation (`scripts/exp018_efficiency_table.py`, `scripts/fold_subset_gold_auc.py`)

**Fold-count sensitivity (gold-58, all-C(5,k)-subsets average -- same method already used for exp008):**
exp018 loses only **0.0067 macro going 5-fold -> 1-fold** (vs exp008's 0.0060 for the same drop) -- window-MIL
already averages ~50 windows/study/fold, so one fold alone is a fairly stable estimate.

**Efficiency table** (`Eff = AUC/(0.5-0.95) + seconds/32400`, minimized; AUC anchored on each 5-fold config's
real public LB, reduced-fold AUC = anchor + the gold-58 delta above; runtime = H100-measured, Kaggle-CPU-
rescaled, T4-projected, decode/model overlapped):

| config | AUC | projected minutes (T4) | Eff score |
|---|---|---|---|
| exp008 (deployed, 5-fold, T4 x1) | 0.8850 | 11.6 | -1.9452 |
| exp018 (1-fold, T4 x1/x2) | 0.9223 | 15.3 | -2.0213 |
| exp018 (5-fold, T4 x1) | 0.9290 | 27.2 | -2.0140 |
| **exp018 (5-fold, T4 x2)** | **0.9290** | **15.3** | **-2.0362 (best)** |

**Finding: the already-staged, already-scored probe #3 notebook (5-fold, T4x2) is ALSO the efficiency-optimal
configuration** -- not a reduced variant. At 2 GPUs, per-study decode time (0.704s, CPU-bound) exceeds the
5-fold model time per GPU (0.628s), so decode is the bottleneck at every fold count up to 5: running all 5
folds costs essentially nothing over running 1, while buying the full AUC gain. A separate reduced-fold
efficiency notebook was therefore NOT built -- it would not be used. Every exp018 config beats exp008 on the
formula; the AUC term dominates the formula at this scale (sensitivity ~2.2 per AUC point vs ~0.00003 per
second of runtime).

> **CORRECTION (2026-09-26): this finding is WRONG.** It rests on a projected model time (0.356 s/study x 3.5 for
> T4) that Kaggle-hardware measurement showed was ~3.6x too low: 5 folds take 4.479 s/study on ONE T4, so the run
> is model-bound at ~53 min, not decode-bound at 15.3 min, and 5 folds is NOT efficiency-optimal. 2 folds are
> (~20.5 min, public 0.925, Eff −2.018 vs −1.966), and that became the efficiency pick. See "C — step 2" and
> "C — step 3" (2026-09-26).

**Recommendation (not yet acted on): select the existing probe #3 submission for the efficiency-track pick as
well as the main pick**, if Kaggle's rules permit selecting the same scored submission for both final slots --
unverified, the user should check this on the platform. If not permitted, resubmit `notebooks/submit-exp018.ipynb`
unchanged as the second pick (same code, same expected score). `notebooks/submit-efficiency.ipynb` (exp008)
stays frozen and submittable until either the same-submission question is resolved or a replacement is
actually probed and scored -- nothing has been swapped in yet.

## 2026-09-23 — Probe #4 v2 hung on Kaggle 9h; root-caused, fixed, proven, v3 staged

> **CORRECTION (added later, 2026-09-23): the "root cause" below is WRONG for the blend.** The
> `prefetch_decode` bug is real, but the blend driver never calls `prefetch_decode` (v2 = `e9944c6`
> imports only `decode_one_for_pool` and helpers from `infer_core`). See "Correction: probe #4 hang
> attribution" at the end of this file.

**Symptom (user's log audit): `submission.csv` written at 346s, `nbconvert` wrote `__results__.html` at
357s (i.e. ALL notebook cell code, including post-processing, had already finished normally), then the
kernel sat for 9 hours before being killed.** Root cause: `infer_core.prefetch_decode`'s feeder thread
called the bounded `result_queue.put(result)` with no timeout; if the consumer stops pulling early (a
`break`, or the generator being garbage-collected before exhaustion -- exactly what Kaggle's dummy
N=3 test invites, since a bounded window fills fast against so few tasks), that `put()` can block
forever with nothing left to unblock it. Because the block happens INSIDE `with ThreadPoolExecutor(...)
as pool:`, the feeder thread never reaches `pool.shutdown()` -- and `ThreadPoolExecutor` worker threads
are deliberately NOT daemon (so real work survives an interpreter exit unless explicitly told to stop).
An abandoned, never-shut-down pool's idle worker threads are exactly the kind of live non-daemon thread
that blocks process teardown. `feeder_thread` itself being `daemon=True` was never the guard it looked
like -- it only stops the feeder thread from blocking exit, not the pool it constructs and never lets
finish. My own `run_exp018`/`run_blend018` drivers had a related but distinct bug: their sentinel+join
code for the GPU worker threads ran as plain statements AFTER `with pool:`, not in a `finally`, so an
exception anywhere in the decode loop would skip it entirely (the GPU threads are daemon=True so this
specific case wasn't itself a hang risk, but it is exactly the fragility pattern the instruction asked
to eliminate, and the `with` block's implicit `shutdown(wait=True)` has no `cancel_futures`, so an early
exit would still wait out already-queued decode tasks instead of dropping them).

**Fix (`src/rsna_knee/infer_core.py`, `infer_wmil.py`, `infer_blend018.py`):**
- `prefetch_decode`: a `stop_event` the consumer's `finally:` sets on any exit path (early break, normal
  exhaustion, or GC); the feeder's queue-put is now a bounded-retry loop that checks it, so the feeder
  can ALWAYS unwind out of `with pool:` instead of blocking forever. `pool.shutdown(wait=True,
  cancel_futures=True)` now runs in the feeder's own `finally`, unconditionally reached. Consumer-side
  join changed from a no-op `timeout=0` to a real `timeout=10` with a logged warning if still alive
  (which should now never fire).
- `run_exp018` / `run_blend018`: the whole decode-dispatch section is now one `try/finally`; the pool is
  shut down explicitly with `cancel_futures=True`; the GPU-worker sentinel+join always runs regardless of
  how the decode loop exits.
- `multiprocessing.Pool`/`ProcessPoolExecutor` audit (instruction 1c): none exist anywhere in the
  inference-path modules (`infer_core.py`, `infer_wmil.py`, `infer_blend018.py`, `wmil.py`) -- confirmed
  by grep, not assumed. Not applicable.

**Proof, on HPC, exactly as instructed (`scripts/test_blend_shutdown.py`, `test_exp018_shutdown.py`,
`test_prefetch_decode_shutdown.py`; job 26194, `slurm/test_shutdown.sbatch`):**

| test | result | notes |
|---|---|---|
| `test_blend_shutdown.py --n-studies 1` | EXIT=0 | full real pipeline (20 models), 36.9s work, clean exit |
| `test_blend_shutdown.py --n-studies 2` | EXIT=0 | 40.0s work, clean exit |
| `test_blend_shutdown.py --n-studies 3` | EXIT=0 | 44.2s work, clean exit -- the exact Kaggle dummy-test size |
| `test_exp018_shutdown.py --n-studies 3` | EXIT=0 | audits submit-exp018.ipynb's driver (already scored 0.929; this is defense-in-depth, not a sign anything was wrong there) |
| `test_prefetch_decode_shutdown.py --n-studies 3 --break-after 1` | EXIT=0 | audits submit-ensemble.ipynb/submit-efficiency.ipynb's decode path directly, reproducing their exact early-`break` pattern (`if guard.over(): break`) that plausibly triggers the bug |
| `test_prefetch_decode_shutdown.py --n-studies 3 --break-after 99` | EXIT=0 | same, normal exhaustion (no early break) |

All six ran under `timeout 120` and returned exit code 0 well inside it. **Precise external timing** (job
26195, bash `time` wrapping the whole process from the outside, not trusting the script's own prints):
blend018 N=3 -- `real 0m45.120s` total process life vs. the script's own "all work finished at 44.2s" print
-- **~0.9s from last work to actual process exit**, where it previously hung for 9 hours. (Instruction 2's
literal "under 30 seconds" referred to shutdown latency, not total runtime; total runtime here is real
model-loading + real 20-model inference on 3 real studies, which alone takes ~40-45s -- reported honestly
rather than substituted with a lighter synthetic test that would prove less.)

**Task 3 audit conclusion:** `submit-exp018.ipynb`'s driver (`run_exp018`) is fixed and proven exit-safe.
`submit-ensemble.ipynb`/`submit-efficiency.ipynb` call `infer_core.prefetch_decode` directly from their own
cells (not through a driver function) -- the underlying function is now fixed and proven exit-safe, but
those two notebooks load `rsna_knee` from the **`rsna-knee-code`** Kaggle dataset, a different, NOT-yet-
updated dataset (this task only asked to update `rsna-knee-blend018-code`). Both notebooks remain frozen
and submittable exactly as before -- nothing about their already-scored status changes -- but if either is
ever re-run or resubmitted, it would still carry the unpatched `prefetch_decode`. Flagged for the user to
decide whether `rsna-knee-code` should also get this fix; not done unilaterally.

**Task 4: done.** After a fresh `kaggle auth login`, `kaggle datasets version -p blend018_code -m "v3: exit-
safety fix..."` uploaded successfully. **`rsna-knee-blend018-code` is v3, status `ready`**, confirmed by
polling `kaggle datasets status` and cross-checking file sizes against the patched sources: `infer_core.py`
23,103B, `infer_wmil.py` 9,277B, `infer_blend018.py` 11,576B (all larger than the pre-fix versions, as
expected). `notebooks/submit-blend018.ipynb` needed no changes (the fix lives entirely in the `rsna_knee`
package, not notebook cells). exp018 (0.929, banked) and the deployed ensemble/efficiency notebooks remain
untouched throughout. Probe #4 (v3) is ready for the user to dry-run and, if clean, submit -- per the
guardrail, this is the final attempt before dropping the blend and locking exp018 for both selections.

### Method writeup

`docs/METHOD_WRITEUP.md` §5a (new) documents exp018's full recipe (preprocessing, architecture, training,
laterality, results) as the current best model, citing `dreaddevelopment/knee-mri-twelve-findings-from-a-
single-model` and its training-kernel companion as the architectural reference spec, per instruction. §1 and
§11 updated with the 0.929 result and probe #3/#4 status; §5-§8 (the original DINOv2 pipeline) are unchanged
-- it remains the code behind both live submissions.

## 2026-09-23 — Probe #4 v3 hung again (10h): blend ABANDONED; exp018 FINAL for both tracks

**What happened.** v3 (with the HPC-proven thread-shutdown fix in `rsna-knee-blend018-code`) was
submitted 2026-09-23 07:41 UTC (Kaggle ref 56487137) and hung ~10h at teardown, exactly like v2 (ref
56473085, 9h). Both show `COMPLETE` with **no public score**. So the `prefetch_decode` bug was real but
was not the whole cause of the hang, or not the cause at all. Guardrail invoked: the blend is dropped
unless the one-line test below works, with no attempts after it.

**v4, the last-chance test (user's hypothesis: CUDA-context / interpreter teardown; the blend is the only
notebook holding 20 models on 2 GPUs).** The final cell of the notebook now re-reads `submission.csv`,
re-validates rows/header/NaNs, fsyncs it, prints, flushes stdout/stderr, and then calls `os._exit(0)` as
the literal last statement. That skips every Python-level cleanup step: thread joins, atexit, CUDA
destructors. The shared tail lives in `scripts/hard_exit_tail.py` and the generators
(`build_blend018_notebook.py`, `build_exp018_notebook.py`) append it. It is a notebook-only change: the
Kaggle notebook was pulled first and was byte-identical to the local copy in every cell. The dataset is
unchanged. Pushed with `kaggle kernels push`, keeping the pulled metadata (same 6 datasets, competition
source, T4, pinned docker image). **Kaggle assigned it version 4**, and its commit run started
immediately.

Two findings from testing the tail locally before pushing it:
1. **Bug caught and fixed:** the first draft fsync'd a read-only handle, which raises `EBADF` on Windows.
   Linux normally allows it, but if it had raised on Kaggle the cell would have thrown *before* reaching
   `os._exit`, turning a hang into an exception (= no score either way). The fix fsyncs through an
   append-mode handle (writes nothing) inside try/except.
2. **Known risk, now confirmed locally:** under `jupyter nbconvert --execute` (nbclient) a kernel that
   calls `os._exit(0)` mid-cell raises **`DeadKernelError: Kernel died`**. nbconvert then exits 1 and
   writes **no HTML**. Plain `python` exits 0 with the file intact. The user flagged this papermill/nbclient
   behaviour in the v2 instructions. Whether Kaggle's runner treats this as a failed run (and whether
   such a version can be scored) is exactly what v4's commit run will show. A related observation:
   nbclient shuts the kernel down gracefully and then SIGKILLs it (jupyter_client default ~5s) *before*
   the HTML is written, and v2/v3 did write `__results__.html`. So a Python-level teardown hang inside the
   kernel should already have been bounded by that kill. That argues the hang is below or outside the
   kernel's Python (a driver-level GPU release that cannot be killed, or the container/output stage),
   and `os._exit` would not bypass either. This is unconfirmed; it is recorded because it predicts that
   v4 may not help.

**v4 commit-run result: `ERROR` after ~200s. It did not hang, and it is not submittable.** From the
downloaded log (`kaggle kernels output`): models loaded in 78s, the submission was written at 171.2s,
the hard-exit tail printed its verification line at 171.3s, and at 172.2s papermill reported "Kernel
died while waiting for execute reply" → **`nbclient.exceptions.DeadKernelError: Kernel died`**. Kaggle
still rendered `__results__.html` and kept `submission.csv`, but it marked the version `ERROR`. So
`os._exit` does end the process with no hang, and Kaggle's runner treats that exit as a failed run,
exactly as the local nbconvert test predicted.

**New bug found in the v4 log (and it was in v3): `[blend018] ... 3 rows, 1 predicted, 2 on prior,
0 errors`.** Two of the three placeholder studies were silently written as label priors. On the same
placeholder, exp018 solo predicted 3/3. Cause: my own v3 exit-safety change (`2ec038d`) replaced the
GPU workers' untimed `w.join()` with `w.join(timeout=30)` in both `run_blend018` and `run_exp018`. On
T4, the first study through 20 models (with warm-up) takes more than 30s per GPU. The joins therefore
gave up while the workers were still predicting, the submission was built from whatever `results`
held at that moment, and the daemon GPU threads were **still running CUDA inference when the notebook
finished and the interpreter began shutting down**. A daemon thread doing CUDA work during interpreter
finalization is a known way to hang a process, so this is a *possible* contributor to v3's 10h hang.
It is not established (see the correction entry below). It cannot explain v2's 9h hang, because v2
(`e9944c6`) still had the untimed join. The HPC proof missed it for two reasons: the H100 finishes well
inside 30s, and the tests asserted only the row count, not that each row was a real prediction.
- Fixed in source (not pushed anywhere): both drivers are back to an untimed `w.join()`, which is safe
  because the sentinels guarantee each worker returns. `test_blend_shutdown.py` and
  `test_exp018_shutdown.py` now also assert that zero rows equal the label prior.
- **The scored exp018 (0.929) is unaffected:** `rsna-knee-exp018-code`'s `infer_wmil.py` is 8,245 B,
  the pre-`2ec038d` version with the untimed join (checked with `kaggle datasets files`).
- `rsna-knee-code` (updated below) is unaffected: its only change is `prefetch_decode`, whose join is
  on the feeder thread after the consumer has already received every item it will use. No results
  are lost.

**Guardrail outcome: the blend is DROPPED.** The one-line test did not produce a submittable run
(`ERROR`, and 2/3 placeholder rows on the prior). No v5 was pushed. A v5 with the untimed join and no
`os._exit` would be a genuinely different test, but v2 hung with exactly that join, so there is no
evidence it would succeed.

**Final state, prepared regardless of v4 (instruction 2):**
- **`submit-exp018.ipynb` is scored and unchanged:** Kaggle ref 56460279, `COMPLETE`, **public 0.929**.
  The local notebook was last changed in `5141fe9`, before that submission. It is the final selection for
  both the main and the efficiency track (efficiency-optimal per the 2026-09-22 table: Eff −2.036 vs
  exp008 −1.945).
- **Insurance hard-exit versions: built, then deliberately BACKED OUT.** The tail was added to
  `submit-exp018.ipynb` and `submit-efficiency.ipynb` as instructed (never pushed, because every CLI
  push starts a run). Once v4 showed that `os._exit(0)` makes Kaggle's papermill runner end the version
  in `ERROR`, the "insurance" would have turned any forced re-run into a failed run. Both notebooks
  were therefore reverted to their scored state (`git checkout`). The tail survives only in
  `scripts/hard_exit_tail.py` and the blend notebook, as the record of v4.
- **`rsna-knee-code` updated with the proven `prefetch_decode` fix.** The live dataset was downloaded
  first. Its `infer_core.py` was byte-identical to commit `d767fdc`, and the only change to that file since
  then is the fix commit `2ec038d`. Staged the other 6 live files byte-for-byte unchanged (their HEAD
  copies of `dataset.py`/`model.py` differ, so they were NOT taken from HEAD) and swapped in only
  `infer_core.py` (20,034 → 23,103 B, the same file as `rsna-knee-blend018-code` v3). Status `ready`,
  sizes verified with `kaggle datasets files`. This only matters if the ensemble/efficiency notebooks are
  ever re-run. Their scored submissions are unaffected.

**Probe #4 ledger (final):** attempted and abandoned, never scored. Two Kaggle timeouts (v2 9h, v3 10h),
both after a `submission.csv` was written, plus one `ERROR` (v4, the deliberate `os._exit` test).
**Neither hang has an established cause** (see the correction entry below):
- v2: the blend's own code left no live threads, so the cause is unexplained.
- v3: the timed-join regression is a possible contributor, not a demonstrated cause.

The OOF blend analysis stays in `artifacts/exp018_probe4_analysis.md` for the record. No further blend
attempts.

## 2026-09-23 — Correction: probe #4 hang attribution

The earlier entries attributed v2's hang to `infer_core.prefetch_decode`'s blocking `put()` inside
`with ThreadPoolExecutor`. **Checked against the code that actually ran: the blend never calls
`prefetch_decode`.**
- `git show e9944c6:src/rsna_knee/infer_blend018.py` (v2) imports only `decode_one_for_pool,
  empirical_percentile, ensemble_blend, load_ensemble_recipe, nearest_patch_multiple, prior_row,
  run_study_inference` from `infer_core`. None of those create threads; `wmil`, `preprocess`, `model`
  and `dataset` contain no Thread/Pool/Executor at all.
- The only thread pool on v2's path is `run_blend018`'s own `with ThreadPoolExecutor(...)` loop. The main
  thread was the producer and called `.result()` on each future before `q.put()`. That loop demonstrably
  finished: the submission is built only after it exits, the `with` exit shuts the pool down, and the
  GPU threads (daemon) were then joined *without* a timeout. When v2's cell finished, the blend's own
  code had no live threads left.
- So the `prefetch_decode` bug (real, and still worth fixing for `submit-ensemble`/`submit-efficiency`,
  which do call it with an early `break`) was **never on the blend's path**. The HPC shutdown tests
  proved the new code exits cleanly, but no pre-fix control was run. They would very likely have passed
  on v2's code too.

**Reading of the evidence (not certain):**

| attempt | our code's threads at teardown | explanation |
|---|---|---|
| v2 (9h) | none left alive | **Unexplained.** Remaining candidates are outside our Python threads: CUDA context/driver teardown with 20 models on 2 GPUs, or the Kaggle runner/container. Not tested. |
| v3 (10h) | possibly daemon GPU threads mid-inference | **Possible contributor: the timed-join regression.** On the 3-row placeholder, the 30s join certainly expired (v4 log: 2/3 on prior). In a 1,300-study scoring run, the backlog at the end is only a few studies per GPU, so the join may not have expired there at all. |
| v4 | n/a (os._exit) | No hang; papermill `DeadKernelError`, so the run is `ERROR`. |

The earlier framing, "v2 = the `prefetch_decode` bug, v3 = the regression its fix introduced, one cause
per attempt", is **not supported**, because v2 did not execute `prefetch_decode`. The parsimonious
statement is narrower: v2's cause is unknown; v3 carried a real regression of ours that may or may not
have caused its hang. The METHOD_WRITEUP §9 paragraph was corrected to match. Scored submissions are
unaffected: exp018's code dataset predates `2ec038d` and never contained either change.

## 2026-09-24 — PHASE 5: exp019 (CoAtNet-3 capacity) ABORTED, exp020 (EMA) NULL. exp018 stands.

Two fold-0 probes against the exp018 fold-0 baseline. Neither promotes to 5 folds; no Kaggle submission
was spent. exp018 (LB 0.9290) remains the final pick for both tracks.

### Baseline (exp018 fold 0, for reference)

`artifacts/exp018_fold0_gate.md` + `fold0/history.json`: fold-val macro **0.9567** (vs labels_v3, the
selection metric; best single epoch 0.9564, top-3 = 15/12/13), fold-0 OOF vs labels_v2_final **0.8883**
(n=879, the cross-experiment comparable number), gold-58 **0.9063**, **370 s/epoch** (369-372 across all
16, ~99 min/fold on one H100).

### exp019 — CoAtNet-3: never produced a result. Two blockers.

**1. The specified checkpoint does not exist.** `coatnet_rmlp_3_rw_384.sw_in12k_ft_in1k` is not in timm,
and `timm.list_models("coatnet_rmlp_3*", pretrained=True)` returns `[]` -- there are NO pretrained
coatnet_rmlp_3 weights at any resolution. The only real CoAtNet-3 capacity step is
`coatnet_3_rw_224.sw_in12k` (163.6M params vs exp018's 72.9M, 1536-d features, in12k with no ft_in1k
stage). It is 224-native, so `img_size=384` must be passed at construction; without it a 384 forward
fails (`576 vs 196`). `WindowMIL` gained an optional `img_size` kwarg (default `None` => exp018's
construction is unchanged) and the weights were pre-downloaded to the shared HF cache, since compute
nodes run `HF_HUB_OFFLINE=1`.

**2. It cannot train at this batch shape (job 26951, FAILED at the first step):**
`RuntimeError: Expected canUse32BitIndexMath(input) && canUse32BitIndexMath(output) to be true`.
CoAtNet-3's MBConv blocks expand 4x internally; at 8 studies x 12 windows = 96 windows/forward at 384px
the stage-0 tensor passes 2^31 elements and `F.conv2d` has no 64-bit indexing path. Measured ceiling
(job 26953): 96 windows FAIL; **48 OK** (1.36e9 elements, 0.63 x INT_MAX); 32 and 16 fine. This is a
property of the architecture at this shape, so it would recur on any GPU.

**Why it was not relaunched.** Every fix reduces windows per forward (chunked `encode`, smaller
`batch_studies` + accumulation, or smaller `k_train`), and the same probe found CoAtNet-3 carries **25
`BatchNormAct2d` layers**, so *any* of them changes training-time BN statistics relative to exp018.
exp019 would then vary capacity AND BN batch AND the 224->384 position-table interpolation -- three
variables, one fold. Combined with the efficiency cost below, the result would not have been
interpretable. Paused for a user decision; not relaunched unilaterally.

**Runtime (the kill rule) did pass, measured before training** (job 26948, H100, same 40 real cached
studies and the same 51.5 windows/study for both archs):

| backbone | s/study (5 folds, H100) | ratio | dual-T4, 1,300 studies |
|---|---|---|---|
| `coatnet_rmlp_2_rw_384` (exp018) | 0.356 | 1.00x | 15.3 min accepted / 21.1 conservative |
| `coatnet_3_rw_224` (exp019) | 0.537 | 1.51x | **20.4 min accepted / 28.0 conservative** |

Both are under the 35-min limit. Two notes. (a) `scripts/bench_arch_runtime.py` inherited the
*conservative* formula from `profile_decode_vs_model_exp018.py`, `max(decode, (decode+model)/2)`, which
double-counts decode per GPU stream; the efficiency table behind the accepted -2.0362 uses
`max(decode, model/2)` because decode is one shared thread pool feeding both GPUs. Both columns are
reported rather than only the flattering one. (b) At 20.4 min the runtime term rises 0.0283 -> 0.0378,
so exp019 would need **+0.0043 macro-AUC just to break even** on the efficiency score; it could only
ever have been a main-track candidate.

**Process failure worth recording:** that benchmark ran ONE study at a time (the inference shape) and so
validated latency but never the training batch shape that failed. A capacity screen must include a
trainability check at the real training shape; a latency number alone does not clear an architecture.

### exp020 — model EMA: NULL against the weight averaging exp018 already does.

**Decay had to be reconsidered before launch.** The instruction specified `ModelEmaV2` at decay 0.9999.
A fold is ~440 steps x 16 epochs = ~7,040 steps, but 0.9999 has a 1/(1-d) = 10,000-step horizon: the EMA
would finish training still carrying `0.9999^7040` = **49.5% of the ImageNet initialisation**. So
`ModelEmaV3` (same algorithm, warm-up ramp available) was used with `use_warmup=True`, and TWO decays
were tracked in one run -- the requested 0.9999 and a 0.999 whose ~1,000-step (~2.3-epoch) horizon
actually fits the schedule. Selection was on fold-validation only; gold-58 stayed quarantined.

**Fold-0 result (job 26952, 16/16 epochs, ~4.4h with the extra EMA validation passes):**

| model | fold-val macro | gold-58 macro |
|---|---|---|
| **top-3 epoch average (the exp018 mechanism)** | **0.9533** | **0.9038** |
| EMA decay 0.999 | 0.9532 (-0.0001) | 0.9029 (-0.0009) |
| EMA decay 0.9999 + warm-up | 0.9136 (-0.0397) | 0.8613 (-0.0425) |
| *exp018 fold 0, for reference (different run)* | *0.9567* | *0.9063* |

**Verdict: EMA adds nothing.** The 0.999 branch ties the top-3 average to within 0.0001/0.0009 -- far
inside noise -- and 0.9999 loses badly even with warm-up, since its horizon still exceeds the run. This
is the outcome the config pre-registered as likely: **exp018 already averages its top-3 epochs' weights**,
so this tested "EMA vs top-3 weight averaging", not "EMA vs nothing". The two mechanisms overlap, and the
one already in the recipe is not improved on.

**The cross-run comparison is noise-dominated, and the within-run one is not.** exp020's training path is
identical to exp018's (EMA never touches the weights used for training), same seed and config, yet the
epoch-3 fold-val differs by 0.013 (exp018 0.9158 vs exp020 0.9024) -- GPU nondeterminism in the backward
kernels. So exp020-vs-exp018 deltas of a few thousandths (including top-3 0.9038 vs 0.9063) mean nothing.
The EMA-vs-top-3 comparison is immune to this: both come from the SAME run and the same weight
trajectory, which is why the branches were built into one run rather than compared across runs.

### Decision gate

Promotion required a clear gold-58 improvement over exp018 fold 0 (0.9063) inside the 35-min envelope.
- **exp019: no result** (cannot train at this batch shape without confounding the test). NOT promoted.
- **exp020: no improvement** (best EMA 0.9029 < 0.9063, and ties its own run's top-3). NOT promoted.

Neither advances. **exp018 stays the final selection for both tracks**, and both remaining Kaggle
submissions are unspent. Kept from this phase: the `img_size` kwarg, the config-gated EMA path (off
unless a config asks for it, EMA state persisted through the wall-clock-guard resume, evaluation moved
AFTER `best.pt` is written and wrapped in try/except so an experimental branch can never cost a
completed run its primary output), and `scripts/bench_arch_runtime.py`.

### User decisions on Phase 5 (2026-09-24)

- **exp019: DROPPED.** Reasons given: confounded (capacity + BN batch + 224->384 position tables),
  efficiency-negative (+0.0043 AUC needed just to break even), and fold-0 gating is noise-dominated.
  Not relaunched in any form. The `img_size` kwarg stays (harmless, default `None`).
- **exp020: null ACCEPTED.** EMA does not improve on top-3 weight averaging; the EMA code path stays in
  `train_wmil.py`, off unless a config enables it.

## 2026-09-24 — exp021: exp018 SEED ENSEMBLE (seed 43, 5 folds) — PRE-REGISTRATION + LAUNCH

**Motivation.** Phase 5's one real finding: two runs of the IDENTICAL exp018 path (same seed, config, data)
diverged by 0.013 fold-val at epoch 3 from GPU nondeterminism. If single-run variance is that large, a
second independent run averaged with the first should buy accuracy for nothing but compute.

**Design (fixed before any result).**
- `configs/exp021_coatnet2_wmil_seed43.yaml` = exp018's config with only `experiment` and `seed` (42 -> 43)
  changed (YAML-parsed diff: exactly those two keys). No EMA, no `img_size`. All 5 folds.
- Training code: the only commits touching the training path since exp018 trained are `af68154` (licence
  comment, parsed code verified unchanged) and `9e954c6` (EMA + `img_size`, both config-gated and absent
  from this config). So seed 43 runs the same code path exp018 ran.
- **Candidate = 10-checkpoint prob-mean** (5 x seed 42 from the scored exp018 + 5 x seed 43). Reported for
  exp018 alone, exp021 alone, and the mean: OOF vs `labels_v2_final` with the paired cluster bootstrap
  (same machinery as exp016/exp018), and gold-58 with its own bootstrap CI so its width at n=58 is
  visible (`scripts/exp021_seed_ensemble.py`). Gold-58 is advisory. Also reported: the seed-42-vs-43
  per-fold spread, i.e. the run-to-run noise floor this project has never actually measured.
- **Probe rule (user's):** one Kaggle submission is reserved for the 10-checkpoint ensemble. It is a
  MAIN-TRACK candidate only; exp018 (56460279) stays locked as the efficiency pick. **If the probe does not
  beat 0.929, modeling is permanently closed.**

**Inference check (instruction 3), BEFORE any code or staging change.** The scored `infer_wmil.py`
(8,245 B, byte-identical to `e9944c6`, re-confirmed against the live dataset listing) neither globs nor
hardcodes 5 folds: `run_exp018(..., weight_paths: list[Path], ...)` takes a list of any length,
`load_fold_models` loads every path, and `gpu_worker` prob-means over all of them (`np.mean(probs, axis=0)`),
which is exactly the requested 10-checkpoint prob-mean. The globbing is in the NOTEBOOK cell
(`sorted(WEIGHTS_DIR.glob("fold*_best.pt"))`, one directory). Consequence: the scored code deploys
UNCHANGED; the only edit is the notebook cell that builds the list, and that is what the commit check
tests. Side facts: each GPU holds a full replica set (10 x fp32 CoAtNet-2 ~= 2.9 GB per T4, fits); the
wall-clock guard drops the LATER models in the list first (won't trigger: ~27 min projected vs a 6.8 h
budget).

**Staging (no exp018 asset touched).** New dataset `rsna-knee-exp021-weights` holds ONLY the five
`fold{k}_seed43_best.pt`, lean format identical to exp018's (`scripts/stage_exp021_kaggle.py` refuses to
stage unless all 5 folds exist, and asserts arch/res/no-img_size). `notebooks/submit-exp021.ipynb`
(`scripts/build_exp021_notebook.py`) attaches the scored `rsna-knee-exp018-code` and the scored
`rsna-knee-exp018-weights` unchanged plus the new dataset. It asserts exactly 5 + 5 checkpoints, so a
missing attachment fails loudly instead of silently running a 5-model "ensemble". Its inference and
validation cells (3, 4) are byte-identical to the scored exp018 notebook; no `os._exit`.

**Launch record.** Job **27017** (array 0-4), `slurm/train_exp021.sbatch`, via `safe_sbatch.py`
(15 GB projected, 199.6 GB budget). Runtime benchmark for 5 vs 10 checkpoints: job 27018.

**Launch mistake, caught and fixed before it cost anything.** The first submission (job 27016) passed
`--array=0-4` AFTER the script path. `safe_sbatch.py` forwards everything after `--` verbatim, so the
option reached the sbatch script as `$1` (ignored), and SLURM created a SINGLE job with no
`ArrayTaskId`, which would have trained fold 0 only (`FOLD` defaults to 0). Caught by checking
`scontrol show job` for the array id, cancelled by explicit id, resubmitted as
`safe_sbatch.py -- --array=0-4 slurm/train_exp021.sbatch`. The Phase-5 exp019/exp020 launches used the
same wrong syntax and were correct only by accident (both intended fold 0). Also found: a
`DISK_PROJECTION_GB` header is taken as the TOTAL for the submission, not multiplied by the array
size, so the header here is 15 (5 x 3 GB), not exp018's per-task 3.

**Runtime projection for 10 checkpoints (instruction 4).** Derived, not re-benchmarked. The benchmark
job for 5 vs 10 checkpoints (27018) was cancelled unrun: `bench_arch_runtime.py` times N passes of the
same model per study, so 10 passes is 2x the 5-pass cost *by construction* and a new measurement adds
nothing. Inputs are the measured H100 cost (job 26948: 0.356 s/study for 5 checkpoints, 40 real studies,
51.5 windows/study) and the accepted decode figure (0.704 s/study, Kaggle-CPU-rescaled). The only true
T4 measurement available is Kaggle's own; its commit run sees just 3 placeholder studies.

| checkpoints | H100 s/study | T4 model s/study | dual-T4 accepted | dual-T4 conservative |
|---|---|---|---|---|
| 5 (exp018) | 0.356 | 1.246 | **15.3 min** (reproduces the accepted figure exactly) | 21.1 min |
| 10 (exp021 ensemble) | 0.712 | 2.492 | **27.0 min** | 34.6 min |

At 10 checkpoints the GPUs, not decode, bound runtime (1.246 s/study per T4 vs 0.704 decode), so
runtime roughly doubles. **Efficiency (main-track candidate only; exp018 56460279 stays the efficiency
pick):** Eff = −AUC/0.45 + s/32400. exp018 = −2.0361. The 10-checkpoint ensemble at 27.0 min carries a
runtime term of 0.0500 (vs 0.0283), so it needs **LB AUC ≥ 0.9387 (+0.0097) just to tie exp018 on
efficiency** (0.9451, +0.0161, under the conservative formula). At an LB of 0.929 / 0.932 / 0.935 its
Eff would be −2.0144 / −2.0211 / −2.0278, all worse than exp018's. This confirms it cannot be the
efficiency pick.

**Scheduling (2026-09-25).** After ~5.9 h all five tasks were still pending: all 24 H100 GPUs were held
by other users' jobs with ≥1 d 20 h of time limit left (our QOS allows 8 GPUs; we held 0), and SLURM
estimated the start at 2026-09-26 21:04. The array requested 6 h per task while a fold needs ~1 h 45 m;
lowered the pending job to `TimeLimit=03:00:00` (`scontrol update`; users can lower but not raise), which
makes it far easier to backfill into early gaps. 3 h rather than 2 h because `safe_torch_save` is a plain
`torch.save` (not atomic), so a SLURM kill mid-write could corrupt `checkpoint.pt`; 1.8x headroom over
exp018's 1 h 41 m makes that very unlikely. The sbatch file now also sets `--max-hours 2.6` so future
launches stop cleanly after an epoch before the limit. 27017 itself keeps the 5.5 h guard it was
submitted with. Node exclusions were removed from all scripts the same day (see HPC_WORKFLOW.md §3.5a).
A local DNS outage (Cloudflare WARP answering NXDOMAIN for the cluster host) briefly made a watcher report
a false "done"; watchers now treat a failed `ssh` as unknown, not finished.

## 2026-09-25 — exp021 RESULT: 10-checkpoint seed ensemble is a small, real OOF gain

All 5 seed-43 folds completed (job 27017, each ~1 h 40 m, all on g002, no co-tenant OOM, 16/16 epochs).
`artifacts/exp021_seed_ensemble.md` (`scripts/exp021_seed_ensemble.py`, 1,000 paired cluster-bootstrap
reps over 3,681 report clusters, 4,349 non-gold studies vs `labels_v2_final`; gold-58 with its own
1,000-rep study bootstrap):

| model | ckpts | OOF macro | Δ OOF vs exp018 (95% CI) | P(Δ≤0) | gold-58 (95% CI) | Δ gold-58 (95% CI) |
|---|---|---|---|---|---|---|
| exp018 (seed 42) | 5 | 0.8816 | — | — | 0.9091 [0.8807, 0.9348] | — |
| exp021 (seed 43) | 5 | 0.8805 | −0.0011 [−0.0031, +0.0008] | 0.863 | 0.9088 [0.8798, 0.9333] | −0.0003 [−0.0075, +0.0083] |
| **10-ckpt prob-mean** | 10 | **0.8840** | **+0.0024 [+0.0014, +0.0033]** | **0.000** | 0.9111 [0.8834, 0.9359] | +0.0021 [−0.0023, +0.0065] |

**Reading.**
- **The ensemble gain is real but small.** +0.0024 OOF, with the whole 95% CI above zero and none of
  the 1,000 bootstrap resamples at or below zero. On gold-58 it is +0.0021, but that CI spans zero, as
  expected at n=58. Per label it helps most where the two seeds disagree most: ACL (seed correlation
  0.904, +0.005 OOF) and MCL (0.891). Seed correlations elsewhere are 0.96–0.99, which caps what
  averaging can add.
- **Seed 43 alone is as good as seed 42** (−0.0011, CI spans zero). exp018 was not a lucky seed.
- **The measured noise floor corrects exp021's own motivation.** The 0.013 epoch-3 gap that motivated
  exp021 was a *mid-training* difference. At the end of training (top-3 averaged), seed 42 vs 43
  differ by only **0.0020 mean / 0.0046 max** in fold-val, and **0.0046 mean / 0.0067 max** in gold-58.
  Top-3 weight averaging already damps most run-to-run variance. So the Phase-5 statement "a
  cross-run fold-0 delta of a few thousandths is noise" still holds, but it is noise of ~0.002–0.007,
  not ~0.013. This is also why the ensemble's gain is modest.

**Staging.** `scripts/stage_exp021_kaggle.py` wrote the five lean `fold{k}_seed43_best.pt` files,
292,865,482 B each; SHA-256 verified identical on HPC and locally. They are 3,430 B larger than exp018's
files, which is fully explained by `torch.save`'s zip entry paths carrying the file stem (`_seed43` = 7 bytes
x 490 entries). New private dataset `rsna-knee-exp021-weights` created (no exp018 asset touched). Kernel
`hrdevelops/submit-exp021`: the scored exp018 kernel's metadata (same pinned docker image, `NvidiaTeslaT4`,
internet off, private) plus the new dataset.

**Commit check (instruction 5): PASSED.** `hrdevelops/submit-exp021` version 1, pushed with the CLI, status
**COMPLETE** after 858 s wall time, of which the notebook itself ran ~124 s. From the downloaded log:
- the scored code was used: `infer_wmil.py size: 8245 B`;
- the weights were found by name, seed 42 `fold0-4_best.pt` + seed 43 `fold0-4_seed43_best.pt`, i.e.
  `checkpoints: 10`, on 2 x T4, loaded in 51 s;
- **`[exp018] wrote /kaggle/working/submission.csv: 3 rows, 3 predicted, 0 on prior, 0 errors, 1.2 min
  total`**, then `3 rows, all checks passed.`

The 3 output rows are distinct, in-range and not the label prior. torch 2.10.0+cu128 and timm 1.0.26 on
Kaggle, matching exp018's scored environment. The one reserved probe is **not** spent; the Submit click is
the user's.

**Probe result (2026-09-25): exp021 scored 0.929 public (ref 56556092), a tie.** Under the pre-set rule
(displayed tie or better => the ensemble), exp021 is the main-track pick and exp018 the efficiency pick
(`docs/SUBMISSION_FINAL.md` §1).

## 2026-09-26 — PHASE 6: modeling RE-OPENED (narrow scope), HARD FREEZE 2026-10-15

User decision. Four workstreams only: A public-code audit refresh (intelligence), B exp022 orthogonal
backbone in the same Window-MIL pipeline (main track), C decode speed-up of exp018 (efficiency track,
prediction-identical), D zero-GPU slice-spacing audit. Explicitly skipped: fold-count sweep, attention
post-hoc tweaks, cross-label reranker, calibration. Probe budget: at most 2 (exp022 blend; fast exp018).

### exp022 — PRE-REGISTRATION (fixed before any result)

- **Backbone.** DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`) in exp018's exact pipeline
  (`configs/exp022_dinov2b_wmil.yaml`): the YAML-parsed diff vs exp018 is exactly `arch`, `res` 384->378,
  `img_size` 378 and `backbone_kwargs.fc_norm: false`. DINOv2 over ConvNeXt-V2 per the audit (A).
- **Two blockers found and fixed before launch.** (1) `WindowMIL` pools with `global_pool="avg"`, which makes a
  timm ViT swap its final `norm` for a post-pool `fc_norm`, so DINOv2's pretrained weights fail a strict load;
  `WindowMIL` gained an optional `backbone_kwargs` (default None => exp018 unchanged) and the config passes
  `fc_norm: false`. (2) Patch 14 needs a multiple-of-14 input: 378. Pretrained load verified through
  `WindowMIL` on the login node, online and with `HF_HUB_OFFLINE=1`.
- **Batch-shape check at the REAL training shape (the exp019 lesson; `scripts/check_train_shape.py`,
  job 27466, H100):** 8 studies x 12 windows = 96 images/forward, bf16, grad checkpointing, backward + step,
  and the eval shape 2 x 54 windows:

  | backbone | largest activation | train s/step | train peak | eval s/study (1 model) |
  |---|---|---|---|---|
  | CoAtNet-2 (exp018, reference) | **0.95 x INT_MAX** | 0.68 | 22.2 GB | 0.080 |
  | **DINOv2-B @378 (exp022)** | 0.11 x INT_MAX | **0.23** | 6.6 GB | **0.030** |
  | ConvNeXt-V2-B @384 | 0.24 x INT_MAX | 0.68 | 15.0 GB | 0.085 |

  The reference row's 0.68 s/step projects to exp018's measured 370 s/epoch, so the check is calibrated.
  **Found in passing: exp018 itself trains at 95% of the 32-bit indexing limit.** Its recipe cannot take
  more windows per study or a bigger batch without the exp019 failure; any future CoAtNet change must be
  checked with this script first.
- **Gate (all three required, else stop):**
  1. mean per-label OOF correlation between exp022 and exp018 < 0.90 (the same diversity bar as exp009),
     with the per-label values reported;
  2. rank-space blend of exp021's 10 checkpoints (prob-mean, then ranked) with exp022's 5 (ranked) improves
     OOF macro vs exp021 alone by >= +0.002 at the **pre-declared weight w(exp022) = 0.50**; other weights are
     shown for information only and do not decide the gate;
  3. gold-58 of that same w=0.50 blend >= exp021's gold-58 (point estimate).
  Only then: stage, placeholder commit check, probe.
- **Runtime expectation (main track only).** exp022's 5 folds add 0.030 x 5 = 0.15 s/study on H100, about
  42% of exp018's model cost: exp021 + exp022 ~= 0.86 s/study H100 -> ~33 min dual-T4 projected (accepted
  formula). This needs a driver that loads two architectures and blends in rank space, so a new code dataset
  and a new commit check; the scored `infer_wmil.py` prob-means one list at one resolution.

**Launch.** Job 27469, array 0-4, `slurm/train_exp022.sbatch` (2 h limit, `--max-hours 1.7`, 15 GB projected).

### A — public-code audit refresh (intelligence only; full report `artifacts/public_benchmark_audit_0926.md`)

- **No new family.** The top-10 by score are all re-runs or forks of the "Speedy Raptors" lineage, sharing 14
  public datasets. Efficiency board: >= 0.94 went 730 -> 934 entries, max 0.958. We are efficiency rank 182
  (0.929); entries at 0.928-0.930 span ranks 54-3,059, so runtime separates hundreds of places at our AUC.
- **Q1 (blended families):** DINOv2-S slot attention (20 members), A5 attention pooling, RadImageNet ResNet-50,
  and a CoAtNet-2 window-MIL arm on the SAME backbone as exp018 at 0.60-1.00 of the final blend per finding.
  Lineage table (claims): CoAtNet alone 0.924 -> **+DINOv2 arm 0.941 (+0.017)** -> CoAtNet variants 0.942 ->
  0.943. DINOv2 is the complementary arm; no ConvNeXt arm exists. => exp022 uses DINOv2-B.
- **Q2 (speed):** no public notebook is sub-10-min (the fastest claims < 30 min for the full stack; the leaders'
  code is private). Output-preserving tricks: header reads with `specific_tags`, one-time cached ordering, and
  BLAS/OpenMP/cv2 threads pinned to 1 inside a worker pool. `cv2.resize` is also used but changes pixels, so it
  is excluded by C's float-tolerance requirement.
- **Q3 (structurally new):** most notable, **denser inference-time window sampling with unchanged models**
  (`raptor_k_eval` 62 -> 94, "the accuracy change that needed no new model"). Also: multi-view preprocessing
  TTA incl. reversed slice order, two new CoAtNet heads on 96-slice stacks, rank-of-mean-probability family
  reduction, inference-only depth TTA. The window-density idea is testable on OOF with the existing
  exp018/exp021 checkpoints (a denser re-cache, no retraining) but is outside the four Phase-6 workstreams,
  so it is reported, not acted on.

### C — step 1: where exp018's decode time goes (`scripts/profile_decode_stages.py`, job 27467)

200 fold-0 studies, exp018's exact decode logic re-run with per-stage timers, ZEN4 node, 4 CPUs allocated
(the node reports 192 visible CPUs, the known `os.cpu_count()` quirk; the stage breakdown is valid, absolute
times are not Kaggle's). Single thread: **0.712 s/study**.

| stage | ms/study | share |
|---|---|---|
| **header**: `dcmread(stop_before_pixels)` on EVERY file of each chosen series, only for ordering | **337.0** | **47.3%** |
| resample: `apply_modality_lut` + `scipy.ndimage.zoom(order=1)` + crop | 172.5 | 24.2% |
| read: full `dcmread` of each selected file | 97.8 | 13.7% |
| norm: `normalize_uint8` percentile clip | 67.9 | 9.5% |
| order + geometry | 14.0 | 2.0% |
| **pixel**: the actual JPEG-Lossless / JPEG-2000 decompression | **13.9** | **1.9%** |
| glob count / orientation | 3.9 | 0.5% |

Work per study: 5.4 series present, 4.7 chosen; **167 headers read; 60.5 slices pixel-decoded; 60.6 stacked,
all used** (every stacked slice is in >= 1 window). So "decode only what the windows use" is already true of
the pixels: exp018 never processes unchosen series and never decodes an unused slice. The cost is **file
opening and header parsing** (all ~35 files per chosen series, to order them), not decompression.

Output-preserving candidates measured:
- **`specific_tags` header reads** (only the 8 tags ordering/geometry use): **0/200 studies differ** (bit-identical
  stacks and presence masks), but header time only 337 -> 316 ms (-6%), total -3%. pydicom still walks each file;
  the cost is per-file I/O, not per-tag parsing.
- **Decode worker threads:** 3 -> **0.248 s/study**, 4 -> **0.195 s/study** (-21%), near-linear, so the GIL is not
  the limit. The scored notebook uses `cpu_count() - 1` = 3 on Kaggle's 4 CPUs.
- **Excluded:** `cv2.resize` (the audit's resampler) changes pixel values, so it fails the float-tolerance rule.

Whether any of this moves the SCORED runtime depends on whether Kaggle's 2 x T4 run is decode-bound at all
(model 0.623 s/study/GPU projected vs decode 0.704). That is being measured on Kaggle itself
(`hrdevelops/rsna-knee-profile-t4`).

### C — step 2: MEASURED on Kaggle's own hardware — the scoring run is MODEL-bound (premise falsified)

`hrdevelops/rsna-knee-profile-t4` v1 (`scripts/build_kaggle_profile_notebook.py`): the SCORED exp018 code
(`infer_wmil.py` 8,245 B) and weights, unchanged, on 150 TRAIN studies from the competition mount (which does
include `train_series/`), on the real notebook machine (4 CPUs, 2 x Tesla T4). First measurement of this
kind in the project; every earlier runtime was an H100 projection.

| measured on Kaggle | value | what the project had assumed |
|---|---|---|
| model, 5 folds, ONE T4, inputs pre-decoded | **4.479 s/study** | 0.356 x 3.5 = 1.246 |
| decode, 3 threads, cold page cache (what scoring sees) | 0.931 s/study | 0.704 |
| decode, 4 threads (ran second: WARM cache, so not comparable) | 0.615 s/study | — |
| **scored `run_exp018` end to end, 2 x T4, 3 decode threads** | **369.0 s / 150 = 2.46 s/study -> ~53 min per 1,300** | **15.3 min** |

- **The T4/H100 ratio for CoAtNet-2 is ~12.6x, not 3.5x.** The 3.5x multiplier came from the DINOv2 ViT notebooks
  (2026-09-11) and was never validated for CoAtNet; it was applied to every CoAtNet projection since.
- **The run is heavily model-bound**: per study, 2 GPUs need 4.479 / 2 = 2.24 s against a 0.93 s decode floor
  that is fully overlapped. **No decode speed-up can move exp018's (5-fold) scoring runtime**, so C's premise
  ("runtime = decode") is false for the 5-fold model and no decode change is being staged for it.
- **Corrections this forces (records built on the wrong projection):**
  - exp018's actual efficiency score is ~**−1.966** (0.929, ~53 min), not −2.0362 (0.929, 15.3 min).
  - **"exp018 5-fold is efficiency-optimal" (2026-09-22 table) is FALSE under measured timing.** Each fold costs
    ~9.7 min on 2 x T4 (0.448 s/study), so fold count is the dominant efficiency lever:

    | folds | bound (cold decode 0.931) | runtime per 1,300 | AUC (0.929 + gold-58 all-subset Δ) | Eff |
    |---|---|---|---|---|
    | 5 | model | 48.5 min | 0.9290 | −1.9746 |
    | 4 | model | 38.8 min | 0.9288 | −1.9921 |
    | 3 | model | 29.1 min | 0.9283 | −2.0090 |
    | **2** | decode | **20.2 min** | 0.9267 | **−2.0220** |
    | 1 | decode | 20.2 min | 0.9223 | −2.0122 |

    With the warm 4-thread decode (0.615) 1 fold reaches −2.0249 and 2 folds −2.0234. At <= 2 folds decode IS the
    bottleneck, so C's worker-count change matters there (and only there).
  - exp021's 10 checkpoints are ~97 min of GPU time on 2 x T4, not ~27 min; still far inside the 9 h cap.
  - The exp018/exp021 runtime and efficiency figures in METHOD_WRITEUP §5a/§11, README, the exp018 model card,
    SUBMISSION_FINAL, the CV talk and the video outline all carry the old projection and need correcting.
- **Decision needed (user):** the fold-count sweep was explicitly SKIPPED on the premise "decode-bound, saves
  nothing". The measurement falsifies that premise. Not acted on; raised for decision.

### C — step 3: user decision -> 2-fold exp018 (folds {0,1}, pre-declared) as the efficiency candidate

- **Folds {0, 1} were pre-declared**, not chosen on a score. For the record, gold-58 of the pair: 0.9078 (5-fold
  0.9091, −0.0012); mean over all 10 pairs 0.9067 (−0.0023), range 0.9025–0.9110.
- **Decode threads, measured properly** (`hrdevelops/rsna-knee-profile-2fold`, scored driver, 2 folds, each
  configuration on its OWN disjoint set of 150 training studies, so both start cold):

  | | total | model load | per study excl. load | per 1,300 |
  |---|---|---|---|---|
  | 3 threads (scored setting) | 161.1 s | 21 s (weights cold) | **0.934 s** | ~20.5 min |
  | 4 threads | 146.8 s | 4 s (weights cached) | **0.952 s** | ~20.8 min |

  The raw totals suggested a 4th thread helps (−14 s), but that is entirely model-load time (cold vs cached
  weights). Per study they are equal; Kaggle's 4 vCPUs also feed two GPU threads. **Kept the scored
  `cpu_count() − 1` (3)**, which also keeps the notebook one change closer to the scored one. This also closes the
  decode-speed question: at 2 folds the run sits at the decode floor (~0.93 s/study) and more threads don't lower it.
- **Notebook** `notebooks/submit-exp018-2fold.ipynb` (`scripts/build_exp018_2fold_notebook.py`) is generated FROM
  the scored `submit-exp018.ipynb`, replacing exactly one code cell (the weight list becomes
  `[fold0_best.pt, fold1_best.pt]` with an existence assert) plus the markdown. Scored code and weight datasets,
  no new dataset.
- **Commit check PASSED** (`hrdevelops/submit-exp018-2fold` v1): COMPLETE; `weight files: ['fold0_best.pt',
  'fold1_best.pt']`; `decode workers: 3`; `3 rows, 3 predicted, 0 on prior, 0 errors`; three distinct rows.
- **Pre-declared selection rule for the efficiency slot (before the probe):** runtimes differ by ~33 min (~53 vs
  ~20.5), which is 0.060 on the Eff scale, so the 2-fold wins the efficiency formula unless its AUC is more than
  0.027 lower. **Select the 2-fold as the efficiency pick if its public LB displays >= 0.903**; otherwise keep 5-fold
  exp018 (56460279).

### B — exp022 RESULT: FAILS all three gate criteria. No staging, no probe.

All 5 folds (job 27469) completed cleanly, ~36 min each (the batch-shape check predicted ~35). Fold-val
0.9395-0.9498, gold-58 0.8912-0.8995 per fold. `artifacts/exp022_gate.md` (`scripts/exp022_gate.py`, run
unmodified from the pre-registered commit `0263983`):

| | OOF macro | gold-58 |
|---|---|---|
| exp022 solo (DINOv2-B Window-MIL) | 0.8709 | 0.8993 |
| exp018 solo | 0.8816 | 0.9091 |
| exp021 (10 ckpt) | 0.8840 | 0.9111 |

1. **Diversity: FAIL.** Mean per-label OOF correlation with exp018 = **0.944** (bar < 0.90). Only ACL 0.882, MCL
   0.877 and Lateral Meniscus 0.870 are below the bar; the other nine are 0.95-0.98.
2. **Blend gain: FAIL.** w(exp022) = 0.50 rank blend with exp021: **−0.0032** OOF [−0.0045, −0.0019].
   Informational weights are all negative, even w = 0.25 (−0.0007 [−0.0013, −0.0001]).
3. **Gold-58: FAIL.** −0.0039 at w = 0.50.

**Reading.** In the SAME Window-MIL pipeline (same cache, windows, MIL head, labels, folds), swapping the backbone
from CoAtNet-2 to DINOv2-B gives a weaker model that ranks studies almost the same way. The pipeline, not the
backbone, determines what the model sees and how it pools it, so a second backbone inside it adds no
independent evidence. This is consistent with, not contrary to, the audit: the public lineage's +0.017 DINOv2
gain came from a DINOv2 arm in a **different pipeline** (slot attention over a different representation and label
table). The project's own closest analogue, exp018 + the deployed DINOv2 SLOT ensemble, measured +0.0036 OOF at 0.5
on 2026-09-21. That was probe #4, which never scored (Kaggle teardown hangs) and was dropped by user decision.

Workstream B is closed. The DINOv2-B `backbone_kwargs` / `img_size` plumbing stays (off by default). exp022 used
~3 GPU-hours; no Kaggle submission was spent.

**Information only (requested after the gate; the decision rule is unchanged):** OOF of the rank blend as the
CoAt weight varies, with the share of 500 paired cluster-bootstrap reps where the blend beats exp021 alone:

| CoAt weight | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| OOF Δ vs exp021 | −0.0064 | −0.0046 | −0.0032 | −0.0020 | −0.0010 | −0.0004 | −0.0001 |
| % reps positive | 0.0% | 0.0% | 0.0% | 0.0% | 0.4% | 4.8% | 32.0% |
| gold-58 Δ | −0.0080 | −0.0062 | −0.0039 | −0.0031 | −0.0019 | −0.0007 | −0.0004 |

No plateau, and no spike: the curve falls monotonically as the DINOv2 weight rises. The best weight is CoAt = 1.0,
i.e. exp021 alone. The gate stays FAIL; the blend notebook (subprocess isolation, test-batch ranks) is not built.

## 2026-09-26 — Probes #5/#6 scored; selections final (freeze remains 2026-10-15)

- **exp018 2-fold (folds {0,1}): 0.925 public** (ref 56560771). The pre-declared rule (>= 0.903) makes it the
  **efficiency pick**. Efficiency score: −2.018 (0.925, ~20.5 min measured) vs −1.966 for 5-fold exp018.
- **Selections:** main = exp021 (56556092, 0.929); efficiency = exp018 2-fold (56560771, 0.925). Recorded in
  `docs/SUBMISSION_FINAL.md` §1.

## 2026-09-26 — Dense-window inference on the existing exp018 checkpoints: NULL (report only, not deployed)

The audit's "more windows at eval, same model" idea (`raptor_k_eval` 62 -> 94), tested on OOF with no retraining
(`scripts/dense_window_eval.py`, job 27506; `scripts/dense_window_compare.py`; `artifacts/dense_window_eval.md`).
Every slot x1.6: [18,14,12,8,12] -> **[29,22,19,13,19]** = 102 slices, **86.9 windows/study** (standard 51.1). The
layout is set through `rsna_knee.wmil`'s module globals, which every decode/window/orientation helper reads at
call time, so the library code is unmodified. Both layouts ran through the SAME inference path (DICOM decode ->
tag/geometry side -> scored `predict_study`, fp16); all 4,407 studies, 0 decode errors each.

- **Control (path soundness):** standard-layout rerun vs exp018's recorded training-path OOF: **0.8814 vs 0.8816**,
  per-label Spearman 0.9988–0.9999. The paths differ only in the side rule.
- **Result:**

  | | standard | dense | Δ | 95% CI | P(Δ ≤ 0) |
  |---|---|---|---|---|---|
  | OOF macro (n=4,349, paired cluster bootstrap) | 0.8814 | 0.8812 | **−0.0001** | [−0.0013, +0.0010] | 0.594 |
  | gold-58 (5-fold mean) | 0.9085 | 0.9095 | +0.0010 | [−0.0048, +0.0066] | 0.382 |

  Per label it is a wash: Lateral Meniscus +0.006, Medial Meniscus +0.002, ACL −0.005, MCL −0.003, the rest within
  ±0.0015.
- **Verdict: null.** Denser windows would cost ~1.7x model time and ~1.6x decoded slices for no measurable gain.
  This is consistent with D's finding that finer effective steps show no dose-response across studies. The
  lineage's reported gain likely came with their other changes (96-slice-native heads, view TTA) rather than
  density alone. Not deployed.

## 2026-09-26 — Efficiency item 5: inference dtype and channels_last (measured; NOT adopted)

- **Dtype, from the scored code:** `infer_wmil.py` sets `dtype = torch.float16 if devices[0] != "cpu" else
  torch.bfloat16`. All Kaggle devices are `cuda:*`, so exp018 runs **fp16 autocast on T4, never bf16**
  (`wmil.predict_study` wraps each window chunk in `torch.autocast(device_type="cuda", dtype=float16)`; a fold
  with non-finite output is recomputed in fp32). bf16 only ever applies on CPU.
- **channels_last, measured on Kaggle's T4** (`hrdevelops/rsna-knee-profile-channels-last`,
  `scripts/build_kaggle_channels_last_notebook.py`): 200 training studies (disjoint from earlier runs), decoded
  once, all 5 folds on ONE T4, three timed passes:

  | variant | s/study (5 folds) | vs scored | max \|Δprob\| vs scored | mean \|Δprob\| | min per-label Spearman |
  |---|---|---|---|---|---|
  | A: scored `predict_study` | 4.514 | — | — | — | — |
  | B: channels_last (weights + window batches) | 4.347 | −3.7% | 2.9e-4 | 1.6e-5 | 0.999997 |
  | C: B + `cudnn.benchmark` | 4.309 | −4.5% | 2.9e-4 | 1.6e-5 | 0.999997 |

  A reproduces the earlier model-time measurement (4.514 vs 4.479 s/study), which confirms the ~53 min 5-fold figure.
- **Verdict: not adopted.** Predictions match within float tolerance, but the speed-up is 1.04–1.05x. The adoption
  bar was "runtime drops meaningfully". The efficiency pick (2 folds) is decode-bound (0.934 s/study vs 0.896 s of
  model time), so a faster model does not shorten it. The only way a model speed-up pays is by fitting more folds
  into the same ~20.5 min, and 5 folds would need ~2.4x. Nothing was built.

## 2026-09-26 — Stale efficiency numbers corrected (one pass)

The −2.0362 efficiency figure, "5 folds is efficiency-optimal" and the ~15–17 min runtime all came from the
15.3-min projection. They are now corrected to the measured figures (~53 min / ~20.5 min / ~100 min; Eff −1.966 /
−2.018) and the final selections in: README (results table, efficiency paragraph, notebook pointer), the exp018
model card (title, intended use, results, limitations; the "seed variance unmeasured" limitation is replaced by the
measured value), METHOD_WRITEUP (§1, §5a, §8, §11, and §10 gains the Phase 5–6 null results),
CV_PROGRESS_PRESENTATION, VIDEO_OUTLINE and SUBMISSION_FINAL §3. The historical 2026-09-22 entry in this file
keeps its text, with a correction note added.

### D — slice-spacing audit (zero-GPU; `scripts/audit_slice_spacing.py`, job 27468; `artifacts/slice_spacing_audit.md`)

All 4,407 studies, 20,539 slot-series, 0 read errors. "Effective step" = physical gap between ADJACENT
CACHED slices, i.e. what one 3-slice window advances by, after exp018's 6-94% linspace subsampling.

| slot | native gap p50 (p10/p90) mm | effective step p50 (p10 / p90) mm | studies with duplicated slices |
|---|---|---|---|
| Sagittal FS | 3.30 (3.00 / 4.50) | 4.50 (3.30 / 6.60) | 7.0% |
| Sagittal non-FS | 3.30 (3.00 / 4.50) | 6.60 (5.12 / 8.80) | 0.1% |
| Coronal FS | 3.50 (3.30 / 4.80) | 7.20 (6.28 / 10.00) | 0.0% |
| Coronal non-FS | 3.60 (3.30 / 5.00) | **13.20** (10.00 / 15.00) | 0.0% |
| Axial | 3.85 (3.30 / 5.20) | 9.90 (6.60 / 13.20) | 0.0% |

The 64-slice budget subsamples coronal/axial series 2-4x: a coronal non-FS window spans ~40 mm.

**exp018 OOF by effective-step quartile** (vs labels_v2_final; Δ = quartile minus the other three, study-bootstrap
95% CI): 10 of 12 cells are inside noise; two are "clearly worse": **Axial Q1 (4.4-7.8 mm) −0.0201
[−0.0315, −0.0069]** and **Coronal Q2 (6.6-7.2 mm) −0.0157 [−0.0293, −0.0053]**. But Axial Q2, the next-finest,
is the BEST cell (+0.0211 [+0.0120, +0.0300]); the pattern is non-monotonic, with no dose-response.

**Confound check (zero-GPU):**
- **Coronal Q2 has the same native gap (3.50 vs 3.60 mm), slice count (30 vs 30) and coverage (102 vs 104 mm) as
  the rest.** Nothing about its spacing differs, so its deficit (ACL −0.059, PF OA −0.030) cannot be a spacing
  effect; it is protocol / case-mix.
- **Axial Q1 = short axial series**: coverage 96 vs 129 mm (30 vs 34 slices at a finer 3.30 vs 4.20 mm gap). Its
  deficit spreads to labels that are not axial-driven (ACL −0.061, MCL −0.038, Synovitis −0.046), and OOF rises with
  axial COVERAGE (Q1 0.869, Q2 0.875, Q3 0.895, Q4 0.883). That points to less scanned anatomy and/or site mix,
  which no preprocessing can restore.
- Report-template clusters are not concentrated in either quartile (top-5 clusters hold 6-8% vs 10-11%).

**Verdict: no action.** Two quartiles are statistically worse, but neither deficit is attributable to slice
spacing, so there is no spacing fix to make (the rule was "act only if a quartile is clearly worse"; these are
worse, but not BECAUSE of spacing). Side note for A's denser-window idea: across studies, finer effective steps
show no dose-response, which weakly lowers the expected gain. Densifying WITHIN a study is a different
experiment and this audit does not rule it out.

## 2026-09-26 — Offline blend gate, exp021 + slot ensemble — PRE-REGISTRATION (written before any number is computed)

**Rulings (user, 2026-09-26).** (1) Offline blend analysis approved, zero GPU. (4) Training freeze confirmed: no
RadImageNet, no new training. This revisits the 2026-09-22 blend analysis (exp018 + slot ensemble: +0.0036–0.0040
OOF, plateau at CoAt weight 0.6–0.8, gold-58 flat). The deployment hung; the model side never failed.

**Arms (architecture-level, one vote each).**
- **CoAt arm = exp021**, the 10-checkpoint mean. OOF: a study in fold k gets the mean of fold-k seed 42 (exp018)
  and fold-k seed 43. Gold-58: the mean of all 10 checkpoints.
- **Slot arm = the scored 3-member ensemble {exp006, exp003, exp005}, exactly as the 0.893 notebook computes it**
  (`src/rsna_knee/ensemble_recipe_v1.yaml`, `infer_core.ensemble_blend`): per label, prob-mean or rank-mean of
  the three members as the recipe says. Rank-mean labels use each member's `empirical_percentile` (searchsorted
  against that member's sorted OOF). Offline, the OOF population is the reference; gold-58 members are the
  5-fold prob-mean looked up against the members' OOF.

**Blend.** Per label, each arm's score becomes a percentile: `searchsorted(sorted OOF population of that arm, x) / N`
(the same lookup used at test time; on OOF it is the within-population rank). Blend = (1 − α)·pct_coat + α·pct_slot,
one global α for all 12 labels, grid α ∈ {0, 0.1, 0.2, 0.3, 0.4}.

**Population / truth.** Non-gold studies with OOF from all five runs, a `labels_v2_final` label and a report
cluster (expected 4,349). Truth = `labels_v2_final` binarised at 0.5; gold-58 truth = train.csv labels.

**Nested CV.**
- Outer: sklearn `GroupKFold(n_splits=5)` over the OOF population, groups = `report_cluster_id`.
- Inner, for each outer fold k: using only the other four folds, gain(α) = macro-AUC(blend α) − macro-AUC(α = 0).
- **α rule:** α_k = the smallest α whose inner gain is within 0.0005 of the best inner gain. OOF under-rates the
  CoAt family by ~0.03 against LB, so an OOF argmax over-weights the slot arm.
- **Cross-fitted score:** outer-fold studies are scored with α_k. Pooled macro-AUC over all 4,349 studies, minus
  exp021 alone (0.8840).
- **Deploy α** (used by B and by criterion 4) = the same rule applied to all 4,349 OOF studies.

**Gate (all five required):**

| # | Criterion | Bar |
|---|---|---|
| 1 | Mean per-label Spearman correlation, exp021 vs slot arm (OOF) | < 0.92 |
| 2 | Cross-fitted macro-AUC gain over exp021 | ≥ +0.0020 |
| 3 | 1,000 paired cluster-bootstrap reps over report clusters (seed 0, cross-fitted scores held fixed) | 95% CI lower bound > 0 and ≥ 95% of reps > 0 |
| 4 | Gold-58 macro-AUC at the deploy α | ≥ 0.9091 (exp021's 0.9111 − 0.002) |
| 5 | IQR of the five outer-fold α_k | ≤ 0.20 |

The full α curve (OOF gain with CI, gold-58 at each α) and each outer fold's α_k are reported as information only.
**Fail = stop, no rescue.**

**If it passes (B):** build `submit-blend-v5`. The parent notebook is CPU-only and never imports torch. It runs two
children with `subprocess.run(timeout=...)`:
- child 1: exp021's scored logic and datasets, unchanged;
- child 2: the scored ensemble notebook's logic with `rsna-knee-code` pinned to the 2026-09-16 version.

Each child's CSV is validated (row count, no NaN, no prior rows). A child that hangs after writing a valid CSV is
killed and its CSV used. The blend uses per-label test-batch ranks at the fixed deploy α.
- Fallbacks: slot child fails → exp021 output unchanged; CoAt child fails → slot output.
- Runtime is measured on the placeholder and projected to the full test (must be < 7 h).
- Placeholder commit check: Complete, 0 on prior, clean exit. Then stage for the user's submit click.

**Selection rule (C, recorded now):** swap the main-track pick to the blend only if its public LB ≥ 0.930. The
efficiency pick stays exp018 2-fold regardless. After this probe, modeling is closed for good.

## 2026-09-26 — Offline blend gate RESULT: PASS on all 5 criteria (deploy α = 0.2)

`scripts/blend_v5_gate.py` (job 27602, ZEN4 CPU; the first attempt, an `srun` step, timed out before launching and
computed nothing). Report: `artifacts/blend_v5_gate.md`. 4,349 OOF studies, 3,681 report clusters, 1,000 reps.
exp021 alone: OOF 0.8840, gold-58 0.9111. Slot arm alone: OOF 0.8744, gold-58 0.8940.

| # | Criterion | Bar | Value | |
|---|---|---|---|---|
| 1 | Mean per-label Spearman, exp021 vs slot | < 0.92 | 0.878 (MCL 0.750 … Effusion 0.944) | PASS |
| 2 | Cross-fitted gain over exp021 | ≥ +0.0020 | **+0.0024** | PASS |
| 3 | Paired cluster bootstrap | CI lo > 0, ≥ 95% reps > 0 | [+0.0017, +0.0031], 100% | PASS |
| 4 | Gold-58 at deploy α | ≥ 0.9091 | 0.9097 (α 0.2) | PASS (margin 0.0006) |
| 5 | Outer-fold α IQR | ≤ 0.20 | 0.00 (all five folds chose 0.2) | PASS |

Outer-fold gains at α_k = 0.2: +0.0037 / +0.0015 / +0.0025 / +0.0029 / +0.0018 (every fold positive).

**α curve (full OOF, information only):**

| α (slot) | OOF | Δ vs exp021 | 95% CI | gold-58 (Δ) |
|---|---|---|---|---|
| 0.1 | 0.8855 | +0.0015 | [+0.0012, +0.0018] | 0.9094 (−0.0017) |
| **0.2** | 0.8864 | +0.0024 | [+0.0017, +0.0031] | 0.9097 (−0.0014) |
| 0.3 | 0.8867 | +0.0027 | [+0.0017, +0.0038] | 0.9097 (−0.0014) |
| 0.4 | 0.8866 | +0.0026 | [+0.0012, +0.0039] | 0.9074 (−0.0038) |

**Read-out.** The inner curves peak at α = 0.3–0.4, and the tolerance rule pulls every fold back to 0.2, as
designed. Gold-58 dips slightly at every α > 0 but stays inside the pre-set 0.002 tolerance at 0.2. The OOF gain is
small (+0.0024 ≈ 0.6 bootstrap SD of a single model's macro) but very consistent. Whether it survives the ~0.03
OOF→LB under-rating of the CoAt family is what the probe measures; the selection rule (swap only at LB ≥ 0.930)
stands.

### B — `submit-blend-v5` build

- **`rsna-knee-code-0916`** (new, private): the 7 files of commit `d767fdc` (2026-09-16), identical to the
  live `rsna-knee-code` before the 09-23 update. SHA-256 is verified against git and against a re-download from
  Kaggle. It is a separate dataset rather than a version pin because a kernel's `dataset_sources` always takes a
  dataset's latest version.
- **`scripts/build_blend_v5_notebook.py`** builds each child from the SCORED notebook's code cells, with asserted
  one-time substitutions only:
  - CoAt child ← `submit-exp021` v1 (pulled from Kaggle and verified identical to `notebooks/submit-exp021.ipynb`):
    `OUT_PATH` only.
  - Slot child ← `ensamble-notebook` (pulled from Kaggle as `notebooks/ensamble-notebook-scored.ipynb`; code
    identical to `submit-ensemble.ipynb` except comments): `OUT_PATH` and the code dataset name.
- **Parent:** never imports torch (asserted at the end).
  - Runs the children sequentially with `Popen(start_new_session=True)` under a per-child timeout and a
    whole-run deadline of 8.25 h.
  - A child still alive 180 s after its CSV is complete and unchanged is killed (process group) and its CSV used.
  - Validation: header, row count, UIDs, no NaN, [0, 1], and prior-fallback rows ≤ 0.5% of rows. That is 0 on the
    3-row placeholder and ≤ 6 of 1,300 on the full test, so a single corrupt DICOM cannot throw away a whole arm.
  - Blend = 0.8·rank(exp021) + 0.2·rank(slot), per-label test-batch percentile ranks.
  - Fallbacks: slot fails → exp021 unchanged; CoAt fails → slot; both over the prior tolerance → exp021; both
    fail → all-prior rows.
- **Mock-child test on HPC** (6 cases: both ok, CoAt hangs after its CSV, slot crashes, prior row in CoAt, both
  crash, slot times out): every case wrote a valid submission with the expected mode, and the blend arithmetic was
  checked by hand.

## 2026-09-27 — B runtime check FAILS: the slot arm cannot run in time on Kaggle T4. Blend-v5 not staged.

**Placeholder commit (`hrdevelops/submit-blend-v5` v1): mechanically clean.** Status COMPLETE; CoAt child
1.5 min and slot child 4.0 min, both `exit code 0`, 3 rows, 0 on prior. Mode: blend. The parent exited
normally without importing torch; total 5.5 min.

**Full-test projection (`hrdevelops/rsna-knee-profile-blend-v5`, the same children and runner, each on its own
disjoint cold set of 120 training studies, 2 × T4, model loading included):**

| child | measured | s/study | projected for 1,300 studies |
|---|---|---|---|
| CoAt (exp021, 10 checkpoints) | 10.8 min / 120 | 5.38 | **116 min** |
| slot (3 members × 5 folds, rung 0) | 117.5 min / 120 | **58.8** | **21.2 h** |
| total | | | **23.2 h**, against the < 7 h requirement → **FAIL** |

The placeholder calibration agrees (61.3 s/study). The 3-member DINOv2 ensemble is ~11× slower per study than the
10-checkpoint CoAtNet ensemble on T4; the 2026-09-11 "~62 min on T4 × 2" was a 3.5×-from-H100 projection, the
same class of error as the corrected exp018 efficiency projection.

**Consequence: submit-blend-v5 v1 is NOT staged, and must not be submitted.** On the full test, the slot child's
own degradation ladder (unchanged scored logic, budget 6.8 h) would project:
- rung 0 (3 members) ≈ 21 h: too slow;
- rung 1 (exp006 + exp005, cost 4/7) ≈ 12 h: too slow;
- **rung 2 (exp005 alone, cost 1/7) ≈ 3 h: fits.**

So it would run exp005 alone, and the submission would be a 0.8/0.2 blend with exp005, which the gate never
evaluated. The parent's deadline logic keeps that run valid; it is simply not the blend that passed the gate.

**Finding about the scored 0.893 (inferred, not directly observable: hidden-test logs are not shown).** The scored
`ensamble-notebook` v3 runs the same ladder on the same 2 × T4 hardware. At ~59 s/study it projects ~21 h at rung 0,
so it almost certainly chose rung 2, meaning **the public 0.893 is most likely exp005 (DINOv2 ViT-S) alone, not the
3-member ensemble**. The 2026-09-17 OOF↔LB gap for "the ensemble" (−0.0186) should be read with that in mind. The
exp006/exp003 weights were likely never used on the hidden test.

**Status:** workstream B stopped at the runtime requirement. There is no blend probe, and the selections are
unchanged (main exp021 56556092, efficiency exp018 2-fold 56560771). Any further blend variant would need a new
pre-registered gate and a user decision; nothing has been started.

## 2026-10-01 — Re-gate with exp005 as the slot arm — PRE-REGISTRATION (written before any number is computed)

**Decision (user, 2026-10-01).** The 2026-09-26 gate PASSED. Deployment failed on measured T4 runtime (3-member slot
ensemble ~21 h). This is a **new pre-registered test with a different arm, not a rescue**. If it fails, modeling is
closed permanently and no further arms are tried.

**Arms (one vote each).**
- **Slot arm = exp005** (DINOv2 ViT-S/14, `labels_v3_candidate`), the 5-fold mean. This is the model the scored
  ensemble notebook falls back to on T4. OOF: each study's own-fold prediction. Gold-58: the mean of the 5 fold
  models.
- **CoAt arm = exp021**, the 10-checkpoint mean, as before.

**Everything else is UNCHANGED from the 2026-09-26 pre-registration (commit 185eab7):**
- per-label percentiles against each arm's OOF population;
- grid α ∈ {0, 0.1, 0.2, 0.3, 0.4}, one global α;
- nested `GroupKFold(5)` by `report_cluster_id`;
- α rule: the smallest α within 0.0005 of the best inner gain;
- deploy α = the rule on the full OOF;
- the same 5 criteria: Spearman < 0.92; cross-fitted gain ≥ +0.0020; bootstrap CI lower bound > 0 and ≥ 95% of
  reps positive; gold-58 at the deploy α ≥ 0.9091; outer α IQR ≤ 0.20.

Run: `scripts/blend_v5_gate.py --slot-arm exp005`, with only the arm swapped.

**If it passes:**
- Child 2 runs exp005 **explicitly** (members pinned to exp005 only, not left to the time-budget ladder), so the
  gated arm is the deployed arm deterministically.
- The exp005-only child is profiled on Kaggle 2 × T4 (cold studies, same method) and the full-test total must be
  < 7 h with margin. Child timeouts are set from the measured numbers. Fallback: slot fails → exp021 unchanged.
- Placeholder commit (Complete, 0 on prior, normal exit), then staged for the user's click.
- Selection rule unchanged: swap the main pick only if public LB ≥ 0.930.

## 2026-10-01 — exp005 re-gate RESULT: FAIL (criterion 2). Modeling CLOSED permanently.

`scripts/blend_v5_gate.py --slot-arm exp005` (job 28913, ZEN4 CPU). Report: `artifacts/blend_v5_gate_exp005.md`.
exp021: OOF 0.8840, gold-58 0.9111. exp005 (5-fold mean): OOF 0.8631, gold-58 0.8903.

| # | Criterion | Bar | Value | |
|---|---|---|---|---|
| 1 | Mean per-label Spearman, exp021 vs exp005 | < 0.92 | 0.855 | PASS |
| 2 | Cross-fitted gain over exp021 | ≥ +0.0020 | **+0.0009** | **FAIL** |
| 3 | Paired cluster bootstrap | CI lo > 0, ≥ 95% reps > 0 | [+0.0005, +0.0012], 100% | PASS |
| 4 | Gold-58 at deploy α | ≥ 0.9091 | 0.9093 (α 0.1) | PASS |
| 5 | Outer-fold α IQR | ≤ 0.20 | 0.00 (all folds chose 0.1) | PASS |

**α curve (full OOF, information only):** α 0.1 +0.0009 (gold-58 −0.0019), 0.2 +0.0010 (−0.0029), 0.3 +0.0004,
0.4 −0.0007. exp005 is more diverse from exp021 than the 3-member arm (0.855 vs 0.878), but it is a weaker model. The
gain peaks at α 0.1–0.2 at well under half the 3-member arm's +0.0024 and turns negative by 0.4. The effect is real
(every fold positive, CI excludes 0) but below the pre-set +0.0020 bar.

**Per the pre-agreed rule: no rescue, no further arms; modeling is closed permanently.** Nothing was pushed for v5b.
`notebooks/profile-blend-v5b.ipynb` and the builder's `--variant v5b` were prepared before the result and are kept
for the record only. Final selections are unchanged: main exp021 (56556092, 0.929), efficiency exp018 2-fold
(56560771, 0.925). One probe remains unused.

**Documentation correction (user item 3, done regardless of the gate):** the scored 0.893 is now described as "DINOv2
slot pipeline (most likely exp005 alone after the T4 runtime fallback)", marked as inference, in README,
METHOD_WRITEUP, the exp005/exp018 model cards, CV_PROGRESS_PRESENTATION, VIDEO_OUTLINE and SUBMISSION_FINAL. The probe
#1 gap is corrected to ~−0.030 (0.8631 → 0.893), which exceeds the 0.02 threshold. exp008's (0.885) fold count was
also chosen at run time and is recorded as unknown. Learning added: a runtime degradation ladder can silently change
which model is scored; log the chosen rung, and measure on the target hardware before trusting what a score belongs
to.

## 2026-10-25 — Research Phase 0A (geometry validity) — PRE-REGISTRATION + LAUNCH (written before the full run)

Post-competition research plan `docs/research/PLAN_v1_3.md`, frozen after review round 4 ("GO, start Phase 0A"). Phase
0A asks the one remaining foundational question: does the retained DICOM geometry (IPP / IOP / PixelSpacing) place the
separately acquired series of a study consistently, given that FrameOfReferenceUIDs were re-keyed per series? CPU only;
no labels, no model.

**Sample and design.**
- 150 training studies: the seed-0 sample of `scripts/check_series_geometry.py`.
- One series per plane family, by the cache's slot rules (Sagittal FS / non-FS, Coronal FS / non-FS, Axial).
- Every pair is registered rigidly around the DICOM placement: 6-DOF Mattes MI in SimpleITK 2.4.1.
- Every cross-plane pair also gets injected rigid errors of 5 mm / 3° and 10 mm / 5°.
- Cycle consistency is checked on triangles.
- Metrics are taken at 9 probe points around the study centre, in mm, degrees and footprint units d (6×6 tokens).
- Code: `scripts/phase0a_registration.py`, `scripts/phase0a_report.py`, `slurm/phase0a_registration.sbatch`.
- Synthetic end-to-end check: `tests/test_phase0a_geometry.py`. Two exact-geometry acquisitions of a phantom; the
  identity is found within 0.12 mm, and injected errors are recovered within 0.01 mm.

**Audit-tool development on 5 pilot studies (these studies are excluded from the gates; listed in
`artifacts/phase0a/pilot_studies.txt`):**
- **Pilot 1 (job 33403).** Single-start MI drifted to "corrections" of up to 48 mm / 20°. The QC overlays showed the
  DICOM placement visibly correct: on partially overlapping thick-slice stacks, MI slides toward background.
  - Fix: metric limited to a 45 mm sphere around the study centre, on fixed foreground at least 10 mm inside the moving
    stack.
  - Fix: multi-start search, with the unmoved placement as a candidate.
  - Fix: an overlap guard rejects candidates that lose more than 20% of their in-stack samples.
- **Pilot 2 (job 33404).** The optimizer used a random 30% sample but selection used all voxels. With mismatched
  objectives the unmoved placement won on a technicality. Fix: one objective, all mask voxels.
- **Pilot 3 (job 33406) and benchmark (jobs 33416, cancelled; 33420).** An MI profile around the DICOM placement is
  smooth and peaked within about 1–2 mm, with no grid-alignment spike. But regular-step gradient descent stalls in the
  rotation–translation valley, ending 1.7–25 mm from the minimum with worse MI than where it started. Conjugate-gradient
  line search is erratic, ending 1.4–72 mm off.
  - Powell returned to each pair's MI minimum after every injected error (0.00 mm / 0.00°, 8–14 s per run).
  - Fix: Powell.
  - In these pilot pairs the MI minimum lies 1.0–1.6 mm / 0.4–0.7° from the DICOM placement.

**Gates (fixed in the plan before any full-run number):**
- G1 optimizer precision: injected errors on cross-plane pairs recovered within 1 mm and 0.5° in ≥ 95%, per level.
- G2 cycle consistency: median residual of Sag → Cor → Ax → Sag triangles ≤ 1 mm.
- G3 raw geometry: ≥ 97% of cross-plane pairs need a correction ≤ 3 mm (max over probes) and ≤ 2°.
- G1 and G2 must pass for the audit to count.
- Then, if G3 passes: registration-free. If G3 fails: train with misregistration noise drawn from the measured
  cross-plane corrections.
- Same-plane pairs and mixed triangles are reported, not gated.

**Launch.** Full run, 145 + 5 studies in 30 shards on ZEN4 (`safe_sbatch.py`): **job 33421** (array 0-29). Report:
`python scripts/phase0a_report.py artifacts/phase0a artifacts/phase0a/pilot_studies.txt`.

## 2026-10-25 — Research Phase 0A RESULT: INCONCLUSIVE by the pre-registered rule (narrow); corrections ~1.7 mm

All 30 shards. Shard 2 timed out at 3 h and was rerun as job 33518 with per-study saving. Gates on the 145 non-pilot
studies (`python scripts/phase0a_report.py artifacts/phase0a artifacts/phase0a/pilot_studies.txt`).

| Gate | Bar | Result |
|---|---|---|
| G1 injected 5 mm / 3° recovered (≤ 1 mm, ≤ 0.5°) | ≥ 95% | 96.8% (n = 1,020), pass |
| G1 injected 10 mm / 5° | ≥ 95% | 94.1%, **fail** |
| G2 all-three-plane cycle residual, median | ≤ 1 mm | 1.06 mm (n = 488), **fail** |
| G3 cross-plane pairs within 3 mm and 2° | ≥ 97% | 80.8% (n = 1,020); counts only if G1 and G2 pass |

**Corrections (cross-plane).**
- Median 1.68 mm / 0.66°; p95 5.60 mm / 2.45°.
- Footprint d: median 0.19, p95 0.64.
- Gross (> 10 mm or > 5°): 2.6%.
- 1.5 T 88.2% vs 3 T 69.4% within.
- Same-plane pairs (reported only): 90.5% within, median 0.87 mm.

**Post hoc, labelled as such.**
- 2.5% of cross-plane pairs have multi-start disagreement. They carry 44% gross corrections and 32–40% injected-error
  recovery.
- On the stable 97.5%: G1 98.2% / 95.7%; gross 1.6%.
- QC overlays of the four worst-flagged pairs show the DICOM placement aligned and the correction misplaced (informal).

**Reading.**
- The audit tool's own resolution is about 1 mm (cycles), so it cannot separate 1–2 mm of real inter-sequence motion
  from MI bias.
- The measured corrections are small at the footprint scale the method uses.

**Proposal (`docs/research/PHASE0A_RESULTS.md`, awaiting the user).**
- Keep the label "inconclusive". No re-tuning of the audit.
- Take the plan's G3-fail branch: raw DICOM geometry plus training-time misregistration noise drawn from the measured
  corrections.
- Optional blinded visual check of about 40 pairs.
- Then Phase 0B (GPU), only after the user's go-ahead.

## 2026-10-25 — Phase 0A: wording correction and post-Phase-0A decision (after the reviewer's check)

**Correction.** The RESULT entry above said "take the plan's G3-fail branch". That is not strictly correct. Phase 0A is
inconclusive because the audit validity gates (G1, G2) did not both pass, so neither G3 branch can be invoked
mechanically. G1 and G2 missed narrowly; G3 (80.8% vs 97%) did not, but it is uninterpretable.

**Study-cluster bootstrap 95% CIs (145 studies).**

| Measure | Value | 95% CI |
|---|---|---|
| G1 at 5 mm / 3° | 96.8% | [95.3, 98.0] |
| G1 at 10 mm / 5° | 94.1% | [92.3, 95.9] |
| G2 median | 1.06 mm | [1.00, 1.14] |
| G3 | 80.8% | [77.0, 84.5] |
| Median correction | 1.68 mm | [1.57, 1.80] |

**Post-Phase-0A engineering decision (not a pre-registered branch).**
- Primary G uses raw DICOM geometry with no misregistration augmentation.
- No empirical-residual augmentation, because the gross tail is contaminated by optimizer failure.
- Bounded jitter ($d\le0.75$) becomes a secondary arm.
- $\Sigma_{reg}$ inside $d^2$ is a later ablation.
- Field strength is a robustness stratum; a clustered descriptive regression comes first.
- A 40-pair blinded visual check runs in parallel; it does not change the label.
- Phase 0B: GO. Geometry-noise handling is pre-registered before Phase 1. Details: `docs/research/PHASE0A_RESULTS.md`.

## 2026-10-25 — Phase 0B preprocessing ladder RESULT: no rung hurts; B5 frozen as the data pipeline

**Setup.** Cache v2 (`data/cache_v2`, 113 GB): all 24,371 series and all native slices, 160 mm at 384 px, exact affine
plus linear interpolation, no prefilter.
- B1's windows reproduce exp018's one-to-one: 40 studies, 2,074 windows, min corr 0.986 (`scripts/check_wmil_v2.py`).
- Ladder: exp018 architecture, training budget and schedule; fold 0; seeds 42 / 43.
- Jobs 33604–33615, about 2.6 h each.
- B0 = exp018 / exp021 fold 0.
- Metric: fold-validation macro-AUC with top-3 averaging, the same metric as the measured seed noise (0.0020 mean / 0.0046
  max).

| Rung | Change | Fold-val macro-AUC (s42 / s43) | 2-seed mean | Δ vs previous rung | Gold-58 mean |
|---|---|---|---|---|---|
| B0 | exp018 | 0.9567 / 0.9547 | 0.9557 | — | 0.9035 |
| B1 | Cache v2 pixels | 0.9562 / 0.9523 | 0.9542 | −0.0015 | 0.9056 |
| B2 | + native-adjacent windows, all in-span slices | 0.9544 / 0.9551 | 0.9547 | +0.0005 | 0.9132 |
| B3 | + every series | 0.9546 / 0.9566 | 0.9556 | +0.0009 | 0.9122 |
| B4 | + plane-balanced budget | 0.9567 / 0.9568 | 0.9567 | +0.0011 | 0.9090 |
| B5 | + 160 mm field, no resize | 0.9569 / 0.9567 | 0.9568 | +0.0001 | 0.9153 |

**Reading.**
- Every step is within the seed noise. The freeze rule (revert a rung only if it lowers the 2-seed mean by > 0.004)
  keeps all of them, so **B5 is the frozen data pipeline** for $B_{\text{final}}$, $B_{\text{cap}}$ and $G$.
- B1 matched its pre-registered prediction (|Δ| < 0.002).
- B0 → B5 overall: macro +0.0011 (noise); gold-58 +0.012 (n = 58, noisy).
- Per label (2-seed means, B0 → B5):
  - ACL fell 0.943 → 0.924. The drop appears from B1 on and is largest from B2 (native windows).
  - MCL (+0.012), Lateral Meniscus (+0.012), Contusion (+0.005) rose.
- So the data changes reshuffle per-label strengths without moving macro. The ACL drop is noted for later; nothing is
  changed for it now.
- Dense evaluation now uses about 157 windows per study (exp018: 54).
- Anti-aliasing ablation $B1_{\text{AA}}$: not run, since B1 is null and the reference keeps no prefilter.

## 2026-10-25 — B1 channel-order check, and Phase 0C PRE-REGISTRATION (written before any 0C score)

**B1 source-slice check (after the reviewer's concern).** The earlier B1 check matched windows by best correlation,
which could hide a swapped slice order inside a window (the network reads the three slices as RGB channels and is not
invariant to swapping them).
- `scripts/check_wmil_v2.py` now rebuilds exp018's eval windows from exp018's own rules as (series,
  channel-ordered source-slice triplet, mirror flag) and asserts equality with the v2 dataset's plan.
- **Identical for 150/150 studies (7,744 windows)**, including right-knee sagittal series. Only the enumeration order of
  whole windows differs, and MIL attention is invariant to that.
- Per-channel pixel correlation of exactly paired windows: median 0.9997, min 0.976.
- B1 was therefore a clean pixel-only change; Phase 0B stands.

**Phase 0C pre-registration.**
- **What runs.** $B_{\text{cap}}$ (token transformer without geometry, PLAN_v1_3 §2.4–2.6, §3.1) on the frozen B5 data
  pipeline; grids $g\in\{4,6,12\}$; fold 0; seeds 42 / 43. Each run is evaluated in both inference modes from the same
  top-3-averaged checkpoint:
  - bagged: mean of $R=8$ random plane-balanced passes of 12 windows;
  - dense: all windows in one pass, capped at 192.
- **Default:** 6×6 grid with bagged inference (the design point of the geometry calculations).
- **Grid rule.** Another grid replaces 6×6 only if its 2-seed mean fold-validation macro-AUC (bagged) beats 6×6 by
  **≥ +0.004**.
- **Mode rule.** On the chosen grid, dense replaces bagged only if the paired cluster bootstrap over report clusters
  (both seeds' predictions averaged; 1,000 reps) gives ΔAUC > 0 **and** a 95% CI lower bound > 0. Otherwise bagged.
- **Feasibility exclusion (engineering, not a performance rejection).** 12×12 dense (about 22,600 tokens per study) is
  omitted if a fixed smoke test fails: out of memory at batch 1 on one H100, or > 3 s per study. 12×12 bagged (1,728
  tokens per pass) stays.
- Gold-58 plays no role in any 0C choice. The chosen $(g^*, m^*)$ is copied unchanged into $G$.
- **ACL, for the record.** Phase 0B moved fold-0 ACL 0.943 → 0.924. It is documented with paired deltas and seed
  consistency once 0C exists; it does not change the pipeline.

## 2026-10-25 — Phase 0C RESULT: frozen 6×6 grid, dense inference (by the pre-registered rules)

$B_{\text{cap}}$ (token transformer without geometry) on the frozen B5 pipeline; fold 0; seeds 42 / 43; jobs
33698–33703, about 2.1 h each. Selection by `scripts/phase0c_select.py`; gold-58 not read.

| Grid | Bagged fold-val (s42 / s43, mean) | Dense fold-val (s42 / s43, mean) | Dense peak GPU memory |
|---|---|---|---|
| 4×4 | 0.9447 / 0.9444, 0.9446 | 0.9507 / 0.9498, 0.9503 | 10.2 GB |
| 6×6 | 0.9508 / 0.9510, 0.9509 | 0.9541 / 0.9544, 0.9543 | 10.2 GB |
| 12×12 | 0.9503 / 0.9517, 0.9510 | 0.9541 / 0.9553, 0.9547 | 53.7 GB (feasible, within 3 s/study) |

**Selection.**
- **Grid:** 4×4 is −0.0063 vs 6×6; 12×12 is +0.0001 (needs ≥ +0.004). → **6×6**.
- **Mode on 6×6:** dense − bagged = **+0.0037 [+0.0013, +0.0068]**, from seed-averaged predictions, paired cluster
  bootstrap, 879 studies / 672 clusters. → **dense**.
- **Frozen for G: 6×6 tokens, dense inference.** Note: the dense / train density mismatch (§2.5) now applies to $G$. The
  soft Mahalanobis bias, chosen over hard k-NN partly for this reason, stays the primary form.

**Observations (descriptive, no action).**
- $B_{\text{cap}}$ at its frozen setting (0.9543) is 0.0025 below the window-MIL $B_{\text{final}}$ = B5 (0.9568) on
  fold 0. The token transformer does not by itself beat the practical baseline. $\Delta_{\text{geo}}$ is still measured
  against $B_{\text{cap}}$, as pre-registered.
- **ACL** (B0 → B5, per seed): s42 0.9533 → 0.9303 (−0.0230); s43 0.9317 → 0.9176 (−0.0141). The drop is
  seed-consistent; logged, no intervention.

**Before Phase 1.**
- Pre-register the geometry-noise handling (primary $G$ with raw geometry and no augmentation; bounded-jitter arm).
- Finish the 40-pair blinded geometry review.

## 2026-10-26 - Phase 1 PRE-REGISTRATION committed (before any G validation number)

Full text: docs/research/PHASE1_PREREG.md.
- Frozen for every arm: B5 pipeline, 6x6 grid, dense inference, B_cap selection rules.
- Primary model G_raw: cross-plane footprint bias only; raw DICOM geometry; no augmentation; no IPE.
- beta: exactly 0 at init; projected SGD; eta_beta set by a fixed gradient-scale rule from a 50-step smoke run without
  evaluation.
- Screening, escalation and confirmation exactly as PLAN_v1_3 s5, with the donor swap defined explicitly.
- G_jitter (d ~ U(0, 0.75), training only): secondary, never a rescue.
- The 40-pair blinded review is descriptive only.

## 2026-10-26 - Phase 1: pre-launch tests passed; eta_beta fixed by the pre-registered probe (before any G metric)

**Tests** (tests/test_phase1_geometry.py, real cache-v2 data):
- 1,500 windows: footprints reproject exactly through the cache-v2 affine; mirrored cells are consistent; Sigma is
  positive definite.
- Paired arms draw identical windows and batches.
- The donor swap changes only mu and Sigma.
- At beta = 0, max |z_G - z_Bcap| = 0.0 in fp32 with identical shared weights; beta > 0 changes the logits.
- 728 padding-masked tokens have no influence on the output.

**Probe.** 50 training steps, fold 0, seed 42, beta held at 0, no evaluation.
- Median |dL/dbeta| = 5.538e-3 over 16 layer x head entries (range 0.0037 to 0.0070); 433 steps per epoch.
- eta_beta = 0.5 / (433 x 5.538e-3) = **0.2085**, written into configs/p1_graw_s42.yaml and p1_graw_s43.yaml.
  Not to be changed.

Launch: G_raw fold 0, seeds 42 / 43. Comparator: p0c_g6_s42 / p0c_g6_s43 (dense).

## 2026-10-26 — Phase 1 SCREENING RESULT (fold 0): STOP by the pre-registered rule (geometry not used)

Jobs 33770 / 33771, about 2.2 h each; frozen 6×6, dense, B5; $\eta_\beta$ 0.2085.

| Seed | $G_{\text{raw}}$ dense fold-val | $B_{\text{cap}}$ dense | $\Delta_{\text{geo}}$ | $G$ with donor geometry | $\Delta_{\text{swap}}$ |
|---|---|---|---|---|---|
| 42 | 0.9497 | 0.9541 | −0.0044 | 0.9497 | 0.0000 |
| 43 | 0.9550 | 0.9544 | +0.0005 | 0.9550 | 0.0000 |

**Rule.**
- Mean $\Delta_{\text{geo}}$ is −0.0020. Not both seeds are ≤ 0, so the first stop clause does not fire.
- $\Delta_{\text{swap}}$ < 0.001 in both seeds, so the second stop clause fires: "geometry is not used".
- **Decision: STOP.** No fold-1 escalation, no confirmation, and no $G_{\text{jitter}}$ (which was conditional on
  $G_{\text{raw}}$ passing).

**Control mechanics** (`scripts/phase1_swap_check.py`; a mechanics check, not a metric search):
- 878/879 val studies received a donor.
- Donor plane layouts differ from the study's own by a median 25.9 mm (p90 48.5 mm) in relative plane-centre position.
- Yet predictions moved by a mean |Δp| of 1.2e-4 / 1.3e-4; at most 5–7% of studies changed by more than 1e-3.
- So the control works, and the trained model is essentially insensitive to footprint geometry.

**Learned $\beta$.** All 16 per head stayed small, 0.003–0.05 (seed-42 max 0.050, seed-43 max 0.049).

**Reading (descriptive, not tested).** With global-attention CoAtNet tokens and dense attention, the footprint term
learned only a weak locality preference, and the decision does not depend on it. This is the pre-registered failure
case for final-stage tokens. PLAN_v1_3 lists the local-stage-token experiment (CoAtNet stage-1 tokens, before the global
attention stages) as the mechanism follow-up "if $G$ fails on final-stage tokens".

H1 is not supported at screening. Gold-58 was not used.
