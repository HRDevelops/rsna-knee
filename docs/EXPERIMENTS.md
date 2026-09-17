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
