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
