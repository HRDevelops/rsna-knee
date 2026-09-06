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
