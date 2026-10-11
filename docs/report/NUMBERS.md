# Number sources for `main.tex` (Dec 5 report)

Every number in the skeleton, and where it comes from. "EXP" = `docs/EXPERIMENTS.md`; "MW" = `docs/METHOD_WRITEUP.md`.
Numbers that are **computed** rather than copied are marked and have a script. Nothing here is estimated by hand
except the two flagged rows.

## Task and data

| Number | Source |
|---|---|
| 12 findings; macro-AUC; GPU <= 9 h; ~1,300 test studies | `CLAUDE.md` s1 |
| 4,407 studies; 58 gold; ~82% report-vs-image agreement | `CLAUDE.md` s2 |
| 24,371 series; 5.53 mean per study (3-14); sagittal 9,864 / coronal 8,609 / axial 5,898 | `CLAUDE.md` s3 |
| Gold-58 positive rates (ACL .41 ... Effusion .60) | `CLAUDE.md` s3 |
| 3,681 report clusters, 5 folds | EXP 2026-09-20 results block |
| Bootstrap SD of a solo macro ~0.0041; bar +0.0082 | EXP ablation ledger, "Conventions" |

## Label engine

| Number | Source |
|---|---|
| `labels_v2_final` macro LOO 0.823 | EXP ledger LE3 |
| v3 chosen-blend LOO 0.904; v4 0.892 (-0.012) | EXP 2026-09-08 entry; ledger LE4, LE5 |
| v3 acceptance: OOF vs original labels 0.8631 vs 0.8489; gold-58 +0.0139 | EXP ledger LE4 |
| Labels alone: +0.0142 OOF (exp005 vs exp002) | EXP ledger B |

## Models and Table I

| Number | Source |
|---|---|
| OOF ladder: ViT-S 0.8631, ViT-B 0.8681, 3-member ensemble 0.8744 | EXP ledger B |
| exp005 gold-58 0.8903 (5-fold mean) and OOF 0.8631 | EXP 2026-10-01 re-gate result (`artifacts/blend_v5_gate_exp005.md`) |
| exp008 OOF 0.8708, gold-58 0.8882, LB 0.885 | EXP ledger B; MW s11 |
| exp018 OOF 0.8816, gold-58 0.9091, LB 0.929, ~53 min | EXP 2026-09-21 result; 2026-09-22 probe #3; 2026-09-26 profiling |
| exp021 OOF 0.8840, gold-58 0.9111 [0.8834, 0.9359], LB 0.929, ~100 min | EXP 2026-09-25 result and probe |
| exp018 2-fold gold-58 0.9078, LB 0.925, ~20.5 min | `README.md` results table; EXP 2026-09-26 probes #5/#6 |
| exp021 over exp018: +0.0024 [+0.0014, +0.0033]; gold-58 +0.0021 (CI spans zero) | EXP 2026-09-25 result |
| LB-minus-OOF gaps: ~+0.030 (0.8631 -> 0.893, corrected), +0.047 (exp018), exp008 +0.014 to +0.020 | EXP 2026-10-01 correction; 2026-09-22 |
| 0.893 most likely exp005 alone; 3 members ~59 s/study, ~21 h on T4 | EXP 2026-09-27 (profile), 2026-10-01 (inference, **not confirmed**) |
| Laterality: 99.4% of the 1,491 studies (34%) resolved; 99.0-99.1% agreement | MW s5a; `artifacts/exp018_cache_audit.md` |

## Table II (ablation ledger)

| Row | Source |
|---|---|
| 448 px, T=32, ACL init, aux-loss w 0.1 / 0.5 | EXP ledger C; 2026-09-20 results block (exp012 interval) |
| TTA -0.0014; per-label recipe +0.0010; 6-member ensemble +0.0020 [+0.0011, +0.0027], 2.9x cost | EXP ledger D |
| Dense windows -0.0001 [-0.0013, +0.0010] | EXP 2026-09-26 dense-window entry |
| Seed 43 alone -0.0011 [-0.0031, +0.0008]; 10-checkpoint +0.0024 [+0.0014, +0.0033] | EXP 2026-09-25 |
| DINOv2-B window model: OOF 0.8709 vs 0.8816, so -0.0107 | EXP 2026-09-26 exp022. **The -0.0107 is computed here (0.8709 - 0.8816); no interval was recorded** |
| EMA 0.999 vs top-3 average: -0.0009 gold-58 (fold 0 only) | EXP 2026-09-24 exp020. **Fold-0 gold-58, not OOF** |
| + DINOv2-B window model w 0.5: -0.0032 [-0.0045, -0.0019] | EXP 2026-09-26 exp022 |
| + 3-member slot ensemble alpha 0.2: +0.0024 [+0.0017, +0.0031]; ~21 h | EXP 2026-09-26 gate result; 2026-09-27 runtime check |
| + exp005 alpha 0.1: +0.0009 [+0.0005, +0.0012]; bar +0.0020 | EXP 2026-10-01 re-gate result |
| DINOv3-S -0.0252; ConvNeXt-V2 corr 0.9296; 3D ResNet killed | EXP ledger B |

## Per-label table, efficiency, attention

| Number | Source |
|---|---|
| Per-label AUC (OOF and gold-58, exp018 and exp021) | **computed**: `scripts/report_tables.py` -> `docs/report/tables/per_label_auc.tex` (asserts the four macro values) |
| Efficiency: -1.945 (exp008, projected 11.6 min), -1.966 (5-fold), -2.018 (2-fold) | `artifacts/exp018_efficiency_table.md` (exp008); computed `AUC/(0.5-0.95)+t/32400` for the other two; EXP 2026-09-26 |
| 5-fold ~4.5 s/study on one T4; channels_last -3.7%; 4th decode thread no gain | EXP 2026-09-26 efficiency entries |
| Attention shares (62% / 45% / 57%), effective windows (16-40 of ~52), seed agreement 0.85-0.95 | **computed**: `docs/figures/attention_slot_mass_gold58.csv` (`scripts/fig_attention.py`; extraction `scripts/viz_attention.py`, job 30522) |
| Top-4 windows hold 22-48% of attention (six selected studies) | printed on `attention_windows_gold58.png` |

## Known soft spots to resolve before submission

- `refs.bib`: the Qiu et al. 2024 entry was checked against Crossref on 2026-10-11 (DOI 10.1038/s41467-024-51888-4,
  volume 15, article 7637, nine authors). The other entries were written from memory and have been read through by a
  reviewer, but check each against the source before submission.
- The 0.893 -> exp005 attribution is an inference; keep the wording "most likely".
- The attention-extraction predictions reproduce exp021's stored gold-58 predictions for 56 of 58 studies to < 0.005;
  two studies differ (max 0.21, on MCL/ACL). The likely cause is the deployed side-resolution rule (tag/geometry)
  versus the report-validated side used in training; not confirmed. Macro AUC from the extraction is 0.9104 vs 0.9111.
