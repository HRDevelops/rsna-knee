# Laterality resolution — DICOM tag x report cross-check

Tag/report agreement on the 796 studies where BOTH a consistent tag side and an unambiguous report side exist: **784/796 (98.5%)**.

## Final side source breakdown

| Source | n studies |
|---|---|
| tag | 2178 |
| none_needs_classifier | 1491 |
| report | 712 |
| conflict_unresolved | 24 |
| tag_bilateral | 1 |
| conflict_resolved_by_report | 1 |

## Tag-conflict studies (25 total)

**Read this before treating these as resolved or as data errors.** A study-level L/R conflict across series is exactly what you'd see from a genuinely BILATERAL exam (both knees imaged in one study) — which breaks the whole one-side-per-study assumption the flip-right-knees pipeline (CLAUDE.md §5) is built on. Almost all of these have no report-stated side either (report=UNKNOWN), consistent with a bilateral study rather than a mislabel. **Recommendation: handle these per-SERIES, not per-study** — look up each series' own Laterality tag and flip/route it individually, rather than forcing one study-wide side decision. Don't silently pick a side and move on.

| study_uid | tag_values_seen | report_side | resolution |
|---|---|---|---|
| 1.2.826.0.1.3680043.8.498.11596665780617412984766822528862741584 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.11924999889245769866426422681253524920 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.12130868132537772155475981935868588761 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.12333996498811456012744466468462396081 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.13030662056306690969977140713179461363 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.13242536312535334908597493894251783308 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.22407175411826914385667795929161924645 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.30248682294786626375369374249874501600 | L,R | L | conflict_resolved_by_report |
| 1.2.826.0.1.3680043.8.498.38056024157303576597111203925091929489 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.38303182087206935163155565441234078389 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.40898220266193841898869905714214454387 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.42396216793489016293032290339618185275 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.46172451572836863687073454997158931662 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.52204390785696944340286127542242341202 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.57239450717434070615919278253974598533 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.58811528407321581206563412079643094538 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.62420632149863295140175740643640707423 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.66210455934011506415244876770725282769 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.70124572708965486791612654433965515063 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.73215265387578908341609746559916587894 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.78119223228067919465880137763593966456 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.83030371526667747898709613600825477966 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.88274753211076645976496220746740420894 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.88752039058074948027322975455607258941 | L,R | UNKNOWN | conflict_unresolved |
| 1.2.826.0.1.3680043.8.498.91125742927944703615970502025043944089 | L,R | UNKNOWN | conflict_unresolved |

## Studies with NO side source (1491) — need the trained classifier

These 1491 studies have neither a usable DICOM Laterality tag nor a report-extracted side. They depend entirely on the report-supervised side classifier (EXECUTION_PLAN Phase 2.3) once it's trained on the 2891 studies that already have a side label from tag/report.

## Bilateral-tagged studies (1) — special-case, do not laterality-normalize blindly

- 1.2.826.0.1.3680043.8.498.11504079342934513987748646905909982588
