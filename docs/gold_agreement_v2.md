# Gold-58 label agreement

## qwen2.5-72b-instruct-awq_v2

Macro ROC-AUC vs gold-58: **0.846** (macro Cohen's kappa: 0.585)

| Target | n | Gold pos rate | Pred pos rate | Accuracy | F1 | Kappa | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Synovitis | 58/58 | 0.47 | 0.62 | 0.569 | 0.603 | 0.152 | 0.634 |
| Contusion | 58/58 | 0.33 | 0.47 | 0.655 | 0.565 | 0.294 | 0.718 |
| Lateral OA | 58/58 | 0.19 | 0.26 | 0.828 | 0.615 | 0.508 | 0.802 |
| PF OA | 58/58 | 0.36 | 0.26 | 0.862 | 0.778 | 0.682 | 0.808 |
| Lateral Meniscus | 58/58 | 0.40 | 0.33 | 0.862 | 0.810 | 0.703 | 0.864 |
| Fracture | 58/58 | 0.31 | 0.33 | 0.845 | 0.757 | 0.643 | 0.865 |
| Effusion | 58/58 | 0.60 | 0.67 | 0.793 | 0.838 | 0.554 | 0.866 |
| Baker's | 58/58 | 0.21 | 0.28 | 0.897 | 0.786 | 0.719 | 0.886 |
| Medial OA | 58/58 | 0.26 | 0.33 | 0.828 | 0.706 | 0.586 | 0.894 |
| MCL | 58/58 | 0.16 | 0.12 | 0.897 | 0.625 | 0.566 | 0.902 |
| Medial Meniscus | 58/58 | 0.45 | 0.45 | 0.897 | 0.885 | 0.791 | 0.933 |
| ACL | 58/58 | 0.41 | 0.40 | 0.914 | 0.894 | 0.821 | 0.979 |

**Below the 0.80 gold-agreement exit criterion (CLAUDE.md §6 step 1.5): Synovitis, Contusion.** Expected candidates: Synovitis, Contusion (reports under-specify these) — anything else here needs a rubric fix.
