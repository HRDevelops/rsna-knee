# Gold-58 label agreement

## Qwen2.5-72B-Instruct-AWQ

Macro ROC-AUC vs gold-58: **0.851** (macro Cohen's kappa: 0.598)

| Target | n | Gold pos rate | Pred pos rate | Accuracy | F1 | Kappa | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Synovitis | 58/58 | 0.47 | 0.28 | 0.672 | 0.558 | 0.324 | 0.657 |
| Contusion | 58/58 | 0.33 | 0.50 | 0.724 | 0.667 | 0.448 | 0.732 |
| Lateral OA | 58/58 | 0.19 | 0.28 | 0.810 | 0.593 | 0.474 | 0.807 |
| PF OA | 58/58 | 0.36 | 0.26 | 0.862 | 0.778 | 0.682 | 0.829 |
| Effusion | 58/58 | 0.60 | 0.74 | 0.759 | 0.821 | 0.464 | 0.853 |
| Fracture | 58/58 | 0.31 | 0.29 | 0.845 | 0.743 | 0.632 | 0.855 |
| Lateral Meniscus | 58/58 | 0.40 | 0.33 | 0.862 | 0.810 | 0.703 | 0.863 |
| Medial OA | 58/58 | 0.26 | 0.34 | 0.810 | 0.686 | 0.554 | 0.883 |
| Baker's | 58/58 | 0.21 | 0.28 | 0.897 | 0.786 | 0.719 | 0.887 |
| Medial Meniscus | 58/58 | 0.45 | 0.47 | 0.879 | 0.868 | 0.757 | 0.930 |
| MCL | 58/58 | 0.16 | 0.12 | 0.897 | 0.625 | 0.566 | 0.938 |
| ACL | 58/58 | 0.41 | 0.41 | 0.931 | 0.917 | 0.858 | 0.983 |

**Below the 0.80 gold-agreement exit criterion (CLAUDE.md §6 step 1.5): Synovitis, Contusion.** Expected candidates: Synovitis, Contusion (reports under-specify these) — anything else here needs a rubric fix.
