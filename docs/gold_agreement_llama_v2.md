# Gold-58 label agreement

## llama-3.3-70b-instruct-awq

Macro ROC-AUC vs gold-58: **0.870** (macro Cohen's kappa: 0.601)

| Target | n | Gold pos rate | Pred pos rate | Accuracy | F1 | Kappa | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Synovitis | 58/58 | 0.47 | 0.31 | 0.707 | 0.622 | 0.398 | 0.766 |
| Contusion | 58/58 | 0.33 | 0.59 | 0.707 | 0.679 | 0.447 | 0.816 |
| Effusion | 58/58 | 0.60 | 0.78 | 0.793 | 0.850 | 0.533 | 0.822 |
| Lateral OA | 58/58 | 0.19 | 0.29 | 0.793 | 0.571 | 0.443 | 0.827 |
| PF OA | 58/58 | 0.36 | 0.36 | 0.828 | 0.762 | 0.627 | 0.832 |
| Fracture | 58/58 | 0.31 | 0.26 | 0.810 | 0.667 | 0.536 | 0.838 |
| Lateral Meniscus | 58/58 | 0.40 | 0.36 | 0.862 | 0.818 | 0.707 | 0.871 |
| Baker's | 58/58 | 0.21 | 0.33 | 0.845 | 0.710 | 0.611 | 0.884 |
| Medial OA | 58/58 | 0.26 | 0.38 | 0.810 | 0.703 | 0.571 | 0.886 |
| Medial Meniscus | 58/58 | 0.45 | 0.47 | 0.879 | 0.868 | 0.757 | 0.949 |
| MCL | 58/58 | 0.16 | 0.19 | 0.931 | 0.800 | 0.759 | 0.969 |
| ACL | 58/58 | 0.41 | 0.50 | 0.914 | 0.906 | 0.828 | 0.982 |

**Below the 0.80 gold-agreement exit criterion (CLAUDE.md §6 step 1.5): Synovitis.** Expected candidates: Synovitis, Contusion (reports under-specify these) — anything else here needs a rubric fix.
