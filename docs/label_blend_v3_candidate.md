# Label blend v3 CANDIDATE — image (exp002 OOF) + report (labels_v2_final) blend, per-label weight chosen by strict LOO on gold-58

**NOT DEPLOYED — candidate for review, no retrain has happened on this yet.**

Macro LOO AUC: chosen blend **0.904** vs report-only 0.915 vs image-only 0.864

| Target | Direction | w_image | Report-only LOO | Image-only LOO | Chosen-blend LOO | Mean confidence |
|---|---|---|---|---|---|---|
| ACL | report-heavy | **0.1** | 0.980 | n/a (not in grid) | 0.982 | 0.703 |
| MCL | report-heavy | **0.3** | 0.850 | n/a (not in grid) | 0.959 | 0.607 |
| Medial Meniscus | image-heavy | **0.6** | n/a (not in grid) | 0.929 | 0.947 | 0.740 |
| Lateral Meniscus | image-heavy | **0.5** | n/a (not in grid) | 0.800 | 0.863 | 0.692 |
| Medial OA | image-heavy | **1.0** | n/a (not in grid) | 0.901 | 0.901 | 0.721 |
| Lateral OA | image-heavy | **0.8** | n/a (not in grid) | 0.787 | 0.812 | 0.720 |
| PF OA | image-heavy | **0.5** | n/a (not in grid) | 0.838 | 0.909 | 0.628 |
| Effusion | image-heavy | **1.0** | n/a (not in grid) | 0.948 | 0.948 | 0.678 |
| Synovitis | image-heavy | **0.6** | n/a (not in grid) | 0.789 | 0.809 | 0.600 |
| Baker's | image-heavy | **1.0** | n/a (not in grid) | 0.958 | 0.958 | 0.708 |
| Contusion | image-heavy | **0.9** | n/a (not in grid) | 0.877 | 0.895 | 0.696 |
| Fracture | image-heavy | **0.5** | n/a (not in grid) | 0.814 | 0.861 | 0.651 |

4407 studies. Strict LOO throughout (each held-out gold study's blend prediction comes from calibrators fit on the OTHER 57 only). Weight search was constrained to the pre-registered direction per label (image-heavy [0.5-1.0] for the 10 labels where exp002's gold58 AUC beat the label engine's own LOO AUC; report-heavy [0.0-0.5] for ACL/MCL) — LOO only picked the magnitude within that direction, not whether to flip it.

## Full weight-grid LOO AUC per label

| Target | w=0.0 | w=0.1 | w=0.2 | w=0.3 | w=0.4 | w=0.5 | w=0.6 | w=0.7 | w=0.8 | w=0.9 | w=1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ACL | 0.980 | 0.982 | 0.977 | 0.974 | 0.974 | 0.973 |  |  |  |  |  |
| MCL | 0.850 | 0.955 | 0.957 | 0.959 | 0.952 | 0.948 |  |  |  |  |  |
| Medial Meniscus |  |  |  |  |  | 0.945 | 0.947 | 0.945 | 0.944 | 0.942 | 0.929 |
| Lateral Meniscus |  |  |  |  |  | 0.863 | 0.858 | 0.856 | 0.845 | 0.815 | 0.800 |
| Medial OA |  |  |  |  |  | 0.899 | 0.899 | 0.898 | 0.896 | 0.899 | 0.901 |
| Lateral OA |  |  |  |  |  | 0.807 | 0.808 | 0.808 | 0.812 | 0.805 | 0.787 |
| PF OA |  |  |  |  |  | 0.909 | 0.905 | 0.898 | 0.883 | 0.875 | 0.838 |
| Effusion |  |  |  |  |  | 0.922 | 0.937 | 0.938 | 0.938 | 0.943 | 0.948 |
| Synovitis |  |  |  |  |  | 0.797 | 0.809 | 0.804 | 0.802 | 0.802 | 0.788 |
| Baker's |  |  |  |  |  | 0.902 | 0.913 | 0.948 | 0.955 | 0.955 | 0.958 |
| Contusion |  |  |  |  |  | 0.866 | 0.884 | 0.889 | 0.887 | 0.895 | 0.877 |
| Fracture |  |  |  |  |  | 0.861 | 0.853 | 0.847 | 0.836 | 0.832 | 0.814 |