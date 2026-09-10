# Label blend v4 CANDIDATE — labels_v3_candidate + exp005 OOF, round-2 noisy student

**NOT DEPLOYED — candidate for review, no retrain has happened on this yet.**

Macro LOO AUC: v4 chosen blend **0.892**

| Target | Direction | w_image | v3-only LOO | exp005-only LOO | Chosen-blend LOO | Mean confidence |
|---|---|---|---|---|---|---|
| ACL | report-heavy | **0.3** | 0.956 | n/a (not in grid) | 0.964 | 0.759 |
| MCL | report-heavy | **0.0** | 0.957 | n/a (not in grid) | 0.957 | 0.684 |
| Medial Meniscus | image-heavy | **0.5** | n/a (not in grid) | 0.940 | 0.946 | 0.882 |
| Lateral Meniscus | image-heavy | **0.5** | n/a (not in grid) | 0.764 | 0.847 | 0.752 |
| Medial OA | image-heavy | **1.0** | n/a (not in grid) | 0.916 | 0.916 | 0.906 |
| Lateral OA | image-heavy | **0.5** | n/a (not in grid) | 0.721 | 0.768 | 0.924 |
| PF OA | image-heavy | **0.5** | n/a (not in grid) | 0.797 | 0.898 | 0.767 |
| Effusion | image-heavy | **0.5** | n/a (not in grid) | 0.938 | 0.957 | 0.886 |
| Synovitis | image-heavy | **0.5** | n/a (not in grid) | 0.765 | 0.792 | 0.873 |
| Baker's | image-heavy | **0.5** | n/a (not in grid) | 0.886 | 0.900 | 0.856 |
| Contusion | image-heavy | **0.9** | n/a (not in grid) | 0.891 | 0.893 | 0.914 |
| Fracture | image-heavy | **0.7** | n/a (not in grid) | 0.854 | 0.865 | 0.813 |

4407 studies. Strict LOO throughout, same methodology and pre-registered per-label direction as `build_labels_v3_candidate.py`.
