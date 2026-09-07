# Label blend v2 FINAL — per-label strategy selection by strict LOO on gold-58

Macro LOO AUC of the CHOSEN per-label strategy: **0.823**

| Target | Chosen | Source weights | Qwen-only LOO AUC | Llama-only LOO AUC | Blend LOO AUC | Mean confidence |
|---|---|---|---|---|---|---|
| ACL | **blend** | qwen=0.5, llama=0.5 (agreement-weighted confidence) | 0.950 | 0.963 | 0.978 | 0.711 |
| MCL | **blend** | qwen=0.5, llama=0.5 (agreement-weighted confidence) | 0.807 | 0.839 | 0.943 | 0.723 |
| Medial Meniscus | **blend** | qwen=0.5, llama=0.5 (agreement-weighted confidence) | 0.888 | 0.877 | 0.892 | 0.925 |
| Lateral Meniscus | **qwen_only** | qwen=1.0, llama=0.0 | 0.812 | 0.789 | 0.801 | 1.000 |
| Medial OA | **blend** | qwen=0.5, llama=0.5 (agreement-weighted confidence) | 0.856 | 0.853 | 0.865 | 0.885 |
| Lateral OA | **blend** | qwen=0.5, llama=0.5 (agreement-weighted confidence) | 0.718 | 0.721 | 0.727 | 0.868 |
| PF OA | **blend** | qwen=0.5, llama=0.5 (agreement-weighted confidence) | 0.746 | 0.755 | 0.788 | 0.805 |
| Effusion | **blend** | qwen=0.5, llama=0.5 (agreement-weighted confidence) | 0.801 | 0.749 | 0.815 | 0.809 |
| Synovitis | **llama_only** | qwen=0.0, llama=1.0 | 0.341 | 0.711 | 0.630 | 1.000 |
| Baker's | **llama_only** | qwen=0.0, llama=1.0 | 0.857 | 0.862 | 0.855 | 1.000 |
| Contusion | **blend** | qwen=0.5, llama=0.5 (agreement-weighted confidence) | 0.588 | 0.715 | 0.731 | 0.732 |
| Fracture | **qwen_only** | qwen=1.0, llama=0.0 | 0.747 | 0.650 | 0.747 | 1.000 |

4407 studies. All LOO numbers are strict leave-one-out (each held-out gold study's prediction comes from calibrator(s) fit on the OTHER 57 only) — this is what decided which strategy is deployed per label, not a post-hoc rationalization.
