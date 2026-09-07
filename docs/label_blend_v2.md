# Label blend v2 — Qwen + Llama, Platt-calibrated on gold-58

| Target | Qwen LOO AUC | Llama LOO AUC | Qwen LOO Brier | Llama LOO Brier | Inter-LLM correlation (all 4407) | Label agreement scale |
|---|---|---|---|---|---|---|
| ACL | 0.950 | 0.963 | 0.075 | 0.058 | 0.663 | 0.832 |
| MCL | 0.807 | 0.839 | 0.096 | 0.062 | 0.593 | 0.796 |
| Medial Meniscus | 0.888 | 0.877 | 0.097 | 0.106 | 0.943 | 0.971 |
| Lateral Meniscus | 0.812 | 0.789 | 0.127 | 0.128 | 0.913 | 0.956 |
| Medial OA | 0.856 | 0.853 | 0.121 | 0.122 | 0.888 | 0.944 |
| Lateral OA | 0.718 | 0.721 | 0.128 | 0.123 | 0.816 | 0.908 |
| PF OA | 0.746 | 0.755 | 0.124 | 0.144 | 0.834 | 0.917 |
| Effusion | 0.801 | 0.749 | 0.162 | 0.163 | 0.847 | 0.924 |
| Synovitis | 0.523 | 0.711 | 0.251 | 0.212 | 0.545 | 0.772 |
| Baker's | 0.857 | 0.862 | 0.086 | 0.093 | 0.842 | 0.921 |
| Contusion | 0.588 | 0.715 | 0.210 | 0.168 | 0.739 | 0.870 |
| Fracture | 0.747 | 0.650 | 0.135 | 0.157 | 0.861 | 0.930 |

4407 studies blended. Confidence weight floors: per-study >= 0.1, per-label scale >= 0.2 (a fully-disagreeing label still contributes some training signal, never fully zeroed).
