# Results on `benchmark\unimer_hwe`

| model | n | exact_% | within1_% | within2_% | similarity | token_error_rate | mean_s_per_image | median_s_per_image | load_s | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| texo | 100 | 57.0 | 75.0 | 81.0 | 0.906 | 0.109 | 0.048 | 0.045 | 0.8 | 0 |
| texteller | 100 | 94.0 | 99.0 | 99.0 | 0.987 | 0.013 | 1.614 | 1.466 | 5.2 | 0 |
| qwen3.5-mathparser | 100 | 71.0 | 83.0 | 90.0 | 0.949 | 0.057 | 9.366 | 8.806 | 4.3 | 0 |
| qwen3-vl | 100 | 65.0 | 76.0 | 84.0 | 0.939 | 0.069 | 8.64 | 7.78 | 3.8 | 0 |

exact / within1 / within2: % of formulas with 0 / <=1 / <=2 token edits after normalization (CROHME ExpRate). similarity: 1 - edit distance / length (1 = perfect). Times are per image, excluding model loading.
