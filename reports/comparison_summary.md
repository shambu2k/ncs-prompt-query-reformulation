# Final Comparison Summary

## Validation Split

| Method | Clean MRR | Adv MRR | Robustness Drop |
| --- | --- | --- | --- |
| Raw Query BM25 | 0.1654 | 0.0518 | 0.1136 |
| Fixed Prompt Rewrite | 0.1878 | 0.0474 | 0.1404 |
| Evolved Prompt Rewrite | 0.1878 | 0.0474 | 0.1404 |

## Test Split

| Method | Clean MRR | Adv MRR | Robustness Drop |
| --- | --- | --- | --- |
| Raw Query BM25 | 0.1338 | 0.0421 | 0.0917 |
| Fixed Prompt Rewrite | 0.1537 | 0.0382 | 0.1155 |
| Evolved Prompt Rewrite | 0.1537 | 0.0382 | 0.1155 |

## Evolution Progress

| Iteration | Prompt ID | Dev MRR | Accepted |
| --- | --- | --- | --- |
| 0 | evolve_v1_seed | 0.0556 | yes |
| 1 | evolve_v1_iter001_20260420T153837 | 0.0352 | no |
| 2 | evolve_v1_iter002_20260420T153909 | 0.0357 | no |
| 3 | evolve_v1_iter003_20260420T153942 | 0.0389 | no |
| 4 | evolve_v1_iter004_20260420T154015 | 0.0366 | no |
| 5 | evolve_v1_iter005_20260420T154048 | 0.0421 | no |
