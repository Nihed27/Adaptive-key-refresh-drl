# Results (50 unseen test episodes, 1000 steps each)

Total cost = −return (lower is better). ± = 95% confidence interval.

| Scenario | Policy | Total cost ↓ | Avg. link compromise ↓ | Refreshes | Network lifetime | Estimation MAE |
|---|---|---|---|---|---|---|
| Honest monitoring | Periodic (T=20) | 103.8 ± 5.3 | 0.054 | 50.0 | 1000 | 0.0077 |
| Honest monitoring | Threshold (θ=0.020) | 67.3 ± 4.1 | 0.026 | 41.2 | 1000 | 0.0070 |
| Honest monitoring | DQN (ours) | 75.3 ± 4.8 | 0.044 | 30.9 | 1000 | 0.0072 |
| Byzantine + mean agg. | Periodic (T=20) | 104.5 ± 4.6 | 0.054 | 50.0 | 1000 | 0.0309 |
| Byzantine + mean agg. | Threshold (θ=0.120) | 109.8 ± 6.3 | 0.089 | 21.0 | 1000 | 0.0348 |
| Byzantine + mean agg. | DQN (ours) | 83.7 ± 5.0 | 0.044 | 40.1 | 1000 | 0.0324 |
| Byzantine + trust agg. | Periodic (T=20) | 104.5 ± 4.6 | 0.054 | 50.0 | 1000 | 0.0079 |
| Byzantine + trust agg. | Threshold (θ=0.040) | 93.4 ± 4.4 | 0.057 | 36.5 | 1000 | 0.0076 |
| Byzantine + trust agg. | DQN (ours) | 73.1 ± 3.9 | 0.033 | 40.1 | 1000 | 0.0075 |

## Event-driven vs periodic reporting (Byzantine + trust, threshold policy)

| Reporting | Messages / episode | Energy used | Total cost |
|---|---|---|---|
| periodic | 500,000 | 0.366 | 108.4 |
| event | 3,842 | 0.267 | 98.5 |

Event-driven reporting sends **99.2% fewer messages** for the same information.
