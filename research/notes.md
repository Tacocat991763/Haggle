# Research log

Setup:
- Dev set: 3,825 rows, run as 5-fold stratified CV with seed 42.
- Holdout: 675 rows (15%), untouched, stratified by target decile.
- Splits are persisted in `experiments/splits.json`.
- All scores are RMSLE, computed as RMSE of log1p with nonnegative clipping.
- Holdout predictions are the average of the five fold models. Holdout is checked only at milestones and never used for selection.

## Hardware and environment
- Apple M4 (10 cores), 16 GB. Python 3.12 venv.
- torch-based models (TabICL, TabDPT, TabPFN, MLP) must run in a different process from LightGBM/XGBoost. On macOS a libomp conflict makes them deadlock at 0% CPU.

## Findings so far

1. **Feature engineering barely matters for GBDTs** (`abl_*` runs, 3 seeds).
   - All family effects are within ±0.003.
   - For XGB, ratio and interaction features help a little; activity features hurt a little. The best set is all families minus activity (0.6518 against 0.6565 raw).
   - For CatBoost, raw features are best (0.6577). Every engineered family is neutral or slightly worse.
   - Conclusion: line counts are coarsened to multiples of 100 and labels are self-reported, so most hand-crafted ratios add noise.
2. **Tabular foundation models are the strongest single learners and add diversity.**

   | Model | CV RMSLE | CV time |
   |---|---|---|
   | TabICL v2 (log + ratio features) | 0.6503 | ~125 s/fold |
   | TabDPT 1.3 (log features) | 0.6508 | ~45 s/fold |
   | TabPFN v2 (fold 0 only) | 0.6525 | ~340 s/fold |

   - TabPFN v2 is dropped as too slow and weaker.
   - TabDPT does not improve with 16 ensembles, 3 seeds, a robust normalizer, missing indicators or dropping the ordinal codes. Quantile-normal normalization hurts (0.662).
3. **Tuning (Optuna TPE, 80 trials):**
   - XGB goes from 0.6518 to 0.6466 (3 seeds). The best config is deep (7), slow learning rate (0.0059), 1,393 trees and strong row subsampling (0.49).
   - LGB tuned: 0.6501. It gets **zero blend weight** next to XGB.
4. **Ridge, ExtraTrees and MLP** are 0.67–0.73 and get zero blend weight.
5. **Residuals** (`research/residuals.md`):
   - Predictions are calibrated per predicted-hours bin (|bias| < 0.06). Calibration slope is 1.02.
   - The bias by *actual* hours is regression to the mean from label noise, not a fixable model bias.
   - Error grows with project size and activity.
   - Category, language and framework biases are small (the largest, react-native at -0.33, has n=30).
   - No correction was applied.

## Milestone 1: XGB-tuned + TabDPT + TabICL blend

| | CV (nested) | Holdout |
|---|---|---|
| CatBoost raw (≈ previous submission) | 0.6577 | 0.6993 |
| Tuned XGB alone | 0.6466 | 0.6920 |
| Blend xgb/tabdpt/tabicl, weights 0.54/0.22/0.24 | **0.6427** | **0.6874** |

The improvement holds on the holdout (-0.012).

Holdout RMSLE runs about 0.04 above CV for every model. The holdout's target distribution makes it harder, which shows how much a 675-row sample can swing. That is consistent with the leaderboard (0.7548) sitting well above the previous model's validation score.

## Additional experiments after milestone 1

| Experiment | CV | Verdict |
|---|---|---|
| LightGBM, Optuna 60 trials | 0.6501 | Zero blend weight next to XGB |
| CatBoost, Optuna 28 trials (stopped at a plateau) | 0.6575 | Not competitive |
| MLP, Optuna 40 trials | 0.6684 | Blend weight about 0.03, dropped |
| TabICL log + ratio, no target encoding | 0.6517 | Worse than with target encoding |
| **TabICL log + ratio, 4 estimators** | 0.6514 | About 5× faster than 8 estimators, nearly the same score. **Selected** |
| TabDPT, 16 ensembles / 3 seeds / robust normalizer / missing indicators / no codes | 0.6508–0.6523 | No gain over the default |

### Ensemble structure (Phase 6), nested CV on members xgb_tuned, tabdpt_log, tabicl_logratio_n4

| Combiner | Nested CV |
|---|---|
| Equal weights | 0.6435 |
| **Nonnegative weights summing to 1** | **0.6427** |
| Ridge stacking (positive) | 0.6430 |
| Weights by prediction-size tercile | 0.6432 |
| Weights per project category | 0.6430 |
| Weights + linear calibration | 0.6430 |

The adaptive and stacked variants are rejected: none beats fixed weights. This matches the residual analysis, which found no systematic segment bias. TabICL adds only about 0.001 (CV and holdout both improve: 0.6879 to 0.6869), so it is the last and optional member.

## Final submission
- Members and weights: XGB tuned (5 seeds) 0.556, TabDPT 0.314, TabICL (4 estimators, batch_size 1) 0.129.
- Clip to [0.1, 140.7].
- XGB is retrained on all 4,500 rows. The foundation models use all 4,500 rows as context.
- Expected CV 0.6427 and holdout 0.6869, against 0.6577 and 0.6993 for the previous submission.
- Runtime on M4, offline sandbox: 69 s for 1,500 test rows, 256 s for 4,500.
- Peak memory: about 6 GB from TabICL, which sizes its batches to the memory available. TabDPT uses 0.85 GB.
- A 480 s budget with per-member kill deadlines guarantees a valid `predictions.csv` under 10 minutes. Members are dropped if the grader is slow.

## Ideas not yet tried
- More TabICL/TabDPT feature representations (for example raw counts with a quantile transform, or fewer columns).
- Repeated-CV Optuna for XGB, to reduce selection noise.
- Training on the unrounded `exp(log_hours)` target.
