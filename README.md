# Project Time Predictor (Haggle)

This model predicts how many hours someone spent on a coding project. It works from repository metadata: commit history, code size, languages, frameworks, assets, README and hosting.

## Submission

```bash
pip install -r requirements.txt   # torch (CPU), tabdpt, tabicl; the GBDT libraries are preinstalled
python predict.py                 # reads train.csv + test.csv, writes predictions.csv
```

`predict.py` is self-contained, CPU-only and works fully offline. It blends three models in `log1p(hours)` space:

| Member | Weight | What it is |
|---|---|---|
| XGBoost (5 seeds, Optuna-tuned) | 0.556 | Pickled in `models/final/`. Retrained from `train.csv` automatically if library versions differ. |
| TabDPT 1.3 | 0.314 | Tabular foundation model, in-context on `train.csv`. Weights are bundled under `models/tabdpt/`. |
| TabICL v2 regressor (4 estimators) | 0.129 | Tabular foundation model, in-context on `train.csv`. Weights are bundled under `models/tabicl/`. |

Predictions are clipped to the training label range [0.1, 140.7] hours.

The weights are stored as `.partNN` files under 100 MB each, because of GitHub's file-size limit. They are reassembled on first run.

**Time safety.** The evaluation limit is 10 minutes, and `predict.py` uses a 480-second budget.
- Members run cheapest first.
- `predictions.csv` is rewritten after each member finishes, so a valid file exists within seconds.
- Each member runs in its own process with a deadline, and is killed if it would overrun.
- The blend is renormalized over the members that finished.

**Benchmarks** (Apple M4, CPU, network blocked, empty HuggingFace cache):

| Test rows | Runtime |
|---|---|
| 50 | 56 s |
| 1,500 | 69 s |
| 4,500 | about 4 min |

## Results

5-fold CV on a 3,825-row development split, plus a separate untouched 675-row holdout:

| Model | CV RMSLE | Holdout RMSLE |
|---|---|---|
| Previous submission (CatBoost, raw features) | 0.6577 | 0.6993 |
| Tuned XGBoost | 0.6466 | 0.6920 |
| **Final blend** | **0.6427** | **0.6869** |

The previous submission scored 0.7548 on the public leaderboard.

Write-ups:
- Full analysis: [research/dataset_analysis.md](research/dataset_analysis.md)
- Experiment log: [research/notes.md](research/notes.md)
- Every experiment ranked: [research/leaderboard.md](research/leaderboard.md)
- Error analysis: [research/residuals.md](research/residuals.md)

## Layout

```
predict.py            submission entry point
train.py              refits the pickled members on all of train.csv from models/final/config.json
src/features.py       FeatureBuilder: feature families, fitted on the training portion only
src/models.py         model wrappers (CatBoost, LightGBM, XGBoost, Ridge, ExtraTrees, MLP, TabPFN, TabICL, TabDPT)
src/validation.py     fixed holdout + 5-fold splits (experiments/splits.json)
src/runner.py         experiment runner: OOF/holdout predictions, results.jsonl, leaderboard
src/ensemble.py       nonnegative weight fitting, stacking, nested-CV blend evaluation
src/pipeline.py       train/predict pipeline with process isolation and timeouts
src/metrics.py        RMSLE
experiments/          experiment scripts, Optuna tuner (tune.py), results.jsonl, residual analysis
research/             findings and leaderboard
models/final/         ensemble config and pickled XGBoost
legacy/               previous CatBoost submission (rollback: python legacy/predict_catboost_v1.py)
```

## Reproducing

```bash
python experiments/e01_baselines.py        # model-family baselines
python experiments/e02_ablation.py         # feature-family ablations
python experiments/tune.py xgb 80 log missing ratio rel interact
python experiments/e03_tfm.py tabicl tabdpt
python experiments/make_config.py "xgb_log+missing+ratio+rel+interact_tuned:0.556:5" tabdpt_log:0.314 tabicl_logratio_n4:0.129
python train.py
```

## Licenses and attribution

- TabDPT weights: Layer 6 AI, Apache-2.0.
- TabICL: Soda team at Inria, BSD-3-Clause.

See the `LICENSE` and `NOTICE` files in `models/tabdpt/` and `models/tabicl/`. TabPFN v2 was evaluated during research but is not part of the submission.
