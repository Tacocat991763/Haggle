"""Resumable Optuna tuning on the fixed dev folds. Usage: python experiments/tune.py <model> <n_trials> [families...]

Progress is stored in experiments/optuna.db after every trial. The best params are re-run
through src.runner so they appear in results.jsonl/leaderboard with OOF predictions.
"""
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import optuna

from src.features import FeatureBuilder
from src.metrics import rmse_log
from src.models import MODELS
from src.runner import data, run

model = sys.argv[1]
n_trials = int(sys.argv[2])
families = tuple(sys.argv[3:]) if len(sys.argv) > 3 else ()
fam_tag = "+".join(families) or "raw"

dev, hold, folds = data()
y = np.log1p(dev.hours_spent.values)
cache = []
for k in range(5):
    tr, va = folds != k, folds == k
    fb = FeatureBuilder(families).fit(dev[tr])
    cache.append((fb.transform(dev[tr]), fb.transform(dev[va]), tr, va, fb.cat_cols))


def space(t):
    if model == "xgb":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 2500, log=True),
                    learning_rate=t.suggest_float("learning_rate", 0.005, 0.08, log=True),
                    max_depth=t.suggest_int("max_depth", 2, 8),
                    min_child_weight=t.suggest_float("min_child_weight", 1, 60, log=True),
                    subsample=t.suggest_float("subsample", 0.4, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 0.01, 50, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    gamma=t.suggest_float("gamma", 1e-4, 1.0, log=True))
    if model == "lgb":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 2500, log=True),
                    learning_rate=t.suggest_float("learning_rate", 0.005, 0.08, log=True),
                    num_leaves=t.suggest_int("num_leaves", 4, 64, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 5, 120, log=True),
                    subsample=t.suggest_float("subsample", 0.4, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 0.01, 50, log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
                    cat_smooth=t.suggest_float("cat_smooth", 1, 100, log=True),
                    min_data_per_group=t.suggest_int("min_data_per_group", 5, 200, log=True))
    if model == "cat":
        return dict(iterations=t.suggest_int("iterations", 300, 3000, log=True),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    depth=t.suggest_int("depth", 3, 8),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 50, log=True),
                    random_strength=t.suggest_float("random_strength", 0.1, 10, log=True),
                    bagging_temperature=t.suggest_float("bagging_temperature", 0, 2),
                    rsm=t.suggest_float("rsm", 0.2, 1.0),
                    one_hot_max_size=t.suggest_categorical("one_hot_max_size", [2, 10]))
    if model == "et":
        return dict(n_estimators=400, min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 30, log=True),
                    max_features=t.suggest_float("max_features", 0.1, 1.0))
    if model == "mlp":
        return dict(hidden=t.suggest_categorical("hidden", [32, 64, 128, 256]), layers=t.suggest_int("layers", 1, 3),
                    dropout=t.suggest_float("dropout", 0.0, 0.5), lr=t.suggest_float("lr", 1e-4, 1e-2, log=True),
                    wd=t.suggest_float("wd", 1e-5, 1e-1, log=True), epochs=t.suggest_int("epochs", 20, 150, log=True),
                    batch=t.suggest_categorical("batch", [64, 128, 256]))
    if model == "ridge":
        return dict(alpha=t.suggest_float("alpha", 0.1, 1000, log=True))
    raise ValueError(model)


def objective(t):
    p = space(t)
    oof = np.zeros(len(y))
    t0 = time.time()
    for k, (Xtr, Xva, tr, va, cats) in enumerate(cache):
        oof[va] = MODELS[model](p, cats, seed=0).fit(Xtr, y[tr]).predict(Xva)
        if k == 1:  # prune hopeless configs early
            part = rmse_log(oof[folds <= 1], y[folds <= 1])
            t.report(part, 1)
            if t.should_prune():
                raise optuna.TrialPruned()
    t.set_user_attr("seconds", time.time() - t0)
    return rmse_log(oof, y)


name = f"{model}_{fam_tag}"
study = optuna.create_study(study_name=name, storage="sqlite:///experiments/optuna.db", load_if_exists=True,
                            direction="minimize", sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
                            pruner=optuna.pruners.PercentilePruner(60, n_startup_trials=10))
optuna.logging.set_verbosity(optuna.logging.WARNING)
left = n_trials - len([t for t in study.trials if t.state.is_finished()])
if left > 0:
    study.optimize(objective, n_trials=left, callbacks=[lambda s, t: print(
        f"trial {t.number} {t.value if t.value else 'pruned'} best={s.best_value:.5f}", flush=True)])
print("best", study.best_value, study.best_params)
run(f"{name}_tuned", model, study.best_params, families=families, seeds=(0, 1, 2),
    note=f"optuna best of {len(study.trials)} trials")
