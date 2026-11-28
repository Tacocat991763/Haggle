"""Blending of OOF predictions (log1p space) with honest, nested evaluation."""
import numpy as np
from scipy.optimize import minimize

from .metrics import rmse_log
from .runner import OOF_DIR, data


def load(ids):
    oof = np.column_stack([np.load(OOF_DIR / f"{i}.npy") for i in ids])
    hold = np.column_stack([np.load(OOF_DIR / f"{i}_hold.npy") for i in ids])
    return oof, hold


def fit_weights(P, y, l2=0.0):
    """Nonnegative weights summing to one minimizing RMSE in log space (optional shrink toward equal)."""
    k = P.shape[1]
    eq = np.full(k, 1 / k)

    def f(w):
        return np.mean((P @ w - y) ** 2) + l2 * np.sum((w - eq) ** 2)

    r = minimize(f, eq, method="SLSQP", bounds=[(0, 1)] * k,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return r.x


def nested_cv(P, y, folds, fitter):
    """Fit the combiner on 4 folds of OOF preds, score on the 5th. Returns (rmse, oof_blend)."""
    out = np.zeros(len(y))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        comb = fitter(P[tr], y[tr])
        out[va] = comb(P[va])
    return rmse_log(out, y), out


def weights_fitter(l2=0.0):
    def fitter(P, y):
        w = fit_weights(P, y, l2)
        return lambda Q: Q @ w
    return fitter


def ridge_fitter(alpha=1.0):
    from sklearn.linear_model import Ridge

    def fitter(P, y):
        m = Ridge(alpha=alpha, positive=True).fit(P, y)
        return m.predict
    return fitter


def report(ids, verbose=True):
    dev, hold, folds = data()
    y = np.log1p(dev.hours_spent.values)
    P, H = load(ids)
    singles = {i: rmse_log(P[:, j], y) for j, i in enumerate(ids)}
    eq = rmse_log(P.mean(1), y)
    nw, _ = nested_cv(P, y, folds, weights_fitter())
    w = fit_weights(P, y)
    res = dict(singles=singles, equal=eq, nested_weights=nw, weights=dict(zip(ids, np.round(w, 3))))
    if verbose:
        for i, s in singles.items():
            print(f"  {i:36s} {s:.5f}  w={res['weights'][i]}")
        print(f"  equal-weight {eq:.5f} | nested optimized weights {nw:.5f}")
    return res
