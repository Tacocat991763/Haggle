import numpy as np


def rmsle(pred_hours, true_hours):
    """RMSLE in original hours space with predictions clipped to be nonnegative."""
    p = np.maximum(np.asarray(pred_hours, dtype=float), 0)
    t = np.asarray(true_hours, dtype=float)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(t)) ** 2)))


def rmse_log(pred_log, true_log):
    """RMSE in log1p space; equals RMSLE when pred_log >= 0."""
    p = np.maximum(np.asarray(pred_log, dtype=float), 0)
    return float(np.sqrt(np.mean((p - np.asarray(true_log, dtype=float)) ** 2)))
