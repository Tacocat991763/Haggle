"""Fixed splits: an untouched holdout plus 5-fold CV on the development portion."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = ROOT / "experiments" / "splits.json"
SEED = 42
HOLDOUT_FRAC = 0.15
N_FOLDS = 5
DROP = ["project_id", "hours_spent", "log_hours", "hours_bucket"]


def load_train():
    df = pd.read_csv(ROOT / "train.csv", dtype={"project_id": str})
    return df


def _strata(y_log):
    return pd.qcut(y_log, 10, labels=False, duplicates="drop")


def make_splits(df):
    """Create (once) and persist holdout ids and fold assignment keyed by project_id."""
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text())
    y = np.log1p(df.hours_spent.values)
    idx = np.arange(len(df))
    dev, hold = train_test_split(idx, test_size=HOLDOUT_FRAC, random_state=SEED, stratify=_strata(y))
    folds = {}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(skf.split(dev, _strata(y[dev]))):
        for i in dev[va]:
            folds[df.project_id.iloc[i]] = k
    splits = {"holdout": sorted(df.project_id.iloc[hold].tolist()), "folds": folds}
    SPLIT_FILE.write_text(json.dumps(splits))
    return splits


def get_data():
    """Return (dev_df, holdout_df, fold_array) with stable ordering."""
    df = load_train()
    s = make_splits(df)
    hold_mask = df.project_id.isin(set(s["holdout"]))
    dev = df[~hold_mask].reset_index(drop=True)
    hold = df[hold_mask].reset_index(drop=True)
    folds = dev.project_id.map(s["folds"]).astype(int).values
    return dev, hold, folds
