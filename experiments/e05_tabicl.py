import sys, os; sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from src.runner import run
LR = dict(families=("log", "ratio"), keep_raw_skewed=False)
run("tabicl_logratio_note", "tabicl", {"te": False}, note="log+ratio, no TE", **LR)
run("tabicl_logratio_n4", "tabicl", {"n_estimators": 4}, note="log+ratio, 4 estimators (speed)", **LR)
run("tabdpt_logratio_note", "tabdpt", {"te": False}, note="log+ratio, no TE", **LR)
