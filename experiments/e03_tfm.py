import sys, os; sys.path.insert(0, ".")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
from src.runner import run
which = sys.argv[1:] or ["tabicl", "tabdpt"]
for m in which:
    run(f"{m}_log", m, {}, families=("log",), keep_raw_skewed=False, note="log feats, raw skewed dropped")
    run(f"{m}_logratio", m, {}, families=("log", "ratio"), keep_raw_skewed=False, note="log+ratio")
    run(f"{m}_log_note", m, {"te": False}, families=("log",), keep_raw_skewed=False, note="no target enc")
