"""Haggle submission: reads train.csv + test.csv next to this file, writes predictions.csv.

Ensemble of tuned XGBoost (pickled, retrained from train.csv if library versions differ) and the
in-context tabular foundation models TabDPT and TabICL (bundled offline weights), blended in
log1p space with CV-fitted weights. CPU only, no network.

Time safety (evaluation limit is 10 minutes):
  * members run cheapest first; predictions.csv is (re)written after every finished member,
    so a valid file exists as soon as the first member (XGBoost, a few seconds) is done;
  * every member runs in its own process with a deadline derived from TIME_BUDGET and is
    killed if it would overrun; the blend is renormalised over the members that finished.
"""
import os
import sys
import time
from pathlib import Path

T0 = time.time()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pandas as pd

from src.pipeline import CLIP_HI, CLIP_LO, FINAL, blend, ensure_weights, load_config, predict_member, run_isolated

TIME_BUDGET = float(os.environ.get("HAGGLE_TIME_BUDGET", 480))  # seconds; hard limit is 600
FALLBACK_HOURS = 6.0  # training median, used only if every member fails


def write(test, hours):
    hours = np.nan_to_num(np.asarray(hours, dtype=float), nan=FALLBACK_HOURS, posinf=CLIP_HI, neginf=CLIP_LO)
    out = pd.DataFrame({"project_id": test["project_id"], "hours_spent": hours.clip(CLIP_LO, CLIP_HI)})
    tmp = ROOT / "predictions.csv.tmp"
    out.to_csv(tmp, index=False)
    tmp.replace(ROOT / "predictions.csv")


def main():
    train_path, test_path = ROOT / "train.csv", ROOT / "test.csv"
    test = pd.read_csv(test_path, dtype={"project_id": str})
    write(test, np.full(len(test), FALLBACK_HOURS))  # never leave the run without a valid file
    ensure_weights()
    cfg = load_config()
    preds, weights = [], []
    for m in cfg["members"]:  # config lists members cheapest first
        remaining = TIME_BUDGET - (time.time() - T0)
        if remaining < 15:
            print(f"skip {m['name']}: time budget exhausted", flush=True)
            continue
        t = time.time()
        try:
            p = np.asarray(run_isolated(predict_member, m, str(train_path), str(test_path), str(FINAL),
                                        timeout=remaining), dtype=float)
        except Exception as e:
            print(f"skip {m['name']}: {e}", flush=True)
            continue
        if p.shape != (len(test),) or not np.all(np.isfinite(p)):
            print(f"skip {m['name']}: invalid output", flush=True)
            continue
        preds.append(p)
        weights.append(m["weight"])
        write(test, blend(preds, weights))
        print(f"{m['name']}: {time.time() - t:.1f}s (total {time.time() - T0:.1f}s)", flush=True)
    print(f"Generated {len(test)} predictions from {len(preds)} members in {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
