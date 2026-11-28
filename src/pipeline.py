"""Train/predict pipeline for the final ensemble, shared by train.py and predict.py.

Each member is described in models/final/config.json:
  {"name", "model", "params", "families", "keep_raw_skewed", "seeds", "weight", "kind": "saved"|"icl"}
"saved" members (GBDTs) are trained by train.py and pickled with the library versions used;
predict.py reuses them only when the installed versions match, otherwise it retrains them from
train.csv with the same config (a few seconds). "icl" members (TabDPT/TabICL) are in-context
learners: they are fitted on train.csv at prediction time from the bundled offline weights.
Every member runs in its own spawned process (torch and the GBDT OpenMP runtimes can deadlock on macOS).
"""
import glob
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "models" / "final"
CLIP_LO, CLIP_HI = 0.1, 140.7
LIBS = ["xgboost", "lightgbm", "catboost", "sklearn", "numpy", "pandas"]


def lib_versions():
    out = {}
    for name in LIBS:
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def offline_env():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


def ensure_weights():
    """Reassemble weights stored as <100MB parts (GitHub limit) into models/<name>/<file>."""
    for parts_dir in [ROOT / "models" / "tabdpt", ROOT / "models" / "tabicl"]:
        for first in glob.glob(str(parts_dir / "*.part00")):
            target = Path(first[: -len(".part00")])
            if target.exists():
                continue
            tmp = Path(tempfile.mkstemp(dir=parts_dir)[1])
            with open(tmp, "wb") as out:
                for p in sorted(glob.glob(str(target) + ".part*")):
                    with open(p, "rb") as f:
                        shutil.copyfileobj(f, out)
            tmp.replace(target)


def _fit_member(member, train):
    from src.features import FeatureBuilder
    from src.models import MODELS
    fb = FeatureBuilder(member["families"], member.get("keep_raw_skewed", True)).fit(train)
    X = fb.transform(train)
    y = np.log1p(train["hours_spent"].values)
    models = [MODELS[member["model"]](member["params"], fb.cat_cols, seed=s).fit(X, y) for s in member["seeds"]]
    return fb, models


def train_member(member, train_path, out_dir):
    """Fit a saved member on all of train.csv and pickle it."""
    offline_env()
    train = pd.read_csv(train_path, dtype={"project_id": str})
    fb, models = _fit_member(member, train)
    with open(Path(out_dir) / f"{member['name']}.pkl", "wb") as f:
        pickle.dump({"fb": fb, "models": models, "versions": lib_versions()}, f)
    return member["name"]


def predict_member(member, train_path, test_path, model_dir):
    """Return log1p-hour predictions for test.csv from one member."""
    offline_env()
    if member["model"] in ("tabdpt", "tabicl", "tabpfn", "mlp"):
        import torch
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    test = pd.read_csv(test_path, dtype={"project_id": str})
    bundle = None
    pkl = Path(model_dir) / f"{member['name']}.pkl"
    if member.get("kind") == "saved" and pkl.exists():
        try:
            with open(pkl, "rb") as f:
                b = pickle.load(f)
            if b["versions"] == lib_versions():
                bundle = b
        except Exception:
            bundle = None
    if bundle is None:
        train = pd.read_csv(train_path, dtype={"project_id": str})
        fb, models = _fit_member(member, train)
    else:
        fb, models = bundle["fb"], bundle["models"]
    X = fb.transform(test)
    return np.mean([m.predict(X) for m in models], axis=0)


def _child(q, fn, args):
    try:
        q.put(("ok", fn(*args)))
    except BaseException as e:  # report any failure to the parent
        q.put(("err", repr(e)))


def run_isolated(fn, *args, timeout=None):
    """Run fn(*args) in a fresh spawned process. Returns its result, or raises TimeoutError/RuntimeError.

    On timeout the child is killed, so a slow member can never push predict.py past its budget.
    """
    import multiprocessing as mp
    import queue as _queue
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, fn, args), daemon=True)
    p.start()
    import time as _time
    deadline = _time.time() + (timeout if timeout is not None else 1e9)
    while True:
        try:
            status, val = q.get(timeout=1.0)
            break
        except _queue.Empty:
            if not p.is_alive():  # crashed or OOM-killed without reporting
                try:
                    status, val = q.get(timeout=2.0)
                    break
                except _queue.Empty:
                    raise RuntimeError(f"worker died (exit code {p.exitcode})")
            if _time.time() > deadline:
                p.kill()
                p.join(5)
                raise TimeoutError(f"{getattr(fn, '__name__', fn)} exceeded {timeout:.0f}s")
    p.join(10)
    if p.is_alive():
        p.kill()
    if status != "ok":
        raise RuntimeError(val)
    return val


def load_config():
    return json.loads((FINAL / "config.json").read_text())


def blend(preds, weights):
    z = sum(w * p for w, p in zip(weights, preds)) / sum(weights)
    z = np.clip(z, np.log1p(CLIP_LO), np.log1p(CLIP_HI))
    return np.expm1(z)
