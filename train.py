"""Build the final submission artifacts from models/final/config.json.

Usage: python train.py
Fits every "saved" member on the full train.csv (dev + holdout) and pickles it to models/final/.
"icl" members need no training; they read train.csv at prediction time.
To change the ensemble, edit config.json (members are chosen from experiments/results.jsonl;
see research/leaderboard.md and research/notes.md).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FINAL, load_config, run_isolated, train_member

if __name__ == "__main__":
    cfg = load_config()
    for m in cfg["members"]:
        if m.get("kind") == "saved":
            t = time.time()
            run_isolated(train_member, m, str(ROOT / "train.csv"), str(FINAL))
            print(f"trained {m['name']} in {time.time() - t:.1f}s")
