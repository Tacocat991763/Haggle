"""Segment OOF errors. Usage: python experiments/residuals.py <oof_id | blend.npy path> > research/residuals.md"""
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from src.runner import OOF_DIR, data

arg = sys.argv[1]
dev, hold, folds = data()
y = np.log1p(dev.hours_spent.values)
p = np.load(arg if arg.endswith(".npy") else OOF_DIR / f"{arg}.npy")
r = np.maximum(p, 0) - y
d = dev.copy()
d["res"], d["sq"] = r, r ** 2


def seg(name, key):
    g = d.groupby(key, observed=True).agg(n=("res", "size"), bias=("res", "mean"), rmsle=("sq", lambda s: np.sqrt(s.mean())))
    g = g[g.n >= 15].round(3)
    print(f"\n### {name}\n\n" + g.to_markdown())


print(f"# Residual analysis for `{arg}`\n\nOverall RMSLE {np.sqrt((r**2).mean()):.4f}; residual = pred - actual (log1p space); positive bias = overestimate.")
seg("Actual hours", pd.cut(dev.hours_spent, [0, 1, 3, 10, 30, 100, 200]))
seg("Predicted hours", pd.cut(np.expm1(p), [0, 1, 3, 10, 30, 100, 1000]))
seg("Project category", d.project_category.fillna("NA"))
seg("Primary language (top)", d.primary_language)
seg("Framework", d.framework)
seg("Lines of code", pd.cut(d.lines_of_code, [-1, 0, 500, 2000, 10000, 1e9]))
seg("Active commit days", pd.cut(d.active_commit_days, [0, 1, 3, 10, 30, 1e9]))
seg("Commit count", pd.cut(d.num_commits, [0, 1, 5, 20, 100, 1e9]))
seg("Author prior projects", d.author_prior_projects)
seg("pct_code_in_first_commit (bulk initial commit)", pd.cut(d.pct_code_in_first_commit.fillna(-1), [-2, -0.5, 0.2, 0.5, 0.9, 1.01]))
seg("Missing source-file stats", d.avg_file_length.isna())
seg("has_backend / has_database", d.has_backend.astype(str) + "/" + d.has_database.astype(str))
seg("uses_game_engine", d.uses_game_engine)
seg("Clone/tutorial or template", d.is_clone_or_tutorial.astype(str) + "/" + d.is_forked_from_template.astype(str))
print("\n### Calibration slope\n")
b = np.polyfit(p, y, 1)
print(f"Regressing actual on predicted (log space): slope {b[0]:.3f}, intercept {b[1]:.3f} (slope > 1 = predictions too compressed).")
