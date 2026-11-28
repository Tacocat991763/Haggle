import sys; sys.path.insert(0, ".")
from src.runner import run
from src.features import ALL_FAMILIES
S = (0, 1, 2)
for m in ["xgb", "cat"]:
    p = {} if m == "xgb" else dict(iterations=1500, depth=6, learning_rate=0.03)
    run(f"abl_{m}_raw", m, p, families=(), seeds=S, note="ablation: raw only")
    run(f"abl_{m}_all", m, p, families=ALL_FAMILIES, seeds=S, note="ablation: all")
    for f in ALL_FAMILIES:
        run(f"abl_{m}_only_{f}", m, p, families=(f,), seeds=S, note=f"ablation: raw+{f}")
        run(f"abl_{m}_minus_{f}", m, p, families=tuple(x for x in ALL_FAMILIES if x != f), seeds=S, note=f"ablation: all-{f}")
