import sys; sys.path.insert(0, ".")
from src.runner import run
run("cat_raw_d6", "cat", dict(iterations=1500, depth=6, learning_rate=0.03), families=(), note="~current submission config")
run("cat_all", "cat", dict(iterations=1500, depth=6, learning_rate=0.03), note="all feature families")
run("lgb_all", "lgb", {}, note="defaults")
run("xgb_all", "xgb", {}, note="defaults")
run("ridge_all", "ridge", dict(alpha=30), note="")
run("et_all", "et", {}, note="")
run("mlp_all", "mlp", {}, note="")
