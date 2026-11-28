"""Write models/final/config.json from experiment ids + weights. Usage: python experiments/make_config.py id:weight[:seeds] ..."""
import json, sys
sys.path.insert(0, ".")
from src.runner import load_results
recs = {r["id"]: r for r in load_results()}
members = []
for arg in sys.argv[1:]:
    parts = arg.split(":")
    r = recs[parts[0]]
    seeds = list(range(int(parts[2]))) if len(parts) > 2 else r["seeds"]
    members.append(dict(name=parts[0].replace("+", "_"), source=parts[0], model=r["model"], params=r["params"],
                        families=r["families"], keep_raw_skewed=r["keep_raw_skewed"], seeds=seeds,
                        weight=float(parts[1]), kind="icl" if r["model"] in ("tabdpt", "tabicl", "tabpfn") else "saved"))
json.dump({"members": members, "clip": [0.1, 140.7]}, open("models/final/config.json", "w"), indent=1)
print(json.dumps(members, indent=1)[:1500])
