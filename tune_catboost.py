from datetime import datetime
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
import logging
from catboost import CatBoostRegressor

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path(f"outputs/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=output_dir / "logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

df = pd.read_csv("train.csv")

inputs = df.drop(columns=[
    "project_id",
    "hours_spent",
    "log_hours",
    "hours_bucket"
])

categorical_columns = inputs.select_dtypes(
    exclude="number"
).columns.tolist()

for column in categorical_columns:
    inputs[column] = (
        inputs[column]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

split = int(len(df) * 0.8)

X = inputs.iloc[:split]
Y = np.log1p(df["hours_spent"].iloc[:split])

Z = inputs.iloc[split:]
W = np.log1p(df["hours_spent"].iloc[split:])

configurations = list(product(
    [3, 4, 6, 8],
    [0.01, 0.03],
    [3, 10, 30]
))

best_score = float("inf")
results = []

try:
    for trial, (depth, lr, reg) in enumerate(
        configurations, start=1
    ):
        message = (
            f"Trial {trial}/{len(configurations)} | "
            f"Depth: {depth} | LR: {lr} | L2: {reg}"
        )

        print(message, flush=True)
        logging.info(message)

        model = CatBoostRegressor(
            iterations=5000,
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=reg,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            task_type="CPU",
            allow_writing_files=False
        )

        model.fit(
            X,
            Y,
            cat_features=categorical_columns,
            eval_set=(Z, W),
            early_stopping_rounds=200,
            use_best_model=True,
            verbose=False
        )

        predictions = np.maximum(model.predict(Z), 0)

        score = np.sqrt(
            np.mean((predictions - W.to_numpy()) ** 2)
        )

        results.append({
            "trial": trial,
            "depth": depth,
            "learning_rate": lr,
            "l2_leaf_reg": reg,
            "iterations": model.tree_count_,
            "RMSLE": score
        })

        message = (
            f"Trial {trial} | "
            f"RMSLE: {score:.6f} | "
            f"Trees: {model.tree_count_}"
        )

        print(message, flush=True)
        logging.info(message)

        model.save_model(str(
            output_dir / f"model_{trial}_{score:.6f}.cbm"
        ))

        if score < best_score:
            best_score = score

            model.save_model(str(
                output_dir / "model_BEST.cbm"
            ))

            print(f"NEW BEST: {best_score:.6f}")
            logging.info(f"NEW BEST: {best_score:.6f}")

        pd.DataFrame(results).sort_values(
            "RMSLE"
        ).to_csv(
            output_dir / "results.csv",
            index=False
        )

except KeyboardInterrupt:
    print("Search manually stopped.")
    logging.warning("Search manually stopped.")

if results:
    leaderboard = pd.DataFrame(results).sort_values("RMSLE")

    print("\nResults:")
    print(leaderboard.to_string(index=False))

    print(f"\nBest RMSLE: {best_score:.6f}")
    logging.info(f"Best RMSLE: {best_score:.6f}")