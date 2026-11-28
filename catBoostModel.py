from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from catboost import CatBoostRegressor

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = Path(f"outputs/{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=output_dir / "logs.txt",
    encoding="utf-8",
    filemode="a",
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

logging.info("Program started")

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

Y = np.log1p(
    df["hours_spent"].iloc[:split].to_numpy()
)

Z = inputs.iloc[split:]

W = np.log1p(
    df["hours_spent"].iloc[split:].to_numpy()
)

print(f"Training input shape: {X.shape}")
print(f"Validation input shape: {Z.shape}")
print(f"Number of features: {X.shape[1]}")

logging.info(f"Training input shape: {X.shape}")
logging.info(f"Validation input shape: {Z.shape}")
logging.info(f"Numerical columns: {list(inputs.select_dtypes(include='number').columns)}")
logging.info(f"Categorical columns: {categorical_columns}")


def rmsle(predictions, targets):
    predictions = np.maximum(predictions, 0)

    return np.sqrt(
        np.mean(
            (np.log1p(predictions) - np.log1p(targets)) ** 2
        )
    )


model = CatBoostRegressor(
    iterations=20000,
    depth=6,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="RMSE",
    l2_leaf_reg=3,
    task_type="CPU",
    random_seed=42,
    train_dir=str(output_dir),
    save_snapshot=True,
    snapshot_file=str(output_dir.resolve() / "training.snapshot"),
    snapshot_interval=60
)

logging.info("Model initialized")
logging.info(f"Model parameters: {model.get_params()}")


class TrainingLogger:
    def __init__(self):
        self.best_score = float("inf")
        self.best_epoch = 0

    def after_iteration(self, info):
        epoch = info.iteration

        train_score = info.metrics["learn"]["RMSE"][-1]
        val_score = info.metrics["validation"]["RMSE"][-1]

        if val_score < self.best_score:
            self.best_score = val_score
            self.best_epoch = epoch

            logging.info(
                f"New best RMSLE: {self.best_score:.6f} "
                f"at iteration {epoch}"
            )

        if epoch % 10 == 0:
            message = (
                f"Iteration: {epoch} | "
                f"Train RMSLE: {train_score:.4f} | "
                f"Val RMSLE: {val_score:.4f} | "
                f"Best RMSLE: {self.best_score:.4f}"
            )

            print(message)
            logging.info(message)

        return val_score > 0.10


training_logger = TrainingLogger()

try:
    logging.info("Starting training")

    model.fit(
        X,
        Y,
        cat_features=categorical_columns,
        eval_set=(Z, W),
        verbose=False,
        early_stopping_rounds=500,
        use_best_model=True,
        callbacks=[training_logger]
    )

except KeyboardInterrupt:
    print("Training manually stopped.")
    logging.warning("Training manually stopped.")

if model.is_fitted():
    best_epoch = model.tree_count_

    filename = output_dir / "model_BEST.cbm"
    model.save_model(str(filename))

    print(f"Saved best model to {filename}")
    logging.info(f"Saved best model to {filename}")

    for epoch in range(1000, best_epoch + 1, 1000):
        checkpoint = model.copy()
        checkpoint.shrink(epoch)

        checkpoint_predictions = np.maximum(
            np.expm1(checkpoint.predict(Z)),
            0
        )

        checkpoint_score = rmsle(
            checkpoint_predictions,
            np.expm1(W)
        )

        filename = output_dir / (
            f"model_{epoch}_{checkpoint_score:.4f}.cbm"
        )

        checkpoint.save_model(str(filename))

        print(f"Saved model to {filename}")
        logging.info(f"Saved model to {filename}")

    train_predictions = np.maximum(
        np.expm1(model.predict(X)),
        0
    )

    predictions = np.maximum(
        np.expm1(model.predict(Z)),
        0
    )

    train_score = rmsle(train_predictions, np.expm1(Y))
    score = rmsle(predictions, np.expm1(W))

    filename = output_dir / "model_FINAL.cbm"
    model.save_model(str(filename))

    print(f"Saved model to {filename}")
    print(f"Best iteration: {best_epoch}")
    print(f"Training RMSLE: {train_score:.6f}")
    print(f"Validation RMSLE: {score:.6f}")

    logging.info(f"Saved model to {filename}")
    logging.info(f"Best iteration: {best_epoch}")
    logging.info(f"Training RMSLE: {train_score:.6f}")
    logging.info(f"Validation RMSLE: {score:.6f}")

    importance = pd.DataFrame({
        "feature": inputs.columns,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        output_dir / "feature_importance.csv",
        index=False
    )

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    logging.info(
        f"Feature importance:\n{importance.to_string(index=False)}"
    )

else:
    logging.warning("No fitted model available to export")
    print("No fitted model available. Check the training snapshot.")

logging.info("Program finished")