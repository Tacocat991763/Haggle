from datetime import datetime
import pandas as pd
import torch
from torch import nn
import logging
from pathlib import Path
import pickle

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


def rmsle(predictions, targets):
    predictions = torch.clamp(predictions, min=0)

    return torch.sqrt(
        torch.mean(
            (torch.log1p(predictions) - torch.log1p(targets)) ** 2
        )
    )


df = pd.read_csv("train.csv")

inputs = df.drop(columns=[
    "project_id",
    "hours_spent",
    "log_hours",
    "hours_bucket"
])

split = int(len(df) * 0.8)

train_inputs = inputs.iloc[:split].copy()
val_inputs = inputs.iloc[split:].copy()

numerical_columns = inputs.select_dtypes(
    include="number"
).columns

categorical_columns = inputs.select_dtypes(
    exclude="number"
).columns

medians = train_inputs[numerical_columns].median().fillna(0)

train_inputs[numerical_columns] = (
    train_inputs[numerical_columns].fillna(medians)
)

val_inputs[numerical_columns] = (
    val_inputs[numerical_columns].fillna(medians)
)

means = train_inputs[numerical_columns].mean()

stds = (
    train_inputs[numerical_columns]
    .std()
    .fillna(1)
    .replace(0, 1)
)

train_inputs[numerical_columns] = (
    train_inputs[numerical_columns] - means
) / stds

val_inputs[numerical_columns] = (
    val_inputs[numerical_columns] - means
) / stds

for column in categorical_columns:
    train_inputs[column] = (
        train_inputs[column]
        .astype("string")
        .fillna("__MISSING__")
    )

    val_inputs[column] = (
        val_inputs[column]
        .astype("string")
        .fillna("__MISSING__")
    )

train_inputs = pd.get_dummies(
    train_inputs,
    columns=categorical_columns,
    dtype=float
)

val_inputs = pd.get_dummies(
    val_inputs,
    columns=categorical_columns,
    dtype=float
)

val_inputs = val_inputs.reindex(
    columns=train_inputs.columns,
    fill_value=0
)

with open(output_dir / "preprocessing.pkl", "wb") as f:
    pickle.dump({
        "numerical_columns": list(numerical_columns),
        "categorical_columns": list(categorical_columns),
        "medians": medians,
        "means": means,
        "stds": stds,
        "feature_columns": list(train_inputs.columns)
    }, f)

X = torch.tensor(
    train_inputs.to_numpy(),
    dtype=torch.float32
)

Y = torch.tensor(
    df["hours_spent"].iloc[:split].to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

Z = torch.tensor(
    val_inputs.to_numpy(),
    dtype=torch.float32
)

W = torch.tensor(
    df["hours_spent"].iloc[split:].to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

print(f"Training input shape: {X.shape}")
print(f"Validation input shape: {Z.shape}")
print(f"Number of input features: {X.shape[1]}")

logging.info(f"Training input shape: {X.shape}")
logging.info(f"Validation input shape: {Z.shape}")
logging.info(f"Number of input features: {X.shape[1]}")
logging.info(f"Numerical columns: {list(numerical_columns)}")
logging.info(f"Categorical columns: {list(categorical_columns)}")

model = nn.Sequential(
    nn.Linear(X.shape[1], 128),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Softplus()
)


def loss_fn(predictions, targets):
    return torch.mean(
        (torch.log1p(predictions) - torch.log1p(targets)) ** 2
    )


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=100,
    min_lr=1e-8
)

logging.info("Model, loss function, and optimizer initialized")
logging.info(f"Model architecture: {model}")
logging.info(f"Loss function: {loss_fn}")
logging.info(f"Optimizer: {optimizer}")
logging.info("Learning rate scheduler initialized")

epoch = 0
max_epochs = 20000
early_stopping_patience = 1800

model.eval()

with torch.no_grad():
    score = rmsle(model(Z), W)

best_score = score.item()
best_epoch = 0

torch.save(
    model.state_dict(),
    output_dir / "model_BEST.pth"
)

try:
    logging.info("Starting training loop")

    while score.item() > 0.10: #and epoch < max_epochs:
        model.train()

        predictions = model(X)
        loss = loss_fn(predictions, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            train_score = rmsle(model(X), Y)
            predictions = model(Z)
            score = rmsle(predictions, W)

        if not torch.isfinite(score).item():
            logging.error("Non-finite RMSLE detected")
            print("Non-finite RMSLE detected. Stopping training.")
            break

        scheduler.step(score.item())

        epoch += 1

        if score.item() < best_score:
            best_score = score.item()
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                output_dir / "model_BEST.pth"
            )

            logging.info(
                f"New best RMSLE: {best_score:.6f} "
                f"at epoch {best_epoch}"
            )

        if epoch % 10 == 0:
            message = (
                f"Epoch: {epoch} | "
                f"Train RMSLE: {train_score.item():.4f} | "
                f"Val RMSLE: {score.item():.4f} | "
                f"Best RMSLE: {best_score:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

            print(message)
            logging.info(message)

        if epoch % 1000 == 0:
            filename = output_dir / (
                f"model_{epoch}_{score.item():.4f}.pth"
            )

            torch.save(model.state_dict(), filename)

            message = (
                f"Saved model to {filename} at epoch {epoch}"
            )

            print(message)
            logging.info(message)

        if epoch - best_epoch >= early_stopping_patience:
            message = (
                f"Early stopping at epoch {epoch}. "
                f"Best RMSLE: {best_score:.6f} "
                f"at epoch {best_epoch}"
            )

            print(message)
            logging.info(message)
            break

except KeyboardInterrupt:
    print("Training manually stopped. Restoring best model...")
    logging.warning(
        "Training manually stopped. Restoring best model..."
    )

model.load_state_dict(
    torch.load(
        output_dir / "model_BEST.pth",
        weights_only=True
    )
)

model.eval()

with torch.no_grad():
    predictions = model(Z)
    score = rmsle(predictions, W)
    train_score = rmsle(model(X), Y)

filename = output_dir / "model_FINAL.pth"

torch.save(model.state_dict(), filename)

print(f"Saved model to {filename}")
print(f"Total epochs: {epoch}")
print(f"Best epoch: {best_epoch}")
print(f"Training RMSLE: {train_score.item():.6f}")
print(f"Validation RMSLE: {score.item():.6f}")

logging.info(f"Saved model to {filename}")
logging.info(f"Total epochs: {epoch}")
logging.info(f"Best epoch: {best_epoch}")
logging.info(f"Training RMSLE: {train_score.item():.6f}")
logging.info(f"Validation RMSLE: {score.item():.6f}")
logging.info("Program finished")