from datetime import datetime
import pandas as pd
import torch
from torch import nn
import logging
from pathlib import Path

logging.basicConfig(
    filename="logs.txt",
    encoding="utf-8",
    filemode="a",
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

logging.info("Program started")
#logging.warning("Something might be off")
#logging.error("An error occurred")


def rmsle(predictions, targets):
    predictions = torch.clamp(predictions, min=0)

    return torch.sqrt(
        torch.mean(
            (torch.log1p(predictions) - torch.log1p(targets)) ** 2
        )
    )

df = pd.read_csv("train.csv")

numerical = df.select_dtypes(include="number")
inputs = numerical.drop(columns=["hours_spent", "log_hours"])

split = int(len(df) * 0.8)

inputs = inputs.fillna(inputs.iloc[:split].median()).fillna(0)

X = torch.tensor(
    inputs.iloc[:split].to_numpy(),
    dtype=torch.float32
)

Y = torch.tensor(
    df["hours_spent"].iloc[:split].to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

Z = torch.tensor(
    inputs.iloc[split:].to_numpy(),
    dtype=torch.float32
)

W = torch.tensor(
    df["hours_spent"].iloc[split:].to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

model = nn.Sequential(
    nn.Linear(45, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Softplus()
)

def loss_fn(predictions, targets):
    return torch.mean(
        (torch.log1p(predictions) - torch.log1p(targets)) ** 2
    )
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

logging.info("Model, loss function, and optimizer initialized")
logging.info(f"Model architecture: {model}")
logging.info(f"Loss function: {loss_fn}")
logging.info(f"Optimizer: {optimizer}")

loss = torch.tensor(float("inf"))
epoch = 0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=5000,
    min_lr=1e-8
)

logging.info("Learning rate scheduler initialized")

model.train()

score = torch.tensor(float("inf"))

try:
    logging.info("Starting training loop")
    while score.item() > 0.10: #and epoch < 20000:
        model.train()

        predictions = model(X)
        loss = loss_fn(predictions, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            predictions = model(Z)
            score = rmsle(predictions, W)

        scheduler.step(score.item())

        epoch += 1

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Loss: {loss.item():.4f} | "
                f"RMSLE: {score.item():.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )
            logging.info(
                f"Epoch: {epoch} | "
                f"Loss: {loss.item():.4f} | "
                f"RMSLE: {score.item():.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.8f}"
            )

        if epoch % 1000 == 0:
            Path("outputs").mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"outputs/model_{timestamp}.pth"
            torch.save(model.state_dict(), filename)
            print(f"Saved model to {filename} at epoch {epoch}")
            logging.info(f"Saved model to {filename} at epoch {epoch}")

except KeyboardInterrupt:
    print("Training manually stopped. Saving model...")
    logging.warning("Training manually stopped. Saving model...")

model.eval()
print(Z)
print(W)
with torch.no_grad():
  print(model(Z))

Path("outputs").mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"outputs/model_{timestamp}.pth"
torch.save(model.state_dict(), filename)
print(f"Saved model to {filename}")
logging.info(f"Saved model to {filename}")
logging.info("Program finished")


with torch.no_grad():
    predictions = model(Z)
    score = rmsle(predictions, W)

print("RMSLE:", score.item())
