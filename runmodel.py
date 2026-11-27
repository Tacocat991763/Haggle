import torch
import torch.nn as nn
import pandas as pd

df = pd.read_csv("train.csv")

numerical = df.select_dtypes(include="number")

inputs = numerical.drop(columns=["hours_spent", "log_hours"])

inputs = inputs.fillna(inputs.median()).fillna(0)

X = torch.tensor(
    inputs.to_numpy(),
    dtype=torch.float32
)

Y = torch.tensor(
    df["hours_spent"].to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

def rmsle(predictions, targets):
    predictions = torch.clamp(predictions, min=0)

    return torch.sqrt(
        torch.mean(
            (torch.log1p(predictions) - torch.log1p(targets)) ** 2
        )
    )

model = nn.Sequential(
    nn.Linear(45,500),
    nn.ReLU(),
    nn.Linear(500,500),
    nn.ReLU(),
    nn.Linear(500,500),
    nn.ReLU(),
    nn.Linear(500, 1)
)

model.load_state_dict(
    torch.load("model_2026-11-27_01-03-47.pth", weights_only=True)
)

model.eval()

with torch.no_grad():
    predictions = model(X)
    score = rmsle(predictions, Y)

print("RMSLE:", score.item())