from datetime import datetime
import pandas as pd
import torch
from torch import nn

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

# Calculate medians using training rows only
inputs = inputs.fillna(inputs.iloc[:-1].median()).fillna(0)

X = torch.tensor(inputs.iloc[:-1].to_numpy(), dtype=torch.float32)

Y = torch.tensor(
    df["hours_spent"].iloc[:-1].to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

Z = torch.tensor(inputs.iloc[-1:].to_numpy(), dtype=torch.float32)

W = torch.tensor(
    df["hours_spent"].iloc[-1:].to_numpy(),
    dtype=torch.float32
).unsqueeze(1)

model = nn.Sequential(
    nn.Linear(45, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, 1),
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss = torch.tensor(float("inf"))
epoch = 0

model.train()
while loss.item() > 0.1:
  predictions = model(X)
  loss = loss_fn(predictions, Y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  epoch += 1
  if epoch % 10 == 0:
    print(epoch, loss.item())

model.eval()
print(X[431])
print(Y[431])
with torch.no_grad():
  print(model(X[431]))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"model_{timestamp}.pth"
torch.save(model.state_dict(), filename)
print(f"Saved model to {filename}")

with torch.no_grad():
    predictions = model(Z)
    score = rmsle(predictions, W)

print("RMSLE:", score.item())
