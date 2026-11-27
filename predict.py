from pathlib import Path
import pandas as pd
import torch
from torch import nn
import pickle

base_dir = Path(__file__).resolve().parent

model_path = base_dir / "model_BEST.pth"
preprocessing_path = base_dir / "preprocessing.pkl"

with open(preprocessing_path, "rb") as f:
    preprocessing = pickle.load(f)

numerical_columns = preprocessing["numerical_columns"]
categorical_columns = preprocessing["categorical_columns"]
medians = preprocessing["medians"]
means = preprocessing["means"]
stds = preprocessing["stds"]
feature_columns = preprocessing["feature_columns"]

df = pd.read_csv(
    base_dir / "test.csv",
    dtype={"project_id": str}
)

project_ids = df["project_id"].copy()

inputs = df.drop(columns=[
    "project_id",
    "hours_spent",
    "log_hours",
    "hours_bucket"
], errors="ignore")

inputs[numerical_columns] = (
    inputs[numerical_columns].fillna(medians)
)

inputs[numerical_columns] = (
    inputs[numerical_columns] - means
) / stds

for column in categorical_columns:
    inputs[column] = (
        inputs[column]
        .astype("string")
        .fillna("__MISSING__")
    )

inputs = pd.get_dummies(
    inputs,
    columns=categorical_columns,
    dtype=float
)

inputs = inputs.reindex(
    columns=feature_columns,
    fill_value=0
)

X = torch.tensor(
    inputs.to_numpy(),
    dtype=torch.float32
)

model = nn.Sequential(
    nn.Linear(X.shape[1], 128),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Softplus()
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

with torch.inference_mode():
    predictions = model(X)
    predictions = torch.clamp(predictions, min=0)
    predictions = predictions.reshape(-1).cpu().numpy()

results = pd.DataFrame({
    "project_id": project_ids,
    "hours_spent": predictions
})

results.to_csv(
    base_dir / "predictions.csv",
    index=False
)

print(f"Generated {len(results)} predictions")