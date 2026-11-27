from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

base_dir = Path(__file__).resolve().parent

model = CatBoostRegressor()
model.load_model(str(base_dir / "model_BEST.cbm"))

df = pd.read_csv(
    base_dir / "test.csv",
    dtype={"project_id": str}
)

inputs = df[model.feature_names_].copy()

for index in model.get_cat_feature_indices():
    column = inputs.columns[index]

    inputs[column] = (
        inputs[column]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

predictions = np.maximum(
    np.expm1(model.predict(inputs)),
    0
)

results = pd.DataFrame({
    "project_id": df["project_id"],
    "hours_spent": predictions
})

results.to_csv(
    base_dir / "predictions.csv",
    index=False
)

print(f"Generated {len(results)} predictions")