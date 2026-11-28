import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model("model_BEST.cbm")

df = pd.read_csv("train.csv")

inputs = df[model.feature_names_].copy()

for index in model.get_cat_feature_indices():
    column = inputs.columns[index]
    inputs[column] = (
        inputs[column]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

split = int(len(df) * 0.8)

Z = inputs.iloc[split:]
W = df["hours_spent"].iloc[split:].to_numpy()

predictions = np.maximum(
    np.expm1(model.predict(Z)),
    0
)

score = np.sqrt(
    np.mean(
        (np.log1p(predictions) - np.log1p(W)) ** 2
    )
)

print("Predictions:", predictions)
print("Actual:", W)
print("RMSLE:", score)