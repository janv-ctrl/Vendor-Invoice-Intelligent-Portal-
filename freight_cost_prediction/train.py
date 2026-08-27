"""Train the freight cost regression model on real invoice data.
Run: python freight_cost_prediction/train.py"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "invoices.csv")

NUM_FEATURES = ["Quantity", "Dollars", "lead_time_days", "vendor_avg_freight_loo"]
CAT_FEATURES = ["VendorName"]
TARGET = "Freight"

X = df[NUM_FEATURES + CAT_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

candidates = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
}

results = {}
best_name, best_pipe, best_r2 = None, None, -np.inf

for name, model in candidates.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    results[name] = dict(MAE=round(mae, 2), RMSE=round(rmse, 2), R2=round(r2, 3))
    print(f"{name:15s} MAE={mae:10.2f}  RMSE={rmse:10.2f}  R2={r2:.3f}")
    if r2 > best_r2:
        best_name, best_pipe, best_r2 = name, pipe, r2

print(f"\nBest model: {best_name} (R2={best_r2:.3f})")

MODEL_DIR = ROOT / "freight_cost_prediction" / "models"
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(best_pipe, MODEL_DIR / "predict_freight_model.pkl")
joblib.dump(results, MODEL_DIR / "metrics.pkl")
print(f"Saved -> {MODEL_DIR / 'predict_freight_model.pkl'}")
