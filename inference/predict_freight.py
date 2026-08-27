from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "freight_cost_prediction" / "models" / "predict_freight_model.pkl"
VENDOR_STATS_PATH = ROOT / "data" / "vendor_stats.csv"


def predict_freight_cost(input_data: dict) -> pd.DataFrame:
    """Predict freight for one invoice using UI values and vendor defaults."""
    quantity = input_data["Quantity"]
    dollars = input_data["Dollars"]
    vendor = input_data.get("VendorName")

    stats = pd.read_csv(VENDOR_STATS_PATH)
    vendor_stats = stats.iloc[0] if vendor is None else stats[stats["VendorName"] == vendor].iloc[0]

    features = pd.DataFrame([{
        "Quantity": quantity,
        "Dollars": dollars,
        "lead_time_days": input_data.get("lead_time_days", vendor_stats["avg_lead_time_days"]),
        "vendor_avg_freight_loo": input_data.get("vendor_avg_freight_loo", vendor_stats["avg_freight"]),
        "VendorName": vendor or vendor_stats["VendorName"],
    }])
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(features)
    return pd.DataFrame({"Predicted_Freight": prediction})
