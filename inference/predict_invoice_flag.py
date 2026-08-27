from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
VENDOR_STATS_PATH = ROOT / "data" / "vendor_stats.csv"


def predict_invoice_flag(input_data: dict) -> pd.DataFrame:
    """Predict whether one invoice needs manual approval."""
    stats = pd.read_csv(VENDOR_STATS_PATH)
    vendor = input_data.get("VendorName")
    vendor_stats = stats.iloc[0] if vendor is None else stats[stats["VendorName"] == vendor].iloc[0]

    quantity = input_data["invoice_quantity"]
    dollars = input_data["invoice_dollars"]
    freight = input_data["Freight"]
    features = pd.DataFrame([{
        "Quantity": quantity,
        "Dollars": dollars,
        "Freight": freight,
        "lead_time_days": input_data.get("lead_time_days", vendor_stats["avg_lead_time_days"]),
        "pay_delay_days": input_data.get("pay_delay_days", vendor_stats["avg_pay_delay_days"]),
        "freight_pct_of_dollars": freight / dollars if dollars else 0,
        "vendor_avg_freight_loo": input_data.get("vendor_avg_freight_loo", vendor_stats["avg_freight"]),
        "vendor_avg_dollars_loo": input_data.get("vendor_avg_dollars_loo", vendor_stats["avg_dollars"]),
        "vendor_invoice_count": input_data.get("vendor_invoice_count", vendor_stats["invoice_count"]),
        "VendorName": vendor or vendor_stats["VendorName"],
    }])
    model = joblib.load(MODEL_PATH)
    prediction = model.predict(features)
    return pd.DataFrame({"Predicted_Flag": prediction})
