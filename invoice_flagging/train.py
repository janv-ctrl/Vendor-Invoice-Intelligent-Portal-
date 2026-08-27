"""Train the invoice risk-flagging (needs-approval) classifier on real data.
Run: python invoice_flagging/train.py"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "invoices.csv")

NUM_FEATURES = [
    "Quantity", "Dollars", "Freight", "lead_time_days", "pay_delay_days",
    "freight_pct_of_dollars", "vendor_avg_freight_loo", "vendor_avg_dollars_loo",
    "vendor_invoice_count",
]
CAT_FEATURES = ["VendorName"]
TARGET = "needs_approval"

X = df[NUM_FEATURES + CAT_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

# class_weight="balanced" matters here: needs_approval is rare (~7% of invoices)
candidates = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42, class_weight="balanced"),
}

results = {}
best_name, best_pipe, best_f1 = None, None, -1

for name, model in candidates.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    results[name] = dict(Accuracy=round(acc, 3), Precision=round(prec, 3),
                          Recall=round(rec, 3), F1=round(f1, 3))
    print(f"{name:20s} Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    if f1 > best_f1:
        best_name, best_pipe, best_f1 = name, pipe, f1

print(f"\nBest model: {best_name} (F1={best_f1:.3f})")
print(f"Base rate (needs_approval): {y.mean():.2%}  <- beat this, not 'accuracy'")

MODEL_DIR = ROOT / "invoice_flagging" / "models"
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(best_pipe, MODEL_DIR / "predict_flag_invoice.pkl")
joblib.dump(results, MODEL_DIR / "metrics.pkl")
print(f"Saved -> {MODEL_DIR / 'predict_flag_invoice.pkl'}")
