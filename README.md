# 📦 Vendor Invoice Intelligence Portal v2

**Freight Cost Prediction & Approval-Routing Prediction — built on a real vendor invoice dataset**

An end-to-end ML system for finance/procurement teams: predicts the expected
freight cost on a vendor invoice, and predicts whether an invoice is likely
to require manual approval sign-off, so reviewers can focus attention before
an invoice even lands rather than auditing everything after the fact.

## Data Source

This uses a real `inventory.db` (SQLite) from a beverage distributor — the
`vendor_invoice` table has 5,543 real invoices across 129 vendors, with
`VendorName`, `Quantity`, `Dollars`, `Freight`, `PODate`, `InvoiceDate`,
`PayDate`, and `Approval` (the name of whoever signed off, or blank).
No synthetic data is used anywhere in this version.

`data/prepare_data.py` extracts `vendor_invoice`, engineers:
- `lead_time_days` = InvoiceDate − PODate
- `pay_delay_days` = PayDate − InvoiceDate
- `freight_pct_of_dollars` = Freight ÷ Dollars
- `needs_approval` = 1 if `Approval` is non-null (the real target for the risk model)
- `vendor_avg_freight_loo` / `vendor_avg_dollars_loo` — **leave-one-out** vendor
  baselines, so a vendor's own invoice never leaks into its own baseline feature

and writes `data/invoices.csv` + `data/vendor_stats.csv` (plain vendor
averages, used by the app to auto-fill fields for a brand-new invoice).

## ⚠️ An important finding, not a bug

The approval-routing classifier scores near-perfect (F1 ≈ 1.0). That looked
like leakage at first, so I checked: for vendors that ever get flagged (14 of
129), the flag is **almost a clean dollar-amount threshold specific to that
vendor** — e.g. Bacardi invoices flip from unflagged to flagged right around
$236K–$252K, consistently. The model isn't overfitting noise; it's recovering
a real (if informal and inconsistent-across-vendors) approval policy that
already exists in this data. That's a legitimate, useful finding on its own —
it's the kind of thing that surfaces "vendor A needs $150K sign-off but
vendor B doesn't get flagged until $400K, is that intentional?" — but it's
also why the classification metrics look almost too good, and I'd rather
flag that explicitly than let a 1.0 F1 pass without explanation.

## Models Used

**Freight Cost Prediction (regression)** — target: `Freight`
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor ✅ best (R² ≈ 0.97)

**Approval-Routing Prediction (classification)** — target: `needs_approval`
- Logistic Regression (F1 ≈ 0.93)
- Decision Tree Classifier ✅ best (F1 ≈ 1.00 — see note above)
- Random Forest Classifier (F1 ≈ 1.00)

All classifiers use `class_weight="balanced"` since only ~6.75% of invoices
require approval.

## Project Structure

```
vendor-invoice-intelligence-portal-v2/
├── data/
│   ├── prepare_data.py     # extracts + engineers features from inventory.db
│   ├── inventory.db        # <- put your real db here (not included, too large to ship)
│   ├── invoices.csv        # generated
│   └── vendor_stats.csv    # generated
├── freight_cost_prediction/
│   ├── train.py
│   └── models/predict_freight_model.pkl
├── invoice_flagging/
│   ├── train.py
│   └── models/predict_flag_invoice.pkl
├── app.py                  # Streamlit UI
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

# put your inventory.db in data/, or pass a path directly:
python data/prepare_data.py --db-path /path/to/inventory.db

python freight_cost_prediction/train.py
python invoice_flagging/train.py

streamlit run app.py
```

## App Features

- **Single Invoice Check** — pick a vendor, enter quantity/amount/timing,
  get a predicted freight cost and an approval-needed probability with a
  gauge chart.
- **Batch Scoring** — upload a CSV of invoices (`VendorName, Quantity,
  Dollars, lead_time_days, pay_delay_days`), get all of them scored,
  download results.
- **Portfolio Dashboard** — freight cost distribution, riskiest vendors by
  approval rate, and both models' metrics side by side.

## Evaluation Metrics

- Regression: MAE, RMSE, R²
- Classification: Accuracy, Precision, Recall, F1 (shown live in the dashboard tab)

## Live Demo : https://vendor-invoice-intelligent.streamlit.app/
