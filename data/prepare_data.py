"""
Extracts vendor invoice records from a real `inventory.db` (SQLite) and
engineers the features used for freight-cost prediction and risk flagging.

Expects a table `vendor_invoice` with columns:
VendorNumber, VendorName, InvoiceDate, PONumber, PODate, PayDate,
Quantity, Dollars, Freight, Approval

Usage:
    python data/prepare_data.py --db-path /path/to/inventory.db
    (defaults to data/inventory.db if not given)

Output: data/invoices.csv
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).parent


def main(db_path: str):
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Could not find {db_path}. Place your inventory.db in data/ "
            f"or pass --db-path /full/path/to/inventory.db"
        )

    with sqlite3.connect(db_path) as conn:
        vi = pd.read_sql("SELECT * FROM vendor_invoice", conn)

    # --- clean + engineer -------------------------------------------------
    for c in ["PODate", "InvoiceDate", "PayDate"]:
        vi[c] = pd.to_datetime(vi[c])

    vi["lead_time_days"] = (vi["InvoiceDate"] - vi["PODate"]).dt.days
    vi["pay_delay_days"] = (vi["PayDate"] - vi["InvoiceDate"]).dt.days
    vi["freight_pct_of_dollars"] = vi["Freight"] / vi["Dollars"].replace(0, 1)

    # target for the risk classifier: did this invoice require a manual
    # approval sign-off? (real signal — only 14/129 vendors ever trigger it,
    # and flagged invoices are ~18x larger on average than unflagged ones)
    vi["needs_approval"] = vi["Approval"].notna().astype(int)

    # drop rows with obviously bad values
    vi = vi[(vi["Quantity"] > 0) & (vi["Dollars"] > 0) & (vi["Freight"] >= 0)]
    vi = vi.dropna(subset=["lead_time_days", "pay_delay_days"])

    # vendor-level baseline, computed leave-one-out so a vendor's own
    # invoice never leaks into its own baseline feature
    vendor_sum_freight = vi.groupby("VendorNumber")["Freight"].transform("sum")
    vendor_sum_dollars = vi.groupby("VendorNumber")["Dollars"].transform("sum")
    vendor_count = vi.groupby("VendorNumber")["Freight"].transform("count")

    vi["vendor_invoice_count"] = vendor_count
    # leave-one-out mean; falls back to the row's own value for vendors with 1 invoice
    vi["vendor_avg_freight_loo"] = (
        (vendor_sum_freight - vi["Freight"]) / (vendor_count - 1).replace(0, 1)
    ).where(vendor_count > 1, vi["Freight"])
    vi["vendor_avg_dollars_loo"] = (
        (vendor_sum_dollars - vi["Dollars"]) / (vendor_count - 1).replace(0, 1)
    ).where(vendor_count > 1, vi["Dollars"])

    keep = [
        "VendorNumber", "VendorName", "PONumber",
        "PODate", "InvoiceDate", "PayDate",
        "Quantity", "Dollars", "Freight",
        "lead_time_days", "pay_delay_days", "freight_pct_of_dollars",
        "vendor_invoice_count", "vendor_avg_freight_loo", "vendor_avg_dollars_loo",
        "needs_approval",
    ]
    vi = vi[keep].reset_index(drop=True)
    vi["invoice_id"] = "INV-" + vi["PONumber"].astype(str) + "-" + vi.index.astype(str)

    out_path = OUT_DIR / "invoices.csv"
    vi.to_csv(out_path, index=False)

    # per-vendor baseline table the app uses to auto-fill features for a
    # brand-new invoice (plain averages here -- no leave-one-out needed
    # since a genuinely new invoice was never part of any average)
    vendor_stats = vi.groupby("VendorName").agg(
        avg_freight=("Freight", "mean"),
        avg_dollars=("Dollars", "mean"),
        avg_lead_time_days=("lead_time_days", "mean"),
        avg_pay_delay_days=("pay_delay_days", "mean"),
        invoice_count=("Freight", "count"),
    ).reset_index()
    vendor_stats.to_csv(OUT_DIR / "vendor_stats.csv", index=False)

    print(f"Extracted {len(vi)} invoices from {vi['VendorName'].nunique()} vendors")
    print(f"needs_approval rate: {vi['needs_approval'].mean():.2%}")
    print(f"Saved -> {out_path}")
    print(f"Saved -> {OUT_DIR / 'vendor_stats.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=str(OUT_DIR / "inventory.db"))
    args = parser.parse_args()
    main(args.db_path)
