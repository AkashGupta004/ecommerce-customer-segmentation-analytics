"""
preprocessing.py
================
Module responsible for data cleaning, anomaly handling, datetime standardization,
and transaction classification (purchases vs. returns/cancellations).

Key Responsibilities:
- Preserve raw data integrity.
- Step-by-step cleaning with row count audit and clear business rationale.
- Robust handling of missing Customer IDs, duplicates, and invalid prices.
- Analysis and isolation of returns/cancellations (Credit Notes with 'C' prefix).
"""

from typing import Dict, Tuple, Any, List
import pandas as pd
import numpy as np


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to lower_snake_case for code readability.
    """
    column_mapping = {
        "Invoice": "invoice",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "price",
        "Customer ID": "customer_id",
        "Country": "country",
    }
    df = df.rename(columns=column_mapping)
    return df


def clean_ecommerce_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes end-to-end data cleaning with full audit trail and business rationale.

    Parameters
    ----------
    data : pd.DataFrame
        Raw combined retail transactions DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]
        - cleaned_transactions: All valid transactions with known customers, including returns
                                (used for net customer monetary calculations).
        - valid_purchases: Subset with positive quantity and price (used for basket analysis).
        - audit_report: Complete step-by-step record of rows removed and justifications.
    """
    # 0. Preserve raw data
    raw_data = data.copy()
    initial_rows = len(raw_data)
    audit_steps: List[Dict[str, Any]] = []

    print("[Preprocessing] Starting data cleaning pipeline...")
    print(f"  Initial raw record count: {initial_rows:,}")

    # Step 1: Handle Missing Customer IDs
    # Business Rationale: Customer segmentation and CRM retention strategies inherently require
    # unique customer entity identifiers. Transactions without Customer ID represent guest checkout
    # or unauthenticated bulk orders which cannot be tracked across time or assigned to customer personas.
    has_cid_mask = raw_data["Customer ID"].notnull()
    df_step1 = raw_data[has_cid_mask].copy()
    removed_missing_cid = initial_rows - len(df_step1)
    audit_steps.append({
        "step": "1. Remove Missing Customer IDs",
        "rows_before": initial_rows,
        "rows_after": len(df_step1),
        "rows_removed": removed_missing_cid,
        "pct_removed": (removed_missing_cid / initial_rows) * 100,
        "rationale": "Customer segmentation requires known customer identities to track cross-session behavior.",
    })
    print(f"  Step 1: Removed {removed_missing_cid:,} rows with missing Customer ID ({audit_steps[-1]['pct_removed']:.2f}%).")

    # Step 2: Remove Exact Duplicate Records
    # Business Rationale: Exact duplicate rows across all fields are technical anomalies resulting from
    # network retransmissions, POS system resyncs, or duplicate logging. Keeping them artificially distorts
    # transaction volume and monetary totals.
    rows_before_dedup = len(df_step1)
    df_step2 = df_step1.drop_duplicates().copy()
    removed_duplicates = rows_before_dedup - len(df_step2)
    audit_steps.append({
        "step": "2. Remove Exact Duplicates",
        "rows_before": rows_before_dedup,
        "rows_after": len(df_step2),
        "rows_removed": removed_duplicates,
        "pct_removed": (removed_duplicates / rows_before_dedup) * 100,
        "rationale": "Exact duplicate rows represent technical logging or retransmission glitches.",
    })
    print(f"  Step 2: Removed {removed_duplicates:,} exact duplicate rows ({audit_steps[-1]['pct_removed']:.2f}%).")

    # Step 3: Column Normalization & Type Conversion
    df_step3 = normalize_column_names(df_step2)
    df_step3["customer_id"] = df_step3["customer_id"].astype("int64").astype(str)
    df_step3["invoice"] = df_step3["invoice"].astype(str).str.strip()
    df_step3["stock_code"] = df_step3["stock_code"].astype(str).str.strip()
    df_step3["invoice_date"] = pd.to_datetime(df_step3["invoice_date"])

    # Step 4: Handle Invalid / Non-positive Prices
    # Business Rationale: Items with Price <= 0 (e.g. manual adjustments, promotional giveaways, damaged stock)
    # do not reflect market transactions and distort customer monetary valuation.
    rows_before_price = len(df_step3)
    valid_price_mask = df_step3["price"] > 0
    df_step4 = df_step3[valid_price_mask].copy()
    removed_invalid_price = rows_before_price - len(df_step4)
    audit_steps.append({
        "step": "3. Remove Non-positive Prices",
        "rows_before": rows_before_price,
        "rows_after": len(df_step4),
        "rows_removed": removed_invalid_price,
        "pct_removed": (removed_invalid_price / rows_before_price) * 100,
        "rationale": "Prices <= 0 indicate zero-value samples, damaged stock write-offs, or system credits.",
    })
    print(f"  Step 3: Removed {removed_invalid_price:,} rows with Price <= 0 ({audit_steps[-1]['pct_removed']:.2f}%).")

    # Step 5: Flag Cancellations & Returns
    # Business Rationale: Returns are identified by Invoices starting with 'C' and negative quantities.
    # Rather than silently dropping returns, we flag them explicitly:
    # - Net customer spend = sum(quantity * price), which correctly subtracts returns.
    # - Positive purchases (quantity > 0) are used for order frequency, basket size, and product analysis.
    df_step4["is_cancellation"] = df_step4["invoice"].str.startswith("C") | (df_step4["quantity"] < 0)
    df_step4["revenue"] = df_step4["quantity"] * df_step4["price"]

    # Valid positive purchases subset
    valid_purchases = df_step4[~df_step4["is_cancellation"] & (df_step4["quantity"] > 0)].copy()

    audit_summary = {
        "initial_rows": initial_rows,
        "cleaned_total_rows": len(df_step4),
        "total_valid_purchases": len(valid_purchases),
        "total_cancellations": int(df_step4["is_cancellation"].sum()),
        "unique_customers_retained": int(df_step4["customer_id"].nunique()),
        "cleaning_steps": audit_steps,
    }

    print("-" * 80)
    print(f"[Preprocessing] Data cleaning complete:")
    print(f"  - Total Cleaned Transactions (including returns): {len(df_step4):,}")
    print(f"  - Positive Purchase Transactions: {len(valid_purchases):,}")
    print(f"  - Return/Cancellation Transactions: {audit_summary['total_cancellations']:,}")
    print(f"  - Unique Customers Tracked: {audit_summary['unique_customers_retained']:,}")
    print("-" * 80)

    return df_step4, valid_purchases, audit_summary


if __name__ == "__main__":
    from data_loading import load_raw_data
    raw, _ = load_raw_data()
    cleaned, purchases, audit = clean_ecommerce_data(raw)
