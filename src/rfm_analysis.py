"""
rfm_analysis.py
===============
Module responsible for computing customer-level RFM (Recency, Frequency, Monetary) metrics,
engineering behavioral e-commerce features, and assigning benchmark RFM quintile scores.

Features Engineered (per Customer ID):
- recency: Days since most recent transaction relative to reference date (max date + 1 day).
- frequency: Count of distinct valid purchase invoices.
- monetary: Net total monetary spend (positive purchases minus returns/cancellations).
- gross_monetary: Total spend on positive purchases only.
- total_quantity: Net total physical units purchased.
- avg_order_value: Average net spending per purchase invoice (monetary / frequency).
- product_diversity: Count of distinct products (stock codes) purchased.
- customer_tenure: Days between customer's first purchase and reference date.
- avg_days_between_purchases: (customer_tenure - recency) / (frequency - 1) for repeat buyers.
- return_count: Count of cancellation/return invoices.
- return_rate: Ratio of cancelled items to total items ordered.
"""

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np


def compute_rfm_features(
    cleaned_df: pd.DataFrame,
    valid_purchases_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes comprehensive customer-level RFM and behavioral metrics.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        All cleaned transactions including returns.
    valid_purchases_df : pd.DataFrame
        Subset containing only valid positive purchase transactions.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - rfm_df: Customer-level DataFrame for all 5,942 customers.
        - eligible_rfm_df: Subset of customers with net_spend > 0 and frequency > 0 (for ML clustering).
    """
    print("[RFM Analysis] Computing customer-level features...")

    # Reference Date: 1 day after the latest transaction in the dataset
    max_date = cleaned_df["invoice_date"].max()
    reference_date = max_date + pd.Timedelta(days=1)
    print(f"  Latest transaction date: {max_date}")
    print(f"  Calculated reference date: {reference_date}")

    # 1. Base Net Monetary & Net Quantity from all cleaned transactions
    net_agg = cleaned_df.groupby("customer_id").agg(
        net_monetary=("revenue", "sum"),
        net_quantity=("quantity", "sum"),
        total_rows=("invoice", "count"),
        country=("country", lambda x: x.mode()[0] if len(x) > 0 else "Unknown"),
        first_date=("invoice_date", "min"),
        last_date=("invoice_date", "max"),
        return_invoices=("invoice", lambda x: x[x.str.startswith("C")].nunique()),
        return_quantity=("quantity", lambda x: abs(x[x < 0].sum())),
    ).reset_index()

    # 2. Purchase-specific metrics from valid positive purchases
    purchase_agg = valid_purchases_df.groupby("customer_id").agg(
        frequency=("invoice", "nunique"),
        gross_monetary=("revenue", "sum"),
        gross_quantity=("quantity", "sum"),
        product_diversity=("stock_code", "nunique"),
        first_purchase_date=("invoice_date", "min"),
        last_purchase_date=("invoice_date", "max"),
    ).reset_index()

    # Merge aggregations
    rfm = pd.merge(net_agg, purchase_agg, on="customer_id", how="left")
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["gross_monetary"] = rfm["gross_monetary"].fillna(0.0)
    rfm["gross_quantity"] = rfm["gross_quantity"].fillna(0).astype(int)
    rfm["product_diversity"] = rfm["product_diversity"].fillna(0).astype(int)

    # 3. Recency & Tenure Calculations
    # Recency: days since most recent purchase (or most recent interaction)
    rfm["recency"] = (reference_date - rfm["last_date"]).dt.days
    rfm["customer_tenure"] = (reference_date - rfm["first_date"]).dt.days

    # 4. Engineered Ratios
    # Monetary (Primary): Net monetary spend
    rfm["monetary"] = rfm["net_monetary"].round(2)

    # Average Order Value (AOV)
    rfm["avg_order_value"] = np.where(
        rfm["frequency"] > 0,
        (rfm["monetary"] / rfm["frequency"]).round(2),
        0.0
    )

    # Average days between purchases for multi-purchase customers
    rfm["avg_days_between_purchases"] = np.where(
        rfm["frequency"] > 1,
        ((rfm["customer_tenure"] - rfm["recency"]) / (rfm["frequency"] - 1)).round(1),
        0.0
    )

    # Return rate (% of ordered physical units returned)
    total_physical_ordered = rfm["gross_quantity"] + rfm["return_quantity"]
    rfm["return_rate"] = np.where(
        total_physical_ordered > 0,
        ((rfm["return_quantity"] / total_physical_ordered) * 100).round(2),
        0.0
    )

    # 5. Rule-based RFM Scoring (Quintiles 1 to 5)
    # Filter customers with valid positive spend and at least 1 purchase for quintiles
    eligible_mask = (rfm["monetary"] > 0) & (rfm["frequency"] > 0)
    eligible_rfm = rfm[eligible_mask].copy()

    # Recency quintiles (lower recency = better customer -> higher score 5)
    eligible_rfm["r_score"] = pd.qcut(
        eligible_rfm["recency"], q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # Frequency quintiles (higher frequency = higher score 5, rank method='first' to handle ties)
    eligible_rfm["f_score"] = pd.qcut(
        eligible_rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Monetary quintiles (higher spend = higher score 5)
    eligible_rfm["m_score"] = pd.qcut(
        eligible_rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Composite RFM Score string and numerical average
    eligible_rfm["rfm_score_str"] = (
        eligible_rfm["r_score"].astype(str)
        + eligible_rfm["f_score"].astype(str)
        + eligible_rfm["m_score"].astype(str)
    )
    eligible_rfm["rfm_score_num"] = (
        (eligible_rfm["r_score"] + eligible_rfm["f_score"] + eligible_rfm["m_score"]) / 3.0
    ).round(2)

    # Standard Rule-based RFM Segment Mapping (Industry Standard)
    def assign_rfm_segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Recent Customers"
        elif r >= 3 and f <= 3 and m >= 3:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3:
            return "At-Risk Customers"
        elif r <= 2 and f <= 2 and m >= 3:
            return "Cannot Lose Them"
        elif r <= 2 and f <= 2:
            return "Hibernating / Churned"
        else:
            return "Promising / Needs Attention"

    eligible_rfm["rfm_rule_segment"] = eligible_rfm.apply(assign_rfm_segment, axis=1)

    # Merge RFM scores back to main rfm DataFrame for full coverage
    rfm = pd.merge(
        rfm,
        eligible_rfm[["customer_id", "r_score", "f_score", "m_score", "rfm_score_str", "rfm_score_num", "rfm_rule_segment"]],
        on="customer_id",
        how="left"
    )
    rfm["rfm_rule_segment"] = rfm["rfm_rule_segment"].fillna("Defected / Net Refund")

    print(f"  Total customer profiles generated: {len(rfm):,}")
    print(f"  Eligible customers for ML clustering (Net Spend > 0, Orders >= 1): {len(eligible_rfm):,}")
    print(f"  Ineligible customers (Net Spend <= 0 or 0 orders): {len(rfm) - len(eligible_rfm):,}")

    return rfm, eligible_rfm


if __name__ == "__main__":
    from data_loading import load_raw_data
    from preprocessing import clean_ecommerce_data
    raw, _ = load_raw_data()
    cleaned, purchases, _ = clean_ecommerce_data(raw)
    rfm, eligible = compute_rfm_features(cleaned, purchases)
    print(eligible[["recency", "frequency", "monetary", "rfm_rule_segment"]].head())
