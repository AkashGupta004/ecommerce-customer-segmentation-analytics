"""
visualization.py
================
Module responsible for generating publication-quality visualizations for Exploratory
Data Analysis (EDA), RFM distributions, K-Means clustering evaluation, PCA projection,
and customer business segment comparisons.

Features:
- Standardized visual design system (palette, readable typography, formatted ticks).
- Automated export to `outputs/figures/` (PNG, 300 DPI).
- Dual support for static Matplotlib/Seaborn and interactive Plotly charts.
"""

import os
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np


# Visual configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#D1D5DB",
    "axes.linewidth": 1.2,
    "grid.color": "#E5E7EB",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

PALETTE_PRIMARY = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626", "#0891B2", "#4F46E5"]
SEGMENT_COLORS = {
    "Champions / High-Value Loyal": "#059669",
    "Champions": "#059669",
    "Loyal Customers": "#2563EB",
    "Potential Loyalists": "#0891B2",
    "Recent / Potential Loyalists": "#0891B2",
    "Recent Customers": "#7C3AED",
    "At-Risk Customers": "#D97706",
    "Hibernating / Churned": "#DC2626",
    "Defected / Net Refund": "#6B7280",
}


def get_figures_dir() -> str:
    """Returns absolute path to outputs/figures directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


# ==============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA) CHARTS
# ==============================================================================

def plot_monthly_revenue_trend(cleaned_df: pd.DataFrame, save_name: str = "monthly_revenue_trend.png") -> plt.Figure:
    """
    Plots monthly revenue trend across the 2-year timeline (2009-12 to 2011-12),
    highlighting holiday season Q4 peaks and year-over-year revenue trajectory.
    """
    df = cleaned_df.copy()
    df["year_month"] = df["invoice_date"].dt.to_period("M")
    monthly = df.groupby("year_month").agg(
        net_revenue=("revenue", "sum"),
        order_count=("invoice", "nunique")
    ).reset_index()
    monthly["year_month_str"] = monthly["year_month"].astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)

    # Revenue bars
    bars = ax1.bar(
        range(len(monthly)),
        monthly["net_revenue"] / 1000,
        color="#2563EB",
        alpha=0.85,
        width=0.55,
        label="Net Revenue (£ Thousands)"
    )
    ax1.set_title("Monthly Net Revenue & Order Volume Trends (2009 - 2011)", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("Year-Month", fontsize=12, labelpad=10)
    ax1.set_ylabel("Net Revenue (£ Thousands)", fontsize=12, color="#1E3A8A")
    ax1.set_xticks(range(len(monthly)))
    ax1.set_xticklabels(monthly["year_month_str"], rotation=45, ha="right", fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x:,.0f}k"))

    # Order volume secondary axis
    ax2 = ax1.twinx()
    line = ax2.plot(
        range(len(monthly)),
        monthly["order_count"],
        color="#DC2626",
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Unique Invoices"
    )
    ax2.set_ylabel("Order Count", fontsize=12, color="#991B1B")
    ax2.grid(False)

    # Annotate peak month
    max_rev_idx = monthly["net_revenue"].idxmax()
    max_rev = monthly.loc[max_rev_idx, "net_revenue"]
    ax1.annotate(
        f"Peak: £{max_rev:,.0f}\n({monthly.loc[max_rev_idx, 'year_month_str']})",
        xy=(max_rev_idx, max_rev / 1000),
        xytext=(max_rev_idx - 1.5, (max_rev / 1000) * 1.05),
        arrowprops=dict(facecolor="#1E3A8A", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10,
        fontweight="bold",
        color="#1E3A8A"
    )

    fig.tight_layout()
    fig.savefig(os.path.join(get_figures_dir(), save_name), dpi=300)
    plt.close(fig)
    return fig


def plot_revenue_by_country(cleaned_df: pd.DataFrame, save_name: str = "revenue_by_country.png") -> plt.Figure:
    """
    Plots top 10 countries by net revenue and the UK vs. International revenue contribution.
    """
    country_rev = cleaned_df.groupby("country")["revenue"].sum().sort_values(ascending=False).reset_index()
    top10 = country_rev.head(10).copy()

    total_rev = country_rev["revenue"].sum()
    uk_rev = country_rev.loc[country_rev["country"] == "United Kingdom", "revenue"].sum()
    intl_rev = total_rev - uk_rev

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300, gridspec_kw={"width_ratios": [2, 1]})

    # Top 10 bar chart
    bars = ax1.barh(top10["country"][::-1], top10["revenue"][::-1] / 1000, color="#059669", alpha=0.85)
    ax1.set_title("Top 10 Countries by Net Revenue", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Net Revenue (£ Thousands)", fontsize=11)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x:,.0f}k"))

    for bar in bars:
        w = bar.get_width()
        ax1.text(w + (top10['revenue'].max()/1000)*0.01, bar.get_y() + bar.get_height()/2, f"£{w:,.1f}k",
                 va="center", fontsize=9, fontweight="bold")

    # Donut chart UK vs International
    shares = [uk_rev, intl_rev]
    labels = [f"UK ({uk_rev/total_rev*100:.1f}%)", f"International ({intl_rev/total_rev*100:.1f}%)"]
    ax2.pie(shares, labels=labels, colors=["#2563EB", "#D97706"], startangle=140,
            wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2})
    ax2.set_title("Revenue Share: UK vs. Global", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(os.path.join(get_figures_dir(), save_name), dpi=300)
    plt.close(fig)
    return fig


def plot_top_products_and_customers(
    valid_purchases_df: pd.DataFrame,
    save_name: str = "top_products_and_customers.png"
) -> plt.Figure:
    """
    Plots Top 10 Products by revenue and Top 10 Customers by revenue side by side.
    """
    top_prod = valid_purchases_df.groupby("description")["revenue"].sum().nlargest(10).reset_index()
    top_cust = valid_purchases_df.groupby("customer_id")["revenue"].sum().nlargest(10).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Top products
    # Clean product description length
    top_prod["short_desc"] = top_prod["description"].astype(str).str.slice(0, 28)
    bars1 = ax1.barh(top_prod["short_desc"][::-1], top_prod["revenue"][::-1] / 1000, color="#7C3AED", alpha=0.85)
    ax1.set_title("Top 10 Products by Revenue", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Revenue (£ Thousands)", fontsize=11)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x:,.0f}k"))

    # Top customers
    bars2 = ax2.barh([f"Cust #{c}" for c in top_cust["customer_id"][::-1]], top_cust["revenue"][::-1] / 1000, color="#0891B2", alpha=0.85)
    ax2.set_title("Top 10 Customers by Revenue (Pareto Concentration)", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Revenue (£ Thousands)", fontsize=11)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x:,.0f}k"))

    fig.tight_layout()
    fig.savefig(os.path.join(get_figures_dir(), save_name), dpi=300)
    plt.close(fig)
    return fig


def plot_distributions_and_frequency(
    valid_purchases_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    save_name: str = "distributions_eda.png"
) -> plt.Figure:
    """
    Plots Order Quantity distribution, Average Order Value (AOV) distribution,
    and Customer Purchase Frequency distribution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # 1. Order Quantity per line item (log scale)
    sns.histplot(valid_purchases_df["quantity"], bins=50, log_scale=True, color="#2563EB", ax=axes[0], kde=True)
    axes[0].set_title("Order Quantity per Line Item (Log Scale)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Quantity (Units)", fontsize=11)
    axes[0].set_ylabel("Transaction Count", fontsize=11)

    # 2. Average Order Value (AOV) distribution (capped at 99th percentile for clean display)
    aov_filtered = rfm_df.loc[(rfm_df["avg_order_value"] > 0) & (rfm_df["avg_order_value"] <= rfm_df["avg_order_value"].quantile(0.98)), "avg_order_value"]
    sns.histplot(aov_filtered, bins=40, color="#059669", ax=axes[1], kde=True)
    axes[1].set_title(f"Customer Average Order Value (AOV) (≤ 98th Pct)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("AOV (£)", fontsize=11)
    axes[1].set_ylabel("Customer Count", fontsize=11)
    axes[1].axvline(aov_filtered.median(), color="#DC2626", linestyle="--", label=f"Median: £{aov_filtered.median():.1f}")
    axes[1].legend()

    # 3. Customer Purchase Frequency (Invoices per customer)
    freq_filtered = rfm_df.loc[(rfm_df["frequency"] > 0) & (rfm_df["frequency"] <= 25), "frequency"]
    sns.histplot(freq_filtered, discrete=True, color="#7C3AED", ax=axes[2])
    axes[2].set_title("Purchase Frequency Distribution (Orders ≤ 25)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Total Lifetime Invoices", fontsize=11)
    axes[2].set_ylabel("Customer Count", fontsize=11)

    fig.tight_layout()
    fig.savefig(os.path.join(get_figures_dir(), save_name), dpi=300)
    plt.close(fig)
    return fig


def plot_seasonal_patterns(valid_purchases_df: pd.DataFrame, save_name: str = "seasonal_patterns.png") -> plt.Figure:
    """
    Plots purchasing patterns across Day of Week and Hour of Day.
    """
    df = valid_purchases_df.copy()
    df["day_name"] = df["invoice_date"].dt.day_name()
    df["hour"] = df["invoice_date"].dt.hour

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]  # Saturday typically no trading
    day_counts = df.groupby("day_name")["revenue"].sum().reindex(days_order).dropna()
    hour_counts = df.groupby("hour")["revenue"].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=300)

    # Day of week
    ax1.bar(day_counts.index, day_counts.values / 1000, color="#4F46E5", alpha=0.85)
    ax1.set_title("Net Revenue by Day of Week", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Revenue (£ Thousands)", fontsize=11)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x:,.0f}k"))

    # Hour of day
    ax2.plot(hour_counts.index, hour_counts.values / 1000, marker="o", color="#D97706", linewidth=2.5)
    ax2.fill_between(hour_counts.index, hour_counts.values / 1000, color="#D97706", alpha=0.15)
    ax2.set_title("Net Revenue by Hour of Day", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Hour (24-hour clock)", fontsize=11)
    ax2.set_ylabel("Revenue (£ Thousands)", fontsize=11)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x:,.0f}k"))
    ax2.set_xticks(range(7, 21))

    fig.tight_layout()
    fig.savefig(os.path.join(get_figures_dir(), save_name), dpi=300)
    plt.close(fig)
    return fig


# ==============================================================================
# SECTION 6, 7 & 9: CLUSTERING, EVALUATION & PCA VISUALIZATIONS
# ==============================================================================

def plot_elbow_and_silhouette(eval_results: Dict[str, Any], save_name: str = "clustering_evaluation.png") -> plt.Figure:
    """
    Plots the Elbow Curve (WCSS / Inertia) and Silhouette Scores across candidate k values.
    """
    k_vals = eval_results["k_values"]
    inertias = eval_results["inertias"]
    silhouettes = eval_results["silhouette_scores"]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

    # Elbow curve (Inertia)
    color1 = "#2563EB"
    ax1.plot(k_vals, inertias, marker="o", linewidth=2.5, color=color1, label="Inertia (WCSS)")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12, labelpad=8)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Silhouette Score secondary axis
    ax2 = ax1.twinx()
    color2 = "#059669"
    ax2.plot(k_vals, silhouettes, marker="s", linewidth=2.5, linestyle="--", color=color2, label="Silhouette Score")
    ax2.set_ylabel("Silhouette Score", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    # Title & optimal point marker
    best_k = eval_results["best_k_silhouette"]
    best_sil = max(silhouettes)
    ax2.scatter([best_k], [best_sil], color="#DC2626", s=120, zorder=5, label=f"Max Silhouette (k={best_k})")

    plt.title("Cluster Count Optimization: Elbow Method vs. Silhouette Score", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(get_figures_dir(), save_name), dpi=300)
    plt.close(fig)
    return fig


def plot_pca_clusters_2d(
    X_pca: np.ndarray,
    labels: np.ndarray,
    segment_names: List[str],
    explained_variance: List[float],
    save_name: str = "pca_customer_clusters.png"
) -> plt.Figure:
    """
    Plots customer clusters projected onto the first two Principal Components (PCA).
    """
    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Segment": segment_names
    })

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

    # Scatter plot
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Segment",
        palette=PALETTE_PRIMARY[:len(set(segment_names))],
        alpha=0.65,
        s=35,
        edgecolor=None,
        ax=ax
    )

    ev1 = explained_variance[0] * 100
    ev2 = explained_variance[1] * 100
    ax.set_title("Customer Segmentation 2D Projection (PCA)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel(f"Principal Component 1 ({ev1:.1f}% Variance Explained)", fontsize=12)
    ax.set_ylabel(f"Principal Component 2 ({ev2:.1f}% Variance Explained)", fontsize=12)
    ax.legend(title="Customer Segment", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(get_figures_dir(), save_name), dpi=300)
    plt.close(fig)
    return fig


def plot_segment_summary_cards(
    summary_table: pd.DataFrame,
    save_name: str = "segment_profiles_comparison.png"
) -> plt.Figure:
    """
    Plots comparison bar charts of Customer Count % vs Revenue % and Average Monetary vs Recency.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # 1. Customer share vs Revenue share
    x = np.arange(len(summary_table))
    width = 0.35
    axes[0].bar(x - width/2, summary_table["pct_customers"], width, label="% of Total Customers", color="#3B82F6", alpha=0.9)
    axes[0].bar(x + width/2, summary_table["pct_revenue"], width, label="% of Total Revenue", color="#10B981", alpha=0.9)
    axes[0].set_title("Customer Share vs. Revenue Contribution", fontsize=14, fontweight="bold", pad=12)
    axes[0].set_ylabel("Percentage (%)", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_table["segment_name"], rotation=20, ha="right", fontsize=9)
    axes[0].legend()

    # 2. Average Spend vs Average Recency
    axes[1].scatter(
        summary_table["avg_recency"],
        summary_table["avg_monetary"],
        s=summary_table["pct_revenue"] * 45 + 100,  # size by revenue share
        c=PALETTE_PRIMARY[:len(summary_table)],
        alpha=0.85,
        edgecolors="black"
    )
    axes[1].set_title("Segment Value Map: Spend vs. Recency (Bubble Size = Revenue %)", fontsize=14, fontweight="bold", pad=12)
    axes[1].set_xlabel("Average Recency (Days since last order - Lower is better)", fontsize=11)
    axes[1].set_ylabel("Average Monetary Spend (£)", fontsize=11)
    axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f"£{y:,.0f}"))

    for _, row in summary_table.iterrows():
        axes[1].annotate(
            row["segment_name"],
            (row["avg_recency"], row["avg_monetary"]),
            xytext=(0, 10),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9,
            ha="center"
        )

    fig.tight_layout()
    fig.savefig(os.path.join(get_figures_dir(), save_name), dpi=300)
    plt.close(fig)
    return fig
