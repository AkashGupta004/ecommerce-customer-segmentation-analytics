"""
app.py
======
Interactive Executive Business Dashboard for E-Commerce Customer Segmentation & Business Analytics.
Built with Streamlit & Plotly.

Run via:
    streamlit run app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page config
st.set_page_config(
    page_title="E-Commerce Customer Analytics & Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive look
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dashboard_data():
    """Loads cleaned transactions and customer segments with caching."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    segments_path = os.path.join(base_dir, "outputs", "customer_segments.csv")
    cleaned_path = os.path.join(base_dir, "outputs", "cleaned_transactions.parquet")

    # Fallbacks if running from workspace root
    if not os.path.exists(segments_path):
        segments_path = os.path.join(base_dir, "Ecommerce", "outputs", "customer_segments.csv")
        cleaned_path = os.path.join(base_dir, "Ecommerce", "outputs", "cleaned_transactions.parquet")

    if not os.path.exists(segments_path):
        st.error(f"Customer segments data not found at '{segments_path}'. Please run `python main.py` first.")
        st.stop()

    customers = pd.read_csv(segments_path)
    customers["customer_id"] = customers["customer_id"].astype(str)

    # Load cleaned transactions
    if os.path.exists(cleaned_path):
        transactions = pd.read_parquet(cleaned_path)
    else:
        # Fallback to minimal placeholder if transactions parquet not yet ready
        transactions = pd.DataFrame()

    return customers, transactions


# Load data
customers_df, transactions_df = load_dashboard_data()

# ==============================================================================
# SIDEBAR FILTERS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/isometric/100/shopping-cart-loaded.png", width=64)
st.sidebar.title("Analytics Filters")

# Country filter
all_countries = ["All Countries"] + sorted(customers_df["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Filter by Country", all_countries, index=0)

# Segment filter
all_segments = sorted(customers_df["segment_name"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Customer Segments",
    options=all_segments,
    default=all_segments
)

# Apply filters to customer DataFrame
filtered_cust = customers_df.copy()
if selected_country != "All Countries":
    filtered_cust = filtered_cust[filtered_cust["country"] == selected_country]
if selected_segments:
    filtered_cust = filtered_cust[filtered_cust["segment_name"].isin(selected_segments)]

# Monetary spend filter slider
min_spend, max_spend = float(filtered_cust["monetary"].min()), float(filtered_cust["monetary"].max())
if min_spend < max_spend:
    spend_range = st.sidebar.slider(
        "Net Spend Range (£)",
        min_value=min_spend,
        max_value=max_spend,
        value=(min_spend, max_spend),
        step=50.0
    )
    filtered_cust = filtered_cust[
        (filtered_cust["monetary"] >= spend_range[0]) & (filtered_cust["monetary"] <= spend_range[1])
    ]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Customers:** `{len(filtered_cust):,}` / `{len(customers_df):,}`")
st.sidebar.caption("E-Commerce Customer Segmentation & Analytics Dashboard v1.0")

# ==============================================================================
# HEADER & EXECUTIVE METRICS
# ==============================================================================
st.title("🛒 E-Commerce Customer Segmentation & Executive Analytics")
st.markdown("Comprehensive business intelligence, customer behavior modeling, and retention strategy for Online Retail II.")

total_rev = filtered_cust["monetary"].sum()
total_cust = len(filtered_cust)
total_orders = filtered_cust["frequency"].sum()
avg_aov = filtered_cust["avg_order_value"].mean() if total_cust > 0 else 0
repeat_rate = (filtered_cust["frequency"] > 1).sum() / total_cust * 100 if total_cust > 0 else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Net Revenue</div>
        <div class="metric-value">£{total_rev:,.0f}</div>
        <div class="metric-delta">Net of Returns</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Customers</div>
        <div class="metric-value">{total_cust:,}</div>
        <div class="metric-delta">Identified Entities</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Purchase Invoices</div>
        <div class="metric-value">{int(total_orders):,}</div>
        <div class="metric-delta">Completed Orders</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Order Value (AOV)</div>
        <div class="metric-value">£{avg_aov:,.1f}</div>
        <div class="metric-delta">Per Customer Invoices</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Repeat Rate</div>
        <div class="metric-value">{repeat_rate:.1f}%</div>
        <div class="metric-delta">Multiple Purchases</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==============================================================================
# DASHBOARD TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Overview & Sales Trends",
    "👥 Customer Segments & Cluster Insights",
    "🔎 Customer 360 & Individual Drilldown",
    "💡 Strategic Playbook & Recommendations"
])

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW & SALES TRENDS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Macro Sales Trends & Regional Breakdown")
    col1, col2 = st.columns([3, 2])

    with col1:
        if not transactions_df.empty:
            df_t = transactions_df.copy()
            df_t["year_month"] = pd.to_datetime(df_t["invoice_date"]).dt.to_period("M").astype(str)
            monthly = df_t.groupby("year_month").agg(
                revenue=("revenue", "sum"),
                orders=("invoice", "nunique")
            ).reset_index()

            fig_monthly = px.bar(
                monthly,
                x="year_month",
                y="revenue",
                title="Monthly Net Revenue (£) with Order Volume Overlay",
                labels={"year_month": "Month", "revenue": "Net Revenue (£)"},
                color_discrete_sequence=["#2563EB"]
            )
            fig_monthly.add_trace(
                go.Scatter(
                    x=monthly["year_month"],
                    y=monthly["orders"] * 100,  # scaled for dual visibility
                    name="Orders (x100)",
                    mode="lines+markers",
                    line=dict(color="#DC2626", width=2.5)
                )
            )
            fig_monthly.update_layout(hovermode="x unified", template="plotly_white", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            st.info("Loading transaction details...")

    with col2:
        # Country distribution
        top_countries = filtered_cust.groupby("country")["monetary"].sum().nlargest(10).reset_index()
        fig_country = px.bar(
            top_countries,
            x="monetary",
            y="country",
            orientation="h",
            title="Top 10 Countries by Net Revenue",
            labels={"monetary": "Revenue (£)", "country": "Country"},
            color="monetary",
            color_continuous_scale="Viridis"
        )
        fig_country.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
        st.plotly_chart(fig_country, use_container_width=True)

    # Top products table
    if not transactions_df.empty:
        st.write("#### 🏆 Top 10 Best Selling Products by Net Revenue")
        top_prod = (
            transactions_df[transactions_df["quantity"] > 0]
            .groupby(["stock_code", "description"])
            .agg(total_revenue=("revenue", "sum"), total_units=("quantity", "sum"))
            .nlargest(10, "total_revenue")
            .reset_index()
        )
        top_prod["total_revenue"] = top_prod["total_revenue"].apply(lambda x: f"£{x:,.2f}")
        top_prod["total_units"] = top_prod["total_units"].apply(lambda x: f"{x:,}")
        st.dataframe(top_prod, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: CUSTOMER SEGMENTS & CLUSTER INSIGHTS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Machine Learning Customer Segmentation Analysis")

    seg_c1, seg_c2 = st.columns([1, 1])

    with seg_c1:
        # Donut Chart: Customer Share by Segment
        seg_counts = filtered_cust["segment_name"].value_counts().reset_index()
        seg_counts.columns = ["segment_name", "count"]
        fig_donut = px.pie(
            seg_counts,
            names="segment_name",
            values="count",
            hole=0.45,
            title="Customer Distribution by Persona Segment",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_donut.update_traces(textinfo="percent+label")
        fig_donut.update_layout(showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with seg_c2:
        # Revenue Share by Segment
        seg_rev = filtered_cust.groupby("segment_name")["monetary"].sum().reset_index()
        fig_rev = px.pie(
            seg_rev,
            names="segment_name",
            values="monetary",
            hole=0.45,
            title="Revenue Contribution by Persona Segment",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_rev.update_traces(textinfo="percent+label")
        fig_rev.update_layout(showlegend=False)
        st.plotly_chart(fig_rev, use_container_width=True)

    # 3D RFM Interactive Scatter Plot
    st.write("#### 🌌 3D Interactive RFM Behavioral Space")
    sample_size = min(3000, len(filtered_cust))
    sampled_rfm = filtered_cust.sample(sample_size, random_state=42) if len(filtered_cust) > sample_size else filtered_cust

    fig_3d = px.scatter_3d(
        sampled_rfm,
        x="recency",
        y="frequency",
        z="monetary",
        color="segment_name",
        hover_data=["customer_id", "avg_order_value", "country"],
        labels={"recency": "Recency (Days)", "frequency": "Frequency (Orders)", "monetary": "Monetary (£)"},
        title=f"3D Customer Segmentation (Interactive Sample of {len(sampled_rfm):,} Customers)",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.7,
        log_z=True,  # Log monetary to compress outliers for great 3D visualization
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig_3d, use_container_width=True)

    # Detailed Summary Table
    st.write("#### 📋 Segment Metrics Breakdown Table")
    summary = (
        filtered_cust.groupby("segment_name")
        .agg(
            customers=("customer_id", "count"),
            total_revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_aov=("avg_order_value", "mean"),
            avg_tenure=("customer_tenure", "mean"),
        )
        .reset_index()
    )
    total_c = summary["customers"].sum()
    total_r = summary["total_revenue"].sum()
    summary["pct_customers"] = (summary["customers"] / total_c * 100).round(1).astype(str) + "%"
    summary["pct_revenue"] = (summary["total_revenue"] / total_r * 100).round(1).astype(str) + "%"
    summary["total_revenue"] = summary["total_revenue"].apply(lambda x: f"£{x:,.2f}")
    summary["avg_monetary"] = summary["avg_monetary"].apply(lambda x: f"£{x:,.2f}")
    summary["avg_aov"] = summary["avg_aov"].apply(lambda x: f"£{x:,.2f}")
    summary["avg_recency"] = summary["avg_recency"].round(1).astype(str) + " days"
    summary["avg_frequency"] = summary["avg_frequency"].round(1).astype(str) + " orders"
    summary["avg_tenure"] = summary["avg_tenure"].round(1).astype(str) + " days"

    st.dataframe(summary, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 3: CUSTOMER 360 & DRILLDOWN
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Customer 360 Profile Explorer")
    search_id = st.text_input("Enter Customer ID (e.g. 14911, 12346, 17841, 18102, 14646):", value="14911")

    if search_id:
        cust_match = customers_df[customers_df["customer_id"] == search_id.strip()]
        if not cust_match.empty:
            c = cust_match.iloc[0]
            st.success(f"Customer Found: **ID #{c['customer_id']}** — Assigned Persona: **{c['segment_name']}**")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Recency", f"{c['recency']} days ago")
            p2.metric("Purchase Frequency", f"{int(c['frequency'])} orders")
            p3.metric("Net Lifetime Spend", f"£{c['monetary']:,.2f}")
            p4.metric("Avg Order Value", f"£{c['avg_order_value']:,.2f}")

            p5, p6, p7, p8 = st.columns(4)
            p5.metric("Country", f"{c['country']}")
            p6.metric("Customer Tenure", f"{c['customer_tenure']} days")
            p7.metric("Product Diversity", f"{int(c.get('product_diversity', 0))} items")
            p8.metric("Return Rate", f"{c.get('return_rate', 0):.1f}%")

            # Show transaction history if available
            if not transactions_df.empty:
                cust_trans = transactions_df[transactions_df["customer_id"] == search_id.strip()]
                st.write(f"#### Transaction History ({len(cust_trans):,} line items)")
                st.dataframe(
                    cust_trans[["invoice", "invoice_date", "stock_code", "description", "quantity", "price", "revenue"]].head(100),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning(f"Customer ID '{search_id}' not found in dataset.")

# ------------------------------------------------------------------------------
# TAB 4: STRATEGIC RECOMMENDATIONS & ACTIONABLE PLAYBOOK
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Actionable Business Strategy & Retention Playbook")

    st.markdown("""
    Based on empirical clustering of **5,939 unique customers** and **£17.8M+ in net retail transactions**,
    here is the data-driven marketing and retention playbook:
    """)

    strat1, strat2 = st.columns(2)

    with strat1:
        st.markdown("""
        ### 👑 1. Champions / High-Value Loyalists
        * **Who they are:** Low recency (<30 days), high frequency (8+ orders), massive monetary spend.
        * **Revenue Impact:** Generate the vast majority (>50%) of all business profits.
        * **Action Items:**
          - **Dedicated Account Management:** White-glove concierge customer support.
          - **VIP Loyalty & Early Access:** Offer priority access to new collections and holiday catalogs.
          - **Surprise & Delight:** Free premium gifts or handwritten thank-you notes.
          - **Do NOT discount aggressively:** They buy on value, reliability, and service, not discount coupons.

        ---
        ### 🚀 2. Recent Customers / Potential Loyalists
        * **Who they are:** High recency (bought recently) but lower frequency (1-2 orders).
        * **Action Items:**
          - **Automated Onboarding Sequence:** Post-purchase educational email series (how to use, styling ideas).
          - **Second-Purchase Incentive:** Time-limited 10% discount on their second purchase within 30 days.
          - **Product Recommendations:** Curated suggestions based on the category of their first purchase.
        """)

    with strat2:
        st.markdown("""
        ### ⚠️ 3. At-Risk Customers
        * **Who they are:** Used to order frequently or spend high amounts, but haven't purchased in 150-300+ days.
        * **Risk:** High probability of defection to competitors.
        * **Action Items:**
          - **Win-Back Campaigns:** High-incentive promo codes ("We miss you — enjoy 20% off your next order").
          - **Customer Feedback Survey:** Short 2-question survey to understand why they stopped buying.
          - **New Arrivals Showcase:** Re-engage with catalog updates in their favorite historical categories.

        ---
        ### ❄️ 4. Hibernating / Churned Customers
        * **Who they are:** High recency (300-700+ days), single or very few small purchases.
        * **Action Items:**
          - **Aggressive Clearance Promotions:** Annual or seasonal clearance sales.
          - **Low-Cost Automated Channels:** Email remarketing only (avoid expensive direct mail or ad retargeting).
          - **List Hygiene:** Sunset unengaged contacts after 24 months to maintain high email deliverability.
        """)

    st.markdown("---")
    st.write("#### ⚖️ Pareto Risk & Revenue Concentration")
    top_10pct_cust = int(len(customers_df) * 0.1)
    top_10pct_rev = customers_df.nlargest(top_10pct_cust, "monetary")["monetary"].sum()
    total_all_rev = customers_df["monetary"].sum()
    top_10pct_share = (top_10pct_rev / total_all_rev) * 100

    st.info(f"""
    **Pareto Principle Observation:** The top **10% of customers ({top_10pct_cust:,} accounts)** contribute
    **{top_10pct_share:.1f}% of total net revenue (£{top_10pct_rev:,.0f})**.
    Losing even 5% of Champion tier accounts would severely impact annual company EBITDA.
    Retention must be prioritized above new acquisition.
    """)
