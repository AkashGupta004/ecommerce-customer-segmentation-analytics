# 🛒 E-Commerce Customer Segmentation & Business Analytics

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A complete, production-grade, end-to-end **Customer Segmentation and Business Analytics** system developed on the **Online Retail II** dataset (UK online retail enterprise, Dec 2009 – Dec 2011).

This project integrates customer-level **RFM (Recency, Frequency, Monetary)** feature engineering, statistical transformations for right-skewed transaction data, **unsupervised machine learning (K-Means & Hierarchical Agglomerative Clustering)**, **PCA 2D projection**, and an interactive **Streamlit executive dashboard**.

---

## 📑 Table of Contents
1. [Project Overview & Business Problem](#-project-overview--business-problem)
2. [Project Architecture & Repository Structure](#-project-architecture--repository-structure)
3. [Technologies Used](#-technologies-used)
4. [Dataset Description](#-dataset-description)
5. [Quickstart & Installation Guide](#-quickstart--installation-guide)
6. [How to Run](#-how-to-run)
   - [Running the Master Pipeline](#1-running-the-master-pipeline)
   - [Launching the Interactive Streamlit Dashboard](#2-launching-the-interactive-streamlit-dashboard)
   - [Running the Jupyter Notebook](#3-running-the-jupyter-notebook)
7. [End-to-End Methodology](#-end-to-end-methodology)
8. [Empirical Results & Customer Segments](#-empirical-results--customer-segments)
9. [Actionable Retention & Marketing Playbook](#-actionable-retention--marketing-playbook)
10. [Generated Visualizations](#-generated-visualizations)
11. [Limitations & Future Roadmap](#-limitations--future-roadmap)

---

## 🎯 Project Overview & Business Problem

### The Business Challenge
In digital commerce, treating all customers identically results in wasted marketing expenditure, generic discount discounting that erodes profit margins, and unmitigated customer churn. 

### Solution
By analyzing empirical behavioral patterns—**Recency** (how recently a customer bought), **Frequency** (how often they buy), and **Monetary Value** (how much net revenue they generate)—this project automatically segments **5,939 unique customer entities** across **1,067,371 transactions** into actionable business cohorts.

### Key Highlights
* **100% Real-World Execution:** Fully executed and verified on the complete two-year dataset (£16.38M+ net customer sales).
* **Transparent Returns Auditing:** Reconciled **18,390 cancellation / return credit notes**, accurately reflecting net profitability rather than inflated gross sales.
* **Mathematically Grounded ML:** Solved severe feature skewness via $\log(1+x)$ transformations, optimized cluster counts via **Elbow Method (WCSS)** and **Silhouette Analysis** ($k=4$, Silhouette = **0.3657**), and benchmarked against **Hierarchical Clustering**.
* **Pareto Concentration:** Revealed that **20.13% of customers (Champions) generate 73.68% of total company revenue**.
* **Interactive Tooling:** Accompanied by a modern Streamlit application featuring interactive 3D RFM exploration and individual Customer 360 lookup.

---

## 📂 Project Architecture & Repository Structure

```
Ecommerce/
├── data/
│   └── online_retail_II.xlsx             # Raw dataset workbook (Year 2009-2010 & Year 2010-2011)
├── notebooks/
│   └── customer_segmentation.ipynb       # 46-cell end-to-end guided Jupyter Notebook
├── src/
│   ├── __init__.py                       # Package initialization
│   ├── data_loading.py                   # Multi-sheet Excel loader, path resolution & parquet cache
│   ├── preprocessing.py                  # Audit pipeline: nulls, deduplication, returns & prices
│   ├── rfm_analysis.py                   # Customer RFM metrics, engineered ratios & quintiles
│   ├── clustering.py                     # Skewness handling, K-Means, Hierarchical, PCA & Profiling
│   └── visualization.py                  # High-res publication charts (Matplotlib/Seaborn/Plotly)
├── outputs/
│   ├── figures/                          # 8 exported 300-DPI publication charts
│   │   ├── monthly_revenue_trend.png
│   │   ├── revenue_by_country.png
│   │   ├── top_products_and_customers.png
│   │   ├── distributions_eda.png
│   │   ├── seasonal_patterns.png
│   │   ├── clustering_evaluation.png
│   │   ├── pca_customer_clusters.png
│   │   └── segment_profiles_comparison.png
│   ├── reports/                          # Audit summaries & model metrics
│   │   ├── cleaning_audit_report.json
│   │   ├── model_comparison.json
│   │   └── segment_summary.csv
│   ├── cleaned_transactions.parquet      # Pre-cleaned transaction records for instant dashboard load
│   ├── raw_combined_cache.parquet        # Parquet cache of full combined dataset
│   └── customer_segments.csv             # Customer-level features, cluster labels & personas
├── app.py                                # Interactive Streamlit Business Dashboard
├── main.py                               # Master end-to-end CLI pipeline runner
├── build_notebook.py                     # Automated notebook generation script
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Git exclusion rules
└── README.md                             # Comprehensive technical & business documentation
```

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12+ / 3.14 (Anaconda) | Primary computational runtime |
| **Data Manipulation** | Pandas 2.x, NumPy 1.24+ | Vectorized tabular wrangling, grouping, and metrics |
| **Spreadsheet Engine** | OpenPyXL | Ingesting multi-sheet Excel files |
| **Machine Learning** | Scikit-Learn 1.3+, SciPy | K-Means, Agglomerative Clustering, PCA, Silhouette, Skewness |
| **Visualization** | Matplotlib, Seaborn | High-resolution publication figures (300 DPI) |
| **Interactive UI** | Streamlit 1.30+, Plotly 5.15+ | Real-time executive BI dashboard and 3D scatter plots |

---

## 📊 Dataset Description

The dataset is `online_retail_II.xlsx`, an Excel workbook containing retail transactions for a UK-based online gift and homeware retailer between **01/12/2009 and 09/12/2011**:

* **Sheet 1:** `Year 2009-2010` (525,461 records)
* **Sheet 2:** `Year 2010-2011` (541,910 records)
* **Total Transactions:** **1,067,371 rows, 8 columns**

| Column | Type | Description |
| :--- | :--- | :--- |
| `Invoice` | String / Object | 6-digit order identifier. Invoices starting with `'C'` designate cancellations / credit notes. |
| `StockCode` | String / Object | Unique product SKU identifier. |
| `Description` | String | Product name / description. |
| `Quantity` | Integer | Quantity of units ordered per line item (negative for returns). |
| `InvoiceDate` | Datetime | Date and timestamp of transaction generation. |
| `Price` | Float | Unit price (£ Sterling). |
| `Customer ID` | Float / String | Unique 5-digit customer account identifier. |
| `Country` | String | Country of residence of the purchasing entity. |

---

## 🚀 Quickstart & Installation Guide

### Prerequisites
* Python 3.10+ installed (Anaconda or standard Python).
* Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/AkashGupta004/ecommerce-customer-segmentation-analytics.git
cd ecommerce-customer-segmentation-analytics
```

### 2. Create and Activate a Virtual Environment (Recommended)
**On Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run

### 1. Running the Master Pipeline
To execute the complete end-to-end analytical pipeline from data loading, cleaning, feature engineering, clustering, PCA, and figure generation:
```powershell
python main.py
```
* **Runtime:** ~10-15 seconds when loading from cache (or ~75s on first cold run from raw Excel).
* **Outputs Generated:** Automatically updates `outputs/customer_segments.csv`, all figures in `outputs/figures/`, and audit reports in `outputs/reports/`.

### 2. Launching the Interactive Streamlit Dashboard
Launch the executive web application:
```powershell
streamlit run app.py
```
Once launched, open your web browser at `http://localhost:8501`.

**Dashboard Features:**
* **Executive Overview:** Net Revenue (£), Total Customers, Total Invoices, Average Order Value (AOV), and Repeat Rate.
* **Interactive Filters:** Slice by Country, Customer Segment, and Net Spend Range slider.
* **3D RFM Explorer:** Interactive Plotly 3D scatter plot of Recency vs. Frequency vs. Monetary space.
* **Customer 360 Drilldown:** Search any individual Customer ID (e.g. `14911`, `18102`, `14646`, `12346`) to inspect their RFM scores, segment badge, tenure, return rate, and complete line-item transaction history.
* **Strategic Playbook:** Quantified recommendations and Pareto risk evaluation.

### 3. Running the Jupyter Notebook
Open the notebook in VS Code or Jupyter Lab:
```powershell
jupyter notebook notebooks/customer_segmentation.ipynb
```
Follow the 11 sequential sections, each with executable code and accompanying markdown analysis.

---

## 🔬 End-to-End Methodology

### 1. Data Ingestion & Unification
Both Excel sheets (`Year 2009-2010` and `Year 2010-2011`) are extracted via `pd.read_excel(..., sheet_name=None)` and vertically concatenated with `pd.concat()`, yielding **1,067,371 rows**.

### 2. Data Cleaning & Return Auditing
| Step | Rows Before | Rows After | Rows Removed | Business Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **1. Missing Customer IDs** | 1,067,371 | 824,364 | 243,007 (22.77%) | Customer segmentation requires persistent customer identities to track repeat purchases across sessions. |
| **2. Exact Duplicates** | 824,364 | 797,885 | 26,479 (3.21%) | Technical retransmissions or POS duplicate logging that artificially inflate volume. |
| **3. Non-positive Prices** | 797,885 | 797,815 | 70 (0.01%) | Non-commercial entries (damaged stock, zero-value samples, manual credit adjustments). |

#### Treatment of Returns & Cancellations
* Invoices prefixed with `'C'` and negative quantities (**18,390 credit note rows**) are **retained** in `cleaned_transactions.parquet`.
* Line item revenue is calculated as $\text{Revenue} = \text{Quantity} \times \text{Price}$. Because returns have negative quantity, customer net spend correctly deducts refunds, avoiding overestimating customer worth.
* Positive transactions are isolated for basket size and order frequency calculations.

### 3. Customer RFM Feature Engineering
* **Reference Date:** `2011-12-10 12:50:00` (1 day after the latest recorded transaction: `2011-12-09`).
* **Recency ($R$):** Days elapsed since the customer's last transaction: $\text{Recency} = \text{Reference Date} - \max(\text{InvoiceDate})$.
* **Frequency ($F$):** Total count of unique completed purchase invoices.
* **Monetary ($M$):** Net lifetime monetary spend across all transactions.
* **Engineered Ratios:** Average Order Value ($\text{AOV} = M / F$), Customer Tenure (days since first purchase), Average Days Between Purchases, Product Diversity (distinct SKUs), and Return Rate (% items refunded).

### 4. Skewness Mitigation & Normalization
Customer features in e-commerce exhibit severe right-skewness:
* **Recency:** Skewness = `0.91` $\rightarrow$ After $\log(1+x)$: **`-0.47`**
* **Frequency:** Skewness = `12.60` $\rightarrow$ After $\log(1+x)$: **`1.00`**
* **Monetary:** Skewness = `26.64` $\rightarrow$ After $\log(1+x)$: **`0.26`**

Features are subsequently normalized using `StandardScaler` ($\mu = 0, \sigma = 1$).

### 5. Machine Learning Clustering Optimization
* Evaluated candidate cluster counts from $k=2$ to $k=8$:
  - $k=2$: WCSS = `8,499.6`, Silhouette = `0.4395` (Broad active vs inactive split).
  - $k=3$: WCSS = `6,260.7`, Silhouette = `0.3479`.
  - **$k=4$: WCSS = `4,845.9`, Silhouette = `0.3657` (Optimal operational elbow & local silhouette peak)**.
  - $k=5$: WCSS = `4,029.7`, Silhouette = `0.3431`.
  - $k=6$: WCSS = `3,490.5`, Silhouette = `0.3330`.
* **Benchmark:** Hierarchical Agglomerative Clustering (Ward Linkage) at $k=4$ yielded Silhouette = `0.3285`, confirming K-Means achieves superior structural separation.

### 6. Dimensionality Reduction (PCA)
* Projected standardized 3D features onto 2 Principal Components:
  - **PC1:** Explains **76.63%** of variance (aligns with Volume, Frequency & Monetary Spend).
  - **PC2:** Explains **18.52%** of variance (aligns with Recency & Dormancy).
  - **Total Variance Explained:** **95.14%** in 2 dimensions.

---

## 📈 Empirical Results & Customer Segments

The final K-Means model partitions the **5,837 active, eligible customers** into 4 distinct behavioral segments, plus a fifth operational cohort of 102 defected/refund-only accounts:

| Cluster | Segment Persona | Customers | Customer % | Total Net Revenue (£) | Revenue % | Avg Recency | Avg Frequency | Avg Monetary (£) | Avg AOV (£) | Avg Tenure |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3** | **👑 Champions / High-Value Loyal** | 1,175 | 20.13% | **£12,070,863.72** | **73.68%** | 25.6 days | 19.5 orders | £10,273.08 | £474.21 | 619.2 days |
| **1** | **⚠️ At-Risk Customers** | 1,447 | 24.79% | **£2,632,123.42** | **16.07%** | 221.9 days | 5.1 orders | £1,819.02 | £435.95 | 575.2 days |
| **2** | **🌟 Loyal Customers** | 1,257 | 21.54% | **£1,069,355.86** | **6.53%** | 27.0 days | 3.1 orders | £850.72 | £308.76 | 283.6 days |
| **0** | **❄️ Hibernating / Churned** | 1,958 | 33.54% | **£610,285.02** | **3.73%** | 391.7 days | 1.4 orders | £311.69 | £244.99 | 444.7 days |
| **-1** | **Defected / Inactive Refund** | 102 | 1.72% | -£23,491.50 | -0.14% | 280.4 days | 0.0 orders | -£230.31 | £0.00 | 382.1 days |

---

## 💡 Actionable Retention & Marketing Playbook

```
                         CUSTOMER STRATEGY MATRIX
                     High Value               Low Value
               ┌───────────────────────┬───────────────────────┐
   Active      │   👑 CHAMPIONS        │   🌟 LOYAL CUSTOMERS  │
   (<60 days)  │  • White-glove VIP    │  • Product cross-sell │
               │  • No heavy discounts │  • Tier upgrade perks │
               ├───────────────────────┼───────────────────────┤
   Lapsed      │   ⚠️ AT-RISK          │   ❄️ HIBERNATING      │
   (>180 days) │  • High-urgency win-  │  • Automated email    │
               │    back offers (20%)  │    clearance blasts   │
               │  • Direct sales call  │  • Sunset unengaged   │
               └───────────────────────┴───────────────────────┘
```

### 1. Champions (20.1% Customers | 73.7% Revenue)
* **Goal:** Zero churn, enterprise VIP protection.
* **Tactics:** Dedicated account management, exclusive preview access to new seasonal lines, concierge phone support.
* **Caution:** Do NOT discount heavily; they buy based on catalog breadth, reliability, and service.

### 2. Loyal Customers (21.5% Customers | 6.5% Revenue)
* **Goal:** Elevate order value and purchase frequency.
* **Tactics:** Automated product recommendations based on past purchase categories, bundle discounts ("Buy 3 for £X"), free shipping threshold incentives.

### 3. At-Risk Customers (24.8% Customers | 16.1% Revenue)
* **Goal:** Immediate reactivation before permanent defection.
* **Tactics:** Automated 3-stage win-back email sequence with increasing discount incentives (10% $\rightarrow$ 15% $\rightarrow$ 20%), feedback surveys to diagnose why ordering halted.

### 4. Hibernating (33.5% Customers | 3.7% Revenue)
* **Goal:** Low-cost opportunistic re-engagement.
* **Tactics:** Annual clearance sales, warehouse outlet emails. Sunset contacts inactive for >24 months to preserve email sender reputation.

---

## 🖼️ Generated Visualizations

All 8 figures are automatically generated and saved to `outputs/figures/`:
1. `monthly_revenue_trend.png`: 24-month revenue trajectory and order volume overlay.
2. `revenue_by_country.png`: Top 10 countries and UK vs. International donut chart.
3. `top_products_and_customers.png`: Bestselling SKUs and Top 10 customer Pareto concentration.
4. `distributions_eda.png`: Quantity log-distribution, Customer AOV distribution, and Order frequency.
5. `seasonal_patterns.png`: Revenue by Day of Week and Hour of Day.
6. `clustering_evaluation.png`: Dual-axis Elbow Curve (WCSS) and Silhouette Score ($k=2..8$).
7. `pca_customer_clusters.png`: 2D PCA projection of clusters with explained variance ratios.
8. `segment_profiles_comparison.png`: Customer count share vs. Revenue share and Spend vs. Recency bubble map.

---

## 🔮 Limitations & Future Roadmap

### Limitations
1. **Unidentified Checkouts:** 22.8% of transactions lacked a Customer ID and could not be linked to longitudinal customer personas.
2. **Gross Margin vs Revenue:** Analysis is based on net revenue; incorporating Cost of Goods Sold (COGS) would enable segmentation by net margin.

### Future Scope
1. **Predictive CLV Modeling:** Implement BG/NBD (Beta-Geometric / Negative Binomial Distribution) and Gamma-Gamma models to predict 90-day transaction probabilities.
2. **Market Basket Mining:** Implement FP-Growth algorithm to identify frequently co-purchased item pairs for real-time recommendation engines.
3. **CRM Webhook Integration:** Sync segment tags automatically with Klaviyo, Mailchimp, or Salesforce Marketing Cloud.

---

## 📜 License & Acknowledgments
* **Dataset Source:** UCI Machine Learning Repository / Online Retail II.
* **License:** MIT Open Source License.
