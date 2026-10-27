"""
main.py
=======
Master orchestration script for the E-Commerce Customer Segmentation and Business Analytics project.

Executes all analytical stages sequentially:
1. Data Loading & Combining (both 2009-2010 and 2010-2011 sheets)
2. Preprocessing & Data Cleaning with full audit reporting
3. Customer-level RFM Feature Engineering & Quintile Scoring
4. K-Means Optimization (Elbow & Silhouette) and Hierarchical Clustering Comparison
5. Empirical Business Segment Assignment & Summary Table Generation
6. Dimensionality Reduction (PCA 2D projection)
7. Visualizations Generation (All 8+ publication-quality figures saved to outputs/figures/)
8. Export of Final Data Products (customer_segments.csv, summary reports, and cleaned transactions)
"""

import os
import sys
import json
import time

# Ensure src modules are resolvable
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np

from data_loading import load_raw_data, inspect_data, print_inspection_report
from preprocessing import clean_ecommerce_data
from rfm_analysis import compute_rfm_features
from clustering import (
    prepare_clustering_features,
    evaluate_cluster_candidates,
    fit_final_kmeans,
    fit_hierarchical_clustering,
    compute_pca,
    assign_business_segments,
)
from visualization import (
    plot_monthly_revenue_trend,
    plot_revenue_by_country,
    plot_top_products_and_customers,
    plot_distributions_and_frequency,
    plot_seasonal_patterns,
    plot_elbow_and_silhouette,
    plot_pca_clusters_2d,
    plot_segment_summary_cards,
)


def run_full_pipeline():
    start_total = time.time()
    print("=" * 80)
    print("   STARTING END-TO-END E-COMMERCE SEGMENTATION & ANALYTICS PIPELINE")
    print("=" * 80)

    # Output paths
    outputs_dir = os.path.join(current_dir, "outputs")
    reports_dir = os.path.join(outputs_dir, "reports")
    figures_dir = os.path.join(outputs_dir, "figures")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # STAGE 1: Data Loading & Initial Inspection
    # --------------------------------------------------------------------------
    print("\n>>> STAGE 1: Data Loading & Inspection...")
    raw_data, _ = load_raw_data(use_cache=True)
    inspection_summary = inspect_data(raw_data)
    print_inspection_report(inspection_summary)

    # --------------------------------------------------------------------------
    # STAGE 2: Preprocessing & Data Cleaning
    # --------------------------------------------------------------------------
    print("\n>>> STAGE 2: Data Cleaning & Returns Auditing...")
    cleaned_df, valid_purchases_df, audit_summary = clean_ecommerce_data(raw_data)

    # Save audit report to JSON
    with open(os.path.join(reports_dir, "cleaning_audit_report.json"), "w") as f:
        json.dump(audit_summary, f, indent=2)

    # Save cleaned transactions for dashboard performance
    cleaned_cache = os.path.join(outputs_dir, "cleaned_transactions.parquet")
    cleaned_df.to_parquet(cleaned_cache, index=False)
    print(f"  Saved cleaned transactions to: {cleaned_cache}")

    # --------------------------------------------------------------------------
    # STAGE 3: Exploratory Data Analysis & Visualizations
    # --------------------------------------------------------------------------
    print("\n>>> STAGE 3: Generating Exploratory Visualizations...")
    plot_monthly_revenue_trend(cleaned_df)
    plot_revenue_by_country(cleaned_df)
    plot_top_products_and_customers(valid_purchases_df)
    plot_seasonal_patterns(valid_purchases_df)
    print(f"  Saved EDA charts to: {figures_dir}")

    # --------------------------------------------------------------------------
    # STAGE 4: RFM Feature Engineering
    # --------------------------------------------------------------------------
    print("\n>>> STAGE 4: Customer-level RFM Feature Engineering...")
    rfm_all, eligible_rfm = compute_rfm_features(cleaned_df, valid_purchases_df)
    plot_distributions_and_frequency(valid_purchases_df, eligible_rfm)

    # --------------------------------------------------------------------------
    # STAGE 5: Machine Learning Clustering & Evaluation
    # --------------------------------------------------------------------------
    print("\n>>> STAGE 5: Clustering, Model Evaluation & Optimization...")
    feature_cols = ["recency", "frequency", "monetary"]
    X_scaled, scaler, trans_report = prepare_clustering_features(eligible_rfm, feature_cols)

    # Evaluate k=2 to 8
    eval_results = evaluate_cluster_candidates(X_scaled, k_range=range(2, 9), random_state=42)
    plot_elbow_and_silhouette(eval_results)

    # Fit final K-Means (k=4)
    chosen_k = 4
    kmeans_model, km_labels = fit_final_kmeans(X_scaled, n_clusters=chosen_k, random_state=42)

    # Comparative evaluation: Hierarchical Agglomerative Clustering
    agg_model, agg_labels, agg_sil = fit_hierarchical_clustering(X_scaled, n_clusters=chosen_k)
    km_sil = eval_results["silhouette_scores"][eval_results["k_values"].index(chosen_k)]

    model_comparison = {
        "kmeans_k4": {"silhouette": km_sil, "inertia": eval_results["inertias"][eval_results["k_values"].index(chosen_k)]},
        "hierarchical_k4": {"silhouette": agg_sil, "linkage": "ward"},
    }
    with open(os.path.join(reports_dir, "model_comparison.json"), "w") as f:
        json.dump(model_comparison, f, indent=2)

    # --------------------------------------------------------------------------
    # STAGE 6: Customer Segment Interpretation & Persona Assignment
    # --------------------------------------------------------------------------
    print("\n>>> STAGE 6: Segment Interpretation & Profiling...")
    segmented_eligible, summary_table = assign_business_segments(eligible_rfm, km_labels)

    # Merge segmented labels back to full customer universe (including 102 defected/refunded)
    final_customer_segments = pd.merge(
        rfm_all,
        segmented_eligible[["customer_id", "cluster", "segment_name"]],
        on="customer_id",
        how="left"
    )
    final_customer_segments["cluster"] = final_customer_segments["cluster"].fillna(-1).astype(int)
    final_customer_segments["segment_name"] = final_customer_segments["segment_name"].fillna("Defected / Inactive Refund")

    # Export customer_segments.csv
    segments_csv_path = os.path.join(outputs_dir, "customer_segments.csv")
    final_customer_segments.to_csv(segments_csv_path, index=False)
    print(f"  Exported customer segments to: {segments_csv_path}")

    # Export segment summary table
    summary_csv_path = os.path.join(reports_dir, "segment_summary.csv")
    summary_table.to_csv(summary_csv_path, index=False)
    print(f"  Exported segment summary table to: {summary_csv_path}")

    # Plot summary comparison
    plot_segment_summary_cards(summary_table)

    # --------------------------------------------------------------------------
    # STAGE 7: Dimensionality Reduction via PCA
    # --------------------------------------------------------------------------
    print("\n>>> STAGE 7: PCA Dimensionality Reduction...")
    X_pca, pca = compute_pca(X_scaled, n_components=2, random_state=42)
    plot_pca_clusters_2d(
        X_pca=X_pca,
        labels=km_labels,
        segment_names=segmented_eligible["segment_name"].tolist(),
        explained_variance=pca.explained_variance_ratio_.tolist()
    )

    # --------------------------------------------------------------------------
    # Pipeline Summary
    # --------------------------------------------------------------------------
    elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print("                     PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Pipeline Runtime : {elapsed:.2f} seconds")
    print(f"Total Transactions     : {len(raw_data):,}")
    print(f"Cleaned Transactions   : {len(cleaned_df):,}")
    print(f"Total Customer Profiles: {len(final_customer_segments):,}")
    print(f"Optimal Cluster Count  : k = {chosen_k} (Silhouette = {km_sil:.4f})")
    print("-" * 80)
    print("CUSTOMER SEGMENT SUMMARY TABLE:")
    print(summary_table.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
