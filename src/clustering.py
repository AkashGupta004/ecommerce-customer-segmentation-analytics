"""
clustering.py
=============
Module responsible for customer segmentation using Machine Learning:
- Feature transformation (handling skewness with log1p)
- Feature standardization (StandardScaler)
- Outlier analysis
- Cluster evaluation (Elbow Method & Silhouette Scores)
- Model fitting (K-Means & Hierarchical Agglomerative Clustering)
- Empirical cluster profiling & business segment mapping
- Dimensionality reduction via PCA (Principal Component Analysis)
"""

from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA


def prepare_clustering_features(
    rfm_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, StandardScaler, Dict[str, Any]]:
    """
    Prepares RFM features for clustering by analyzing skewness, applying log1p
    transformation to normalize distributions, and standardizing to mean=0, std=1.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Eligible customer DataFrame (with monetary > 0 and frequency > 0).
    feature_cols : Optional[List[str]]
        Features to include, defaults to ['recency', 'frequency', 'monetary'].

    Returns
    -------
    Tuple[np.ndarray, StandardScaler, Dict[str, Any]]
        - X_scaled: Standardized numerical array ready for clustering.
        - scaler: Fitted StandardScaler instance.
        - transformation_report: Skewness before and after transformation.
    """
    if feature_cols is None:
        feature_cols = ["recency", "frequency", "monetary"]

    raw_features = rfm_df[feature_cols].copy()

    # Calculate initial skewness
    skew_before = {col: float(skew(raw_features[col])) for col in feature_cols}

    # Apply log1p transformation to mitigate severe right-skewness
    log_features = np.log1p(raw_features)
    skew_after = {col: float(skew(log_features[col])) for col in feature_cols}

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(log_features)

    transformation_report = {
        "features": feature_cols,
        "skew_before": skew_before,
        "skew_after": skew_after,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    print("[Clustering] Feature Transformation Summary:")
    for f in feature_cols:
        print(f"  - {f:<12}: Skew before = {skew_before[f]:>7.2f}  ==>  Skew after = {skew_after[f]:>6.2f}")

    return X_scaled, scaler, transformation_report


def evaluate_cluster_candidates(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates candidate cluster counts using the Elbow Method (Inertia/WCSS)
    and Silhouette Scores.

    Parameters
    ----------
    X_scaled : np.ndarray
        Standardized feature matrix.
    k_range : range, default range(2, 9)
        Range of k values to evaluate.
    random_state : int, default 42
        Reproducibility seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary with k_values, inertias, and silhouette_scores.
    """
    print(f"[Clustering] Evaluating K-Means candidate clusters k={k_range.start}..{k_range.stop-1}...")
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_scaled, labels))
        inertias.append(inertia)
        silhouette_scores.append(sil)
        print(f"  k = {k}: WCSS / Inertia = {inertia:>9.1f} | Silhouette Score = {sil:.4f}")

    # Determine recommended k (highest silhouette or optimal elbow)
    best_k_sil = k_values[int(np.argmax(silhouette_scores))]
    print(f"[Clustering] Highest Silhouette Score at k = {best_k_sil} ({max(silhouette_scores):.4f})")

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "best_k_silhouette": best_k_sil,
    }


def fit_final_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray]:
    """
    Fits the final K-Means model with fixed seed and assigns cluster labels.
    """
    print(f"[Clustering] Fitting final K-Means model with k = {n_clusters} (random_state={random_state})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"[Clustering] Final K-Means fitted. Silhouette Score = {sil:.4f}")
    return kmeans, labels


def fit_hierarchical_clustering(
    X_scaled: np.ndarray,
    n_clusters: int = 4
) -> Tuple[AgglomerativeClustering, np.ndarray, float]:
    """
    Fits Agglomerative Hierarchical Clustering (Ward linkage) as an alternative benchmark.
    """
    print(f"[Clustering] Fitting Hierarchical Agglomerative Clustering (k = {n_clusters}, linkage='ward')...")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = agg.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels))
    print(f"[Clustering] Hierarchical Clustering fitted. Silhouette Score = {sil:.4f}")
    return agg, labels, sil


def compute_pca(
    X_scaled: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA]:
    """
    Projects standardized customer features into 2D PCA space for visualization.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    print(f"[PCA] Projected into {n_components} dimensions.")
    print(f"  - Explained Variance PC1: {explained_var[0]*100:.2f}%")
    print(f"  - Explained Variance PC2: {explained_var[1]*100:.2f}%")
    print(f"  - Total Variance Explained: {sum(explained_var)*100:.2f}%")
    return X_pca, pca


def assign_business_segments(
    rfm_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_col: str = "cluster"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Interprets empirical cluster metrics (Recency, Frequency, Monetary, AOV)
    and dynamically maps numeric cluster labels to actionable business personas.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Eligible customer DataFrame.
    cluster_labels : np.ndarray
        Array of integer cluster assignments.
    cluster_col : str, default 'cluster'
        Name of the cluster column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - segmented_df: DataFrame with cluster numbers and assigned business persona names.
        - segment_summary: Structured table showing customer counts, revenue shares, and averages.
    """
    df = rfm_df.copy()
    df[cluster_col] = cluster_labels

    # Compute empirical profile per cluster
    profile = df.groupby(cluster_col).agg(
        num_customers=("customer_id", "count"),
        total_revenue=("monetary", "sum"),
        avg_recency=("recency", "mean"),
        median_recency=("recency", "median"),
        avg_frequency=("frequency", "mean"),
        median_frequency=("frequency", "median"),
        avg_monetary=("monetary", "mean"),
        median_monetary=("monetary", "median"),
        avg_aov=("avg_order_value", "mean"),
        avg_tenure=("customer_tenure", "mean"),
    ).reset_index()

    total_customers = profile["num_customers"].sum()
    total_rev = profile["total_revenue"].sum()

    profile["pct_customers"] = ((profile["num_customers"] / total_customers) * 100).round(2)
    profile["pct_revenue"] = ((profile["total_revenue"] / total_rev) * 100).round(2)

    # Dynamic Persona Assignment based on empirical rankings
    # Rank clusters by monetary and recency
    # Highest Spend + Frequency + Low Recency -> Champions / High-Value Loyalists
    # Low Spend + Low Frequency + High Recency -> Hibernating / Churned
    # Moderate Spend + Low Recency -> Potential Loyalists / Recent Shoppers
    # High Recency + Past purchases -> At-Risk Customers
    persona_map = {}
    for idx, row in profile.iterrows():
        c_id = int(row[cluster_col])
        rec = row["avg_recency"]
        freq = row["avg_frequency"]
        mon = row["avg_monetary"]

        if freq >= 8 or mon >= 4000:
            name = "Champions / High-Value Loyal"
        elif rec >= 250 and freq <= 2:
            name = "Hibernating / Churned"
        elif rec >= 150 and freq >= 2:
            name = "At-Risk Customers"
        elif rec < 100 and freq >= 3:
            name = "Loyal Customers"
        elif rec < 100 and freq <= 2:
            name = "Recent / Potential Loyalists"
        else:
            name = f"Segment {c_id}"
        persona_map[c_id] = name

    # Ensure unique names if two overlap
    counts: Dict[str, int] = {}
    final_persona_map: Dict[int, str] = {}
    for c_id, name in persona_map.items():
        if name in counts:
            counts[name] += 1
            final_persona_map[c_id] = f"{name} (Tier {counts[name]})"
        else:
            counts[name] = 1
            final_persona_map[c_id] = name

    df["segment_name"] = df[cluster_col].map(final_persona_map)
    profile["segment_name"] = profile[cluster_col].map(final_persona_map)

    # Clean profile table for presentation
    cols_order = [
        cluster_col, "segment_name", "num_customers", "pct_customers",
        "total_revenue", "pct_revenue", "avg_recency", "avg_frequency",
        "avg_monetary", "avg_aov", "avg_tenure"
    ]
    summary_table = profile[cols_order].sort_values(by="total_revenue", ascending=False).reset_index(drop=True)

    # Round numerical metrics
    summary_table["total_revenue"] = summary_table["total_revenue"].round(2)
    summary_table["avg_recency"] = summary_table["avg_recency"].round(1)
    summary_table["avg_frequency"] = summary_table["avg_frequency"].round(1)
    summary_table["avg_monetary"] = summary_table["avg_monetary"].round(2)
    summary_table["avg_aov"] = summary_table["avg_aov"].round(2)
    summary_table["avg_tenure"] = summary_table["avg_tenure"].round(1)

    return df, summary_table
