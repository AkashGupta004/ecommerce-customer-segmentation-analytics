"""
data_loading.py
================
Module responsible for locating, loading, and performing initial inspection
of the Online Retail II dataset across both operational years (2009-2010 and 2010-2011).

Key Functions:
- locate_dataset(): Resolves dataset filepath across varied workspace layouts.
- load_raw_data(): Reads both Excel sheets and concatenates them vertically.
- inspect_data(): Computes initial descriptive statistics, counts, nulls, and integrity checks.
"""

import os
import time
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np


def locate_dataset(provided_path: Optional[str] = None) -> str:
    """
    Locates the Online Retail II dataset file on the filesystem.
    Checks user-specified path followed by known workspace candidate paths.

    Parameters
    ----------
    provided_path : Optional[str]
        A custom file path provided by the caller.

    Returns
    -------
    str
        Validated absolute path to the dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found in any candidate path.
    """
    if provided_path and os.path.exists(provided_path):
        return os.path.abspath(provided_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workspace_root = os.path.dirname(base_dir)

    candidate_paths = [
        # Relative to project directory
        os.path.join(base_dir, "data", "online_retail_II.xlsx"),
        os.path.join(base_dir, "data", "Online Retail II.xlsx"),
        os.path.join(base_dir, "Online Retail II.xlsx"),
        os.path.join(base_dir, "online_retail_II.xlsx"),
        # Relative to workspace root
        os.path.join(workspace_root, "Ecommerce", "data", "online_retail_II.xlsx"),
        os.path.join(workspace_root, "Ecommerce", "Online Retail II.xlsx"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"Unable to locate 'online_retail_II.xlsx'. Searched locations:\n"
        + "\n".join(f" - {p}" for p in candidate_paths)
    )


def load_raw_data(
    file_path: Optional[str] = None,
    use_cache: bool = True,
    cache_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Loads both sheets from the Online Retail II Excel file and combines them
    vertically into a single pandas DataFrame named `data`.

    Parameters
    ----------
    file_path : Optional[str]
        Optional custom path to the Excel file.
    use_cache : bool, default True
        If True and a pre-saved parquet/feather cache file exists, loads from cache
        for faster execution while retaining exact data fidelity.
    cache_path : Optional[str]
        Custom path for caching the combined DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
        - data: The vertically concatenated pandas DataFrame.
        - sheets_dict: Dictionary containing the original individual sheet DataFrames.
    """
    resolved_path = locate_dataset(file_path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_cache = os.path.join(base_dir, "outputs", "raw_combined_cache.parquet")
    actual_cache = cache_path or default_cache

    # 1. Fast cache loading if available and requested
    if use_cache and os.path.exists(actual_cache):
        try:
            print(f"[Data Loading] Loading combined data from optimized cache: {actual_cache}")
            data = pd.read_parquet(actual_cache)
            print(f"[Data Loading] Successfully loaded {len(data):,} records from cache.")
            return data, {}
        except Exception as e:
            print(f"[Data Loading] Cache read failed ({e}). Falling back to Excel reading.")

    # 2. Read from Excel
    print(f"[Data Loading] Reading Excel file from: {resolved_path}")
    print("[Data Loading] Loading sheets: 'Year 2009-2010' and 'Year 2010-2011'...")
    start_time = time.time()

    # Required specification: pd.read_excel("online_retail_II.xlsx", sheet_name=None)
    sheets_dict = pd.read_excel(resolved_path, sheet_name=None)
    load_duration = time.time() - start_time
    print(f"[Data Loading] Loaded {len(sheets_dict)} sheets in {load_duration:.1f} seconds.")

    for sheet_name, sheet_df in sheets_dict.items():
        print(f"  - Sheet '{sheet_name}': {sheet_df.shape[0]:,} rows, {sheet_df.shape[1]} columns")

    # Combine sheet DataFrames vertically using pd.concat()
    data = pd.concat(sheets_dict.values(), ignore_index=True)
    print(f"[Data Loading] Combined DataFrame shape: {data.shape[0]:,} rows, {data.shape[1]} columns")

    # Ensure consistent data types for parquet compatibility (Invoice has mixed int and 'C' strings)
    data["Invoice"] = data["Invoice"].astype(str)
    data["StockCode"] = data["StockCode"].astype(str)
    data["Description"] = data["Description"].astype(str)
    data["Country"] = data["Country"].astype(str)

    # Save to parquet cache for subsequent sub-second loading
    try:
        os.makedirs(os.path.dirname(actual_cache), exist_ok=True)
        data.to_parquet(actual_cache, index=False)
        print(f"[Data Loading] Cached raw combined dataset to: {actual_cache}")
    except Exception as cache_err:
        print(f"[Data Loading] Warning: Could not write cache file ({cache_err}). Continuing.")

    return data, sheets_dict


def inspect_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs initial exploratory inspection of the dataset.

    Returns dictionary containing:
    - shape: (rows, cols)
    - columns: list of column names
    - dtypes: dictionary of column types
    - missing_values: counts and percentages per column
    - duplicate_count: count of exact duplicate rows
    - unique_counts: unique counts for customers, invoices, products, countries
    - date_range: min and max InvoiceDate
    - cancellation_stats: count of cancellation invoices and negative quantities
    """
    total_rows = len(data)
    missing_counts = data.isnull().sum().to_dict()
    missing_pct = {k: (v / total_rows) * 100 for k, v in missing_counts.items()}

    cancellations = data["Invoice"].astype(str).str.startswith("C")
    neg_quantity = data["Quantity"] < 0
    zero_price = data["Price"] <= 0

    inspection_results = {
        "total_rows": total_rows,
        "total_columns": data.shape[1],
        "columns": data.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_pct,
        "duplicate_rows": int(data.duplicated().sum()),
        "unique_customers": int(data["Customer ID"].nunique()),
        "unique_invoices": int(data["Invoice"].nunique()),
        "unique_products": int(data["StockCode"].nunique()),
        "unique_countries": int(data["Country"].nunique()),
        "date_min": str(data["InvoiceDate"].min()),
        "date_max": str(data["InvoiceDate"].max()),
        "cancellation_invoices_count": int(cancellations.sum()),
        "negative_quantity_count": int(neg_quantity.sum()),
        "invalid_or_zero_price_count": int(zero_price.sum()),
    }

    return inspection_results


def print_inspection_report(summary: Dict[str, Any]) -> None:
    """Prints a structured, formatted data inspection report."""
    print("=" * 80)
    print("               DATA LOADING & INITIAL INSPECTION REPORT")
    print("=" * 80)
    print(f"Total Transactions : {summary['total_rows']:,}")
    print(f"Total Features     : {summary['total_columns']}")
    print(f"Date Range         : {summary['date_min']} to {summary['date_max']}")
    print(f"Exact Duplicates   : {summary['duplicate_rows']:,} rows")
    print("-" * 80)
    print("CARDINALITY:")
    print(f"  - Unique Customers : {summary['unique_customers']:,}")
    print(f"  - Unique Invoices  : {summary['unique_invoices']:,}")
    print(f"  - Unique Products  : {summary['unique_products']:,}")
    print(f"  - Unique Countries : {summary['unique_countries']:,}")
    print("-" * 80)
    print("DATA INTEGRITY & ANOMALIES:")
    print(f"  - Missing Customer IDs    : {summary['missing_counts']['Customer ID']:,} ({summary['missing_percentages']['Customer ID']:.2f}%)")
    print(f"  - Cancellation Invoices   : {summary['cancellation_invoices_count']:,}")
    print(f"  - Negative Quantities     : {summary['negative_quantity_count']:,}")
    print(f"  - Zero or Negative Prices : {summary['invalid_or_zero_price_count']:,}")
    print("=" * 80)


if __name__ == "__main__":
    df, _ = load_raw_data()
    summary = inspect_data(df)
    print_inspection_report(summary)
