"""
Feature Engineering Module
===========================
Combines tabular event features from events.csv with graph-structural
features from GraphFeatureExtractor to produce a single, unified feature
matrix ready for model training.

The resulting DataFrame contains 12 columns:
    Tabular (7):  transaction_amount, account_age_days, device_change_count,
                  ip_country_mismatch, velocity_1hr, avg_txn_amount_30d,
                  failed_login_count_24hr
    Graph   (4):  degree_centrality, component_size, shared_device_count,
                  ip_reuse_count
    Label   (1):  is_fraud

All identifier columns (event_id, account_id, timestamp, device_id,
ip_address, fraud_type) are dropped — they are not model features.

Usage:
    from src.ml.feature_engineering import build_feature_matrix

    df = build_feature_matrix(
        events_path="data/synthetic/events.csv",
        graph_path="data/graphs/account_graph.pkl",
    )
    # df.shape → (10000, 12)
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path so `src.*` imports resolve whether this
# file is run directly (`python src/ml/feature_engineering.py`) or imported
# as a module from a script at project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.graph_features import GraphFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Columns that must exist in the raw CSV for feature engineering to proceed
_REQUIRED_COLUMNS = [
    "event_id",
    "account_id",
    "timestamp",
    "transaction_amount",
    "account_age_days",
    "device_change_count",
    "ip_country_mismatch",
    "velocity_1hr",
    "avg_txn_amount_30d",
    "failed_login_count_24hr",
    "device_id",
    "ip_address",
    "is_fraud",
]

# Identifier columns to strip before returning the feature matrix
_DROP_COLUMNS = [
    "event_id",
    "account_id",
    "timestamp",
    "device_id",
    "ip_address",
    "fraud_type",   # present in synthetic data; not a feature for training
]


def build_feature_matrix(events_path: str, graph_path: str) -> pd.DataFrame:
    """
    Load events, extract graph features, and return a clean feature matrix.

    The function uses an account-level cache so that GraphFeatureExtractor.extract()
    is called once per unique account — not once per row. For 10,000 events
    across ~1,000 unique accounts this reduces graph lookups by ~10×.

    Args:
        events_path: Path to events.csv (output of scripts/generate_data.py)
        graph_path:  Path to account_graph.pkl (output of graph_builder.py)

    Returns:
        pd.DataFrame with exactly 12 columns and one row per event.

    Raises:
        FileNotFoundError: if either input file is missing
        ValueError:        if required columns are absent from the CSV
    """
    # ── 1. Load raw events ────────────────────────────────────────────────────
    events_file = Path(events_path)
    if not events_file.exists():
        raise FileNotFoundError(f"Events CSV not found: {events_file}")

    logger.info(f"Loading events from {events_file} ...")
    df = pd.read_csv(events_file)
    logger.info(f"Loaded {len(df):,} raw events.")

    # ── 2. Validate schema ────────────────────────────────────────────────────
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Events CSV is missing required columns: {missing}")

    # ── 3. Initialise graph feature extractor ─────────────────────────────────
    logger.info("Initialising GraphFeatureExtractor ...")
    extractor = GraphFeatureExtractor(graph_path)

    # ── 4. Build account-level feature cache ─────────────────────────────────
    # Computing graph features per row would be 10× slower than per unique account.
    unique_accounts = df["account_id"].unique()
    logger.info(
        f"Extracting graph features for {len(unique_accounts):,} unique accounts ..."
    )

    cache: dict[str, dict] = {}
    for account_id in unique_accounts:
        cache[account_id] = extractor.extract(account_id)

    logger.info("Graph feature extraction complete.")

    # ── 5. Map graph features back onto every row ─────────────────────────────
    df["degree_centrality"]    = df["account_id"].map(lambda a: cache[a]["degree_centrality"])
    df["component_size"]       = df["account_id"].map(lambda a: cache[a]["component_size"])
    df["shared_device_count"]  = df["account_id"].map(lambda a: cache[a]["shared_device_count"])
    df["ip_reuse_count"]       = df["account_id"].map(lambda a: cache[a]["ip_reuse_count"])

    # ── 6. Drop identifier columns ────────────────────────────────────────────
    # Only drop columns that actually exist to avoid errors on partial datasets
    cols_to_drop = [c for c in _DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # ── 7. Final integrity checks ─────────────────────────────────────────────
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values detected in feature matrix:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("No null values found in feature matrix.")

    logger.info(
        f"Feature matrix ready — shape: {df.shape} | "
        f"fraud rows: {df['is_fraud'].sum():,} / {len(df):,} "
        f"({df['is_fraud'].mean() * 100:.2f}%)"
    )

    return df


# ─── Quick verification ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    df = build_feature_matrix(
        events_path="data/synthetic/events.csv",
        graph_path="data/graphs/account_graph.pkl",
    )

    print("\n" + "=" * 55)
    print("  Feature Matrix — Verification")
    print("=" * 55)
    print(f"  Shape          : {df.shape}")
    print(f"  Columns        : {list(df.columns)}")
    print(f"\n  Fraud breakdown:")
    print(df["is_fraud"].value_counts().to_string())
    print(f"\n  Null values    : {df.isnull().sum().sum()}")
    print(f"\n  Sample row (fraud=1):")
    fraud_row = df[df["is_fraud"] == 1].iloc[0]
    print(fraud_row.to_string())
    print("\n" + "=" * 55)
    print("Step 3.2 complete ✓")
