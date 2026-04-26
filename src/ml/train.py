"""
Model Training Module
======================
Trains an ensemble fraud detector consisting of:
    1. XGBoostClassifier  — supervised, learns from historical fraud labels
    2. IsolationForest    — unsupervised, flags statistical anomalies

Training pipeline (in order):
    1. Build 12-column feature matrix (tabular + graph features)
    2. Stratified 80/20 train/val split (preserves 1.45% fraud rate in each set)
    3. Apply SMOTE to training set ONLY (never validation — data leakage!)
    4. Train XGBoost with scale_pos_weight to handle residual imbalance
    5. Train Isolation Forest with contamination matching our fraud rate
    6. Evaluate both models on the untouched validation set
    7. Save model artifacts to data/models/

Expected metrics on this synthetic dataset:
    XGBoost  Val AUC > 0.90,  Precision > 0.80,  Recall > 0.75
    Isolation Forest anomaly detection rate ~0.75–0.80

Usage (as module):
    from src.ml.train import train
    train("data/synthetic/events.csv", "data/graphs/account_graph.pkl", "data/models/")

Usage (CLI via scripts/train_model.py):
    python scripts/train_model.py --data data/synthetic/events.csv \
                                  --graph data/graphs/account_graph.pkl \
                                  --output data/models/
"""

import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── path bootstrap so this file is importable from project root ───────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.feature_engineering import build_feature_matrix
from src.ml.utils import apply_smote, compute_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train(
    data_path: str,
    graph_path: str,
    output_dir: str,
) -> dict:
    """
    Full training pipeline — returns a dict of evaluation metrics.

    Args:
        data_path:  Path to events.csv
        graph_path: Path to account_graph.pkl
        output_dir: Directory to save model artifacts

    Returns:
        dict with keys: xgb_auc, xgb_precision, xgb_recall, xgb_f1,
                        iso_detection_rate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Build feature matrix ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Building feature matrix ...")
    logger.info("=" * 60)
    df = build_feature_matrix(data_path, graph_path)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    feature_names = list(X.columns)
    logger.info(f"Feature names: {feature_names}")

    # ── 2. Stratified train/val split ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Stratified 80/20 train / val split ...")
    logger.info("=" * 60)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,          # preserves fraud rate in both splits
        random_state=42,
    )
    logger.info(
        f"Train: {len(X_train):,} rows ({y_train.sum()} fraud)  |  "
        f"Val: {len(X_val):,} rows ({y_val.sum()} fraud)"
    )

    # ── 3. SMOTE on training set ONLY ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Applying SMOTE to training set ...")
    logger.info("=" * 60)
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # ── 4. Train XGBoost ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Training XGBoostClassifier ...")
    logger.info("=" * 60)
    spw = compute_class_weights(y_train)   # scale_pos_weight from original (pre-SMOTE) distribution

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=42,
        verbosity=0,           # suppress XGBoost's own INFO spam
    )
    xgb_model.fit(
        X_train_sm, y_train_sm,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("XGBoost training complete.")

    # ── 5. Evaluate XGBoost on validation set ─────────────────────────────────
    y_prob_val = xgb_model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= 0.5).astype(int)

    xgb_auc       = roc_auc_score(y_val, y_prob_val)
    xgb_precision = precision_score(y_val, y_pred_val, zero_division=0)
    xgb_recall    = recall_score(y_val, y_pred_val, zero_division=0)
    xgb_f1        = f1_score(y_val, y_pred_val, zero_division=0)

    print("\n" + "─" * 50)
    print("  XGBoost Validation Results")
    print("─" * 50)
    print(f"  AUC       : {xgb_auc:.4f}")
    print(f"  Precision : {xgb_precision:.4f}")
    print(f"  Recall    : {xgb_recall:.4f}")
    print(f"  F1        : {xgb_f1:.4f}")
    print(classification_report(y_val, y_pred_val, target_names=["Legit", "Fraud"], zero_division=0))

    # ── 6. Train Isolation Forest ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Training Isolation Forest ...")
    logger.info("=" * 60)
    # Isolation Forest is unsupervised — train on ORIGINAL training data (no SMOTE needed)
    # contamination = expected proportion of anomalies, matching our fraud rate
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.015,   # matches our ~1.45% fraud rate
        random_state=42,
    )
    iso_model.fit(X_train)     # trained on original (imbalanced) set — unsupervised
    logger.info("Isolation Forest training complete.")

    # ── 7. Evaluate Isolation Forest ──────────────────────────────────────────
    # predict() returns -1 for anomalies (fraud), +1 for normal (legit)
    iso_preds_val = iso_model.predict(X_val)
    # Convert to binary: anomaly (-1) → 1 (fraud), normal (+1) → 0 (legit)
    iso_binary = np.where(iso_preds_val == -1, 1, 0)

    # Detection rate: of actual fraud cases, how many did Isolation Forest flag?
    actual_fraud_mask = y_val.values == 1
    iso_detection_rate = iso_binary[actual_fraud_mask].mean()

    print("\n" + "─" * 50)
    print("  Isolation Forest Validation Results")
    print("─" * 50)
    print(f"  Contamination     : 0.015")
    print(f"  Fraud detected    : {iso_binary[actual_fraud_mask].sum()} / {actual_fraud_mask.sum()}")
    print(f"  Detection rate    : {iso_detection_rate:.4f}")

    # ── 8. Save model artifacts ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Saving model artifacts ...")
    logger.info("=" * 60)

    xgb_path = output_path / "xgboost_fraud.json"
    iso_path  = output_path / "isolation_forest.pkl"

    xgb_model.save_model(str(xgb_path))
    logger.info(f"XGBoost model saved → {xgb_path}")

    with open(iso_path, "wb") as f:
        pickle.dump(iso_model, f)
    logger.info(f"Isolation Forest saved → {iso_path}")

    # Also save feature names so the scorer knows the column order
    feature_names_path = output_path / "feature_names.pkl"
    with open(feature_names_path, "wb") as f:
        pickle.dump(feature_names, f)
    logger.info(f"Feature names saved  → {feature_names_path}")

    metrics = {
        "xgb_auc": round(xgb_auc, 4),
        "xgb_precision": round(xgb_precision, 4),
        "xgb_recall": round(xgb_recall, 4),
        "xgb_f1": round(xgb_f1, 4),
        "iso_detection_rate": round(float(iso_detection_rate), 4),
    }

    print("\n" + "=" * 50)
    print("  Training Complete — Summary")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print("=" * 50)

    return metrics


# ─── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics = train(
        data_path="data/synthetic/events.csv",
        graph_path="data/graphs/account_graph.pkl",
        output_dir="data/models/",
    )
    print("\nStep 3.4 complete ✓")
