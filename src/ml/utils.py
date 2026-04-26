"""
ML Utilities Module
====================
Small collection of helpers used during model training to handle the
severe class imbalance typical of real-world fraud datasets (~1.5% fraud).

Two functions:
    apply_smote(X, y)         → balanced X_res, y_res (training set only)
    compute_class_weights(y)  → scale_pos_weight ratio for XGBoost

Why class imbalance matters:
    A naive model that predicts "not fraud" for every row achieves 98.5%
    accuracy — yet catches zero fraud. These two techniques work together
    to force the model to learn the minority (fraud) class properly:

    SMOTE:             Synthesises new fraud examples by interpolating
                       between existing ones in feature space. The model
                       sees a richer, more varied fraud signal.

    scale_pos_weight:  Tells XGBoost to penalise missing a fraud case
                       more than falsely flagging a legitimate one.
                       Value = count(negatives) / count(positives) ≈ 65.

CRITICAL — SMOTE must only be applied to the TRAINING SET.
    Applying it to the validation or test set creates data leakage:
    the model would be evaluated on synthetic points derived from the
    same distribution it was trained on, giving falsely inflated recall.
"""

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def apply_smote(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority (fraud) class using SMOTE.

    SMOTE works by:
        1. Picking a minority sample at random
        2. Finding its k nearest minority neighbours in feature space
        3. Drawing a new synthetic point on the line segment between them

    This produces a balanced dataset without simply duplicating rows,
    which would cause the model to memorise specific fraud examples
    rather than learn generalised fraud patterns.

    Args:
        X:            Feature matrix (training set only — never validation)
        y:            Binary labels (0 = legit, 1 = fraud)
        random_state: Seed for reproducibility

    Returns:
        X_res, y_res: Balanced feature matrix and labels as numpy arrays
    """
    neg_count = int((y == 0).sum())
    pos_count = int((y == 1).sum())
    logger.info(
        f"Applying SMOTE — before: {neg_count:,} legit / {pos_count:,} fraud "
        f"(ratio 1:{neg_count // pos_count})"
    )

    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)

    new_pos = int((y_res == 1).sum())
    new_neg = int((y_res == 0).sum())
    logger.info(
        f"After SMOTE — {new_neg:,} legit / {new_pos:,} fraud "
        f"(balanced at 1:1)"
    )

    return X_res, y_res


def compute_class_weights(y: pd.Series | np.ndarray) -> float:
    """
    Compute the scale_pos_weight value for XGBoost.

    XGBoost's scale_pos_weight parameter is the ratio of negatives to
    positives. Setting it tells the model: "each fraud case counts as
    much as scale_pos_weight legitimate cases." This biases gradient
    updates toward correct fraud classification.

    Formula: count(negative) / count(positive)
    Example: 9850 / 150 ≈ 65.7

    Args:
        y: Binary label array (0 = legit, 1 = fraud)

    Returns:
        float: The scale_pos_weight ratio
    """
    neg = int((np.array(y) == 0).sum())
    pos = int((np.array(y) == 1).sum())

    if pos == 0:
        raise ValueError("No positive (fraud) samples found — cannot compute class weights.")

    weight = neg / pos
    logger.info(
        f"Class weight ratio (scale_pos_weight): "
        f"{neg:,} negatives / {pos:,} positives = {weight:.2f}"
    )
    return weight


# ─── Quick verification ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    from src.ml.feature_engineering import build_feature_matrix
    from sklearn.model_selection import train_test_split

    print("\nLoading feature matrix ...")
    df = build_feature_matrix(
        events_path="data/synthetic/events.csv",
        graph_path="data/graphs/account_graph.pkl",
    )

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n" + "=" * 55)
    print("  Utils — Verification")
    print("=" * 55)

    weight = compute_class_weights(y_train)
    print(f"\n  scale_pos_weight : {weight:.2f}")

    X_sm, y_sm = apply_smote(X_train, y_train)
    print(f"\n  SMOTE output shape : X={X_sm.shape}, y={y_sm.shape}")
    print(f"  Class balance      : {dict(zip(*np.unique(y_sm, return_counts=True)))}")

    print("\n" + "=" * 55)
    print("Step 3.3 complete ✓")
