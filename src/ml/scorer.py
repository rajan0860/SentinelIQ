"""
Fraud Scorer Module
====================
High-level inference layer that wraps EnsembleScorer and adds SHAP-based
explanations for every scored event.

Given raw event features (including graph features), this module:
    1. Runs the ensemble (XGBoost + Isolation Forest)
    2. Computes SHAP values to identify the top driving features
    3. Returns a human-readable one-line explanation alongside the score

The SHAP explanation tells analysts *why* a score is high, not just *that* it is.
Example: "velocity_1hr (+0.38), shared_device_count (+0.29)"

SHAP (SHapley Additive exPlanations):
    Based on game theory — each feature's contribution is its average marginal
    contribution across all possible orderings of features. For tree models like
    XGBoost, TreeExplainer computes exact SHAP values in O(TLD²) time where
    T = trees, L = leaves, D = max depth.

Usage:
    scorer = FraudScorer(
        xgb_path="data/models/xgboost_fraud.json",
        iso_path="data/models/isolation_forest.pkl",
        feature_names_path="data/models/feature_names.pkl",
    )
    result = scorer.score_event({
        "transaction_amount": 4850.0,
        "account_age_days": 7,
        ...
    })
    # {
    #   "risk_score": 0.85,
    #   "risk_level": "HIGH",
    #   "xgb_prob": 1.0,
    #   "iso_anom": 0.5,
    #   "flags": ["xgboost_high"],
    #   "explanation": "velocity_1hr (+0.38), shared_device_count (+0.29)"
    # }
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ml.ensemble import EnsembleScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Number of top features to surface in the human-readable explanation
_TOP_N_FEATURES = 2


class FraudScorer:
    """
    Combines EnsembleScorer with SHAP explanations for analyst-facing output.

    The SHAP TreeExplainer is initialised once (expensive) and then used
    cheaply per-event. This is the class that the LangGraph agent, the API,
    and the ingestion pipeline all use to score events.
    """

    def __init__(
        self,
        xgb_path: str,
        iso_path: str,
        feature_names_path: str = None,
    ):
        """
        Load models and initialise the SHAP explainer.

        Args:
            xgb_path:           Path to xgboost_fraud.json
            iso_path:           Path to isolation_forest.pkl
            feature_names_path: Optional path to feature_names.pkl
        """
        logger.info("Initialising FraudScorer ...")

        self.ensemble = EnsembleScorer(
            xgb_path=xgb_path,
            iso_path=iso_path,
            feature_names_path=feature_names_path,
        )
        self.feature_names = self.ensemble.feature_names

        # TreeExplainer is specifically optimised for tree-based models.
        # It computes exact (not approximate) SHAP values — important for
        # reliable explanations in a fraud investigation context.
        logger.info("Building SHAP TreeExplainer ...")
        self.explainer = shap.TreeExplainer(self.ensemble.xgb_model)
        logger.info("FraudScorer ready.")

    def _build_explanation(self, features_dict: dict) -> str:
        """
        Compute SHAP values for one event and return a concise explanation
        highlighting the top N most influential features.

        Args:
            features_dict: Feature dict in the same format as score_event()

        Returns:
            Human-readable string, e.g.:
            "velocity_1hr (+0.38), shared_device_count (+0.29)"
        """
        # Build the input in training column order
        X_df = self.ensemble._features_to_dataframe(features_dict)

        # shap_values shape: (1, n_features) for binary classification
        # We take index [0] to get the per-row array, then [1] for fraud class.
        # For XGBoost with TreeExplainer, shap_values returns a list of arrays
        # for each class — index [1] is the fraud (positive) class.
        raw_shap = self.explainer.shap_values(X_df)

        # TreeExplainer on XGBClassifier may return a 2D array (n_samples, n_features)
        # rather than a list, depending on the version. Handle both shapes.
        if isinstance(raw_shap, list):
            # List of arrays — one per class; take fraud class (index 1)
            shap_row = raw_shap[1][0]
        else:
            # 2D array — single output (fraud probability log-odds)
            shap_row = raw_shap[0]

        # Pair each feature name with its SHAP value and sort by absolute impact
        feature_impacts = list(zip(self.feature_names, shap_row))
        top_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
        top_features = top_features[:_TOP_N_FEATURES]

        # Format: "velocity_1hr (+0.38), shared_device_count (+0.29)"
        parts = [f"{name} ({val:+.2f})" for name, val in top_features if abs(val) > 0.001]
        return ", ".join(parts) if parts else "No dominant features identified."

    def score_event(self, event: dict) -> dict:
        """
        Score a single event and return ensemble result + SHAP explanation.

        Args:
            event: Feature dict. Must contain all 11 model features.
                   Graph features (degree_centrality, component_size, etc.)
                   should be pre-computed by GraphFeatureExtractor and merged
                   into this dict before calling score_event().

        Returns:
            dict with keys:
                risk_score   (float)   0.0 – 1.0
                risk_level   (str)     LOW | MEDIUM | HIGH | CRITICAL
                xgb_prob     (float)   raw XGBoost fraud probability
                iso_anom     (float)   normalised Isolation Forest anomaly score
                flags        (list)    descriptive signal tags
                explanation  (str)     human-readable SHAP feature contribution
        """
        # Run the ensemble scorer
        result = self.ensemble.score(event)

        # Add SHAP explanation (only meaningful when XGBoost sees the event
        # as suspicious — for very low scores the explanation is noise)
        explanation = self._build_explanation(event)
        result["explanation"] = explanation

        return result


# ─── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = FraudScorer(
        xgb_path="data/models/xgboost_fraud.json",
        iso_path="data/models/isolation_forest.pkl",
        feature_names_path="data/models/feature_names.pkl",
    )

    legit_event = {
        "transaction_amount": 85.0,
        "account_age_days": 730,
        "ip_country_mismatch": 0,
        "device_change_count": 0,
        "velocity_1hr": 1,
        "avg_txn_amount_30d": 90.0,
        "failed_login_count_24hr": 0,
        "degree_centrality": 0.0006,
        "component_size": 3,
        "shared_device_count": 0,
        "ip_reuse_count": 0,
    }

    fraud_event = {
        "transaction_amount": 4850.0,
        "account_age_days": 7,
        "ip_country_mismatch": 1,
        "device_change_count": 3,
        "velocity_1hr": 8,
        "avg_txn_amount_30d": 104.0,
        "failed_login_count_24hr": 4,
        "degree_centrality": 0.0006,
        "component_size": 6,
        "shared_device_count": 3,
        "ip_reuse_count": 3,
    }

    print("\n" + "=" * 60)
    print("  FraudScorer (with SHAP) — Verification")
    print("=" * 60)

    for label, event in [("LEGIT event", legit_event), ("FRAUD event", fraud_event)]:
        result = scorer.score_event(event)
        print(f"\n  {label}")
        print(f"    risk_score  : {result['risk_score']}")
        print(f"    risk_level  : {result['risk_level']}")
        print(f"    xgb_prob    : {result['xgb_prob']}")
        print(f"    iso_anom    : {result['iso_anom']}")
        print(f"    flags       : {result['flags']}")
        print(f"    explanation : {result['explanation']}")

    print("\n" + "=" * 60)
    print("Step 3.6 complete ✓")
