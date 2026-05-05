"""
Ensemble Scorer Module
=======================
Combines XGBoost (supervised) and Isolation Forest (unsupervised) predictions
into a single unified risk_score (0.0 – 1.0) at inference time.

Why an ensemble?
    - XGBoost is trained on historical fraud labels. It is highly precise
      on patterns it has seen before but can miss novel attack vectors.
    - Isolation Forest is unsupervised — it flags statistical anomalies
      regardless of whether it has seen that pattern before.
    - Weighting 70% XGBoost + 30% Isolation Forest gives strong precision
      while maintaining sensitivity to new fraud patterns.

Ensemble formula:
    xgb_prob  = XGBoost.predict_proba(row)[1]          # 0–1, probability of fraud
    iso_raw   = IsolationForest.score_samples(row)[0]  # lower = more anomalous
    iso_norm  = (iso_raw - min) / (max - min)          # normalise to 0–1
    iso_anom  = 1 - iso_norm                           # invert: 1 = anomaly
    risk_score = 0.7 * xgb_prob + 0.3 * iso_anom

Risk levels:
    CRITICAL  ≥ 0.90
    HIGH      0.75 – 0.90
    MEDIUM    0.40 – 0.75
    LOW       < 0.40

Usage:
    scorer = EnsembleScorer(
        xgb_path="data/models/xgboost_fraud.json",
        iso_path="data/models/isolation_forest.pkl",
    )
    result = scorer.score({
        "transaction_amount": 4850.0,
        "account_age_days": 12,
        ...
    })
    # {"risk_score": 0.93, "risk_level": "CRITICAL",
    #  "flags": ["xgboost_high", "isolation_forest_anomaly"]}
"""

import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensemble weighting — XGBoost is the primary supervised signal
_XGB_WEIGHT = 0.7
_ISO_WEIGHT  = 0.3

# Weights when GNN is available (three-model ensemble)
_XGB_WEIGHT_WITH_GNN = 0.40
_GNN_WEIGHT          = 0.35
_ISO_WEIGHT_WITH_GNN = 0.25

# Risk thresholds
_THRESHOLD_CRITICAL = 0.90
_THRESHOLD_HIGH     = 0.75
_THRESHOLD_MEDIUM   = 0.40


def _risk_level(score: float) -> str:
    """Map a continuous risk score to a categorical risk level string."""
    if score >= _THRESHOLD_CRITICAL:
        return "CRITICAL"
    elif score >= _THRESHOLD_HIGH:
        return "HIGH"
    elif score >= _THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


class EnsembleScorer:
    """
    Loads trained XGBoost and Isolation Forest models and combines their
    outputs into a single risk score at inference time.

    The Isolation Forest score_samples() output range varies by dataset.
    We calibrate the normalisation bounds once at init using a small grid
    of synthetic scores so the iso_anom term is always 0–1.
    """

    def __init__(self, xgb_path: str, iso_path: str, feature_names_path: str = None, gnn_path: str = None):
        """
        Load both model artifacts from disk.

        Args:
            xgb_path:           Path to xgboost_fraud.json
            iso_path:           Path to isolation_forest.pkl
            feature_names_path: Optional path to feature_names.pkl — if
                                provided, features dict is reordered to
                                match training column order automatically.
            gnn_path:           Optional path to gnn_fraud.pt — if provided
                                and the file exists, GNN becomes the third
                                ensemble member.
        """
        # Load XGBoost
        xgb_file = Path(xgb_path)
        if not xgb_file.exists():
            raise FileNotFoundError(f"XGBoost model not found: {xgb_file}")
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(xgb_file))
        logger.info(f"XGBoost model loaded from {xgb_file}")

        # Load Isolation Forest
        iso_file = Path(iso_path)
        if not iso_file.exists():
            raise FileNotFoundError(f"Isolation Forest model not found: {iso_file}")
        with open(iso_file, "rb") as f:
            self.iso_model = pickle.load(f)
        logger.info(f"Isolation Forest model loaded from {iso_file}")

        # Load feature names (optional but strongly recommended)
        self.feature_names = None
        if feature_names_path:
            fn_file = Path(feature_names_path)
            if fn_file.exists():
                with open(fn_file, "rb") as f:
                    self.feature_names = pickle.load(f)
                logger.info(f"Feature names loaded: {self.feature_names}")

        # Calibrate Isolation Forest score range so normalisation is stable.
        # Use a named DataFrame so sklearn doesn't warn about missing feature names
        # (the model was fitted with a DataFrame, so inference must match).
        dummy_cols = self.feature_names if self.feature_names else \
                     [f"f{i}" for i in range(self.iso_model.n_features_in_)]
        dummy = pd.DataFrame(
            np.zeros((100, self.iso_model.n_features_in_)),
            columns=dummy_cols,
        )
        scores = self.iso_model.score_samples(dummy)
        self._iso_min = scores.min()
        self._iso_max = scores.max()
        logger.info(
            f"Isolation Forest score range calibrated: "
            f"[{self._iso_min:.4f}, {self._iso_max:.4f}]"
        )

        # Load GNN model (optional — graceful fallback if not available)
        self.gnn_model = None
        self._gnn_available = False
        if gnn_path:
            gnn_file = Path(gnn_path)
            if gnn_file.exists():
                try:
                    import torch
                    from src.ml.gnn_model import create_fraud_gnn

                    checkpoint = torch.load(str(gnn_file), map_location="cpu", weights_only=False)
                    metadata = checkpoint["metadata"]
                    hidden_channels = checkpoint.get("hidden_channels", 64)

                    self.gnn_model = create_fraud_gnn(metadata, hidden_channels=hidden_channels)
                    self.gnn_model.load_state_dict(checkpoint["model_state_dict"])
                    self.gnn_model.eval()
                    self._gnn_available = True
                    logger.info(f"GNN model loaded from {gnn_file} — three-model ensemble active.")
                except Exception as e:
                    logger.warning(f"Failed to load GNN model: {e}. Falling back to XGB + IF.")
            else:
                logger.info(f"GNN model not found at {gnn_file} — using XGB + IF ensemble.")

    def _features_to_dataframe(self, features: dict) -> pd.DataFrame:
        """
        Convert a feature dict to a 1-row DataFrame in training column order.

        Using a named DataFrame (rather than a raw numpy array) ensures both
        XGBoost and IsolationForest receive properly labelled columns and avoids
        sklearn feature-name warnings.
        """
        if self.feature_names:
            try:
                row = {col: [features[col]] for col in self.feature_names}
            except KeyError as e:
                raise ValueError(f"Missing feature in input: {e}")
        else:
            row = {k: [v] for k, v in features.items()}
        return pd.DataFrame(row)

    def score(self, features: dict) -> dict:
        """
        Run the ensemble on a single event's feature dict.

        Args:
            features: dict mapping feature name → value.
                      Must contain all 11 model features
                      (transaction_amount, account_age_days, etc.)

        Returns:
            {
                "risk_score":  float  (0.0 – 1.0),
                "risk_level":  str    ("LOW" | "MEDIUM" | "HIGH" | "CRITICAL"),
                "xgb_prob":    float  (raw XGBoost fraud probability),
                "iso_anom":    float  (normalised Isolation Forest anomaly score),
                "flags":       list[str],
            }
        """
        X = self._features_to_dataframe(features)

        # ── XGBoost probability ───────────────────────────────────────────────
        xgb_prob = float(self.xgb_model.predict_proba(X)[0][1])

        # ── Isolation Forest anomaly score ────────────────────────────────────
        iso_raw  = float(self.iso_model.score_samples(X)[0])
        # Normalise to 0–1 using calibrated range
        range_   = self._iso_max - self._iso_min
        iso_norm = (iso_raw - self._iso_min) / range_ if range_ != 0 else 0.5
        iso_norm = float(np.clip(iso_norm, 0.0, 1.0))
        iso_anom = 1.0 - iso_norm   # invert: high value = more anomalous

        # ── Weighted ensemble ─────────────────────────────────────────────────
        # GNN probability is injected externally by FraudScorer when available.
        # At the EnsembleScorer level, we use two-model weights as the baseline.
        # FraudScorer.score_event() overrides risk_score with three-model weights
        # when the GNN is loaded.
        risk_score = _XGB_WEIGHT * xgb_prob + _ISO_WEIGHT * iso_anom
        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        level      = _risk_level(risk_score)

        # ── Flags — descriptive signals for the agent ─────────────────────────
        flags = []
        if xgb_prob >= 0.75:
            flags.append("xgboost_high")
        elif xgb_prob >= 0.40:
            flags.append("xgboost_medium")

        if iso_anom >= 0.70:
            flags.append("isolation_forest_anomaly")

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": level,
            "xgb_prob":   round(xgb_prob, 4),
            "iso_anom":   round(iso_anom, 4),
            "flags":      flags,
        }


# ─── Quick verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = EnsembleScorer(
        xgb_path="data/models/xgboost_fraud.json",
        iso_path="data/models/isolation_forest.pkl",
        feature_names_path="data/models/feature_names.pkl",
    )

    # Scenario A — clearly legitimate event
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

    # Scenario B — high-risk synthetic identity ring event
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

    print("\n" + "=" * 55)
    print("  Ensemble Scorer — Verification")
    print("=" * 55)

    for label, event in [("LEGIT event", legit_event), ("FRAUD event", fraud_event)]:
        result = scorer.score(event)
        print(f"\n  {label}")
        print(f"    risk_score  : {result['risk_score']}")
        print(f"    risk_level  : {result['risk_level']}")
        print(f"    xgb_prob    : {result['xgb_prob']}")
        print(f"    iso_anom    : {result['iso_anom']}")
        print(f"    flags       : {result['flags']}")

    print("\n" + "=" * 55)
    print("Step 3.5 complete ✓")
