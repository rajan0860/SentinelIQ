"""
tests/test_scorer.py
====================
Unit tests for the ensemble ML scorer (EnsembleScorer + FraudScorer / SHAP).

These tests load the pre-trained model artifacts from data/models/ so they
require that `python scripts/train_model.py` has been run at least once.
If the artifacts are missing the tests are skipped rather than failing, so
CI pipelines that don't include a training step don't break.
"""

import os
import sys
import pytest

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── paths ─────────────────────────────────────────────────────────────────────
XGB_PATH           = "data/models/xgboost_fraud.json"
ISO_PATH           = "data/models/isolation_forest.pkl"
FEATURE_NAMES_PATH = "data/models/feature_names.pkl"

MODELS_EXIST = (
    os.path.exists(XGB_PATH)
    and os.path.exists(ISO_PATH)
    and os.path.exists(FEATURE_NAMES_PATH)
)

skip_if_no_models = pytest.mark.skipif(
    not MODELS_EXIST,
    reason="Model artifacts not found — run `python scripts/train_model.py` first.",
)

# ── shared fixtures ───────────────────────────────────────────────────────────

LEGIT_EVENT = {
    "transaction_amount":      85.0,
    "account_age_days":        730.0,
    "ip_country_mismatch":     0.0,
    "device_change_count":     0.0,
    "velocity_1hr":            1.0,
    "avg_txn_amount_30d":      90.0,
    "failed_login_count_24hr": 0.0,
    "degree_centrality":       0.0006,
    "component_size":          3.0,
    "shared_device_count":     0.0,
    "ip_reuse_count":          0.0,
}

FRAUD_EVENT = {
    "transaction_amount":      4850.0,
    "account_age_days":        7.0,
    "ip_country_mismatch":     1.0,
    "device_change_count":     3.0,
    "velocity_1hr":            8.0,
    "avg_txn_amount_30d":      104.0,
    "failed_login_count_24hr": 4.0,
    "degree_centrality":       0.0006,
    "component_size":          6.0,
    "shared_device_count":     3.0,
    "ip_reuse_count":          3.0,
}


# ── EnsembleScorer tests ──────────────────────────────────────────────────────

class TestEnsembleScorer:

    @pytest.fixture(scope="class")
    def scorer(self):
        from src.ml.ensemble import EnsembleScorer
        return EnsembleScorer(
            xgb_path=XGB_PATH,
            iso_path=ISO_PATH,
            feature_names_path=FEATURE_NAMES_PATH,
        )

    @skip_if_no_models
    def test_score_returns_required_keys(self, scorer):
        result = scorer.score(LEGIT_EVENT)
        assert "risk_score"  in result
        assert "risk_level"  in result
        assert "xgb_prob"    in result
        assert "iso_anom"    in result
        assert "flags"       in result

    @skip_if_no_models
    def test_risk_score_is_bounded(self, scorer):
        for event in [LEGIT_EVENT, FRAUD_EVENT]:
            result = scorer.score(event)
            assert 0.0 <= result["risk_score"] <= 1.0, (
                f"risk_score {result['risk_score']} is out of [0, 1]"
            )

    @skip_if_no_models
    def test_legit_event_scores_low(self, scorer):
        result = scorer.score(LEGIT_EVENT)
        assert result["risk_score"] < 0.75, (
            f"Expected legit event to score below 0.75, got {result['risk_score']}"
        )

    @skip_if_no_models
    def test_fraud_event_scores_high(self, scorer):
        result = scorer.score(FRAUD_EVENT)
        assert result["risk_score"] >= 0.50, (
            f"Expected fraud event to score >= 0.50, got {result['risk_score']}"
        )

    @skip_if_no_models
    def test_risk_level_matches_score(self, scorer):
        result = scorer.score(FRAUD_EVENT)
        score = result["risk_score"]
        level = result["risk_level"]

        if score >= 0.90:
            assert level == "CRITICAL"
        elif score >= 0.75:
            assert level == "HIGH"
        elif score >= 0.40:
            assert level == "MEDIUM"
        else:
            assert level == "LOW"

    @skip_if_no_models
    def test_flags_is_list(self, scorer):
        result = scorer.score(LEGIT_EVENT)
        assert isinstance(result["flags"], list)

    @skip_if_no_models
    def test_missing_feature_raises_value_error(self, scorer):
        incomplete = {k: v for k, v in LEGIT_EVENT.items() if k != "velocity_1hr"}
        with pytest.raises((ValueError, KeyError)):
            scorer.score(incomplete)


# ── FraudScorer (with SHAP) tests ─────────────────────────────────────────────

class TestFraudScorer:

    @pytest.fixture(scope="class")
    def fraud_scorer(self):
        from src.ml.scorer import FraudScorer
        return FraudScorer(
            xgb_path=XGB_PATH,
            iso_path=ISO_PATH,
            feature_names_path=FEATURE_NAMES_PATH,
        )

    @skip_if_no_models
    def test_score_event_returns_explanation(self, fraud_scorer):
        result = fraud_scorer.score_event(LEGIT_EVENT)
        assert "explanation" in result
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0

    @skip_if_no_models
    def test_explanation_contains_feature_name(self, fraud_scorer):
        """SHAP explanation should reference at least one known feature name."""
        result = fraud_scorer.score_event(FRAUD_EVENT)
        explanation = result["explanation"]
        known_features = [
            "transaction_amount", "account_age_days", "velocity_1hr",
            "ip_country_mismatch", "device_change_count", "shared_device_count",
        ]
        assert any(feat in explanation for feat in known_features), (
            f"Explanation '{explanation}' doesn't mention any known feature."
        )

    @skip_if_no_models
    def test_score_event_includes_all_ensemble_keys(self, fraud_scorer):
        result = fraud_scorer.score_event(LEGIT_EVENT)
        for key in ["risk_score", "risk_level", "xgb_prob", "iso_anom", "flags", "explanation"]:
            assert key in result, f"Missing key: {key}"


# ── ML Utils tests ────────────────────────────────────────────────────────────

class TestMLUtils:

    def test_compute_class_weights_returns_float(self):
        import numpy as np
        from src.ml.utils import compute_class_weights
        y = np.array([0, 0, 0, 0, 1])  # 4 negatives, 1 positive
        weight = compute_class_weights(y)
        assert isinstance(weight, float)
        assert weight == 4.0

    def test_compute_class_weights_raises_on_no_positives(self):
        import numpy as np
        from src.ml.utils import compute_class_weights
        y = np.array([0, 0, 0, 0])  # all negatives
        with pytest.raises(ValueError):
            compute_class_weights(y)

    def test_apply_smote_balances_classes(self):
        import numpy as np
        import pandas as pd
        from src.ml.utils import apply_smote
        X = pd.DataFrame(np.random.rand(100, 5))
        y = np.array([0] * 95 + [1] * 5)  # 95 negatives, 5 positives
        X_res, y_res = apply_smote(X, y)
        # After SMOTE, classes should be balanced
        assert (y_res == 0).sum() == (y_res == 1).sum()

    def test_apply_smote_skips_when_minority_too_small(self):
        import numpy as np
        import pandas as pd
        from src.ml.utils import apply_smote
        X = pd.DataFrame(np.random.rand(10, 5))
        y = np.array([0] * 9 + [1] * 1)  # only 1 positive sample
        X_res, y_res = apply_smote(X, y)
        # SMOTE should skip and return original data
        assert len(X_res) == len(X)
        assert len(y_res) == len(y)

    def test_apply_smote_skips_when_no_majority_class(self):
        import numpy as np
        import pandas as pd
        from src.ml.utils import apply_smote
        X = pd.DataFrame(np.random.rand(5, 5))
        y = np.array([1, 1, 1, 1, 1])  # all positives
        X_res, y_res = apply_smote(X, y)
        # SMOTE should skip
        assert len(X_res) == len(X)
