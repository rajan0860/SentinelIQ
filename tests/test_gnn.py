"""
GNN Component Tests
====================
Unit tests for the GraphSAGE GNN integration:
  - HeteroData construction from NetworkX + events
  - Model architecture (forward pass shape)
  - Graceful fallback when GNN artifacts are missing
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Skip decorators ──────────────────────────────────────────────────────────

def _torch_available():
    try:
        import torch
        import torch_geometric
        return True
    except ImportError:
        return False


def _data_available():
    """Check if synthetic data + graph exist for integration tests."""
    events = os.path.exists("data/synthetic/events.csv")
    graph = os.path.exists("data/graphs/account_graph.pkl")
    return events and graph


skip_if_no_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch / PyTorch Geometric not installed"
)

skip_if_no_data = pytest.mark.skipif(
    not _data_available(),
    reason="Synthetic data / graph files not found"
)


# ── GNN Data Tests ───────────────────────────────────────────────────────────

class TestGNNData:

    @skip_if_no_torch
    @skip_if_no_data
    def test_build_hetero_data_returns_heterodata(self):
        """HeteroData object is correctly constructed from NetworkX + CSV."""
        from torch_geometric.data import HeteroData
        from src.ml.gnn_data import build_hetero_data

        data = build_hetero_data(
            events_path="data/synthetic/events.csv",
            graph_path="data/graphs/account_graph.pkl",
        )

        assert isinstance(data, HeteroData)

    @skip_if_no_torch
    @skip_if_no_data
    def test_hetero_data_has_correct_node_types(self):
        """HeteroData must contain account, device, and ip node types."""
        from src.ml.gnn_data import build_hetero_data

        data = build_hetero_data(
            events_path="data/synthetic/events.csv",
            graph_path="data/graphs/account_graph.pkl",
        )

        assert "account" in data.node_types
        assert "device" in data.node_types
        assert "ip" in data.node_types

    @skip_if_no_torch
    @skip_if_no_data
    def test_hetero_data_account_has_features_and_labels(self):
        """Account nodes must have feature tensor x and label tensor y."""
        from src.ml.gnn_data import build_hetero_data

        data = build_hetero_data(
            events_path="data/synthetic/events.csv",
            graph_path="data/graphs/account_graph.pkl",
        )

        assert data["account"].x is not None
        assert data["account"].y is not None
        assert data["account"].x.shape[0] == data["account"].y.shape[0]
        # 7 tabular features per account
        assert data["account"].x.shape[1] == 7

    @skip_if_no_torch
    @skip_if_no_data
    def test_hetero_data_has_edge_types(self):
        """HeteroData must have forward + reverse edges for both relationships."""
        from src.ml.gnn_data import build_hetero_data

        data = build_hetero_data(
            events_path="data/synthetic/events.csv",
            graph_path="data/graphs/account_graph.pkl",
        )

        edge_types = data.edge_types
        assert ("account", "uses_device", "device") in edge_types
        assert ("account", "uses_ip", "ip") in edge_types
        assert ("device", "rev_uses_device", "account") in edge_types
        assert ("ip", "rev_uses_ip", "account") in edge_types

    @skip_if_no_torch
    @skip_if_no_data
    def test_hetero_data_has_train_val_masks(self):
        """Account nodes must have boolean train and val masks."""
        from src.ml.gnn_data import build_hetero_data

        data = build_hetero_data(
            events_path="data/synthetic/events.csv",
            graph_path="data/graphs/account_graph.pkl",
        )

        assert data["account"].train_mask is not None
        assert data["account"].val_mask is not None
        # Masks should cover all accounts, no overlap
        total = data["account"].train_mask.sum() + data["account"].val_mask.sum()
        assert total == data["account"].num_nodes


# ── GNN Model Tests ──────────────────────────────────────────────────────────

class TestGNNModel:

    @skip_if_no_torch
    @skip_if_no_data
    def test_gnn_forward_produces_correct_output_shape(self):
        """Model output should be [num_accounts, 2] (binary classification)."""
        import torch
        from src.ml.gnn_data import build_hetero_data
        from src.ml.gnn_model import create_fraud_gnn

        data = build_hetero_data(
            events_path="data/synthetic/events.csv",
            graph_path="data/graphs/account_graph.pkl",
        )

        model = create_fraud_gnn(data.metadata(), hidden_channels=32)
        model.eval()

        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)

        assert "account" in out
        assert out["account"].shape[0] == data["account"].num_nodes
        assert out["account"].shape[1] == 2  # binary: [legit, fraud]

    @skip_if_no_torch
    def test_create_fraud_gnn_is_callable(self):
        """create_fraud_gnn should accept metadata and return a Module."""
        import torch
        from src.ml.gnn_model import create_fraud_gnn

        # Minimal metadata mimicking a simple graph
        metadata = (
            ["account", "device", "ip"],
            [
                ("account", "uses_device", "device"),
                ("account", "uses_ip", "ip"),
                ("device", "rev_uses_device", "account"),
                ("ip", "rev_uses_ip", "account"),
            ],
        )

        model = create_fraud_gnn(metadata, hidden_channels=16)
        assert isinstance(model, torch.nn.Module)


# ── Ensemble Fallback Tests ─────────────────────────────────────────────────

class TestEnsembleFallback:

    def test_ensemble_works_without_gnn(self):
        """EnsembleScorer must work when gnn_path is None (backward compat)."""
        xgb_path = "data/models/xgboost_fraud.json"
        iso_path = "data/models/isolation_forest.pkl"
        feat_path = "data/models/feature_names.pkl"

        if not (os.path.exists(xgb_path) and os.path.exists(iso_path)):
            pytest.skip("Model artifacts not found")

        from src.ml.ensemble import EnsembleScorer

        scorer = EnsembleScorer(
            xgb_path=xgb_path,
            iso_path=iso_path,
            feature_names_path=feat_path,
            gnn_path=None,
        )

        assert scorer._gnn_available is False
        assert scorer.gnn_model is None

    def test_ensemble_fallback_on_missing_gnn_file(self):
        """EnsembleScorer should gracefully skip when gnn_path doesn't exist."""
        xgb_path = "data/models/xgboost_fraud.json"
        iso_path = "data/models/isolation_forest.pkl"
        feat_path = "data/models/feature_names.pkl"

        if not (os.path.exists(xgb_path) and os.path.exists(iso_path)):
            pytest.skip("Model artifacts not found")

        from src.ml.ensemble import EnsembleScorer

        scorer = EnsembleScorer(
            xgb_path=xgb_path,
            iso_path=iso_path,
            feature_names_path=feat_path,
            gnn_path="data/models/NONEXISTENT_gnn_fraud.pt",
        )

        assert scorer._gnn_available is False

    def test_scorer_includes_gnn_prob_key(self):
        """FraudScorer output must always include gnn_prob (0.0 if GNN disabled)."""
        xgb_path = "data/models/xgboost_fraud.json"
        iso_path = "data/models/isolation_forest.pkl"
        feat_path = "data/models/feature_names.pkl"

        if not (os.path.exists(xgb_path) and os.path.exists(iso_path)):
            pytest.skip("Model artifacts not found")

        from src.ml.scorer import FraudScorer

        scorer = FraudScorer(
            xgb_path=xgb_path,
            iso_path=iso_path,
            feature_names_path=feat_path,
            gnn_path=None,
        )

        event = {
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

        result = scorer.score_event(event)
        assert "gnn_prob" in result
        assert result["gnn_prob"] == 0.0  # GNN disabled, should be 0.0
