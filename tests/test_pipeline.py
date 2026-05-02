"""
tests/test_pipeline.py
======================
Integration tests for the IngestionPipeline.

These tests use real synthetic data files (if available) or create
minimal temporary fixtures. The LangGraph agent and ChromaDB are mocked
so tests run without Ollama or a live vector store.

Tests are skipped if the synthetic data or model artifacts don't exist,
so CI pipelines that don't run the full setup don't break.
"""

import json
import os
import sys
import tempfile
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
EVENTS_PATH   = _PROJECT_ROOT / "data" / "synthetic" / "events.csv"
GRAPH_PATH    = _PROJECT_ROOT / "data" / "graphs" / "account_graph.pkl"
MODELS_EXIST  = (
    (_PROJECT_ROOT / "data" / "models" / "xgboost_fraud.json").exists()
    and (_PROJECT_ROOT / "data" / "models" / "isolation_forest.pkl").exists()
)
DATA_EXISTS   = EVENTS_PATH.exists() and GRAPH_PATH.exists()

skip_if_no_data   = pytest.mark.skipif(not DATA_EXISTS,   reason="Synthetic data not found — run generate_data.py and ingest_and_run.py first.")
skip_if_no_models = pytest.mark.skipif(not MODELS_EXIST,  reason="Model artifacts not found — run train_model.py first.")


# ── Minimal synthetic fixture ─────────────────────────────────────────────────

def _make_minimal_events_csv(tmp_path: Path, n_legit: int = 20, n_fraud: int = 5) -> str:
    """Creates a minimal events CSV for testing without real data."""
    import random
    rows = []
    for i in range(n_legit):
        rows.append({
            "event_id": f"TXN-LEGIT-{i:04d}",
            "account_id": f"ACC-{i:05d}",
            "timestamp": "2026-01-01 12:00:00",
            "transaction_amount": 100.0,
            "account_age_days": 365,
            "device_id": f"DEV-{i:04d}",
            "ip_address": f"1.2.3.{i % 255}",
            "ip_country_mismatch": 0,
            "device_change_count": 0,
            "velocity_1hr": 1,
            "avg_txn_amount_30d": 100.0,
            "failed_login_count_24hr": 0,
            "is_fraud": 0,
            "fraud_type": None,
        })
    for i in range(n_fraud):
        rows.append({
            "event_id": f"TXN-FRAUD-{i:04d}",
            "account_id": f"ACC-FRAUD-{i:05d}",
            "timestamp": "2026-01-01 12:00:00",
            "transaction_amount": 4500.0,
            "account_age_days": 5,
            "device_id": "DEV-SHARED-001",  # shared device = fraud ring signal
            "ip_address": "9.9.9.9",
            "ip_country_mismatch": 1,
            "device_change_count": 3,
            "velocity_1hr": 8,
            "avg_txn_amount_30d": 100.0,
            "failed_login_count_24hr": 4,
            "is_fraud": 1,
            "fraud_type": "account_takeover",
        })
    df = pd.DataFrame(rows)
    path = str(tmp_path / "test_events.csv")
    df.to_csv(path, index=False)
    return path


# ── Pipeline summary structure tests ─────────────────────────────────────────

class TestPipelineSummaryStructure:
    """
    Tests that the pipeline returns a correctly structured summary dict.
    Uses mocked scorer and agent so no model artifacts are needed.
    """

    def test_pipeline_summary_has_required_keys(self, tmp_path):
        events_path = _make_minimal_events_csv(tmp_path)
        graph_path  = str(tmp_path / "test_graph.pkl")
        scored_path = str(tmp_path / "scored.json")

        mock_scorer = MagicMock()
        mock_scorer.score_event.return_value = {
            "risk_score": 0.10, "risk_level": "LOW",
            "xgb_prob": 0.10, "iso_anom": 0.10,
            "flags": [], "explanation": "No dominant features.",
        }
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "degree_centrality": 0.0, "component_size": 1,
            "shared_device_count": 0, "ip_reuse_count": 0,
        }

        with (
            patch("src.ingestion.pipeline.FraudScorer",          return_value=mock_scorer),
            patch("src.ingestion.pipeline.GraphFeatureExtractor", return_value=mock_extractor),
            patch("src.ingestion.pipeline.InvestigationAgent"),
        ):
            from src.ingestion.pipeline import IngestionPipeline
            pipeline = IngestionPipeline()
            summary = pipeline.run(
                events_path=events_path,
                graph_output_path=graph_path,
                scored_output_path=scored_path,
                embed_cases=False,
                max_agent_cases=0,
            )

        required_keys = [
            "events_loaded", "events_scored", "events_flagged",
            "cases_investigated", "cases_added_to_queue", "errors",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key '{key}' in pipeline summary"

    def test_pipeline_events_loaded_matches_csv_rows(self, tmp_path):
        events_path = _make_minimal_events_csv(tmp_path, n_legit=10, n_fraud=2)
        graph_path  = str(tmp_path / "test_graph.pkl")
        scored_path = str(tmp_path / "scored.json")

        mock_scorer = MagicMock()
        mock_scorer.score_event.return_value = {
            "risk_score": 0.10, "risk_level": "LOW",
            "xgb_prob": 0.10, "iso_anom": 0.10,
            "flags": [], "explanation": "",
        }
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "degree_centrality": 0.0, "component_size": 1,
            "shared_device_count": 0, "ip_reuse_count": 0,
        }

        with (
            patch("src.ingestion.pipeline.FraudScorer",          return_value=mock_scorer),
            patch("src.ingestion.pipeline.GraphFeatureExtractor", return_value=mock_extractor),
            patch("src.ingestion.pipeline.InvestigationAgent"),
        ):
            from src.ingestion.pipeline import IngestionPipeline
            pipeline = IngestionPipeline()
            summary = pipeline.run(
                events_path=events_path,
                graph_output_path=graph_path,
                scored_output_path=scored_path,
                embed_cases=False,
                max_agent_cases=0,
            )

        assert summary["events_loaded"] == 12
        assert summary["events_scored"] == 12

    def test_pipeline_no_flagged_events_when_all_low_risk(self, tmp_path):
        events_path = _make_minimal_events_csv(tmp_path, n_legit=10, n_fraud=0)
        graph_path  = str(tmp_path / "test_graph.pkl")
        scored_path = str(tmp_path / "scored.json")

        mock_scorer = MagicMock()
        mock_scorer.score_event.return_value = {
            "risk_score": 0.05, "risk_level": "LOW",
            "xgb_prob": 0.05, "iso_anom": 0.05,
            "flags": [], "explanation": "",
        }
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "degree_centrality": 0.0, "component_size": 1,
            "shared_device_count": 0, "ip_reuse_count": 0,
        }

        with (
            patch("src.ingestion.pipeline.FraudScorer",          return_value=mock_scorer),
            patch("src.ingestion.pipeline.GraphFeatureExtractor", return_value=mock_extractor),
            patch("src.ingestion.pipeline.InvestigationAgent"),
        ):
            from src.ingestion.pipeline import IngestionPipeline
            pipeline = IngestionPipeline()
            summary = pipeline.run(
                events_path=events_path,
                graph_output_path=graph_path,
                scored_output_path=scored_path,
                embed_cases=False,
                max_agent_cases=0,
            )

        assert summary["events_flagged"] == 0
        assert summary["cases_investigated"] == 0

    def test_pipeline_writes_scored_events_json(self, tmp_path):
        events_path = _make_minimal_events_csv(tmp_path)
        graph_path  = str(tmp_path / "test_graph.pkl")
        scored_path = str(tmp_path / "scored.json")

        mock_scorer = MagicMock()
        mock_scorer.score_event.return_value = {
            "risk_score": 0.10, "risk_level": "LOW",
            "xgb_prob": 0.10, "iso_anom": 0.10,
            "flags": [], "explanation": "",
        }
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "degree_centrality": 0.0, "component_size": 1,
            "shared_device_count": 0, "ip_reuse_count": 0,
        }

        with (
            patch("src.ingestion.pipeline.FraudScorer",          return_value=mock_scorer),
            patch("src.ingestion.pipeline.GraphFeatureExtractor", return_value=mock_extractor),
            patch("src.ingestion.pipeline.InvestigationAgent"),
        ):
            from src.ingestion.pipeline import IngestionPipeline
            pipeline = IngestionPipeline()
            pipeline.run(
                events_path=events_path,
                graph_output_path=graph_path,
                scored_output_path=scored_path,
                embed_cases=False,
                max_agent_cases=0,
            )

        assert Path(scored_path).exists(), "scored_events.json was not written"
        with open(scored_path) as f:
            scored = json.load(f)
        assert isinstance(scored, list)
        assert len(scored) == 25  # 20 legit + 5 fraud

    def test_pipeline_scored_events_sorted_by_risk_score(self, tmp_path):
        events_path = _make_minimal_events_csv(tmp_path)
        graph_path  = str(tmp_path / "test_graph.pkl")
        scored_path = str(tmp_path / "scored.json")

        # Return varying scores so we can verify sort order
        call_count = [0]
        def varying_score(features):
            call_count[0] += 1
            score = (call_count[0] % 10) / 10.0
            return {
                "risk_score": score, "risk_level": "LOW",
                "xgb_prob": score, "iso_anom": score,
                "flags": [], "explanation": "",
            }

        mock_scorer = MagicMock()
        mock_scorer.score_event.side_effect = varying_score
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "degree_centrality": 0.0, "component_size": 1,
            "shared_device_count": 0, "ip_reuse_count": 0,
        }

        with (
            patch("src.ingestion.pipeline.FraudScorer",          return_value=mock_scorer),
            patch("src.ingestion.pipeline.GraphFeatureExtractor", return_value=mock_extractor),
            patch("src.ingestion.pipeline.InvestigationAgent"),
        ):
            from src.ingestion.pipeline import IngestionPipeline
            pipeline = IngestionPipeline()
            pipeline.run(
                events_path=events_path,
                graph_output_path=graph_path,
                scored_output_path=scored_path,
                embed_cases=False,
                max_agent_cases=0,
            )

        with open(scored_path) as f:
            scored = json.load(f)
        scores = [e["risk_score"] for e in scored]
        assert scores == sorted(scores, reverse=True), "Scored events should be sorted descending"

    def test_pipeline_errors_list_is_empty_on_success(self, tmp_path):
        events_path = _make_minimal_events_csv(tmp_path)
        graph_path  = str(tmp_path / "test_graph.pkl")
        scored_path = str(tmp_path / "scored.json")

        mock_scorer = MagicMock()
        mock_scorer.score_event.return_value = {
            "risk_score": 0.10, "risk_level": "LOW",
            "xgb_prob": 0.10, "iso_anom": 0.10,
            "flags": [], "explanation": "",
        }
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "degree_centrality": 0.0, "component_size": 1,
            "shared_device_count": 0, "ip_reuse_count": 0,
        }

        with (
            patch("src.ingestion.pipeline.FraudScorer",          return_value=mock_scorer),
            patch("src.ingestion.pipeline.GraphFeatureExtractor", return_value=mock_extractor),
            patch("src.ingestion.pipeline.InvestigationAgent"),
        ):
            from src.ingestion.pipeline import IngestionPipeline
            pipeline = IngestionPipeline()
            summary = pipeline.run(
                events_path=events_path,
                graph_output_path=graph_path,
                scored_output_path=scored_path,
                embed_cases=False,
                max_agent_cases=0,
            )

        assert summary["errors"] == [], f"Unexpected errors: {summary['errors']}"


# ── Real data tests (skipped if data not present) ─────────────────────────────

class TestPipelineWithRealData:

    @skip_if_no_data
    @skip_if_no_models
    def test_pipeline_scores_real_events(self, tmp_path):
        """Smoke test: run the full pipeline on real synthetic data."""
        scored_path = str(tmp_path / "scored.json")

        mock_agent = MagicMock()
        mock_agent.investigate.return_value = None  # Don't run LLM

        with patch("src.ingestion.pipeline.InvestigationAgent", return_value=mock_agent):
            from src.ingestion.pipeline import IngestionPipeline
            pipeline = IngestionPipeline()
            summary = pipeline.run(
                events_path=str(EVENTS_PATH),
                graph_output_path=str(GRAPH_PATH),
                scored_output_path=scored_path,
                embed_cases=False,
                max_agent_cases=0,
            )

        assert summary["events_loaded"] > 0
        assert summary["events_scored"] > 0
        assert summary["errors"] == []
