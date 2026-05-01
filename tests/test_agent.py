"""
tests/test_agent.py
===================
Unit tests for the LangGraph InvestigationAgent.

The LLM and RAG retriever are mocked so these tests run without a live
Ollama instance or ChromaDB. They verify the graph routing logic, state
transitions, and report structure — not the quality of LLM output.
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Shared test data ──────────────────────────────────────────────────────────

LOW_RISK_SCORE = {
    "risk_score":  0.30,
    "risk_level":  "LOW",
    "xgb_prob":    0.25,
    "iso_anom":    0.40,
    "flags":       [],
    "explanation": "No dominant features identified.",
}

HIGH_RISK_SCORE = {
    "risk_score":  0.93,
    "risk_level":  "CRITICAL",
    "xgb_prob":    0.95,
    "iso_anom":    0.88,
    "flags":       ["xgboost_high", "isolation_forest_anomaly"],
    "explanation": "velocity_1hr (+0.38), shared_device_count (+0.29)",
}

SAMPLE_EVENT_DATA = {
    "transaction_amount":      4850.0,
    "account_age_days":        7.0,
    "ip_country_mismatch":     1.0,
    "device_change_count":     3.0,
    "velocity_1hr":            8.0,
    "avg_txn_amount_30d":      104.0,
    "failed_login_count_24hr": 4.0,
    "degree_centrality":       0.91,
    "component_size":          6.0,
    "shared_device_count":     3.0,
    "ip_reuse_count":          3.0,
}

MOCK_LLM_RESPONSE = json.dumps({
    "fraud_type":         "Synthetic Identity",
    "confidence":         0.91,
    "evidence_summary":   "New account with high graph centrality sharing devices with flagged accounts.",
    "similar_cases":      ["HIST-001", "HIST-002"],
    "recommended_action": "Block transaction. Freeze account pending manual review.",
})

MOCK_RETRIEVED_CASES = [
    {
        "case_id":            "HIST-001",
        "summary":            "Account showed rapid velocity from foreign IP.",
        "fraud_type":         "account_takeover",
        "outcome":            "confirmed_fraud",
        "recommended_action": "Block device.",
        "evidence":           "IP mismatch, new device",
        "similarity_score":   0.87,
    }
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_llm(response_text: str = MOCK_LLM_RESPONSE):
    """Returns a mock LangChain LLM that returns a fixed response."""
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ── Node-level unit tests ─────────────────────────────────────────────────────

class TestFlagNode:

    def test_low_risk_event_not_flagged(self):
        from src.agent.nodes import flag_node
        state = {
            "event_id":   "EVT-001",
            "event_data": SAMPLE_EVENT_DATA,
            "ml_score":   LOW_RISK_SCORE,
        }
        result = flag_node(state)
        assert result["is_flagged"] is False

    def test_high_risk_event_is_flagged(self):
        from src.agent.nodes import flag_node
        state = {
            "event_id":   "EVT-002",
            "event_data": SAMPLE_EVENT_DATA,
            "ml_score":   HIGH_RISK_SCORE,
        }
        result = flag_node(state)
        assert result["is_flagged"] is True

    def test_threshold_boundary_exactly_at_threshold(self):
        """An event exactly at the threshold should be flagged."""
        from src.agent.nodes import flag_node
        boundary_score = {"risk_score": 0.75, "risk_level": "HIGH", "flags": []}
        state = {"event_id": "EVT-003", "event_data": {}, "ml_score": boundary_score}
        result = flag_node(state)
        assert result["is_flagged"] is True

    def test_missing_risk_score_defaults_to_not_flagged(self):
        from src.agent.nodes import flag_node
        state = {"event_id": "EVT-004", "event_data": {}, "ml_score": {}}
        result = flag_node(state)
        assert result["is_flagged"] is False


class TestRetrieveNode:

    def test_retrieve_node_returns_list(self):
        from src.agent.nodes import retrieve_node
        with patch("src.agent.nodes.retrieve_similar_cases", return_value=MOCK_RETRIEVED_CASES):
            state = {
                "event_id":   "EVT-005",
                "event_data": SAMPLE_EVENT_DATA,
                "ml_score":   HIGH_RISK_SCORE,
            }
            result = retrieve_node(state)
        assert "retrieved_cases" in result
        assert isinstance(result["retrieved_cases"], list)

    def test_retrieve_node_uses_explanation_as_query(self):
        """The retriever should be called with the SHAP explanation text."""
        from src.agent.nodes import retrieve_node
        with patch("src.agent.nodes.retrieve_similar_cases", return_value=[]) as mock_retrieve:
            state = {
                "event_id":   "EVT-006",
                "event_data": SAMPLE_EVENT_DATA,
                "ml_score":   HIGH_RISK_SCORE,
            }
            retrieve_node(state)
            call_args = mock_retrieve.call_args[0][0]
            assert "velocity_1hr" in call_args or "shared_device_count" in call_args


class TestAnalyseNode:

    def test_analyse_node_stores_llm_response(self):
        from src.agent.nodes import analyse_node
        with (
            patch("src.agent.nodes.get_llm", return_value=_make_mock_llm()),
            patch("src.agent.nodes.retrieve_similar_cases", return_value=MOCK_RETRIEVED_CASES),
        ):
            state = {
                "event_id":       "EVT-007",
                "event_data":     SAMPLE_EVENT_DATA,
                "ml_score":       HIGH_RISK_SCORE,
                "retrieved_cases": MOCK_RETRIEVED_CASES,
            }
            result = analyse_node(state)
        assert "llm_raw_response" in result
        assert len(result["llm_raw_response"]) > 0


class TestReportNode:

    def test_report_node_parses_valid_json(self):
        from src.agent.nodes import report_node
        state = {
            "event_id":        "EVT-008",
            "llm_raw_response": MOCK_LLM_RESPONSE,
        }
        result = report_node(state)
        report = result["final_report"]
        assert report is not None
        assert report["case_id"]            == "EVT-008"
        assert report["fraud_type"]         == "Synthetic Identity"
        assert report["confidence"]         == 0.91
        assert "evidence_summary"           in report
        assert "recommended_action"         in report

    def test_report_node_handles_markdown_wrapped_json(self):
        """LLMs sometimes wrap JSON in ```json ... ``` — the node should strip it."""
        from src.agent.nodes import report_node
        wrapped = f"```json\n{MOCK_LLM_RESPONSE}\n```"
        state = {"event_id": "EVT-009", "llm_raw_response": wrapped}
        result = report_node(state)
        assert result["final_report"]["fraud_type"] == "Synthetic Identity"

    def test_report_node_returns_fallback_on_invalid_json(self):
        from src.agent.nodes import report_node
        state = {"event_id": "EVT-010", "llm_raw_response": "This is not JSON at all."}
        result = report_node(state)
        report = result["final_report"]
        assert report is not None
        assert report["case_id"]    == "EVT-010"
        assert report["fraud_type"] == "Parsing Error"

    def test_report_has_all_required_fields(self):
        from src.agent.nodes import report_node
        state = {"event_id": "EVT-011", "llm_raw_response": MOCK_LLM_RESPONSE}
        result = report_node(state)
        report = result["final_report"]
        required = ["case_id", "fraud_type", "confidence", "evidence_summary",
                    "similar_cases", "recommended_action"]
        for field in required:
            assert field in report, f"Missing required field: {field}"


# ── Full graph integration tests ──────────────────────────────────────────────

class TestInvestigationAgent:

    @pytest.fixture
    def agent(self):
        from src.agent.graph import InvestigationAgent
        return InvestigationAgent()

    def test_low_risk_event_returns_none(self, agent):
        with patch("src.agent.nodes.retrieve_similar_cases", return_value=[]):
            report = agent.investigate("EVT-LOW", SAMPLE_EVENT_DATA, LOW_RISK_SCORE)
        assert report is None

    def test_high_risk_event_returns_report(self, agent):
        with (
            patch("src.agent.nodes.retrieve_similar_cases", return_value=MOCK_RETRIEVED_CASES),
            patch("src.agent.nodes.get_llm", return_value=_make_mock_llm()),
        ):
            report = agent.investigate("EVT-HIGH", SAMPLE_EVENT_DATA, HIGH_RISK_SCORE)

        assert report is not None
        assert report["case_id"] == "EVT-HIGH"

    def test_report_confidence_is_bounded(self, agent):
        with (
            patch("src.agent.nodes.retrieve_similar_cases", return_value=MOCK_RETRIEVED_CASES),
            patch("src.agent.nodes.get_llm", return_value=_make_mock_llm()),
        ):
            report = agent.investigate("EVT-CONF", SAMPLE_EVENT_DATA, HIGH_RISK_SCORE)

        assert 0.0 <= report["confidence"] <= 1.0

    def test_report_similar_cases_is_list(self, agent):
        with (
            patch("src.agent.nodes.retrieve_similar_cases", return_value=MOCK_RETRIEVED_CASES),
            patch("src.agent.nodes.get_llm", return_value=_make_mock_llm()),
        ):
            report = agent.investigate("EVT-SIM", SAMPLE_EVENT_DATA, HIGH_RISK_SCORE)

        assert isinstance(report["similar_cases"], list)

    def test_llm_parse_error_still_returns_fallback_report(self, agent):
        """Even if the LLM returns garbage, the agent should return a fallback report."""
        with (
            patch("src.agent.nodes.retrieve_similar_cases", return_value=[]),
            patch("src.agent.nodes.get_llm", return_value=_make_mock_llm("not valid json")),
        ):
            report = agent.investigate("EVT-ERR", SAMPLE_EVENT_DATA, HIGH_RISK_SCORE)

        assert report is not None
        assert report["fraud_type"] == "Parsing Error"
