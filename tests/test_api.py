"""
tests/test_api.py
=================
Integration tests for the FastAPI REST API.

Uses FastAPI's TestClient so no live server is needed. The review queue
and feedback handler are mocked to isolate the API layer from the
filesystem and ChromaDB.
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_CASE = {
    "case_id":            "CASE-API-001",
    "fraud_type":         "Account Takeover",
    "confidence":         0.92,
    "evidence_summary":   "High velocity from foreign IP with new device.",
    "similar_cases":      ["HIST-001"],
    "recommended_action": "Block transaction.",
    "reviewer_notes":     "",
}

MOCK_EVENTS = [
    {
        "event_id":           "EVT-001",
        "account_id":         "ACC-00001",
        "timestamp":          "2026-01-01T12:00:00",
        "transaction_amount": 4850.0,
        "risk_score":         0.93,
        "risk_level":         "CRITICAL",
        "xgb_prob":           0.95,
        "iso_anom":           0.88,
        "flags":              ["xgboost_high"],
        "explanation":        "velocity_1hr (+0.38)",
    }
]


@pytest.fixture(scope="module")
def client():
    """
    Creates a TestClient with the queue and feedback handler mocked
    so tests don't touch the filesystem or ChromaDB.
    """
    mock_queue = MagicMock()
    mock_queue.get_pending_cases.return_value = [MOCK_CASE]
    mock_queue.get_case.side_effect = lambda cid: MOCK_CASE if cid == "CASE-API-001" else None

    mock_feedback = MagicMock()
    mock_feedback.log_decision.return_value = True

    with (
        patch("src.api.routes.cases.queue_manager",   mock_queue),
        patch("src.api.routes.cases.feedback_handler", mock_feedback),
    ):
        from src.api.main import app
        yield TestClient(app)


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthCheck:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_returns_version(self, client):
        data = client.get("/health").json()
        assert "version" in data


# ── Cases endpoints ───────────────────────────────────────────────────────────

class TestCasesEndpoints:

    def test_get_cases_returns_200(self, client):
        response = client.get("/cases/")
        assert response.status_code == 200

    def test_get_cases_returns_list(self, client):
        data = client.get("/cases/").json()
        assert isinstance(data, list)

    def test_get_cases_contains_mock_case(self, client):
        data = client.get("/cases/").json()
        assert any(c["case_id"] == "CASE-API-001" for c in data)

    def test_get_case_by_id_returns_200(self, client):
        response = client.get("/cases/CASE-API-001")
        assert response.status_code == 200

    def test_get_case_by_id_returns_correct_case(self, client):
        data = client.get("/cases/CASE-API-001").json()
        assert data["case_id"] == "CASE-API-001"
        assert data["fraud_type"] == "Account Takeover"

    def test_get_nonexistent_case_returns_404(self, client):
        response = client.get("/cases/NONEXISTENT-999")
        assert response.status_code == 404

    def test_submit_review_returns_200(self, client):
        payload = {
            "case_id":        "CASE-API-001",
            "decision":       "approve",
            "reviewer_notes": "Confirmed fraud.",
            "reviewer_id":    "test_analyst",
        }
        response = client.post("/cases/CASE-API-001/review", json=payload)
        assert response.status_code == 200

    def test_submit_review_returns_success_status(self, client):
        payload = {
            "case_id":  "CASE-API-001",
            "decision": "dismiss",
        }
        data = client.post("/cases/CASE-API-001/review", json=payload).json()
        assert data["status"] == "success"

    def test_submit_review_mismatched_case_id_returns_400(self, client):
        payload = {
            "case_id":  "DIFFERENT-ID",
            "decision": "approve",
        }
        response = client.post("/cases/CASE-API-001/review", json=payload)
        assert response.status_code == 400

    def test_submit_review_nonexistent_case_returns_404(self, client):
        payload = {
            "case_id":  "NONEXISTENT-999",
            "decision": "approve",
        }
        response = client.post("/cases/NONEXISTENT-999/review", json=payload)
        assert response.status_code == 404

    def test_submit_review_invalid_decision_returns_422(self, client):
        payload = {
            "case_id":  "CASE-API-001",
            "decision": "invalid_action",
        }
        response = client.post("/cases/CASE-API-001/review", json=payload)
        assert response.status_code == 422


# ── Events endpoints ──────────────────────────────────────────────────────────

class TestEventsEndpoints:

    def test_get_risk_events_returns_200(self, client):
        response = client.get("/events/risk")
        assert response.status_code == 200

    def test_get_risk_events_returns_list(self, client):
        data = client.get("/events/risk").json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_risk_events_have_required_fields(self, client):
        events = client.get("/events/risk").json()
        required = ["event_id", "account_id", "risk_score", "risk_level"]
        for event in events[:3]:
            for field in required:
                assert field in event, f"Missing field '{field}' in event"

    def test_get_risk_events_sorted_by_risk_score(self, client):
        events = client.get("/events/risk").json()
        scores = [e["risk_score"] for e in events]
        assert scores == sorted(scores, reverse=True), "Events should be sorted by risk_score descending"

    def test_get_nonexistent_event_returns_404(self, client):
        response = client.get("/events/NONEXISTENT-EVT-999")
        assert response.status_code == 404


# ── Ingest endpoint ───────────────────────────────────────────────────────────

class TestIngestEndpoint:

    def test_ingest_returns_200(self, client):
        response = client.post("/ingest")
        assert response.status_code == 200

    def test_ingest_returns_run_id(self, client):
        data = client.post("/ingest").json()
        assert "run_id" in data
        assert len(data["run_id"]) == 8

    def test_ingest_with_embed_cases(self, client):
        response = client.post("/ingest?embed_cases=true")
        assert response.status_code == 200
        data = response.json()
        assert data["embed_cases"] is True
