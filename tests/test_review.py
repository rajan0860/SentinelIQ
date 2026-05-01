"""
tests/test_review.py
====================
Unit tests for the human-in-the-loop review system:
    - ReviewQueueManager  (queue.py)
    - ReviewFeedbackHandler (feedback.py)
    - ReviewDecision schema (schemas.py)

The ChromaDB collection and embedder are mocked so these tests run
without a live vector store or Ollama instance.
"""

import os
import sys
import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Shared fixtures ───────────────────────────────────────────────────────────

MOCK_CASE_1 = {
    "case_id":            "CASE-TEST-001",
    "fraud_type":         "Synthetic Identity",
    "confidence":         0.88,
    "evidence_summary":   "Account shares a device with 5 other accounts.",
    "similar_cases":      ["HIST-001"],
    "recommended_action": "Escalate",
    "reviewer_notes":     "",
}

MOCK_CASE_2 = {
    "case_id":            "CASE-TEST-002",
    "fraud_type":         "Account Takeover",
    "confidence":         0.95,
    "evidence_summary":   "Foreign IP, new device, high velocity.",
    "similar_cases":      ["HIST-002"],
    "recommended_action": "Block transaction",
    "reviewer_notes":     "",
}


@pytest.fixture
def tmp_queue_path(tmp_path):
    """Provides a temporary JSON file path for each test — no shared state."""
    return str(tmp_path / "test_review_queue.json")


@pytest.fixture
def queue_manager(tmp_queue_path):
    from src.review.queue import ReviewQueueManager
    return ReviewQueueManager(queue_path=tmp_queue_path)


# ── ReviewQueueManager tests ──────────────────────────────────────────────────

class TestReviewQueueManager:

    def test_queue_starts_empty(self, queue_manager):
        cases = queue_manager.get_pending_cases()
        assert cases == []

    def test_add_case(self, queue_manager):
        queue_manager.add_case(MOCK_CASE_1)
        cases = queue_manager.get_pending_cases()
        assert len(cases) == 1
        assert cases[0]["case_id"] == "CASE-TEST-001"

    def test_add_multiple_cases(self, queue_manager):
        queue_manager.add_case(MOCK_CASE_1)
        queue_manager.add_case(MOCK_CASE_2)
        cases = queue_manager.get_pending_cases()
        assert len(cases) == 2

    def test_duplicate_prevention(self, queue_manager):
        """Adding the same case_id twice should not create a duplicate."""
        queue_manager.add_case(MOCK_CASE_1)
        queue_manager.add_case(MOCK_CASE_1)
        cases = queue_manager.get_pending_cases()
        assert len(cases) == 1

    def test_get_case_by_id(self, queue_manager):
        queue_manager.add_case(MOCK_CASE_1)
        case = queue_manager.get_case("CASE-TEST-001")
        assert case is not None
        assert case["fraud_type"] == "Synthetic Identity"

    def test_get_case_returns_none_for_unknown_id(self, queue_manager):
        case = queue_manager.get_case("NONEXISTENT-999")
        assert case is None

    def test_remove_case(self, queue_manager):
        queue_manager.add_case(MOCK_CASE_1)
        queue_manager.add_case(MOCK_CASE_2)
        queue_manager.remove_case("CASE-TEST-001")
        cases = queue_manager.get_pending_cases()
        assert len(cases) == 1
        assert cases[0]["case_id"] == "CASE-TEST-002"

    def test_remove_nonexistent_case_does_not_raise(self, queue_manager):
        """Removing a case that doesn't exist should be a no-op."""
        queue_manager.remove_case("NONEXISTENT-999")   # should not raise
        assert queue_manager.get_pending_cases() == []

    def test_queue_persists_to_disk(self, tmp_queue_path):
        """Data written by one manager instance should be readable by another."""
        from src.review.queue import ReviewQueueManager
        mgr1 = ReviewQueueManager(queue_path=tmp_queue_path)
        mgr1.add_case(MOCK_CASE_1)

        mgr2 = ReviewQueueManager(queue_path=tmp_queue_path)
        cases = mgr2.get_pending_cases()
        assert len(cases) == 1
        assert cases[0]["case_id"] == "CASE-TEST-001"

    def test_queue_file_is_valid_json(self, tmp_queue_path, queue_manager):
        queue_manager.add_case(MOCK_CASE_1)
        with open(tmp_queue_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)


# ── ReviewDecision schema tests ───────────────────────────────────────────────

class TestReviewDecisionSchema:

    def test_valid_approve_decision(self):
        from src.review.schemas import ReviewDecision
        d = ReviewDecision(case_id="CASE-001", decision="approve")
        assert d.decision == "approve"
        assert d.reviewer_notes == ""
        assert d.reviewer_id == "system"

    def test_valid_escalate_decision(self):
        from src.review.schemas import ReviewDecision
        d = ReviewDecision(
            case_id="CASE-002",
            decision="escalate",
            reviewer_notes="Needs senior review.",
            reviewer_id="analyst_01",
        )
        assert d.decision == "escalate"
        assert d.reviewer_id == "analyst_01"

    def test_valid_dismiss_decision(self):
        from src.review.schemas import ReviewDecision
        d = ReviewDecision(case_id="CASE-003", decision="dismiss")
        assert d.decision == "dismiss"

    def test_invalid_decision_raises_validation_error(self):
        from src.review.schemas import ReviewDecision
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ReviewDecision(case_id="CASE-004", decision="invalid_action")

    def test_missing_case_id_raises_validation_error(self):
        from src.review.schemas import ReviewDecision
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ReviewDecision(decision="approve")


# ── ReviewFeedbackHandler tests ───────────────────────────────────────────────

class TestReviewFeedbackHandler:

    def _make_handler(self, tmp_queue_path):
        """
        Creates a ReviewFeedbackHandler with a mocked ChromaDB collection
        and embedder so no live services are needed.
        """
        mock_collection = MagicMock()
        mock_embedder   = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 8

        with (
            patch("src.review.feedback.get_vector_store",
                  return_value=(None, mock_collection)),
            patch("src.review.feedback.get_configured_embeddings",
                  return_value=mock_embedder),
        ):
            from src.review.feedback import ReviewFeedbackHandler
            from src.review.queue import ReviewQueueManager

            handler = ReviewFeedbackHandler.__new__(ReviewFeedbackHandler)
            handler.queue_manager = ReviewQueueManager(queue_path=tmp_queue_path)
            handler.collection    = mock_collection
            handler.embedder      = mock_embedder

        return handler, mock_collection

    def test_decision_maps_approve_to_confirmed_fraud(self, tmp_queue_path):
        handler, _ = self._make_handler(tmp_queue_path)
        assert handler._map_decision_to_outcome("approve")   == "confirmed_fraud"

    def test_decision_maps_escalate_to_under_investigation(self, tmp_queue_path):
        handler, _ = self._make_handler(tmp_queue_path)
        assert handler._map_decision_to_outcome("escalate")  == "under_investigation"

    def test_decision_maps_dismiss_to_false_positive(self, tmp_queue_path):
        handler, _ = self._make_handler(tmp_queue_path)
        assert handler._map_decision_to_outcome("dismiss")   == "false_positive"

    def test_decision_maps_unknown_to_unknown(self, tmp_queue_path):
        handler, _ = self._make_handler(tmp_queue_path)
        assert handler._map_decision_to_outcome("something") == "unknown"

    def test_log_decision_removes_case_from_queue(self, tmp_queue_path):
        handler, _ = self._make_handler(tmp_queue_path)
        handler.queue_manager.add_case(MOCK_CASE_1)

        success = handler.log_decision({
            "case_id":        "CASE-TEST-001",
            "decision":       "approve",
            "reviewer_notes": "Confirmed fraud ring.",
            "reviewer_id":    "analyst_01",
        })

        assert success is True
        assert handler.queue_manager.get_case("CASE-TEST-001") is None

    def test_log_decision_upserts_to_chromadb(self, tmp_queue_path):
        handler, mock_collection = self._make_handler(tmp_queue_path)
        handler.queue_manager.add_case(MOCK_CASE_1)

        handler.log_decision({
            "case_id":        "CASE-TEST-001",
            "decision":       "approve",
            "reviewer_notes": "Confirmed.",
            "reviewer_id":    "analyst_01",
        })

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["review-CASE-TEST-001"]
        assert call_kwargs["metadatas"][0]["outcome"] == "confirmed_fraud"

    def test_log_decision_fails_gracefully_for_missing_case(self, tmp_queue_path):
        handler, _ = self._make_handler(tmp_queue_path)
        # Don't add the case to the queue first
        success = handler.log_decision({
            "case_id":  "NONEXISTENT-999",
            "decision": "approve",
        })
        assert success is False

    def test_log_decision_embeds_reviewer_notes_in_document(self, tmp_queue_path):
        handler, mock_collection = self._make_handler(tmp_queue_path)
        handler.queue_manager.add_case(MOCK_CASE_1)

        handler.log_decision({
            "case_id":        "CASE-TEST-001",
            "decision":       "escalate",
            "reviewer_notes": "Linked to known fraud ring XYZ.",
            "reviewer_id":    "analyst_02",
        })

        call_kwargs = mock_collection.upsert.call_args[1]
        document_text = call_kwargs["documents"][0]
        assert "Linked to known fraud ring XYZ." in document_text
