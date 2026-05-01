"""
Cases Router
============
Endpoints for managing LangGraph case reports and human reviews.
"""

from fastapi import APIRouter, HTTPException, Path
from typing import List, Dict, Any

from src.review.queue import ReviewQueueManager
from src.review.feedback import ReviewFeedbackHandler
from src.review.schemas import ReviewDecision
from src.agent.reports import CaseReport

router = APIRouter()
queue_manager = ReviewQueueManager()
feedback_handler = ReviewFeedbackHandler()

@router.get("/", response_model=List[Dict[str, Any]])
def get_pending_cases():
    """
    Retrieves all cases currently waiting in the human review queue.
    """
    return queue_manager.get_pending_cases()

@router.get("/{case_id}", response_model=Dict[str, Any])
def get_case(case_id: str = Path(..., description="The ID of the case to retrieve")):
    """
    Retrieves a specific case report from the review queue.
    """
    case = queue_manager.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found in pending queue.")
    return case

@router.post("/{case_id}/review")
def submit_review_decision(decision: ReviewDecision, case_id: str = Path(...)):
    """
    Submits a human review decision for a case.
    This triggers the RAG feedback loop, updating the knowledge base.
    """
    if case_id != decision.case_id:
        raise HTTPException(status_code=400, detail="Path case_id must match body case_id")

    # Explicit 404 before attempting the feedback log
    if not queue_manager.get_case(case_id):
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found in review queue.")

    success = feedback_handler.log_decision(decision.model_dump())
    if not success:
        raise HTTPException(status_code=500, detail="Failed to log decision.")

    return {"status": "success", "message": f"Decision for {case_id} logged to RAG knowledge base."}
