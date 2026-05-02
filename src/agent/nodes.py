"""
Nodes Module
============
Contains the node functions for the LangGraph Investigation Agent.
Each node takes the InvestigationState, performs an action, and
returns updates to the state.
"""

import json
import logging
import os
import time
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import InvestigationState
from src.agent.reports import CaseReport
from src.rag.retriever import retrieve_similar_cases
from src.rag.prompts import INVESTIGATION_SYSTEM_PROMPT, build_investigation_prompt
from src.llm.llm_client import get_llm

logger = logging.getLogger(__name__)

# Maximum LLM call attempts before giving up and returning a fallback report
_LLM_MAX_RETRIES = 3
_LLM_RETRY_DELAY = 2  # seconds between retries


def flag_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Node 1: Flag
    Checks if the ML risk score exceeds the high-risk threshold.
    """
    logger.info(f"--- Node: Flag [{state['event_id']}] ---")

    score_data = state.get("ml_score", {})
    risk_score = score_data.get("risk_score", 0.0)

    threshold = float(os.getenv("RISK_THRESHOLD_HIGH", "0.75"))
    is_flagged = risk_score >= threshold
    logger.info(f"Risk score: {risk_score:.2f} | Threshold: {threshold} | Flagged: {is_flagged}")

    return {"is_flagged": is_flagged}


def retrieve_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Node 2: Retrieve
    Queries the RAG vector store for similar historical cases.
    Logs a warning if the knowledge base is empty (not yet seeded).
    """
    logger.info(f"--- Node: Retrieve [{state['event_id']}] ---")

    explanation = state.get("ml_score", {}).get("explanation", "")
    event_data  = state.get("event_data", {})
    query_text  = explanation if explanation else str(event_data)

    retrieved_cases = retrieve_similar_cases(query_text, k=3)

    if not retrieved_cases:
        logger.warning(
            "No similar cases found in ChromaDB. "
            "Run `python scripts/ingest_and_run.py --embed-cases` to seed the knowledge base."
        )

    return {"retrieved_cases": retrieved_cases}


def analyse_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Node 3: Analyse
    Calls the LLM with the system prompt, event details, ML score,
    and retrieved context. Retries up to _LLM_MAX_RETRIES times on
    transient failures (e.g. Ollama timeout or connection reset).
    """
    logger.info(f"--- Node: Analyse [{state['event_id']}] ---")

    llm = get_llm()
    user_prompt = build_investigation_prompt(
        event_data=state.get("event_data", {}),
        score_data=state.get("ml_score", {}),
        retrieved_cases=state.get("retrieved_cases", []),
    )
    messages = [
        SystemMessage(content=INVESTIGATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    last_error = None
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            logger.info(f"Calling LLM (attempt {attempt}/{_LLM_MAX_RETRIES})...")
            response = llm.invoke(messages)
            # Log only a truncated preview to avoid PII in logs
            preview = response.content[:120].replace("\n", " ")
            logger.debug(f"LLM response preview: {preview}...")
            return {"llm_raw_response": response.content}
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed (attempt {attempt}): {e}")
            if attempt < _LLM_MAX_RETRIES:
                time.sleep(_LLM_RETRY_DELAY)

    # All retries exhausted — return a structured error payload so report_node
    # can produce a meaningful fallback rather than crashing the graph.
    logger.error(f"LLM call failed after {_LLM_MAX_RETRIES} attempts: {last_error}")
    fallback = json.dumps({
        "fraud_type":         "LLM Unavailable",
        "confidence":         0.0,
        "evidence_summary":   f"LLM analysis failed after {_LLM_MAX_RETRIES} attempts. Manual review required.",
        "similar_cases":      [],
        "recommended_action": "Escalate to Engineering — Ollama may be down.",
    })
    return {"llm_raw_response": fallback}


def report_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Node 4: Report
    Parses the LLM's raw JSON response into the CaseReport Pydantic model
    to ensure structured validation, then converts it back to a dict for the state.
    """
    logger.info(f"--- Node: Report [{state['event_id']}] ---")

    raw_response = state.get("llm_raw_response", "{}")
    event_id     = state.get("event_id", "UNKNOWN")

    # Strip markdown code fences that some LLMs add despite instructions
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        report_model = CaseReport(
            case_id=event_id,
            fraud_type=parsed.get("fraud_type", "Unknown"),
            confidence=float(parsed.get("confidence", 0.0)),
            evidence_summary=parsed.get("evidence_summary", "No summary provided."),
            similar_cases=parsed.get("similar_cases", []),
            recommended_action=parsed.get("recommended_action", "Manual Review"),
        )
        final_report = report_model.model_dump()
        logger.info(f"CaseReport validated — fraud_type={final_report['fraud_type']}, confidence={final_report['confidence']}")

    except Exception as e:
        logger.error(f"Failed to parse LLM response into CaseReport: {e}")
        final_report = {
            "case_id":            event_id,
            "fraud_type":         "Parsing Error",
            "confidence":         0.0,
            "evidence_summary":   "The LLM failed to output valid JSON. Manual review required.",
            "similar_cases":      [],
            "recommended_action": "Escalate to Engineering",
            "reviewer_notes":     f"Parse error: {str(e)}",
        }

    return {"final_report": final_report}
