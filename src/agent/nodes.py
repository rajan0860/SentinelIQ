"""
Nodes Module
============
Contains the node functions for the LangGraph Investigation Agent.
Each node takes the InvestigationState, performs an action, and
returns updates to the state.
"""

import os
import json
import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.state import InvestigationState
from src.agent.reports import CaseReport
from src.rag.retriever import retrieve_similar_cases
from src.rag.prompts import INVESTIGATION_SYSTEM_PROMPT, build_investigation_prompt
from src.llm.llm_client import get_llm

logger = logging.getLogger(__name__)

def flag_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Node 1: Flag
    Checks if the ML risk score exceeds the high-risk threshold.
    """
    logger.info(f"--- Node: Flag [{state['event_id']}] ---")
    
    score_data = state.get("ml_score", {})
    risk_score = score_data.get("risk_score", 0.0)
    
    # Load threshold from environment or default to 0.75
    threshold = float(os.getenv("RISK_THRESHOLD_HIGH", "0.75"))
    
    is_flagged = risk_score >= threshold
    logger.info(f"Risk score: {risk_score:.2f} | Threshold: {threshold} | Flagged: {is_flagged}")
    
    return {"is_flagged": is_flagged}

def retrieve_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Node 2: Retrieve
    Queries the RAG vector store for similar historical cases.
    """
    logger.info(f"--- Node: Retrieve [{state['event_id']}] ---")
    
    # Construct a basic query string from the event data and explanation
    event_data = state.get("event_data", {})
    explanation = state.get("ml_score", {}).get("explanation", "")
    
    # Use explanation if present, otherwise fallback to stringified event data
    query_text = explanation if explanation else str(event_data)
    
    # Retrieve top 3 similar cases
    retrieved_cases = retrieve_similar_cases(query_text, k=3)
    
    return {"retrieved_cases": retrieved_cases}

def analyse_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Node 3: Analyse
    Calls the LLM with the system prompt, event details, ML score,
    and retrieved context to analyse the case.
    """
    logger.info(f"--- Node: Analyse [{state['event_id']}] ---")
    
    llm = get_llm()
    
    user_prompt = build_investigation_prompt(
        event_data=state.get("event_data", {}),
        score_data=state.get("ml_score", {}),
        retrieved_cases=state.get("retrieved_cases", [])
    )
    
    messages = [
        SystemMessage(content=INVESTIGATION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    logger.info("Calling LLM for analysis...")
    response = llm.invoke(messages)
    
    raw_response = response.content
    logger.debug(f"LLM Raw Response: {raw_response}")
    
    return {"llm_raw_response": raw_response}

def report_node(state: InvestigationState) -> Dict[str, Any]:
    """
    Node 4: Report
    Parses the LLM's raw JSON response into the CaseReport Pydantic model
    to ensure structured validation, then converts it back to a dict for the state.
    """
    logger.info(f"--- Node: Report [{state['event_id']}] ---")
    
    raw_response = state.get("llm_raw_response", "{}")
    event_id = state.get("event_id", "UNKNOWN")
    
    # Try to clean up the response in case the LLM ignored instructions
    # and added markdown code blocks.
    cleaned_response = raw_response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    
    try:
        parsed_json = json.loads(cleaned_response)
        
        # Validate with Pydantic
        report_model = CaseReport(
            case_id=event_id,
            fraud_type=parsed_json.get("fraud_type", "Unknown"),
            confidence=parsed_json.get("confidence", 0.0),
            evidence_summary=parsed_json.get("evidence_summary", "Failed to parse evidence."),
            similar_cases=parsed_json.get("similar_cases", []),
            recommended_action=parsed_json.get("recommended_action", "Manual Review")
        )
        
        final_report = report_model.model_dump()
        logger.info("Successfully validated CaseReport.")
        
    except Exception as e:
        logger.error(f"Failed to parse LLM response into CaseReport: {e}")
        logger.error(f"Raw response was: {raw_response}")
        # Provide a fallback report
        final_report = {
            "case_id": event_id,
            "fraud_type": "Parsing Error",
            "confidence": 0.0,
            "evidence_summary": "The LLM failed to output valid JSON. Manual review required.",
            "similar_cases": [],
            "recommended_action": "Escalate to Engineering",
            "reviewer_notes": f"Error: {str(e)}"
        }
        
    return {"final_report": final_report}
