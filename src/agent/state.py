"""
State Module
============
Defines the state schema for the LangGraph Investigation Agent.
We use TypedDict to define the state that is passed between nodes.
"""

from typing import TypedDict, Any, Dict, List, Optional

class InvestigationState(TypedDict):
    """
    Represents the state of a single fraud investigation case
    as it moves through the LangGraph nodes.
    """
    event_id: str
    event_data: Dict[str, Any]
    ml_score: Dict[str, Any]      # Contains risk_score, risk_level, explanation
    
    # Updated by flag_node
    is_flagged: bool
    
    # Updated by retrieve_node
    retrieved_cases: List[Dict[str, Any]]
    
    # Updated by analyse_node
    llm_raw_response: str
    
    # Updated by report_node
    final_report: Optional[Dict[str, Any]]
