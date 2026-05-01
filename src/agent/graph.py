"""
Investigation Agent Module
==========================
This module orchestrates the multi-node fraud investigation workflow using LangGraph.
It assembles the analytical nodes (Flag, Retrieve, Analyse, Report) into a
StateGraph and handles the conditional routing logic based on the initial
risk assessment from the ML ensemble.
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.agent.state import InvestigationState
from src.agent.nodes import flag_node, retrieve_node, analyse_node, report_node

logger = logging.getLogger(__name__)

def should_investigate(state: InvestigationState) -> str:
    """
    Conditional edge routing function.
    If flagged, proceed to retrieval. Otherwise, end the graph.
    """
    if state.get("is_flagged"):
        return "retrieve"
    return "end"

class InvestigationAgent:
    """
    Wrapper class for the compiled LangGraph agent.
    """
    def __init__(self):
        # 1. Initialize the StateGraph with our TypedDict
        builder = StateGraph(InvestigationState)
        
        # 2. Add nodes
        builder.add_node("flag", flag_node)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("analyse", analyse_node)
        builder.add_node("report", report_node)
        
        # 3. Add edges
        builder.add_edge(START, "flag")
        
        # Conditional edge: after 'flag', do we investigate or end?
        builder.add_conditional_edges(
            "flag",
            should_investigate,
            {
                "retrieve": "retrieve",
                "end": END
            }
        )
        
        builder.add_edge("retrieve", "analyse")
        builder.add_edge("analyse", "report")
        builder.add_edge("report", END)
        
        # 4. Compile graph
        self.graph = builder.compile()
        logger.info("InvestigationAgent graph compiled successfully.")
        
    def investigate(self, event_id: str, event_data: dict, ml_score: dict) -> dict:
        """
        Main entry point to run an event through the agent.
        
        Args:
            event_id: Unique identifier for the case/event.
            event_data: The raw event details.
            ml_score: The output dictionary from the ML scorer.
            
        Returns:
            The final_report dictionary, or None if not flagged.
        """
        logger.info(f"Starting investigation for {event_id}...")
        
        # Initialize the state
        initial_state = {
            "event_id": event_id,
            "event_data": event_data,
            "ml_score": ml_score,
            "is_flagged": False,
            "retrieved_cases": [],
            "llm_raw_response": "",
            "final_report": None
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Return the resulting report (if it was flagged and investigated)
        report = final_state.get("final_report")
        if report:
            logger.info(f"Investigation complete for {event_id}. Fraud Type: {report.get('fraud_type')}")
        else:
            logger.info(f"Event {event_id} was not flagged. No report generated.")
            
        return report

# Quick verification test
if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    
    agent = InvestigationAgent()
    
    test_event = {
        "transaction_amount": 4850.00,
        "account_age_days": 12,
        "velocity_1hr": 8
    }
    
    test_score = {
        "risk_score": 0.93,
        "risk_level": "CRITICAL",
        "explanation": "High velocity and new account age."
    }
    
    result = agent.investigate("TEST-123", test_event, test_score)
    print("\nFinal Result:")
    pprint.pprint(result)
