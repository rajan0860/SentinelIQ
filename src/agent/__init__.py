"""
Agent Package
=============
LangGraph-based fraud investigation agent.

Nodes:  flag → retrieve → analyse → report
Entry:  InvestigationAgent (graph.py)
State:  InvestigationState TypedDict (state.py)
Output: CaseReport Pydantic model (reports.py)
"""
