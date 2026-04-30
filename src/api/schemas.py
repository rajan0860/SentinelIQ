"""
API Schemas Module
==================
Defines the Pydantic models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.review.schemas import ReviewDecision
from src.agent.reports import CaseReport

class HealthResponse(BaseModel):
    status: str
    version: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language query to ask the RAG knowledge base.")

class QueryResponse(BaseModel):
    answer: str
    retrieved_cases: List[Dict[str, Any]]
