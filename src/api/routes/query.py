"""
Query Router
============
Provides a Natural Language interface to query the RAG knowledge base.
"""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage

from src.api.schemas import QueryRequest, QueryResponse
from src.rag.retriever import retrieve_similar_cases
from src.llm.llm_client import get_llm

router = APIRouter()

SYSTEM_PROMPT = """You are SentinelIQ, an expert fraud investigation AI.
The user is asking a question about the fraud knowledge base.
Use the provided historical cases to answer the question.
If the answer is not in the historical cases, say so. Do not hallucinate.
Keep your answer concise and professional."""

@router.post("/", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """
    Query the RAG knowledge base using natural language.
    """
    try:
        # 1. Retrieve context
        retrieved_cases = retrieve_similar_cases(request.query, k=3)
        
        # 2. Format context for LLM
        context_str = "\n".join(
            [f"- Case {c['case_id']} ({c['outcome']}): {c['summary']}" for c in retrieved_cases]
        )
        
        prompt = f"User Query: {request.query}\n\nHistorical Context:\n{context_str}"
        
        # 3. Call LLM
        llm = get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        
        return QueryResponse(
            answer=response.content,
            retrieved_cases=retrieved_cases
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query knowledge base: {str(e)}")
