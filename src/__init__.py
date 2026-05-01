"""
SentinelIQ — AI-powered fraud detection and investigation platform.

Package structure:
    src.agent       LangGraph investigation agent (flag → retrieve → analyse → report)
    src.api         FastAPI REST backend
    src.dashboard   Streamlit multi-page dashboard
    src.ingestion   Event loading, graph building, and pipeline orchestration
    src.llm         Ollama LLM and embedding client wrappers
    src.ml          Ensemble ML scorer (XGBoost + Isolation Forest + SHAP)
    src.rag         ChromaDB vector store, retriever, and prompts
    src.review      Human-in-the-loop review queue and RAG feedback loop
"""
