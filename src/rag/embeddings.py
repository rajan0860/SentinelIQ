"""
Embeddings Module
=================
Provides the configured embedding model for ChromaDB.
Acts as a bridge between the LLM client and the RAG vector store.
"""

import logging
from src.llm.llm_client import get_embeddings

logger = logging.getLogger(__name__)

def get_configured_embeddings():
    """
    Returns the instantiated OllamaEmbeddings model.
    This abstraction ensures we can swap the underlying embedding provider
    (e.g., to OpenAI or Cohere) without changing the vector store logic.
    """
    logger.info("Fetching configured embeddings for vector store.")
    return get_embeddings()

if __name__ == "__main__":
    emb = get_configured_embeddings()
    print("Embeddings configured successfully.")
