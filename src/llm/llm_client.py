"""
LLM Client Module
=================
Provides a unified wrapper for local AI inference via Ollama.
Instantiates both the Chat model (for investigation) and the
Embedding model (for RAG).

Configuration is read from the .env file.
"""

import logging
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_llm() -> ChatOllama:
    """
    Initialise and return the ChatOllama model instance.
    Uses OLLAMA_BASE_URL and OLLAMA_MODEL from environment variables.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    
    logger.info(f"Initialising ChatOllama with model: {model_name} at {base_url}")
    
    # We set temperature=0 for deterministic, analytical outputs
    # essential for fraud investigation.
    return ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=0.0,
    )

def get_embeddings() -> OllamaEmbeddings:
    """
    Initialise and return the OllamaEmbeddings instance.
    Uses OLLAMA_BASE_URL and OLLAMA_EMBED_MODEL from environment variables.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    
    logger.info(f"Initialising OllamaEmbeddings with model: {embed_model} at {base_url}")
    
    return OllamaEmbeddings(
        base_url=base_url,
        model=embed_model,
    )

# Quick verification
if __name__ == "__main__":
    llm = get_llm()
    embedder = get_embeddings()
    print("LLM Client successfully initialised.")
