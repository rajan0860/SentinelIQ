"""
Vector Store Module
===================
Initialises and manages the ChromaDB instance.
Provides functionality to load and upsert historical fraud cases
so the LangGraph agent can retrieve them as context.
"""

import json
import logging
import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag.embeddings import get_configured_embeddings

load_dotenv()
logger = logging.getLogger(__name__)

# Load vector store configuration
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "fraud_cases")

def get_vector_store():
    """
    Returns the ChromaDB client and the target collection.
    If the collection doesn't exist, it creates it.
    """
    logger.info(f"Initialising ChromaDB at {_PERSIST_DIR}")
    
    # We use PersistentClient to ensure the database survives restarts
    client = chromadb.PersistentClient(path=_PERSIST_DIR)
    
    # We don't pass the LangChain embedding function directly to the
    # chromadb python client during collection creation in this setup.
    # Instead, we'll use LangChain's Chroma wrapper later, OR we can
    # handle embeddings manually before inserting. We'll use the native
    # ChromaDB client here for raw upserts, computing embeddings manually
    # to maintain strict control over the indexing process.
    
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Cosine similarity is standard for text
    )
    
    return client, collection

def upsert_historical_cases(json_path: str = "data/synthetic/historical_cases.json"):
    """
    Loads historical cases from a JSON file, computes embeddings, and
    upserts them into ChromaDB.
    """
    if not os.path.exists(json_path):
        logger.error(f"Historical cases file not found: {json_path}")
        return

    _, collection = get_vector_store()
    
    logger.info(f"Loading historical cases from {json_path}")
    with open(json_path, "r") as f:
        cases = json.load(f)
        
    if not cases:
        logger.warning("No cases found in JSON.")
        return

    # Extract data into lists for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for case in cases:
        # The 'summary' is the main text we want to search against
        documents.append(case["summary"])
        
        # Store the rest as metadata for filtering and agent consumption
        # We need to join lists into strings because ChromaDB metadata
        # only supports strings, ints, or floats.
        metadatas.append({
            "fraud_type": case["fraud_type"],
            "outcome": case["outcome"],
            "recommended_action": case["recommended_action"],
            "evidence": ", ".join(case["evidence"])
        })
        
        # Use the case_id as the primary key
        ids.append(case["case_id"])
        
    logger.info(f"Computing embeddings and upserting {len(documents)} cases...")
    
    # We fetch the embedding model we configured in Step 4.2
    embedder = get_configured_embeddings()
    embeddings = embedder.embed_documents(documents)
    
    # Upsert into Chroma (updates if ID exists, inserts if new)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    logger.info(f"Successfully upserted {len(ids)} documents into '{_COLLECTION_NAME}' collection.")

if __name__ == "__main__":
    upsert_historical_cases()
