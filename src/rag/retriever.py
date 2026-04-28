"""
Retriever Module
================
Provides the search interface over the RAG vector store.
Given a summary of a current suspicious event, it retrieves
the most mathematically similar historical cases to provide
context to the LangGraph agent.
"""

import logging
import os
import sys
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag.vector_store import get_vector_store
from src.rag.embeddings import get_configured_embeddings

logger = logging.getLogger(__name__)

def retrieve_similar_cases(query_text: str, k: int = 3, filter_outcome: str = None) -> List[Dict]:
    """
    Search ChromaDB for historical cases similar to the query text.
    
    Args:
        query_text: A string describing the current event (e.g., "Account ACC-123 has high velocity...")
        k: Number of similar cases to retrieve
        filter_outcome: Optional metadata filter (e.g., "confirmed_fraud") to only retrieve specific outcomes.
        
    Returns:
        A list of dictionaries containing the retrieved case details.
    """
    logger.info(f"Retrieving top {k} similar cases for query: '{query_text[:50]}...'")
    
    _, collection = get_vector_store()
    embedder = get_configured_embeddings()
    
    # 1. Embed the search query using the exact same model we used to embed the documents
    query_embedding = embedder.embed_query(query_text)
    
    # 2. Build the metadata filter if requested
    where_filter = None
    if filter_outcome:
        where_filter = {"outcome": filter_outcome}
        logger.info(f"Applying metadata filter: {where_filter}")
        
    # 3. Perform the similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_filter,
        include=["metadatas", "documents", "distances"]
    )
    
    # 4. Format the output for the agent
    retrieved_cases = []
    
    # Chroma returns lists of lists (because you can send multiple queries at once)
    # We only sent one query, so we access index 0.
    if results['ids'] and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            case_id = results['ids'][0][i]
            document = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] # Cosine distance (lower is more similar)
            
            retrieved_cases.append({
                "case_id": case_id,
                "summary": document,
                "fraud_type": metadata.get("fraud_type"),
                "outcome": metadata.get("outcome"),
                "recommended_action": metadata.get("recommended_action"),
                "evidence": metadata.get("evidence"),
                "similarity_score": round(1.0 - distance, 4) # Convert distance to similarity score
            })
            
    logger.info(f"Retrieved {len(retrieved_cases)} cases.")
    return retrieved_cases

if __name__ == "__main__":
    # Test the retriever
    test_query = "Account exhibited rapid transaction velocity from an IP in CN while registered in US."
    cases = retrieve_similar_cases(test_query, k=2)
    
    print("\n--- Retrieval Results ---")
    for c in cases:
        print(f"Case ID: {c['case_id']} | Similarity: {c['similarity_score']}")
        print(f"Summary: {c['summary']}\n")
