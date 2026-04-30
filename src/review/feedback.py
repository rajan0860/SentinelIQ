"""
Feedback Module
===============
Handles logging human decisions and feeding them back into the
ChromaDB vector store to improve future LangGraph agent retrievals.
"""

import logging
import uuid
from typing import Dict, Any

from src.review.schemas import ReviewDecision
from src.review.queue import ReviewQueueManager
from src.rag.vector_store import get_vector_store
from src.rag.embeddings import get_configured_embeddings

logger = logging.getLogger(__name__)

class ReviewFeedbackHandler:
    """
    Processes human review decisions and updates the RAG knowledge base.
    """
    def __init__(self):
        self.queue_manager = ReviewQueueManager()
        _, self.collection = get_vector_store()
        self.embedder = get_configured_embeddings()

    def _map_decision_to_outcome(self, decision: str) -> str:
        """
        Maps the frontend 'decision' action to the backend 'outcome' label
        used by the RAG system.
        """
        mapping = {
            "approve": "confirmed_fraud",
            "escalate": "under_investigation",
            "dismiss": "false_positive"
        }
        return mapping.get(decision, "unknown")

    def log_decision(self, decision_data: Dict[str, Any]) -> bool:
        """
        Logs a decision, removes the case from the queue, and updates ChromaDB.
        
        Args:
            decision_data: Dictionary matching the ReviewDecision schema.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # 1. Validate the decision payload
            decision = ReviewDecision(**decision_data)
            case_id = decision.case_id
            logger.info(f"Processing review decision for {case_id}: {decision.decision}")
            
            # 2. Retrieve the original case report from the queue before removing it
            case_report = self.queue_manager.get_case(case_id)
            if not case_report:
                logger.error(f"Cannot log decision: Case {case_id} not found in queue.")
                return False
                
            # 3. Remove from queue
            self.queue_manager.remove_case(case_id)
            
            # 4. Prepare document for ChromaDB
            # We construct a rich text summary that the RAG retriever can search against.
            outcome_label = self._map_decision_to_outcome(decision.decision)
            
            document_text = f"Case Report for {case_id}.\n"
            document_text += f"Agent Analysis: {case_report.get('evidence_summary', '')}\n"
            document_text += f"Reviewer Notes: {decision.reviewer_notes}\n"
            document_text += f"Final Outcome: {outcome_label}"
            
            metadata = {
                "fraud_type": case_report.get("fraud_type", "Unknown"),
                "outcome": outcome_label,
                "recommended_action": case_report.get("recommended_action", ""),
                "evidence": case_report.get("evidence_summary", ""),
                "reviewer": decision.reviewer_id
            }
            
            # Use the case_id as the document ID in Chroma so we can update it if reviewed again
            doc_id = f"review-{case_id}"
            
            # 5. Embed and Upsert to ChromaDB
            logger.info(f"Embedding and upserting document {doc_id} to ChromaDB...")
            embedding = self.embedder.embed_query(document_text)
            
            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[metadata]
            )
            
            logger.info(f"Successfully fed case {case_id} back into RAG knowledge base.")
            return True
            
        except Exception as e:
            logger.error(f"Error logging review decision: {e}")
            return False
