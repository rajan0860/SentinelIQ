"""
Test Script for Review Queue and Feedback Loop
"""

import sys
import os
import logging
import pprint

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.review.queue import ReviewQueueManager
from src.review.feedback import ReviewFeedbackHandler
from src.rag.retriever import retrieve_similar_cases

logging.basicConfig(level=logging.INFO)

def run_test():
    print("=== Testing Review Queue & Feedback Loop ===")
    
    # 1. Mock a case report from the agent
    mock_case = {
        "case_id": "REVIEW-TEST-001",
        "fraud_type": "Synthetic Identity",
        "confidence": 0.88,
        "evidence_summary": "Account shares a device with 5 other accounts and has high degree centrality.",
        "recommended_action": "Escalate"
    }
    
    # 2. Add to Queue
    print("\n[1] Adding case to queue...")
    qm = ReviewQueueManager()
    qm.add_case(mock_case)
    
    pending = qm.get_pending_cases()
    print(f"Pending cases in queue: {len(pending)}")
    
    # 3. Simulate human reviewer decision
    print("\n[2] Human Reviewer approves the case as Confirmed Fraud...")
    handler = ReviewFeedbackHandler()
    
    decision_payload = {
        "case_id": "REVIEW-TEST-001",
        "decision": "approve",
        "reviewer_notes": "I manually checked the device ID, it is a known emulator. Definitely a synthetic fraud ring.",
        "reviewer_id": "analyst_rajan"
    }
    
    success = handler.log_decision(decision_payload)
    print(f"Log Decision Success: {success}")
    
    # Verify removal from queue
    pending_after = qm.get_pending_cases()
    print(f"Pending cases after review: {len(pending_after)}")
    
    # 4. Verify RAG feedback loop worked by querying ChromaDB
    print("\n[3] Querying RAG to see if the system learned the new case...")
    query = "Account shares a device and is a known emulator."
    retrieved = retrieve_similar_cases(query, k=1)
    
    print("\nTop Retrieved Case:")
    if retrieved:
        pprint.pprint(retrieved[0])
    else:
        print("Nothing retrieved.")
        
if __name__ == "__main__":
    run_test()
