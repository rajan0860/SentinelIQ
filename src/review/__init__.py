"""
Review Package
==============
Human-in-the-loop review queue and RAG feedback loop.

ReviewQueueManager    → JSON-backed persistent case queue
ReviewFeedbackHandler → maps decisions to outcomes, upserts to ChromaDB
ReviewDecision        → Pydantic schema for approve / escalate / dismiss
"""
