"""
tests/test_rag.py
=================
Unit tests for the RAG pipeline — vector store upsert, retrieval,
metadata filtering, and embedding consistency.

Tests use an isolated in-memory ChromaDB collection so they don't
pollute the production data/chroma store and can run without Ollama
by mocking the embedding model.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CASES = [
    {
        "case_id":            "HIST-001",
        "summary":            "Account ACC-11111 showed rapid velocity from a foreign IP. Device was new.",
        "fraud_type":         "account_takeover",
        "outcome":            "confirmed_fraud",
        "recommended_action": "Block device and freeze account.",
        "evidence":           ["IP mismatch", "New device", "High velocity"],
    },
    {
        "case_id":            "HIST-002",
        "summary":            "Three new accounts shared the same device ID. Synthetic identity ring suspected.",
        "fraud_type":         "synthetic_identity",
        "outcome":            "confirmed_fraud",
        "recommended_action": "Close all linked accounts.",
        "evidence":           ["Shared device", "New accounts", "Coordinated pattern"],
    },
    {
        "case_id":            "HIST-003",
        "summary":            "Customer disputed a transaction that was shipped to their verified address.",
        "fraud_type":         "first_party_fraud",
        "outcome":            "false_positive",
        "recommended_action": "Deny dispute.",
        "evidence":           ["Address match", "No device anomaly"],
    },
]


def _make_fake_embedder(dim: int = 8):
    """
    Returns a mock embedder that produces deterministic fixed-length vectors.
    Avoids needing a live Ollama instance during tests.
    """
    import hashlib

    def _embed(text: str):
        # Deterministic: same text → same vector
        h = hashlib.md5(text.encode()).digest()
        # Repeat bytes to fill `dim` floats
        raw = list(h) * (dim // len(h) + 1)
        vec = [b / 255.0 for b in raw[:dim]]
        return vec

    embedder = MagicMock()
    embedder.embed_documents = lambda docs: [_embed(d) for d in docs]
    embedder.embed_query     = lambda q: _embed(q)
    return embedder


@pytest.fixture
def chroma_collection():
    """
    Provides an isolated in-memory ChromaDB collection for each test.
    Automatically torn down after the test.
    """
    import chromadb
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="test_fraud_cases",
        metadata={"hnsw:space": "cosine"},
    )
    yield collection
    client.delete_collection("test_fraud_cases")


@pytest.fixture
def populated_collection(chroma_collection):
    """
    Collection pre-loaded with SAMPLE_CASES using the fake embedder.
    """
    embedder = _make_fake_embedder()
    documents  = [c["summary"] for c in SAMPLE_CASES]
    embeddings = embedder.embed_documents(documents)
    metadatas  = [
        {
            "fraud_type":         c["fraud_type"],
            "outcome":            c["outcome"],
            "recommended_action": c["recommended_action"],
            "evidence":           ", ".join(c["evidence"]),
        }
        for c in SAMPLE_CASES
    ]
    ids = [c["case_id"] for c in SAMPLE_CASES]

    chroma_collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return chroma_collection, embedder


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVectorStoreUpsert:

    def test_upsert_adds_documents(self, chroma_collection):
        embedder = _make_fake_embedder()
        docs = ["Test fraud case summary."]
        chroma_collection.upsert(
            ids=["TEST-001"],
            documents=docs,
            embeddings=embedder.embed_documents(docs),
            metadatas=[{"fraud_type": "test", "outcome": "confirmed_fraud",
                        "recommended_action": "block", "evidence": "none"}],
        )
        assert chroma_collection.count() == 1

    def test_upsert_is_idempotent(self, chroma_collection):
        """Upserting the same ID twice should not create a duplicate."""
        embedder = _make_fake_embedder()
        docs = ["Duplicate case."]
        meta = [{"fraud_type": "test", "outcome": "confirmed_fraud",
                 "recommended_action": "block", "evidence": "none"}]

        for _ in range(2):
            chroma_collection.upsert(
                ids=["DUP-001"],
                documents=docs,
                embeddings=embedder.embed_documents(docs),
                metadatas=meta,
            )

        assert chroma_collection.count() == 1

    def test_upsert_multiple_cases(self, populated_collection):
        collection, _ = populated_collection
        assert collection.count() == len(SAMPLE_CASES)


class TestRetrieval:

    def test_retrieve_returns_k_results(self, populated_collection):
        collection, embedder = populated_collection
        query_embedding = embedder.embed_query("account takeover foreign IP")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["metadatas", "documents", "distances"],
        )
        assert len(results["ids"][0]) == 2

    def test_retrieve_returns_at_most_collection_size(self, populated_collection):
        """Requesting more results than documents should not raise an error."""
        collection, embedder = populated_collection
        query_embedding = embedder.embed_query("fraud")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=100,   # more than the 3 docs we have
            include=["metadatas", "documents", "distances"],
        )
        assert len(results["ids"][0]) <= len(SAMPLE_CASES)

    def test_retrieve_includes_metadata(self, populated_collection):
        collection, embedder = populated_collection
        query_embedding = embedder.embed_query("synthetic identity ring")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["metadatas", "documents", "distances"],
        )
        meta = results["metadatas"][0][0]
        assert "fraud_type"         in meta
        assert "outcome"            in meta
        assert "recommended_action" in meta

    def test_retrieve_distances_are_valid(self, populated_collection):
        """Cosine distances should be in [0, 2]."""
        collection, embedder = populated_collection
        query_embedding = embedder.embed_query("velocity spike")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["distances"],
        )
        for dist in results["distances"][0]:
            assert 0.0 <= dist <= 2.0, f"Distance {dist} out of expected range."


class TestMetadataFilter:

    def test_filter_by_outcome_confirmed_fraud(self, populated_collection):
        collection, embedder = populated_collection
        query_embedding = embedder.embed_query("fraud")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"outcome": "confirmed_fraud"},
            include=["metadatas"],
        )
        for meta in results["metadatas"][0]:
            assert meta["outcome"] == "confirmed_fraud"

    def test_filter_by_outcome_false_positive(self, populated_collection):
        collection, embedder = populated_collection
        query_embedding = embedder.embed_query("dispute")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"outcome": "false_positive"},
            include=["metadatas"],
        )
        for meta in results["metadatas"][0]:
            assert meta["outcome"] == "false_positive"

    def test_filter_returns_empty_for_unknown_outcome(self, populated_collection):
        collection, embedder = populated_collection
        query_embedding = embedder.embed_query("fraud")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"outcome": "nonexistent_outcome"},
            include=["metadatas"],
        )
        assert len(results["ids"][0]) == 0


class TestRetrieveSimilarCasesFunction:
    """
    Integration-style tests for the public retrieve_similar_cases() function,
    patching the embedder and vector store so no live services are needed.
    """

    def test_retrieve_similar_cases_returns_list(self, populated_collection):
        collection, embedder = populated_collection

        with (
            patch("src.rag.retriever.get_vector_store", return_value=(None, collection)),
            patch("src.rag.retriever.get_configured_embeddings", return_value=embedder),
        ):
            from src.rag.retriever import retrieve_similar_cases
            results = retrieve_similar_cases("account takeover", k=2)

        assert isinstance(results, list)
        assert len(results) <= 2

    def test_retrieve_similar_cases_result_structure(self, populated_collection):
        collection, embedder = populated_collection

        with (
            patch("src.rag.retriever.get_vector_store", return_value=(None, collection)),
            patch("src.rag.retriever.get_configured_embeddings", return_value=embedder),
        ):
            from src.rag.retriever import retrieve_similar_cases
            results = retrieve_similar_cases("synthetic identity", k=1)

        if results:
            case = results[0]
            assert "case_id"          in case
            assert "summary"          in case
            assert "fraud_type"       in case
            assert "outcome"          in case
            assert "similarity_score" in case
            assert 0.0 <= case["similarity_score"] <= 1.0
