"""
LLM Package
===========
Wrappers for local Ollama inference.

get_llm()         → ChatOllama instance (qwen2.5:7b-instruct by default)
get_embeddings()  → OllamaEmbeddings instance (nomic-embed-text by default)

All configuration is read from environment variables (see .env.example).
"""
