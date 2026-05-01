# Contributing to SentinelIQ

Thank you for your interest in contributing to SentinelIQ! This project is designed to be a modular, extensible platform for local-first fraud detection and investigation.

## Project Philosophy

1. **Local-First AI**: All core intelligence should default to local inference (via Ollama). Avoid adding mandatory dependencies on cloud LLM APIs unless explicitly structured as an optional provider.
2. **Explainability by Design**: Every score or decision must be accompanied by an explanation (e.g., SHAP, evidence summary, graph context).
3. **Strict Type Safety**: Use Pydantic models for all data interchange between modules and via the API.
4. **Reproducibility**: Ensure all synthetic data generation and model training is deterministic via fixed seeds.

## Core Architecture Overview

### 1. The ML Ensemble (`src/ml/`)
We use a hybrid of supervised (`XGBoost`) and unsupervised (`Isolation Forest`) models. 
- To add a new model: Implement it in `src/ml/ensemble.py` and update the `EnsembleScorer.score()` method.
- To add new features: Update `src/ml/graph_features.py` (for graph-derived features) or `src/ingestion/event_loader.py` (for raw features).

### 2. The LangGraph Agent (`src/agent/`)
The investigation logic is managed by a LangGraph state machine.
- `state.py`: Defines the `AgentState` TypedDict.
- `nodes.py`: Contains the logic for `flag_node`, `retrieve_node`, `analyse_node`, and `report_node`.
- `graph.py`: Orchestrates the flow and conditional routing.

### 3. The RAG Knowledge Base (`src/rag/`)
We use ChromaDB to store and retrieve historical cases.
- Decisions made in the dashboard are fed back into ChromaDB via `src/review/feedback.py`.
- Future investigations retrieve these decisions to provide better context to the agent.

## How to Contribute

1. **Bug Fixes**: Ensure you add a test case in `tests/` that fails without your fix.
2. **New Features**: 
    - Outline your proposed change in an Issue.
    - Follow the existing module structure (`src/`, `scripts/`, `tests/`).
    - Update the `README.md` if the change affects the setup or architecture.
3. **Testing**: Run the full suite before submitting:
    ```bash
    pytest tests/
    ```

## Coding Standards

- **Docstrings**: Use Google-style docstrings for all classes and functions.
- **Linting**: We follow standard PEP 8 guidelines.
- **Logging**: Use the standard `logging` module. Avoid `print()` statements in `src/`.
