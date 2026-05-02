# Contributing to SentinelIQ

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Project Philosophy

1. **Local-First AI** — All core intelligence defaults to local inference via Ollama. Cloud LLM providers are supported as optional swaps, not defaults.
2. **Explainability by Design** — Every score or decision must be accompanied by an explanation (SHAP, evidence summary, graph context).
3. **Type Safety** — Use Pydantic models for all data interchange between modules and via the API.
4. **Reproducibility** — All synthetic data generation and model training is deterministic via fixed seeds.

---

## Local Development Setup

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/rajan0860/sentineliq.git
cd sentineliq
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install all dependencies (including dev tools)
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env

# 4. Validate setup
python scripts/setup_project.py

# 5. Generate data and train models
python scripts/generate_data.py
python scripts/train_model.py
python scripts/ingest_and_run.py --embed-cases
```

---

## Running the Application

```bash
# Terminal 1 — FastAPI backend (auto-restarts on code changes)
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run src/dashboard/app.py

# API docs
open http://localhost:8000/docs
```

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# Skip tests that require trained model artifacts
pytest tests/ -v -k "not scorer"

# Run a specific test file
pytest tests/test_agent.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

> **Note:** `test_scorer.py` and `test_pipeline.py` (real data tests) require
> `python scripts/train_model.py` and `python scripts/ingest_and_run.py` to have been run first.

---

## Debugging

### Debug the LangGraph Agent
```bash
# Enable verbose LangChain tracing
export LANGCHAIN_VERBOSE=true
python scripts/ingest_and_run.py --max-cases 1
```

### Debug the API
```bash
# Run with debug logging
uvicorn src.api.main:app --reload --log-level debug
```

### Inspect the ChromaDB knowledge base
```python
from src.rag.vector_store import get_vector_store
_, collection = get_vector_store()
print(f"Documents in knowledge base: {collection.count()}")
```

### Inspect the account graph
```bash
python scripts/inspect_graph.py
```

---

## Architecture Overview

```
src/
├── agent/          LangGraph agent: flag → retrieve → analyse → report
├── api/            FastAPI REST backend + scheduled ingestion
├── dashboard/      Streamlit multi-page UI
│   └── components/ Reusable UI widgets (metrics_bar, risk_badge, case_card)
├── ingestion/      Event loading, graph building, pipeline orchestration
├── llm/            Ollama LLM + embedding client wrappers
├── ml/             XGBoost + Isolation Forest + SHAP + graph features
├── rag/            ChromaDB vector store, retriever, prompts
└── review/         Human review queue + RAG feedback loop
```

### Adding a New ML Feature
1. Add the feature to `src/ml/feature_engineering.py` (tabular) or `src/ml/graph_features.py` (graph-derived)
2. Update `src/ingestion/pipeline.py` → `_build_feature_dict()` to include it
3. Retrain: `python scripts/train_model.py`

### Adding a New Agent Node
1. Define the node function in `src/agent/nodes.py`
2. Register it in `src/agent/graph.py` with `builder.add_node()`
3. Add edges to connect it in the graph flow
4. Update `InvestigationState` in `src/agent/state.py` if new state fields are needed

### Adding a New API Endpoint
1. Add the route to the appropriate file in `src/api/routes/`
2. Add request/response Pydantic models to `src/api/schemas.py`
3. Add a test in `tests/test_api.py`

### Adding a New Dashboard Page
1. Create `src/dashboard/pages/your_page.py`
2. Streamlit auto-discovers it — no registration needed
3. Use components from `src/dashboard/components/` for consistency

---

## Code Standards

- **Docstrings** — Google-style for all public classes and functions
- **Type hints** — Required on all function signatures
- **Logging** — Use `logging.getLogger(__name__)`. Never call `logging.basicConfig()` in library modules (only in entry points)
- **No `print()` in `src/`** — Use the logger
- **Pydantic for data** — All inter-module data exchange uses Pydantic models

---

## Pull Request Checklist

- [ ] Tests pass: `pytest tests/`
- [ ] New functionality has tests in `tests/`
- [ ] Docstrings added/updated
- [ ] `README.md` updated if setup or architecture changed
- [ ] No hardcoded paths (use `from src import PROJECT_ROOT`)
- [ ] No `print()` statements in `src/`

---

## Reporting Issues

Please open a GitHub Issue with:
- Python version (`python --version`)
- OS
- Steps to reproduce
- Expected vs actual behaviour
- Relevant log output
