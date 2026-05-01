"""
FastAPI Main Application
========================
Entry point for the SentinelIQ REST API.
Registers all routes and configures CORS.
"""

import logging
import threading
import uuid
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import cases, events, query
from src.api.schemas import HealthResponse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SentinelIQ API",
    description="Backend API for SentinelIQ fraud detection and investigation platform.",
    version="0.1.0",
)

# Allow Streamlit frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cases.router, prefix="/cases", tags=["Cases"])
app.include_router(events.router, prefix="/events", tags=["Events"])
app.include_router(query.router, prefix="/query", tags=["Query"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


def _run_pipeline_background(run_id: str, embed_cases: bool):
    """
    Runs the ingestion pipeline in a background thread so the API
    returns immediately without blocking on the long-running ML + LLM work.
    """
    logger.info(f"[run_id={run_id}] Background ingestion pipeline starting...")
    try:
        # Import here to avoid loading heavy ML deps at startup
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        summary = pipeline.run(embed_cases=embed_cases)
        logger.info(
            f"[run_id={run_id}] Pipeline complete — "
            f"{summary['events_scored']} scored, "
            f"{summary['cases_added_to_queue']} cases queued."
        )
    except Exception as e:
        logger.error(f"[run_id={run_id}] Pipeline failed: {e}")


@app.post("/ingest", tags=["System"])
def trigger_ingestion(embed_cases: bool = False):
    """
    Triggers a manual ingestion + scoring + agent investigation run.

    The pipeline runs in a background thread so this endpoint returns
    immediately. Check GET /cases and GET /events/risk for results.

    Args:
        embed_cases: Set to true on first run to seed ChromaDB with
                     historical fraud cases.
    """
    run_id = uuid.uuid4().hex[:8].upper()
    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(run_id, embed_cases),
        daemon=True,
    )
    thread.start()

    return {
        "status": "Ingestion pipeline started.",
        "run_id": run_id,
        "embed_cases": embed_cases,
        "started_at": datetime.utcnow().isoformat(),
        "message": (
            "Pipeline is running in the background. "
            "Check GET /events/risk and GET /cases for results shortly."
        ),
    }
