"""
FastAPI Main Application
========================
Entry point for the SentinelIQ REST API.
Registers all routes, configures CORS, and starts the background
ingestion scheduler so the pipeline runs automatically every
AGENT_RUN_INTERVAL minutes (default: 15).
"""

import atexit
import logging
import os
import threading
import uuid
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import cases, events, query
from src.api.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SentinelIQ API",
    description="Backend API for SentinelIQ fraud detection and investigation platform.",
    version="0.3.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Defaults to localhost only. Override DASHBOARD_URL in .env for production.
_DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:8501")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_DASHBOARD_URL, "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(cases.router, prefix="/cases", tags=["Cases"])
app.include_router(events.router, prefix="/events", tags=["Events"])
app.include_router(query.router, prefix="/query", tags=["Query"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check endpoint."""
    return {"status": "ok", "version": "0.3.0"}


# ── Background pipeline runner ────────────────────────────────────────────────
def _run_pipeline_background(run_id: str, embed_cases: bool = False) -> None:
    """
    Runs the ingestion pipeline in a background thread so the API
    returns immediately without blocking on the long-running ML + LLM work.
    """
    logger.info(f"[run_id={run_id}] Background ingestion pipeline starting...")
    try:
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        summary = pipeline.run(embed_cases=embed_cases)
        logger.info(
            f"[run_id={run_id}] Pipeline complete — "
            f"{summary['events_scored']} scored, "
            f"{summary['cases_added_to_queue']} cases queued."
        )
    except Exception as e:
        logger.error(f"[run_id={run_id}] Pipeline failed: {e}", exc_info=True)


# ── Scheduled ingestion ───────────────────────────────────────────────────────
_INTERVAL_MINUTES = int(os.getenv("AGENT_RUN_INTERVAL", "15"))

def _scheduled_pipeline_job() -> None:
    """Called by APScheduler every AGENT_RUN_INTERVAL minutes."""
    run_id = uuid.uuid4().hex[:8].upper()
    logger.info(f"[scheduler] Triggering scheduled pipeline run (run_id={run_id})")
    _run_pipeline_background(run_id=run_id, embed_cases=False)


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(
    func=_scheduled_pipeline_job,
    trigger="interval",
    minutes=_INTERVAL_MINUTES,
    id="ingestion_pipeline",
    name=f"Ingestion pipeline (every {_INTERVAL_MINUTES} min)",
    replace_existing=True,
)
scheduler.start()
atexit.register(lambda: scheduler.shutdown(wait=False))
logger.info(f"Scheduled ingestion pipeline: every {_INTERVAL_MINUTES} minutes.")


# ── Manual ingest endpoint ────────────────────────────────────────────────────
@app.post("/ingest", tags=["System"])
def trigger_ingestion(embed_cases: bool = False):
    """
    Manually triggers an ingestion + scoring + agent investigation run.

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

    next_run = scheduler.get_job("ingestion_pipeline").next_run_time
    return {
        "status": "Ingestion pipeline started.",
        "run_id": run_id,
        "embed_cases": embed_cases,
        "started_at": datetime.utcnow().isoformat(),
        "next_scheduled_run": str(next_run) if next_run else "N/A",
        "message": (
            "Pipeline is running in the background. "
            "Check GET /events/risk and GET /cases for results shortly."
        ),
    }
