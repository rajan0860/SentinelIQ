"""
FastAPI Main Application
========================
Entry point for the SentinelIQ REST API.
Registers all routes and configures CORS.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.api.routes import cases, events, query
from src.api.schemas import HealthResponse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="SentinelIQ API",
    description="Backend API for SentinelIQ fraud detection and investigation platform.",
    version="0.1.0",
)

# Allow Streamlit frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the dashboard URL
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

@app.post("/ingest", tags=["System"])
def trigger_ingestion():
    """
    Triggers a manual ingestion run.
    For this prototype, it returns a simulated success response.
    In a full deployment, this would trigger an async celery task or subprocess.
    """
    return {"status": "Ingestion triggered. Check the review queue shortly."}
