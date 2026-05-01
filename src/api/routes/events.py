"""
Events Router
=============
Endpoints for fetching raw event data and their risk scores.

GET /events/risk  — returns scored events from the last pipeline run.
                    Falls back to mock data if no pipeline run has occurred yet,
                    so the dashboard works out of the box before first ingest.

GET /events/{id}  — returns detail for a single event by ID.
"""

import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

# Path written by IngestionPipeline.run() — relative to project root
_SCORED_EVENTS_PATH = os.getenv(
    "SCORED_EVENTS_PATH",
    "data/processed/scored_events.json",
)


def _load_scored_events() -> List[Dict[str, Any]]:
    """
    Load scored events from the JSON file written by the ingestion pipeline.
    Returns an empty list if the file doesn't exist yet.
    """
    path = Path(_SCORED_EVENTS_PATH)
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read scored events from {path}: {e}")
        return []


def _generate_mock_events(count: int = 50) -> List[Dict[str, Any]]:
    """
    Fallback mock data used when no pipeline run has produced real scored events.
    Clearly labelled so it's obvious in the dashboard that this is demo data.
    """
    events = []
    base_time = datetime.now()

    for _ in range(count):
        is_high_risk = random.random() < 0.15

        if is_high_risk:
            risk_score = random.uniform(0.75, 0.98)
            risk_level = "CRITICAL" if risk_score >= 0.90 else "HIGH"
            flags = ["xgboost_high"]
        else:
            risk_score = random.uniform(0.05, 0.45)
            risk_level = "LOW" if risk_score < 0.40 else "MEDIUM"
            flags = []

        events.append({
            "event_id":           f"MOCK-{uuid.uuid4().hex[:8].upper()}",
            "account_id":         f"ACC-{random.randint(1000, 9999)}",
            "timestamp":          (base_time - timedelta(minutes=random.randint(1, 60))).isoformat(),
            "transaction_amount": round(random.uniform(10.0, 5000.0), 2),
            "risk_score":         round(risk_score, 4),
            "risk_level":         risk_level,
            "xgb_prob":           round(random.uniform(0.0, 1.0), 4),
            "iso_anom":           round(random.uniform(0.0, 1.0), 4),
            "flags":              flags,
            "explanation":        "Mock data — run ingest_and_run.py for real scores.",
        })

    events.sort(key=lambda x: x["risk_score"], reverse=True)
    return events


@router.get("/risk", response_model=List[Dict[str, Any]])
def get_risk_events():
    """
    Returns scored events ranked by risk score (highest first).

    Reads from data/processed/scored_events.json produced by the ingestion
    pipeline. Falls back to mock data if the file doesn't exist yet so the
    dashboard is usable before the first pipeline run.
    """
    events = _load_scored_events()

    if events:
        logger.info(f"Returning {len(events)} real scored events.")
        return events

    logger.warning(
        "No scored events found at %s — returning mock data. "
        "Run `python scripts/ingest_and_run.py` to generate real scores.",
        _SCORED_EVENTS_PATH,
    )
    return _generate_mock_events(count=100)


@router.get("/{event_id}", response_model=Dict[str, Any])
def get_event_detail(event_id: str):
    """
    Returns the full scored record for a single event by ID.
    """
    events = _load_scored_events()

    for event in events:
        if event.get("event_id") == event_id:
            return event

    raise HTTPException(
        status_code=404,
        detail=f"Event '{event_id}' not found. "
               "It may not have been ingested yet, or the ID is incorrect.",
    )
