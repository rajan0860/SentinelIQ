"""
Events Router
=============
Endpoints for fetching raw event data and their risk scores.
"""

from fastapi import APIRouter
from typing import List, Dict, Any
import random
import uuid
from datetime import datetime, timedelta

router = APIRouter()

# For the prototype, we generate mock events on the fly to simulate
# a stream of recently scored transactions.
def _generate_mock_events(count: int = 50) -> List[Dict[str, Any]]:
    events = []
    base_time = datetime.now()
    
    for i in range(count):
        # Determine if this mock event is high risk or not
        is_high_risk = random.random() < 0.15 # 15% chance of being high risk in the feed
        
        if is_high_risk:
            risk_score = random.uniform(0.75, 0.98)
            risk_level = "CRITICAL" if risk_score >= 0.90 else "HIGH"
        else:
            risk_score = random.uniform(0.05, 0.45)
            risk_level = "LOW" if risk_score < 0.40 else "MEDIUM"
            
        events.append({
            "event_id": f"EVT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": (base_time - timedelta(minutes=random.randint(1, 60))).isoformat(),
            "transaction_amount": round(random.uniform(10.0, 5000.0), 2),
            "account_id": f"ACC-{random.randint(1000, 9999)}",
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "flags": ["xgboost_high"] if risk_score > 0.8 else []
        })
    
    # Sort by risk score descending
    events.sort(key=lambda x: x["risk_score"], reverse=True)
    return events

@router.get("/risk", response_model=List[Dict[str, Any]])
def get_risk_events():
    """
    Retrieves a list of recently scored events.
    Used to populate the Live Feed and Risk Heatmap on the dashboard.
    """
    # We use mock data here to simulate real-time streaming without a DB.
    return _generate_mock_events(count=100)

@router.get("/{event_id}")
def get_event_detail(event_id: str):
    """
    Retrieves details for a single event.
    """
    # In a real system, this fetches from the feature store / DB.
    return {"event_id": event_id, "status": "Event details mock"}
