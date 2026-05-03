"""
Review Queue Page
=================
The primary workspace for human analysts to review LangGraph investigation reports.
"""

import streamlit as st
import requests

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.dashboard.components.metrics_bar import render_metrics_bar
from src.dashboard.components.case_card import render_case_card

API_URL = "http://localhost:8000"

st.title("⚖️ Case Review Queue")
st.markdown("Review agent investigations and submit decisions to train the system.")

if st.button("🔄 Refresh Queue"):
    st.rerun()

def fetch_cases():
    try:
        res = requests.get(f"{API_URL}/cases/", timeout=5)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ConnectionError:
        st.error("🚨 Could not connect to the FastAPI backend. Is it running on port 8000?")
        return []
    except Exception as e:
        st.error(f"Failed to fetch cases: {e}")
        return []

cases = fetch_cases()

if cases is not None:
    if not cases:
        st.success("🎉 The queue is empty. All cases have been reviewed!")
    else:
        # Summary metrics
        total = len(cases)
        high_conf = sum(1 for c in cases if c.get("confidence", 0) >= 0.90)
        avg_conf = sum(c.get("confidence", 0) for c in cases) / total if total else 0

        render_metrics_bar([
            {"label": "Pending Cases", "value": total},
            {"label": "Critical Confidence (≥90%)", "value": high_conf},
            {"label": "Avg Confidence", "value": f"{avg_conf:.0%}"},
        ])

        st.divider()

        # Sort by confidence descending so highest-risk cases appear first
        cases_sorted = sorted(cases, key=lambda c: c.get("confidence", 0), reverse=True)

        for case in cases_sorted:
            render_case_card(case, api_url=API_URL, on_decision=st.rerun)
