"""
Live Feed Page
==============
Monitors incoming transactions and their assigned risk scores.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Live Feed | SentinelIQ", layout="wide")

from src.dashboard.components.metrics_bar import render_metrics_bar
from src.dashboard.components.risk_badge import risk_badge_html

API_URL = "http://localhost:8000"

st.title("📡 Live Transaction Feed")
st.markdown("Monitoring real-time events processed by the ML Ensemble.")

col_refresh, col_ingest = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh Feed"):
        st.rerun()
with col_ingest:
    if st.button("▶️ Trigger Ingestion Run"):
        try:
            r = requests.post(f"{API_URL}/ingest", timeout=5)
            r.raise_for_status()
            st.success("Ingestion pipeline started in background. Refresh in ~30 seconds.")
        except Exception as e:
            st.error(f"Failed to trigger ingestion: {e}")

try:
    with st.spinner("Fetching latest events..."):
        response = requests.get(f"{API_URL}/events/risk", timeout=10)
        response.raise_for_status()
        events = response.json()

    if events:
        total    = len(events)
        critical = sum(1 for e in events if e.get("risk_level") == "CRITICAL")
        high     = sum(1 for e in events if e.get("risk_level") == "HIGH")
        medium   = sum(1 for e in events if e.get("risk_level") == "MEDIUM")

        render_metrics_bar([
            {"label": "Total Events", "value": f"{total:,}"},
            {"label": "🔴 Critical", "value": critical,
             "delta": f"{critical/total*100:.1f}%", "delta_color": "inverse"},
            {"label": "🟠 High",     "value": high},
            {"label": "🔵 Medium",   "value": medium},
        ])

        st.divider()

        df = pd.DataFrame(events)

        # Keep only columns that exist in the dataframe
        desired_cols = ["timestamp", "event_id", "account_id",
                        "transaction_amount", "risk_score", "risk_level", "flags"]
        available_cols = [c for c in desired_cols if c in df.columns]
        df = df[available_cols]

        def color_risk(val):
            colours = {
                "CRITICAL": "#f85149",
                "HIGH":     "#d29922",
                "MEDIUM":   "#58a6ff",
                "LOW":      "#3fb950",
            }
            c = colours.get(val, "")
            return f"color: {c}; font-weight: bold" if c else ""

        fmt = {"transaction_amount": "${:.2f}", "risk_score": "{:.4f}"}
        styled_df = df.style.map(color_risk, subset=["risk_level"]).format(
            {k: v for k, v in fmt.items() if k in df.columns}
        )

        st.dataframe(styled_df, use_container_width=True, height=600)

    else:
        st.info("No events found. Run `python scripts/ingest_and_run.py` to generate scored events.")

except requests.exceptions.ConnectionError:
    st.error("🚨 Could not connect to the FastAPI backend. Is it running on port 8000?")
except Exception as e:
    st.error(f"Error fetching data: {e}")
