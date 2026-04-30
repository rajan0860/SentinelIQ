"""
Live Feed Page
==============
Monitors incoming transactions and their assigned risk scores.
"""

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Live Feed | SentinelIQ", layout="wide")

API_URL = "http://localhost:8000"

st.title("📡 Live Transaction Feed")
st.markdown("Monitoring real-time events processed by the ML Ensemble.")

# Add a refresh button
if st.button("🔄 Refresh Feed"):
    st.rerun()

try:
    with st.spinner("Fetching latest events..."):
        response = requests.get(f"{API_URL}/events/risk")
        response.raise_for_status()
        events = response.json()
        
    if events:
        # High level metrics
        total = len(events)
        critical = sum(1 for e in events if e.get("risk_level") == "CRITICAL")
        high = sum(1 for e in events if e.get("risk_level") == "HIGH")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events (last 1h)", total)
        col2.metric("Critical Risk", critical, delta=f"{(critical/total)*100:.1f}%", delta_color="inverse")
        col3.metric("High Risk", high)
        
        st.divider()
        
        # Display as a nicely formatted dataframe
        df = pd.DataFrame(events)
        # Reorder columns
        df = df[["timestamp", "event_id", "account_id", "transaction_amount", "risk_score", "risk_level", "flags"]]
        
        # Style the dataframe
        def color_risk(val):
            color = ''
            if val == 'CRITICAL': color = '#f85149'
            elif val == 'HIGH': color = '#d29922'
            elif val == 'MEDIUM': color = '#58a6ff'
            elif val == 'LOW': color = '#3fb950'
            return f'color: {color}; font-weight: bold'
            
        styled_df = df.style.map(color_risk, subset=['risk_level']).format({
            "transaction_amount": "${:.2f}",
            "risk_score": "{:.4f}"
        })
        
        st.dataframe(styled_df, width="stretch", height=600)
        
    else:
        st.info("No events found in the recent stream.")
        
except requests.exceptions.ConnectionError:
    st.error("🚨 Could not connect to the FastAPI backend. Is it running on port 8000?")
except Exception as e:
    st.error(f"Error fetching data: {e}")
