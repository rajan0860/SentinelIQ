"""
Risk Heatmap Page
=================
Visualises risk distribution using Plotly.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Risk Analytics | SentinelIQ", layout="wide")

API_URL = "http://localhost:8000"

st.title("📈 Risk Analytics & Heatmap")
st.markdown("Visualising risk concentration across the transaction stream.")

try:
    response = requests.get(f"{API_URL}/events/risk")
    response.raise_for_status()
    events = response.json()
    
    if events:
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Score Distribution")
            fig1 = px.histogram(
                df, 
                x="risk_score", 
                color="risk_level",
                nbins=20,
                color_discrete_map={
                    "CRITICAL": "#f85149",
                    "HIGH": "#d29922",
                    "MEDIUM": "#58a6ff",
                    "LOW": "#3fb950"
                },
                template="plotly_dark"
            )
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
            
        with col2:
            st.subheader("Transaction Amount vs Risk Score")
            fig2 = px.scatter(
                df,
                x="transaction_amount",
                y="risk_score",
                color="risk_level",
                hover_data=["event_id", "account_id"],
                color_discrete_map={
                    "CRITICAL": "#f85149",
                    "HIGH": "#d29922",
                    "MEDIUM": "#58a6ff",
                    "LOW": "#3fb950"
                },
                template="plotly_dark"
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
            
    else:
        st.info("No data available for visualization.")
        
except Exception as e:
    st.error(f"Error fetching data: {e}")
