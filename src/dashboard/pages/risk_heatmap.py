"""
Risk Heatmap Page
=================
Visualises risk distribution using Plotly.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.title("📈 Risk Analytics & Heatmap")
st.markdown("Visualising risk concentration across the transaction stream.")

if st.button("🔄 Refresh"):
    st.rerun()

try:
    with st.spinner("Loading event data..."):
        response = requests.get(f"{API_URL}/events/risk", timeout=10)
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
                    "HIGH":     "#d29922",
                    "MEDIUM":   "#58a6ff",
                    "LOW":      "#3fb950",
                },
                template="plotly_dark",
                labels={"risk_score": "Risk Score", "count": "Events"},
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
                    "HIGH":     "#d29922",
                    "MEDIUM":   "#58a6ff",
                    "LOW":      "#3fb950",
                },
                template="plotly_dark",
                labels={"transaction_amount": "Transaction Amount ($)", "risk_score": "Risk Score"},
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

        # Risk level breakdown table
        st.divider()
        st.subheader("Risk Level Summary")
        summary = df.groupby("risk_level").agg(
            count=("event_id", "count"),
            avg_score=("risk_score", "mean"),
            avg_amount=("transaction_amount", "mean"),
        ).reset_index().sort_values("avg_score", ascending=False)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    else:
        st.info(
            "📊 No event data available yet.\n\n"
            "**Next steps:**\n"
            "1. Run `python scripts/generate_data.py` to create synthetic events\n"
            "2. Run `python scripts/ingest_and_run.py` to score them\n"
            "3. Or click **▶️ Trigger Ingestion Run** on the Live Feed page"
        )

except requests.exceptions.ConnectionError:
    st.error("🚨 Could not connect to the FastAPI backend. Is it running on port 8000?")
except Exception as e:
    st.error(f"Error fetching data: {e}")
