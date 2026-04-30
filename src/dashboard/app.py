"""
Streamlit Main Application
==========================
Entry point for the SentinelIQ Dashboard.
Sets up the multi-page navigation and injects custom CSS
for a premium dark mode UI.
"""

import streamlit as st

st.set_page_config(
    page_title="SentinelIQ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode CSS
st.markdown("""
<style>
    /* Main Backgrounds */
    .stApp {
        background-color: #0f1115;
        color: #e2e8f0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f0f6fc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics and Cards */
    [data-testid="stMetricValue"] {
        color: #58a6ff;
    }
    .metric-card {
        background-color: #1c2128;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #238636;
        color: white;
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: #8b949e;
    }
    
    /* Warning/Critical Colors */
    .risk-critical { color: #f85149; font-weight: bold; }
    .risk-high { color: #d29922; font-weight: bold; }
    .risk-medium { color: #58a6ff; font-weight: bold; }
    .risk-low { color: #3fb950; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ SentinelIQ Command Center")
st.markdown("Welcome to the SentinelIQ fraud operations platform. Select a view from the sidebar.")

st.info("Ensure the FastAPI backend is running on `http://localhost:8000`.")
