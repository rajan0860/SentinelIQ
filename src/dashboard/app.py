"""
Streamlit Main Application
==========================
Entry point for the SentinelIQ Dashboard.
Sets up the multi-page navigation and injects custom CSS
for a premium dark mode UI.

NOTE: st.set_page_config() is called ONLY here. Individual page files
must NOT call it — Streamlit only allows one call per session and it
must be the first Streamlit command executed.
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

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SentinelIQ")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home",
            "📡 Live Feed",
            "📈 Risk Heatmap",
            "🕸️ Graph View",
            "⚖️ Review Queue",
            "🧠 Knowledge Base",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("v0.2.0 · Local AI · No data leaves your network")

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🛡️ SentinelIQ Command Center")
    st.markdown("Welcome to the SentinelIQ fraud operations platform. Select a view from the sidebar.")
    st.info("Ensure the FastAPI backend is running on `http://localhost:8000`.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📡 Live Feed")
        st.markdown("Monitor incoming transactions and their ML risk scores in real time.")
    with col2:
        st.markdown("### ⚖️ Review Queue")
        st.markdown("Review LangGraph investigation reports and submit approve / escalate / dismiss decisions.")
    with col3:
        st.markdown("### 🧠 Knowledge Base")
        st.markdown("Query historical fraud patterns using natural language over the RAG knowledge base.")

elif page == "📡 Live Feed":
    from src.dashboard.pages import live_feed  # noqa: F401  — executes the page module

elif page == "📈 Risk Heatmap":
    from src.dashboard.pages import risk_heatmap  # noqa: F401

elif page == "🕸️ Graph View":
    from src.dashboard.pages import graph_view  # noqa: F401

elif page == "⚖️ Review Queue":
    from src.dashboard.pages import review_queue  # noqa: F401

elif page == "🧠 Knowledge Base":
    from src.dashboard.pages import query  # noqa: F401
