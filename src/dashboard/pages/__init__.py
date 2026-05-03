"""
Dashboard Pages
===============
Multi-page views for the SentinelIQ dashboard.

Pages are imported and executed by src/dashboard/app.py based on
the sidebar navigation selection. Each page module contains only
Streamlit rendering code — no st.set_page_config() calls (that is
handled once in app.py).

Available pages:
    - live_feed:     Real-time transaction risk feed
    - risk_heatmap:  Risk distribution charts and scatter plots
    - graph_view:    Interactive account-device-IP network graph
    - review_queue:  Human-in-the-loop case review interface
    - query:         Natural language RAG knowledge base query
"""
