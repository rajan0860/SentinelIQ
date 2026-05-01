"""
Dashboard Package
=================
Streamlit multi-page dashboard for SentinelIQ.

Entry point:  src/dashboard/app.py  (streamlit run src/dashboard/app.py)

Pages (auto-discovered by Streamlit from the pages/ directory):
    live_feed.py      Live transaction feed with risk scores
    risk_heatmap.py   Risk distribution analytics and scatter plots
    graph_view.py     Interactive account-device-IP network visualisation
    review_queue.py   Human analyst case review interface
    query.py          Natural language knowledge base query

Components (src/dashboard/components/):
    metrics_bar.py    Horizontal row of st.metric cards
    risk_badge.py     Colour-coded HTML risk level badge
    case_card.py      Full investigation case card with review controls
"""
