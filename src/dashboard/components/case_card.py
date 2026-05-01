"""
Case Card Component
===================
Renders a single investigation case as an expandable card with
evidence summary, similar cases, and a three-button review interface.

Extracted from review_queue.py to allow reuse on other pages.

Usage:
    from src.dashboard.components.case_card import render_case_card

    render_case_card(case, api_url="http://localhost:8000", on_decision=st.rerun)
"""

import streamlit as st
import requests
from typing import Callable, Optional

from src.dashboard.components.risk_badge import risk_badge_html


def render_case_card(
    case: dict,
    api_url: str = "http://localhost:8000",
    on_decision: Optional[Callable] = None,
) -> None:
    """
    Render a single case as a Streamlit expander with review controls.

    Args:
        case:        Case dict from GET /cases (must have case_id, fraud_type,
                     confidence, evidence_summary, similar_cases, recommended_action)
        api_url:     Base URL of the FastAPI backend.
        on_decision: Callback invoked after a successful review submission
                     (typically st.rerun to refresh the queue).
    """
    case_id = case.get("case_id", "UNKNOWN")
    fraud_type = case.get("fraud_type", "Unknown")
    confidence = case.get("confidence", 0.0)
    risk_level = _confidence_to_risk(confidence)

    badge = risk_badge_html(risk_level)
    label = f"**{case_id}** — {fraud_type} {badge} (confidence: {confidence:.0%})"

    with st.expander(label, expanded=False):
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Agent Evidence Summary")
            st.info(case.get("evidence_summary", "No summary available."))

            similar = case.get("similar_cases", [])
            if similar:
                st.subheader("Similar Historical Cases")
                st.write(", ".join(similar))

        with col_right:
            st.subheader("Recommended Action")
            st.warning(case.get("recommended_action", "Manual review required."))

            st.divider()
            st.write("**Submit Review Decision**")

            notes_key = f"notes_{case_id}"
            notes = st.text_area("Reviewer Notes", key=notes_key, height=80)

            c1, c2, c3 = st.columns(3)

            def _submit(decision: str) -> None:
                payload = {
                    "case_id": case_id,
                    "decision": decision,
                    "reviewer_notes": st.session_state.get(notes_key, ""),
                    "reviewer_id": "human_analyst",
                }
                try:
                    r = requests.post(
                        f"{api_url}/cases/{case_id}/review",
                        json=payload,
                        timeout=10,
                    )
                    r.raise_for_status()
                    st.success(f"Decision '{decision}' logged for {case_id}.")
                    if on_decision:
                        on_decision()
                except Exception as e:
                    st.error(f"Failed to submit decision: {e}")

            if c1.button("✅ Approve", key=f"app_{case_id}", help="Confirm as fraud"):
                _submit("approve")
            if c2.button("⚠️ Escalate", key=f"esc_{case_id}", help="Needs deeper review"):
                _submit("escalate")
            if c3.button("❌ Dismiss", key=f"dis_{case_id}", help="Mark as false positive"):
                _submit("dismiss")


def _confidence_to_risk(confidence: float) -> str:
    """Map a confidence float to a risk level string for badge colouring."""
    if confidence >= 0.90:
        return "CRITICAL"
    elif confidence >= 0.75:
        return "HIGH"
    elif confidence >= 0.50:
        return "MEDIUM"
    return "LOW"
