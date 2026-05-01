"""
Risk Badge Component
====================
Renders a colour-coded HTML badge for a risk level string.
Used in tables and case cards across the dashboard.

Usage:
    from src.dashboard.components.risk_badge import risk_badge_html, render_risk_badge

    # Get raw HTML (for use inside st.markdown)
    html = risk_badge_html("CRITICAL")
    st.markdown(html, unsafe_allow_html=True)

    # Or render directly
    render_risk_badge("HIGH")
"""

import streamlit as st

# Colour map matching the dashboard's dark-mode palette
_RISK_COLOURS: dict[str, tuple[str, str]] = {
    "CRITICAL": ("#f85149", "#fff"),
    "HIGH":     ("#d29922", "#fff"),
    "MEDIUM":   ("#58a6ff", "#fff"),
    "LOW":      ("#3fb950", "#fff"),
}

_DEFAULT_COLOUR = ("#8b949e", "#fff")


def risk_badge_html(risk_level: str) -> str:
    """
    Return an inline HTML badge string for the given risk level.

    Args:
        risk_level: One of "CRITICAL", "HIGH", "MEDIUM", "LOW"

    Returns:
        HTML string with a styled <span> badge.
    """
    bg, fg = _RISK_COLOURS.get(risk_level.upper(), _DEFAULT_COLOUR)
    return (
        f'<span style="'
        f"background-color:{bg};"
        f"color:{fg};"
        f"padding:2px 8px;"
        f"border-radius:4px;"
        f"font-size:0.8em;"
        f"font-weight:bold;"
        f'">{risk_level}</span>'
    )


def render_risk_badge(risk_level: str) -> None:
    """
    Render a risk badge directly into the Streamlit page.

    Args:
        risk_level: One of "CRITICAL", "HIGH", "MEDIUM", "LOW"
    """
    st.markdown(risk_badge_html(risk_level), unsafe_allow_html=True)
