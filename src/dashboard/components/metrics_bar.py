"""
Metrics Bar Component
=====================
Reusable top-of-page metrics strip used across multiple dashboard pages.
Renders a row of st.metric cards with consistent styling.

Usage:
    from src.dashboard.components.metrics_bar import render_metrics_bar

    render_metrics_bar([
        {"label": "Total Events", "value": 1024},
        {"label": "Critical Risk", "value": 12, "delta": "-3", "delta_color": "inverse"},
        {"label": "High Risk", "value": 45},
    ])
"""

import streamlit as st
from typing import List, Dict, Any


def render_metrics_bar(metrics: List[Dict[str, Any]]) -> None:
    """
    Render a horizontal row of metric cards.

    Args:
        metrics: List of dicts, each with:
            - label (str):        Metric label
            - value (int|float|str): Metric value
            - delta (str, opt):   Delta string (e.g. "+5%")
            - delta_color (str, opt): "normal" | "inverse" | "off"
            - help (str, opt):    Tooltip text
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        col.metric(
            label=metric["label"],
            value=metric["value"],
            delta=metric.get("delta"),
            delta_color=metric.get("delta_color", "normal"),
            help=metric.get("help"),
        )
