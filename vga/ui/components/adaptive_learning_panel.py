"""Adaptive learning panel (v15.0). Shows historical stage performance and calibration."""
from __future__ import annotations


def render(api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st

    st.subheader("🧠 Adaptive Learning Status")
    st.info("Adaptive calibration uses exponential smoothing (α=0.90) to update SLA thresholds based on historical performance.")
    st.caption("Historical data accumulates across pipeline sessions. Improves stage timing estimates over time.")
