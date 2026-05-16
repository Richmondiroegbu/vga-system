"""Image pipeline status panel (v16.0). Shows S-05 through S-07 status."""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.subheader("🖼️ Image Pipeline Status")

    try:
        identity_data = requests.get(f"{api_base}/identity/{scene_id}", timeout=10).json()
        is_frozen = identity_data.get("is_frozen", False)
        drift = identity_data.get("drift_score", 0.0)
        cum_drift = identity_data.get("cumulative_drift", 0.0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Identity Frozen", "YES ✅" if is_frozen else "NO ⏳")
        col2.metric("Latest Drift", f"{drift:.4f}", delta="✅" if drift <= 0.02 else "❌")
        col3.metric("Cumulative Drift", f"{cum_drift:.4f}", delta="✅" if cum_drift <= 0.15 else "❌")
    except Exception:
        st.info("Identity state not yet available")

    st.caption("Stages: S-05 (FLUX, no LoRA) → S-06 (6A/6B/6C, Consistance LoRA) → S-07 (Z-Image-Turbo)")
