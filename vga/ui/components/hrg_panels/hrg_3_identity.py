"""HRG-3: Identity design review panel. S-03 output review."""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.header("👤 HRG-3: Identity Design Review")
    st.caption("Review character identity and environment design before base image generation.")

    try:
        identity_data = requests.get(f"{api_base}/identity/{scene_id}", timeout=10).json()
        st.subheader("Identity State")
        st.metric("Frozen", str(identity_data.get("is_frozen", False)))
        st.metric("Drift Score", f"{identity_data.get('drift_score', 0.0):.4f}")
        st.json(identity_data)
    except Exception as exc:
        st.warning(f"Identity state not yet available: {exc}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Identity", key=f"hrg3_approve_{job_id}"):
            _submit(api_base, job_id, "HRG-3", scene_id, "approved")
            st.success("Identity design approved")
    with col2:
        if st.button("❌ Reject", key=f"hrg3_reject_{job_id}"):
            _submit(api_base, job_id, "HRG-3", scene_id, "rejected")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
