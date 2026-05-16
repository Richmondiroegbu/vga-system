"""HRG-9: Dialogue QA review panel. S-11 output. Timing ≤ 0.10s (RULE-96)."""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st

    st.header("🎙️ HRG-9: Dialogue Audio QA")
    st.caption("Review synthesized dialogue. Timing error ≤ 0.10s required (RULE-96, CosyVoice3).")
    st.info("Listen to the dialogue audio for each segment. Check timing alignment.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Dialogue", key=f"hrg9_approve_{job_id}"):
            _submit(api_base, job_id, "HRG-9", scene_id, "approved")
            st.success("Dialogue approved")
    with col2:
        if st.button("❌ Reject & Redo", key=f"hrg9_reject_{job_id}"):
            _submit(api_base, job_id, "HRG-9", scene_id, "rejected")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
