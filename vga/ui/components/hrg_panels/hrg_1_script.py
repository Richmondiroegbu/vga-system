"""HRG-1: Script review panel. S-01 output review."""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.header("📝 HRG-1: Script Review")
    st.caption("Review the generated script before proceeding to scene planning.")

    try:
        r = requests.get(f"{api_base}/jobs/{job_id}", timeout=10)
        job = r.json()
        st.json(job)
    except Exception as exc:
        st.error(f"Could not load script data: {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Script", key=f"hrg1_approve_{job_id}"):
            _submit_decision(api_base, job_id, "HRG-1", scene_id, "approved")
            st.success("Script approved — proceeding to scene planning")
    with col2:
        if st.button("❌ Reject & Regenerate", key=f"hrg1_reject_{job_id}"):
            reason = st.text_input("Rejection reason:", key=f"hrg1_reason_{job_id}")
            _submit_decision(api_base, job_id, "HRG-1", scene_id, "rejected", reason)
            st.error("Script rejected — will regenerate")


def _submit_decision(api_base: str, job_id: str, checkpoint: str, scene_id: str, decision: str, reason: str = "") -> None:
    import requests
    try:
        requests.post(
            f"{api_base}/hrg/decision",
            json={"checkpoint": checkpoint, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason},
            timeout=10,
        )
    except Exception:
        pass
