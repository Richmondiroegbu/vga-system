"""HRG-2 (NEW v17.0): Scene/Segment Plan review panel. S-02 output review."""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.header("🎬 HRG-2: Scene & Segment Plan Review")
    st.caption("Review scene/segment plan. Check durations and segment counts before identity design.")

    try:
        r = requests.get(f"{api_base}/jobs/{job_id}", timeout=10)
        job = r.json()
        stage_summary = job.get("stage_summary", {})
        st.metric("Total Segments", stage_summary.get("total_segments", "—"))
        st.metric("Completed Stages", f"{stage_summary.get('completed_stages', 0)}/16")
        st.json(job)
    except Exception as exc:
        st.error(f"Could not load scene plan: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve Plan", key=f"hrg2_approve_{job_id}"):
            _submit(api_base, job_id, "HRG-2", scene_id, "approved")
            st.success("Scene plan approved")
    with col2:
        if st.button("✏️ Edit & Replan", key=f"hrg2_edit_{job_id}"):
            st.info("Edit scene plan via API PATCH, then approve")
    with col3:
        if st.button("❌ Reject", key=f"hrg2_reject_{job_id}"):
            _submit(api_base, job_id, "HRG-2", scene_id, "rejected")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
