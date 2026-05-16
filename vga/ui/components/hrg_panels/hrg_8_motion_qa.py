"""
HRG-8: Motion QA review panel. S-09/S-10 output.
Updated v17.0: shows identity_per_segment CLIP scores. SR-049.
"""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.header("🎞️ HRG-8: Motion QA Review")
    st.caption("Review video segments, continuity scores, and identity validation per segment.")

    try:
        temporal_data = requests.get(f"{api_base}/temporal/{scene_id}", timeout=10).json()
        st.metric("Buffer Frames", temporal_data.get("frame_count", "—"))
        st.metric("Buffer Initialized", str(temporal_data.get("is_initialized", False)))
    except Exception:
        st.warning("Temporal data not yet available")

    try:
        continuity_data = requests.get(f"{api_base}/jobs/{job_id}/temporal", timeout=10).json()
        if continuity_data:
            score = continuity_data.get("overall_continuity_score", 0.0)
            st.metric(
                "Overall Continuity",
                f"{score:.4f}",
                delta="✅ PASS" if score >= 0.90 else "❌ FAIL",
            )
    except Exception:
        pass

    # NEW v17.0: identity_per_segment display (SR-049)
    try:
        identity_data = requests.get(f"{api_base}/identity/{scene_id}", timeout=10).json()
        drift_history = identity_data.get("drift_history", [])
        if drift_history:
            st.subheader("Identity Per Segment (CLIP scores)")
            for i, drift in enumerate(drift_history):
                clip_approx = max(0.0, 1.0 - drift)
                color = "green" if clip_approx >= 0.93 else "red"
                icon = "✅" if clip_approx >= 0.93 else "❌"
                st.markdown(f"Segment {i+1}: CLIP ≈ **{clip_approx:.3f}** {icon}")
    except Exception:
        pass

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Motion QA", key=f"hrg8_approve_{job_id}"):
            _submit(api_base, job_id, "HRG-8", scene_id, "approved")
            st.success("Motion QA approved")
    with col2:
        if st.button("❌ Reject & Regenerate", key=f"hrg8_reject_{job_id}"):
            _submit(api_base, job_id, "HRG-8", scene_id, "rejected")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
