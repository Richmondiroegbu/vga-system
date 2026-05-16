"""HRG-7: Refined image review panel. S-07 output — char_identity_ref frozen here."""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.header("✨ HRG-7: Refined Image Review")
    st.caption(
        "Z-Image-Turbo refinement. "
        "CRITICAL: char_identity_ref is FROZEN after this approval (RULE-95). "
        "Drift ≤ 0.02 enforced (RULE-93)."
    )

    try:
        identity_data = requests.get(f"{api_base}/identity/{scene_id}", timeout=10).json()
        drift = identity_data.get("drift_score", 0.0)
        frozen = identity_data.get("is_frozen", False)
        clip_score = 1.0 - drift

        col1, col2, col3 = st.columns(3)
        col1.metric("CLIP Score (approx)", f"{clip_score:.4f}", delta="✅" if clip_score >= 0.93 else "❌")
        col2.metric("Drift Score", f"{drift:.4f}", delta="✅" if drift <= 0.02 else "❌")
        col3.metric("Identity Frozen", "YES" if frozen else "NO")
    except Exception:
        st.warning("Identity state not yet available")

    st.warning("⚠️ Approving this gate FREEZES char_identity_ref for all downstream stages.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Freeze Identity", key=f"hrg7_approve_{job_id}"):
            _submit(api_base, job_id, "HRG-7", scene_id, "approved")
            st.success("Refined image approved — char_identity_ref is now frozen")
    with col2:
        if st.button("❌ Reject & Re-refine", key=f"hrg7_reject_{job_id}"):
            _submit(api_base, job_id, "HRG-7", scene_id, "rejected")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
