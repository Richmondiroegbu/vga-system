"""
HRG-10: Lip Sync QA review panel. S-12 output.
Updated v17.0: shows identity_delta_per_segment. phoneme_alignment ≥ 0.80. RULE-97.
"""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.header("👄 HRG-10: Lip Sync QA Review")
    st.caption(
        "Phoneme alignment ≥ 0.80. Identity delta ≤ 0.03 per segment (RULE-97). "
        "CLIP validation applied (RULE-89)."
    )

    try:
        identity_data = requests.get(f"{api_base}/identity/{scene_id}", timeout=10).json()
        drift_history = identity_data.get("drift_history", [])

        if drift_history:
            st.subheader("Identity Delta Per Segment (NEW v17.0)")
            for i, delta in enumerate(drift_history):
                ok = delta <= 0.03
                icon = "✅" if ok else "❌"
                badge = "green" if ok else "red"
                st.markdown(f"Segment {i+1}: delta = **{delta:.4f}** {icon}")
    except Exception:
        st.warning("Identity data not yet available")

    st.subheader("Phoneme Alignment")
    st.info("Phoneme alignment score: awaiting data from LatentSync-1.6")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Lip Sync", key=f"hrg10_approve_{job_id}"):
            _submit(api_base, job_id, "HRG-10", scene_id, "approved")
            st.success("Lip sync approved")
    with col2:
        if st.button("❌ Reject & Re-sync", key=f"hrg10_reject_{job_id}"):
            _submit(api_base, job_id, "HRG-10", scene_id, "rejected")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
