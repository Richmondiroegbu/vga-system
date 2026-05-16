"""HRG-6: Identity reinforcement review panel. S-06 (6A/6B/6C) output."""
from __future__ import annotations


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st

    st.header("🎭 HRG-6: Identity Reinforcement Review")
    st.caption("Review multi-angle variants (6A), merged candidate (6B), and scene-expanded image (6C). CLIP ≥ 0.93 required.")

    st.markdown("**Sub-stages completed:**")
    st.text("  6A: Multi-angle variants (5-8 angles, LoRA 0.55)")
    st.text("  6B: Merged consensus image (LoRA 0.60)")
    st.text("  6C: Scene-expanded with environment (LoRA 0.55, full CompositionPlan)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Reinforcement", key=f"hrg6_approve_{job_id}"):
            _submit(api_base, job_id, "HRG-6", scene_id, "approved")
            st.success("Identity reinforcement approved")
    with col2:
        if st.button("❌ Reject & Redo", key=f"hrg6_reject_{job_id}"):
            _submit(api_base, job_id, "HRG-6", scene_id, "rejected")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
