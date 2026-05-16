"""
HRG-4 (NEW v17.0): Scene Composition review panel. All 6 fields editable. S-04 output.
Spec: VGA Streamlit UI Spec v17.2 §HRG-4; SR-047, SR-048
"""
from __future__ import annotations

VALID_CAMERA_ANGLES = [
    "extreme close-up", "close-up", "medium close-up", "medium shot",
    "medium wide shot", "wide shot", "extreme wide shot", "overhead",
    "low angle", "high angle", "dutch angle", "eye level",
]
VALID_MOTION_VECTORS = [
    "stationary", "forward_slow", "forward_medium", "backward_slow",
    "right_slow", "right_medium", "left_slow", "left_medium", "up_slow",
]


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.header("🎥 HRG-4: Scene Composition Review")
    st.caption(
        "Review and edit all 6 CompositionPlan fields. "
        "This gates ALL image/video generation (RULE-88)."
    )

    try:
        plan = requests.get(f"{api_base}/composition/{scene_id}?job_id={job_id}", timeout=10).json()
    except Exception as exc:
        st.error(f"CompositionPlan not available: {exc}")
        return

    st.subheader("Composition Plan Fields (all 6 required)")

    # All 6 editable fields per spec SR-048
    try:
        angle_idx = VALID_CAMERA_ANGLES.index(plan.get("camera_angle", "eye level"))
    except ValueError:
        angle_idx = 0
    camera_angle = st.selectbox("Camera Angle", VALID_CAMERA_ANGLES, index=angle_idx, key=f"hrg4_angle_{scene_id}")
    camera_motion = st.text_input("Camera Motion", value=plan.get("camera_motion", "static"), key=f"hrg4_motion_{scene_id}")
    focus_subject = st.text_input("Focus Subject", value=plan.get("focus_subject", "main_character"), key=f"hrg4_focus_{scene_id}")
    lighting_style = st.text_input("Lighting Style", value=plan.get("lighting_style", "natural"), key=f"hrg4_lighting_{scene_id}")

    try:
        mv_idx = VALID_MOTION_VECTORS.index(plan.get("motion_vector", "stationary"))
    except ValueError:
        mv_idx = 0
    motion_vector = st.selectbox("Motion Vector", VALID_MOTION_VECTORS, index=mv_idx, key=f"hrg4_mv_{scene_id}")
    char_positions = st.text_area(
        "Character Positions (JSON)",
        value=str(plan.get("character_positions", [{"character_id": "main_character", "position": "center", "facing": "camera"}])),
        key=f"hrg4_pos_{scene_id}",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve", key=f"hrg4_approve_{scene_id}"):
            _submit(api_base, job_id, "HRG-4", scene_id, "approved")
            st.success("Composition approved — proceeding to base image generation")
    with col2:
        if st.button("✏️ Edit & Recompose", key=f"hrg4_edit_{scene_id}"):
            import json
            try:
                positions = json.loads(char_positions)
            except Exception:
                positions = plan.get("character_positions", [])
            try:
                requests.patch(
                    f"{api_base}/composition/{scene_id}",
                    json={
                        "scene_id": scene_id,
                        "camera_angle": camera_angle,
                        "camera_motion": camera_motion,
                        "focus_subject": focus_subject,
                        "lighting_style": lighting_style,
                        "motion_vector": motion_vector,
                        "character_positions": positions,
                    },
                    timeout=10,
                )
                st.info("Composition updated — review the changes above then approve")
            except Exception as exc:
                st.error(f"Update failed: {exc}")
    with col3:
        if st.button("❌ Reject", key=f"hrg4_reject_{scene_id}"):
            _submit(api_base, job_id, "HRG-4", scene_id, "rejected")
            st.error("Composition rejected — will regenerate")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
