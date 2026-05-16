# Prompt 16: Streamlit UI — HRG Review Panels
**Category:** UI  
**Files:**
- `vga/ui/app.py`
- `vga/ui/components/hrg_panels/hrg_1_script.py`
- `vga/ui/components/hrg_panels/hrg_2_scene_plan.py` [NEW]
- `vga/ui/components/hrg_panels/hrg_3_identity.py`
- `vga/ui/components/hrg_panels/hrg_4_composition.py` [NEW v17.0]
- `vga/ui/components/hrg_panels/hrg_5_base_images.py`
- `vga/ui/components/hrg_panels/hrg_6_composed_images.py`
- `vga/ui/components/hrg_panels/hrg_7_refined_images.py`
- `vga/ui/components/hrg_panels/hrg_8_motion_qa.py` [updated]
- `vga/ui/components/hrg_panels/hrg_9_voice_qa.py`
- `vga/ui/components/hrg_panels/hrg_10_lipsync_qa.py` [updated]
- `vga/ui/components/hrg_panels/hrg_11_final_qa.py` [updated]
- `vga/ui/components/temporal_engine_panel.py` [NEW v17.0]

## HRG-4 Panel (NEW v17.0 — CompositionPlan Review)
```python
def hrg_4_composition_panel(job_id: str, scene_id: str):
    """
    Display CompositionPlan for human review and editing.
    SR-047: User reviews camera/blocking/motion at HRG-4.
    SR-048: User can EDIT all 6 CompositionPlan fields.
    """
    import streamlit as st
    
    st.header("🎬 HRG-4: Scene Composition Review")
    plan = fetch_composition_plan(job_id, scene_id)
    
    # Display all 6 editable fields
    camera_angle = st.selectbox("Camera Angle", VALID_CAMERA_ANGLES, index=...)
    camera_motion = st.text_input("Camera Motion", value=plan.camera_motion)
    character_positions = st.json_input("Character Positions", value=plan.character_positions)
    focus_subject = st.text_input("Focus Subject", value=plan.focus_subject)
    lighting_style = st.text_input("Lighting Style", value=plan.lighting_style)
    motion_vector = st.selectbox("Motion Vector", VALID_MOTION_VECTORS, ...)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve"):
            post_hrg_decision(job_id, "HRG-4", "approve")
    with col2:
        if st.button("✏️ Edit & Recompose"):
            post_hrg_decision(job_id, "HRG-4", "recompose", {
                "camera_angle": camera_angle, ...
            })
    with col3:
        if st.button("❌ Reject"):
            post_hrg_decision(job_id, "HRG-4", "reject")
```

## HRG-8 Panel (updated — identity_per_segment)
```python
def hrg_8_motion_qa_panel(job_id, scene_id):
    """
    SR-049: Display continuity score with sub-score breakdown.
    NEW v17.0: identity_per_segment scores displayed.
    """
    ...
    # Show identity scores per segment (NEW v17.0)
    st.subheader("Identity Per Segment")
    for i, score in enumerate(data["identity_per_segment"]):
        color = "green" if score >= 0.93 else "red"
        st.markdown(f"Segment {i+1}: :{color}[{score:.3f}] {'✅' if score >= 0.93 else '❌'}")
```

## HRG-11 Panel (updated — SNR + clipping)
```python
def hrg_11_final_qa_panel(job_id, scene_id):
    """
    SR-051: Display SNR and clipping validation (NEW v17.0).
    """
    audio_record = fetch_audio_validation(job_id, scene_id)
    
    snr_ok = audio_record["snr_db"] >= 10.0
    clip_ok = not audio_record["clipping_detected"]
    
    st.metric("SNR", f"{audio_record['snr_db']:.1f} dB", 
              delta="✅ PASS" if snr_ok else "❌ FAIL")
    st.metric("Peak Level", f"{audio_record['peak_db']:.1f} dBFS",
              delta="✅ No Clipping" if clip_ok else "❌ CLIPPING")
```

## TemporalEngine Status Panel (NEW v17.0)
```python
def temporal_engine_panel(job_id, scene_id):
    """Shows TemporalBuffer status, segment generation progress, identity scores."""
    status = fetch_temporal_buffer(job_id)
    
    st.subheader("🕐 Temporal Engine Status")
    st.progress(status["frame_count"] / 5, f"Buffer: {status['frame_count']}/5 frames")
    
    for i, seg in enumerate(status["segments"]):
        model = "Wan2.2" if i == 0 else "SVI Pro 2"
        icon = "✅" if seg["completed"] else "⏳"
        st.text(f"Segment {i+1} ({model}): {icon}")
    
    # Motion state
    st.json(status.get("motion_state", {}))
    
    # Identity per segment
    st.subheader("Identity Validation")
    for i, score in enumerate(status.get("identity_per_segment", [])):
        st.text(f"  Segment {i+1}: CLIP={score:.3f} {'✅' if score >= 0.93 else '❌'}")
```

## Acceptance Criteria
- [ ] HRG-4 panel shows all 6 editable CompositionPlan fields
- [ ] HRG-8 panel shows `identity_per_segment` scores per segment
- [ ] HRG-10 panel shows `identity_delta_per_segment` and phoneme_alignment
- [ ] HRG-11 panel shows SNR badge and clipping status
- [ ] TemporalEngine panel shows buffer frame count and segment progress
- [ ] All 11 HRG panels are correctly numbered and named
