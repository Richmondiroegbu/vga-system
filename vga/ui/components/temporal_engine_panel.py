"""
TemporalEngine status panel (NEW v17.0).
Shows buffer frame count, segment progress, motion state, identity per segment.
Spec: VGA Streamlit UI Spec v17.2 §temporal_engine_panel.py
"""
from __future__ import annotations

from vga.config.settings import settings


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.subheader("🕐 Temporal Engine Status (v17.0)")

    try:
        status = requests.get(f"{api_base}/temporal/{scene_id}", timeout=10).json()
    except Exception as exc:
        st.warning(f"Temporal status unavailable: {exc}")
        return

    frame_count = status.get("frame_count", 0)
    is_initialized = status.get("is_initialized", False)
    segment_index = status.get("segment_index", 0)
    total_segments = status.get("total_segments", 0)

    # Buffer status
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Buffer Frames",
        f"{frame_count}/{settings.TEMPORAL_BUFFER_SIZE}",
        delta="✅ RULE-86" if frame_count == settings.TEMPORAL_BUFFER_SIZE else "⚠️",
    )
    col2.metric("Buffer Initialized", "YES ✅" if is_initialized else "NO ⏳")
    col3.metric(
        "Segments Generated",
        f"{segment_index}/{total_segments}" if total_segments > 0 else "—",
    )

    # Segment progress
    if total_segments > 0:
        st.progress(
            segment_index / total_segments,
            text=f"Segment {segment_index}/{total_segments}",
        )

    # Identity per segment
    try:
        identity_data = requests.get(f"{api_base}/identity/{scene_id}", timeout=10).json()
        drift_history = identity_data.get("drift_history", [])
        if drift_history:
            st.subheader("Identity Validation Per Segment")
            for i, drift in enumerate(drift_history):
                clip_approx = max(0.0, 1.0 - drift)
                icon = "✅" if clip_approx >= settings.CLIP_IDENTITY_THRESHOLD else "❌"
                model = "Wan2.2" if i == 0 else "SVI Pro 2"
                st.text(f"  Segment {i+1} ({model}): CLIP ≈ {clip_approx:.3f} {icon}")
    except Exception:
        pass

    # SVI CFG range reminder
    st.caption(
        f"SVI CFG range: [{settings.SVI_CFG_MIN}, {settings.SVI_CFG_MAX}] | "
        f"LoRA schedule: High={settings.LORA_WEIGHT_HIGH_NOISE} "
        f"Mid={settings.LORA_WEIGHT_MID_NOISE} "
        f"Low={settings.LORA_WEIGHT_LOW_NOISE}"
    )
