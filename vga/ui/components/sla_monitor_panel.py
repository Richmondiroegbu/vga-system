"""SLA monitor panel (v15.0). Shows per-stage SLA compliance."""
from __future__ import annotations

from vga.config.settings import settings


def render(job_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st

    st.subheader("⏱️ SLA Monitor")

    sla_budgets = {
        "S-01": settings.SLA_SCRIPT_MAX_S,
        "S-02": settings.SLA_SCENE_PLAN_MAX_S,
        "S-03": settings.SLA_IDENTITY_DESIGN_MAX_S,
        "S-04": settings.SLA_COMPOSITION_MAX_S,
        "S-05": settings.SLA_BASE_IMAGE_MAX_S,
        "S-08": settings.SLA_SEGMENT_GEN_MAX_S,
        "S-09": settings.SLA_SEGMENT_GEN_CRITICAL_MAX_S,
        "S-11": settings.SLA_LIPSYNC_MAX_S,
        "S-15": settings.SLA_AUDIO_MIX_MAX_S,
        "S-16": settings.SLA_EXPORT_MAX_S,
    }

    for stage_id, budget in sla_budgets.items():
        st.text(f"  {stage_id}: budget = {budget:.0f}s")
