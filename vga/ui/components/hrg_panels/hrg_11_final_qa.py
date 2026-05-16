"""
HRG-11: Final Audio QA review panel. S-15 output.
Updated v17.0: SNR badge + clipping status + mixing levels. RULE-99. SR-051.
"""
from __future__ import annotations

from vga.config.settings import settings


def render(job_id: str, scene_id: str, api_base: str = "http://localhost:8000/api/v1") -> None:
    import streamlit as st
    import requests

    st.header("🎵 HRG-11: Final Audio QA Review")
    st.caption(
        f"SNR ≥ {settings.MIN_SNR_DB}dB required. "
        f"Peaks ≤ {settings.MAX_PEAK_DBFS}dBFS required (RULE-99). "
        "Cross-modal alignment ±0.10s (FR-972)."
    )

    try:
        audio_data = requests.get(f"{api_base}/audio/{scene_id}", timeout=10).json()
        record = audio_data.get("record", audio_data)

        snr_db = record.get("snr_db", 0.0)
        peak_db = record.get("peak_db", 0.0)
        snr_ok = record.get("snr_passed", snr_db >= settings.MIN_SNR_DB)
        clip_ok = record.get("clipping_passed", peak_db <= settings.MAX_PEAK_DBFS)
        clipping = record.get("clipping_detected", False)

        # SNR badge (NEW v17.0)
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "SNR",
            f"{snr_db:.1f} dB",
            delta="✅ PASS" if snr_ok else "❌ FAIL",
        )
        col2.metric(
            "Peak Level",
            f"{peak_db:.1f} dBFS",
            delta="✅ No Clipping" if clip_ok else "❌ CLIPPING",
        )
        col3.metric(
            "Clipping",
            "DETECTED ❌" if clipping else "Clear ✅",
        )

        if not snr_ok:
            st.error(f"SNR {snr_db:.1f}dB is below the {settings.MIN_SNR_DB}dB minimum (RULE-99)")
        if clipping:
            st.error("Clipping detected — peak exceeds 0 dBFS (RULE-99)")

    except Exception as exc:
        st.warning(f"Audio validation data not yet available: {exc}")

    st.subheader("Mixing Levels")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dialogue", f"{settings.DIALOGUE_LEVEL_DB} dB", delta="Highest priority")
    col2.metric("Ambient", f"{settings.AMBIENT_LEVEL_DB} dB")
    col3.metric("Music", f"{settings.MUSIC_LEVEL_DB} dB")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Final Audio → Export", key=f"hrg11_approve_{job_id}"):
            _submit(api_base, job_id, "HRG-11", scene_id, "approved")
            st.success("Final audio approved — proceeding to export (S-16)")
    with col2:
        if st.button("❌ Reject & Remix", key=f"hrg11_reject_{job_id}"):
            _submit(api_base, job_id, "HRG-11", scene_id, "rejected")
            st.error("Rejected — triggering audio remix")


def _submit(api_base, job_id, cp, scene_id, decision, reason=""):
    import requests
    try:
        requests.post(f"{api_base}/hrg/decision", json={"checkpoint": cp, "scene_id": scene_id, "job_id": job_id, "decision": decision, "reason": reason}, timeout=10)
    except Exception:
        pass
