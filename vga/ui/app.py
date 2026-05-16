"""
VGA Streamlit UI — main entry point.
Displays pipeline status, all 11 HRG review panels, and temporal engine status.
Run: streamlit run vga/ui/app.py --server.port 8501
Spec: VGA Streamlit UI Spec v17.2; FR-800–FR-850
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from vga.config.settings import settings

st.set_page_config(
    page_title="VGA v17.2 — Cinematic AI Motivation System",
    page_icon="🎬",
    layout="wide",
)

API_BASE = "http://localhost:8000/api/v1"


def main():
    st.title("VGA v17.2 — Cinematic AI Motivation System")
    st.caption("Mission: *Inspire audiences by telling stories of people who overcame adversity.*")

    tab_pipeline, tab_hrg, tab_temporal, tab_identity, tab_audio = st.tabs([
        "Pipeline Status",
        "HRG Review Gates",
        "Temporal Engine",
        "Identity State",
        "Audio Quality",
    ])

    with tab_pipeline:
        render_pipeline_status()

    with tab_hrg:
        render_hrg_panels()

    with tab_temporal:
        render_temporal_panel()

    with tab_identity:
        render_identity_panel()

    with tab_audio:
        render_audio_panel()


def render_pipeline_status():
    st.header("Pipeline Status")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Schema Version", settings.SCHEMA_VERSION)
    col2.metric("System Version", settings.SYSTEM_VERSION)
    col3.metric("HRG Checkpoints", str(settings.HRG_CHECKPOINT_COUNT))
    col4.metric("Temporal Buffer", f"{settings.TEMPORAL_BUFFER_SIZE} frames")

    st.subheader("Stages")
    stages = [
        ("S-01", "Script Generation"),
        ("S-02", "Scene Planning"),
        ("S-03", "Identity Design"),
        ("S-04", "Composition Planning"),
        ("S-05", "Base Image Generation"),
        ("S-06", "Identity Reinforcement"),
        ("S-07", "Image Refinement"),
        ("S-08", "Video Segment 1 (Wan2.2)"),
        ("S-09", "Temporal Engine (SVI)"),
        ("S-10", "Continuity Validation"),
        ("S-11", "Dialogue (CosyVoice3)"),
        ("S-12", "Lip Sync (LatentSync)"),
        ("S-13", "Ambient Audio (MMAudio)"),
        ("S-14", "Music (MusicGen)"),
        ("S-15", "Audio Mixing"),
        ("S-16", "Export & Assembly"),
    ]
    for sid, name in stages:
        st.text(f"{sid}: {name}")


def render_hrg_panels():
    st.header("Human Review Gates (11 Checkpoints)")

    hrg_names = [
        "HRG-1: Script",
        "HRG-2: Scene Plan (v17.0)",
        "HRG-3: Identity Design",
        "HRG-4: Composition Plan (v17.0)",
        "HRG-5: Base Images",
        "HRG-6: Identity Reinforcement",
        "HRG-7: Refined Image",
        "HRG-8: Motion QA",
        "HRG-9: Dialogue Audio",
        "HRG-10: Lip Sync QA",
        "HRG-11: Final Audio QA",
    ]

    # Check for pending HRG files
    pending = []
    if settings.HRG_DIR.exists():
        for f in settings.HRG_DIR.rglob("HRG-*_*.json"):
            if "_response" not in f.name:
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if data.get("status") == "pending":
                        pending.append(data)
                except Exception:
                    pass

    if pending:
        st.warning(f"{len(pending)} HRG checkpoint(s) awaiting review")
        for item in pending:
            with st.expander(f"{item['checkpoint']} — {item['scene_id']}"):
                st.json(item.get("display_data", {}))

                col1, col2 = st.columns(2)
                if col1.button(f"Approve {item['checkpoint']}", key=f"approve_{item['checkpoint']}_{item['scene_id']}"):
                    _submit_hrg_decision(item["checkpoint"], item["scene_id"], item["job_id"], "approved")
                    st.success("Approved")
                    st.rerun()

                if col2.button(f"Reject {item['checkpoint']}", key=f"reject_{item['checkpoint']}_{item['scene_id']}"):
                    reason = st.text_input("Rejection reason:")
                    _submit_hrg_decision(item["checkpoint"], item["scene_id"], item["job_id"], "rejected", reason)
                    st.error("Rejected")
                    st.rerun()
    else:
        st.info("No HRG checkpoints pending review.")

    st.subheader("All HRG Checkpoints")
    for name in hrg_names:
        st.text(f"  {name}")


def render_temporal_panel():
    st.header("Temporal Engine Status (v17.0)")
    st.metric("Buffer Size", f"{settings.TEMPORAL_BUFFER_SIZE} frames (RULE-86)")
    st.metric("Max Retries/Segment", settings.TEMPORAL_MAX_RETRIES_PER_SEGMENT)
    st.metric("SVI CFG Range", f"[{settings.SVI_CFG_MIN}, {settings.SVI_CFG_MAX}]")

    col1, col2, col3 = st.columns(3)
    col1.metric("LoRA High Noise", f"{settings.LORA_WEIGHT_HIGH_NOISE} (t > 67%)")
    col2.metric("LoRA Mid Noise", f"{settings.LORA_WEIGHT_MID_NOISE} (33–67%)")
    col3.metric("LoRA Low Noise", f"{settings.LORA_WEIGHT_LOW_NOISE} (t ≤ 33%)")

    st.info("TemporalEngine authority: this is the ONLY component that may invoke SVI.")


def render_identity_panel():
    st.header("Identity State (RULE-95)")
    st.metric("CLIP Threshold", settings.CLIP_IDENTITY_THRESHOLD)
    st.metric("Max Drift/Step", settings.CLIP_DRIFT_THRESHOLD)
    st.metric("Cumulative Drift Limit", settings.IDENTITY_CUMULATIVE_DRIFT_THRESHOLD)
    st.metric("LipSync Delta Limit", settings.LIPSYNC_IDENTITY_DELTA_THRESHOLD)
    st.info("char_identity_ref is frozen at S-07 and never recomputed. RULE-95.")


def render_audio_panel():
    st.header("Audio Quality (RULE-99)")
    col1, col2 = st.columns(2)
    col1.metric("Min SNR", f"{settings.MIN_SNR_DB} dB")
    col2.metric("Max Peak", f"{settings.MAX_PEAK_DBFS} dBFS")

    st.subheader("Mixing Levels (RULE-98)")
    st.text(f"  Dialogue: {settings.DIALOGUE_LEVEL_DB} dB (highest priority)")
    st.text(f"  Ambient:  {settings.AMBIENT_LEVEL_DB} dB")
    st.text(f"  Music:    {settings.MUSIC_LEVEL_DB} dB")


def _submit_hrg_decision(
    checkpoint: str,
    scene_id: str,
    job_id: str,
    decision: str,
    reason: str = "",
) -> None:
    """Write HRG decision file for HRGController to pick up."""
    response_file = settings.HRG_DIR / f"{checkpoint}_{scene_id}_response.json"
    settings.HRG_DIR.mkdir(parents=True, exist_ok=True)
    response_file.write_text(
        json.dumps({
            "checkpoint": checkpoint,
            "scene_id": scene_id,
            "decision": decision,
            "reason": reason,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
