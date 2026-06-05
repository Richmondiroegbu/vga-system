#!/usr/bin/env python3
"""
VGA Pipeline Runner — executes a queued job through all 16 stages.

Run from /workspace/vga/:
    python scripts/run_pipeline.py <job_id>

Or create a new job and run:
    python scripts/run_pipeline.py --new \
        --topic "A teacher who built a school" \
        --duration 60
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the vga package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

# Load environment
env_file = Path("/workspace/.env_vga")
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_runner")

API_BASE = os.environ.get("VGA_API_BASE", "http://localhost:8000/api/v1")


def api_get(path: str) -> dict:
    import requests
    r = requests.get(f"{API_BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, data: dict) -> dict:
    import requests
    r = requests.post(f"{API_BASE}{path}", json=data, timeout=30)
    r.raise_for_status()
    return r.json()


def api_patch(path: str, data: dict) -> dict:
    import requests
    r = requests.patch(f"{API_BASE}{path}", json=data, timeout=30)
    r.raise_for_status()
    return r.json()


def update_job_status(job_id: str, status: str, stage: str = None, pct: float = 0.0) -> None:
    """Update job status in the API."""
    try:
        from vga.api.routes.jobs import _jobs
        if job_id in _jobs:
            _jobs[job_id]["status"] = status
            if stage:
                _jobs[job_id]["current_stage"] = stage
            _jobs[job_id]["progress_percent"] = pct
    except Exception:
        pass


def wait_for_hrg(checkpoint_id: str, display_data: dict, job_id: str) -> bool:
    """Wait for human approval at an HRG checkpoint via the API.

    Returns True if approved, False if rejected.
    """
    logger.info("=" * 60)
    logger.info(f"HRG CHECKPOINT: {checkpoint_id}")
    logger.info("Review in Streamlit UI and approve/reject.")
    logger.info("Or press Enter here to auto-approve for testing.")
    logger.info("=" * 60)

    # Write display data to disk for Streamlit to read
    hrg_dir = Path("/workspace/hrg") / job_id
    hrg_dir.mkdir(parents=True, exist_ok=True)
    (hrg_dir / f"{checkpoint_id}_display.json").write_text(
        json.dumps(display_data, indent=2, default=str), encoding="utf-8"
    )
    (hrg_dir / f"{checkpoint_id}_status.json").write_text(
        json.dumps({"status": "pending", "checkpoint": checkpoint_id}), encoding="utf-8"
    )

    # Wait for response file (written by Streamlit) or keyboard input
    response_file = hrg_dir / f"{checkpoint_id}_response.json"
    deadline = time.monotonic() + 300   # 5-minute timeout
    print(f"\n>>> Waiting for HRG-{checkpoint_id} approval (press Enter to auto-approve)...")

    import select
    while time.monotonic() < deadline:
        # Check if response file was written (by Streamlit)
        if response_file.exists():
            resp = json.loads(response_file.read_text(encoding="utf-8"))
            decision = resp.get("decision", "approved")
            response_file.unlink(missing_ok=True)
            logger.info(f"HRG {checkpoint_id}: {decision.upper()}")
            return decision == "approved"

        # Check for keyboard Enter (auto-approve)
        if sys.stdin in select.select([sys.stdin], [], [], 0.5)[0]:
            sys.stdin.readline()
            logger.info(f"HRG {checkpoint_id}: AUTO-APPROVED (Enter pressed)")
            return True

    logger.warning(f"HRG {checkpoint_id}: TIMED OUT — auto-approving")
    return True


def run_pipeline(job_id: str, request: dict) -> None:
    """Run the full VGA pipeline for a job."""
    logger.info(f"Starting pipeline for job {job_id}")
    logger.info(f"Topic: {request.get('topic')}")

    # Initialize singletons
    logger.info("Initializing VGA bootstrap singletons...")
    try:
        from vga.bootstrap import run_bootstrap
        registry = run_bootstrap()
        logger.info("Bootstrap complete.")
    except Exception as exc:
        logger.error(f"Bootstrap failed: {exc}")
        raise

    orchestrator = registry.get("orchestrator")
    hrg = registry.get("hrg_controller")

    # Create initial context
    from vga.state.context_factory import ContextFactory
    ctx = ContextFactory.create_initial(job_id=job_id, scene_id="scene_001")

    output_dir = Path("/workspace/output") / job_id / "scene_001"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── PHASE 1: Narrative Intelligence ─────────────────────────────────────

    # S-01: Script
    logger.info("S-01: Generating script...")
    update_job_status(job_id, "running", "S-01 ScriptAgent", 5.0)
    from vga.agents.script_agent import ScriptAgent
    script_output, ctx = orchestrator.execute_stage(
        ScriptAgent(), {
            "topic": request.get("topic", "A story of resilience"),
            "protagonist_description": request.get("protagonist_description", ""),
            "theme": request.get("theme", "hope and perseverance"),
            "duration_s": request.get("duration_s", 60.0),
        }, ctx
    )
    logger.info(f"Script: '{script_output.title}' — {len(script_output.scenes)} scenes")

    if not wait_for_hrg("HRG-1", {"script": script_output.model_dump()}, job_id):
        logger.warning("HRG-1 rejected — stopping pipeline")
        return

    # S-02: Scene Planning
    logger.info("S-02: Planning scenes...")
    update_job_status(job_id, "running", "S-02 SceneAgent", 10.0)
    from vga.agents.scene_agent import SceneAgent
    scene_plans, ctx = orchestrator.execute_stage(SceneAgent(), script_output, ctx)
    logger.info(f"Generated {len(scene_plans)} scene plans")

    if not wait_for_hrg("HRG-2", {"scene_plans": [p.model_dump() for p in scene_plans]}, job_id):
        return

    # S-03: Identity Design
    logger.info("S-03: Designing character identity...")
    update_job_status(job_id, "running", "S-03 IdentityDesignAgent", 15.0)
    from vga.agents.identity_design_agent import IdentityDesignAgent
    identity_design, ctx = orchestrator.execute_stage(
        IdentityDesignAgent(),
        {"script": script_output, "character_id": "main_character"},
        ctx
    )
    logger.info(f"Identity design complete for character: {identity_design.character_id}")

    if not wait_for_hrg("HRG-3", {"identity": identity_design.model_dump()}, job_id):
        return

    # S-04: Scene Composition
    logger.info("S-04: Generating composition plans...")
    update_job_status(job_id, "running", "S-04 SceneCompositionAgent", 20.0)
    from vga.agents.scene_composition_agent import SceneCompositionAgent
    composition_plan, ctx = orchestrator.execute_stage(
        SceneCompositionAgent(),
        {"scene_plan": scene_plans[0], "identity_design": identity_design.model_dump()},
        ctx
    )
    logger.info(f"Composition plan: camera={composition_plan.camera_angle}, lighting={composition_plan.lighting_style}")

    if not wait_for_hrg("HRG-4", {"composition": composition_plan.model_dump()}, job_id):
        return

    # ─── PHASE 2: Visual Grounding ────────────────────────────────────────────

    # S-05: Base Images
    logger.info("S-05: Generating 6 base images (FLUX.2-klein)...")
    update_job_status(job_id, "running", "S-05 BaseImageAgent", 30.0)
    from vga.agents.base_image_agent import BaseImageAgent
    image_output, ctx = orchestrator.execute_stage(
        BaseImageAgent(),
        {"identity_design": identity_design},
        ctx
    )
    logger.info(f"Generated {len(image_output['images'])} base images, best CLIP: {max(image_output['clip_scores']):.4f}")

    if not wait_for_hrg("HRG-5", {"images_count": len(image_output["images"]), "clip_scores": image_output["clip_scores"]}, job_id):
        return

    # S-06: Identity Reinforcement (A, B, C)
    logger.info("S-06: Running identity reinforcement (6A/6B/6C)...")
    update_job_status(job_id, "running", "S-06 ImageEditAgent", 40.0)
    from vga.agents.image_edit_agent import ImageEditAgent
    edit_output, ctx = orchestrator.execute_stage(
        ImageEditAgent(),
        {**image_output, "identity_design": identity_design.model_dump()},
        ctx
    )

    if not wait_for_hrg("HRG-6", {"scene_expanded": True}, job_id):
        return

    # S-07: Image Refinement + Identity Freeze
    logger.info("S-07: Refining image and freezing identity reference...")
    update_job_status(job_id, "running", "S-07 ImageRefinementAgent", 50.0)
    from vga.agents.image_refinement_agent import ImageRefinementAgent
    refine_output, ctx = orchestrator.execute_stage(
        ImageRefinementAgent(),
        {"scene_expanded_image": edit_output["scene_expanded_image"],
         "identity_design": identity_design.model_dump()},
        ctx
    )
    logger.info(f"Identity FROZEN — refined CLIP: {refine_output['refined_clip_score']:.4f}")

    if not wait_for_hrg("HRG-7", {"refined_clip": refine_output["refined_clip_score"]}, job_id):
        return

    # ─── PHASE 3: Video Generation ────────────────────────────────────────────

    # S-08: Video Segment 1 (Wan2.2)
    logger.info("S-08: Generating Segment_1 with Wan2.2-I2V...")
    update_job_status(job_id, "running", "S-08 VideoSegmentGenerator", 60.0)
    from vga.agents.video_segment_generator import VideoSegmentGenerator
    seg1_output, ctx = orchestrator.execute_stage(
        VideoSegmentGenerator(),
        {"refined_image": refine_output["refined_image"],
         "output_dir": str(output_dir),
         "prompt": identity_design.character_identity},
        ctx
    )
    logger.info(f"Segment_1 generated: {seg1_output['segment_1'].file_path}")

    # S-09: Temporal Engine (SVI autoregressive)
    logger.info("S-09: Generating remaining segments with SVI temporal engine...")
    update_job_status(job_id, "running", "S-09 TemporalEngine", 70.0)
    from vga.temporal.temporal_engine import TemporalEngine
    te = registry.get("temporal_engine")
    remaining_plans = scene_plans[0].segments[1:] if len(scene_plans[0].segments) > 1 else []
    if remaining_plans:
        segments, ctx = te.generate_segments(
            seg1_output["segment_1"], remaining_plans, ctx, output_dir
        )
        all_segments = [seg1_output["segment_1"]] + segments
    else:
        all_segments = [seg1_output["segment_1"]]
    logger.info(f"Generated {len(all_segments)} total segments")

    # S-10: Continuity Validation
    logger.info("S-10: Running continuity validation...")
    update_job_status(job_id, "running", "S-10 ContinuityValidationAgent", 68.0)
    from vga.agents.continuity_validation_agent import ContinuityValidationAgent
    continuity_output, ctx = orchestrator.execute_stage(
        ContinuityValidationAgent(),
        {"video_segments": all_segments, "scene_id": ctx.scene_id,
         "output_dir": str(output_dir)},
        ctx
    )
    logger.info(f"Continuity validation complete — {len(all_segments)} segments validated")

    if not wait_for_hrg("HRG-8", {"segments": len(all_segments),
                                    "continuity_score": continuity_output.get("continuity_score", 0.0)}, job_id):
        return

    # ─── PHASE 4: Audio ───────────────────────────────────────────────────────

    # S-11: Dialogue
    logger.info("S-11: Generating dialogue audio...")
    update_job_status(job_id, "running", "S-11 DialogueAgent", 75.0)
    from vga.agents.dialogue_agent import DialogueAgent
    dialogue_output, ctx = orchestrator.execute_stage(
        DialogueAgent(),
        {"segment_plans": scene_plans[0].segments, "scene_id": ctx.scene_id,
         "output_dir": str(output_dir)},
        ctx
    )

    if not wait_for_hrg("HRG-9", {"dialogue": "generated"}, job_id):
        return

    # S-12: Lip Sync
    logger.info("S-12: Running lip sync...")
    update_job_status(job_id, "running", "S-12 LipSyncAgent", 80.0)
    from vga.agents.lip_sync_agent import LipSyncAgent
    lipsync_output, ctx = orchestrator.execute_stage(
        LipSyncAgent(),
        {**dialogue_output, "video_segments": all_segments},
        ctx
    )

    if not wait_for_hrg("HRG-10", {"lip_sync": "complete"}, job_id):
        return

    # S-13: Ambient Audio
    logger.info("S-13: Generating ambient audio...")
    update_job_status(job_id, "running", "S-13 AmbientAudioAgent", 83.0)
    from vga.agents.ambient_audio_agent import AmbientAudioAgent
    ambient_output, ctx = orchestrator.execute_stage(
        AmbientAudioAgent(),
        {"video_segments": all_segments, "scene_id": ctx.scene_id,
         "output_dir": str(output_dir)},
        ctx
    )

    # S-14: Music
    logger.info("S-14: Generating background music...")
    update_job_status(job_id, "running", "S-14 MusicAgent", 86.0)
    from vga.agents.music_agent import MusicAgent
    music_output, ctx = orchestrator.execute_stage(
        MusicAgent(),
        {"scene_id": ctx.scene_id, "output_dir": str(output_dir),
         "duration_s": request.get("duration_s", 60.0)},
        ctx
    )

    # S-15: Audio Mixing
    logger.info("S-15: Mixing audio tracks...")
    update_job_status(job_id, "running", "S-15 AudioMixingAgent", 89.0)
    from vga.agents.audio_mixing_agent import AudioMixingAgent
    mix_output, ctx = orchestrator.execute_stage(
        AudioMixingAgent(),
        {**lipsync_output, **ambient_output, **music_output,
         "scene_id": ctx.scene_id, "output_dir": str(output_dir)},
        ctx
    )

    if not wait_for_hrg("HRG-11", {"snr_db": mix_output.get("snr_db", 0)}, job_id):
        return

    # ─── PHASE 5: Assembly ────────────────────────────────────────────────────

    # S-16: Assemble final video
    logger.info("S-16: Assembling final video...")
    update_job_status(job_id, "running", "S-16 AssemblyAgent", 95.0)
    from vga.agents.assembly_agent import AssemblyAgent
    from vga.agents.export_agent import ExportAgent
    from vga.agents.quality_agent import QualityAgent

    assembly_output, ctx = orchestrator.execute_stage(
        AssemblyAgent(),
        {"video_segments": all_segments, **mix_output,
         "output_dir": str(output_dir), "job_id": job_id},
        ctx
    )
    export_output, ctx = orchestrator.execute_stage(
        ExportAgent(),
        {**assembly_output, "job_id": job_id},
        ctx
    )
    quality_output, ctx = orchestrator.execute_stage(
        QualityAgent(),
        {**export_output, "job_id": job_id, "scene_id": ctx.scene_id},
        ctx
    )

    update_job_status(job_id, "completed", "S-16 Complete", 100.0)
    final_video = assembly_output.get("output_path", "unknown")
    logger.info(f"{'='*60}")
    logger.info(f"PIPELINE COMPLETE!")
    logger.info(f"Final video: {final_video}")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="VGA Pipeline Runner")
    parser.add_argument("job_id", nargs="?", help="Job ID to process")
    parser.add_argument("--new", action="store_true", help="Create a new job")
    parser.add_argument("--topic", default="A young teacher who built a school from nothing")
    parser.add_argument("--protagonist", default="A woman in her 30s, determined and dignified")
    parser.add_argument("--theme", default="hope, education and resilience")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    if args.new or not args.job_id:
        # Create a new job via API
        logger.info("Creating new job...")
        try:
            resp = api_post("/jobs/", {
                "topic": args.topic,
                "protagonist_description": args.protagonist,
                "theme": args.theme,
                "duration_s": args.duration,
            })
            job_id = resp["job_id"]
            request = resp
            logger.info(f"Job created: {job_id}")
        except Exception as exc:
            logger.error(f"Failed to create job: {exc}")
            sys.exit(1)
    else:
        job_id = args.job_id
        try:
            request = api_get(f"/jobs/{job_id}")
            request["topic"] = request.get("topic", args.topic)
            request["duration_s"] = args.duration
        except Exception as exc:
            logger.error(f"Failed to get job {job_id}: {exc}")
            sys.exit(1)

    try:
        run_pipeline(job_id, request)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
