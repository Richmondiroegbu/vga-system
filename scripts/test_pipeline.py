#!/usr/bin/env python3
"""
VGA full pipeline test: S-01 through S-08.
Run on pod as: python3 /workspace/vga/scripts/test_pipeline.py

Key fix: kills SVI server before S-07 (Z-Image needs ~14GB; SVI server holds ~17GB).
Restarts SVI server before S-09.
"""
import os
import subprocess
import sys
import time
import logging

os.environ["HRG_REVIEW_ENABLED"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONPATH"] = "/workspace"

sys.path.insert(0, "/workspace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_pipeline")


def _free_vram_for_heavy_model():
    """Kill SVI server to free ~17GB VRAM before loading Z-Image or WanWrapper."""
    result = subprocess.run(
        ["pkill", "-f", "vga_svi_server.py"],
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info("SVI server killed — ~17GB VRAM freed")
        time.sleep(5)  # let GPU memory fully release
    else:
        logger.info("SVI server was not running")

    # Also clear any lingering CUDA fragmentation
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 1e9
        logger.info("VRAM free after cleanup: %.1f GB", free)
    except Exception:
        pass


def _assemble_final_video(
    segment_paths: list,
    output_path: str,
    overlap_frames: int = 4,
    fps: int = 15,
) -> None:
    """Concatenate video segments into one coherent video, trimming SVI overlap frames.

    Segment 1 is kept in full.  Segments 2..N each start overlap_frames frames in,
    because SVI continuation mode makes the first N frames replicate the end of the
    previous segment — trimming them yields a seamless join with no duplicated content.
    """
    import shutil

    if not segment_paths:
        logger.error("Assembly: no segments provided")
        return

    if len(segment_paths) == 1:
        shutil.copy2(segment_paths[0], output_path)
        logger.info("Assembly: single segment copied → %s", output_path)
        return

    # Build ffmpeg filter_complex:
    #   [0:v] → pass through unchanged (full Segment_1)
    #   [i:v] for i>0 → trim=start_frame=N,setpts=PTS-STARTPTS  (drop overlap frames)
    # Then concat all trimmed streams.
    filter_parts = []
    for i, _ in enumerate(segment_paths):
        label = f"[v{i}]"
        if i == 0:
            filter_parts.append(f"[0:v]null{label}")
        else:
            filter_parts.append(
                f"[{i}:v]trim=start_frame={overlap_frames},setpts=PTS-STARTPTS{label}"
            )
    concat_inputs = "".join(f"[v{i}]" for i in range(len(segment_paths)))
    filter_complex = (
        ";".join(filter_parts)
        + f";{concat_inputs}concat=n={len(segment_paths)}:v=1:a=0[outv]"
    )

    input_args = []
    for seg_path in segment_paths:
        input_args += ["-i", seg_path]

    cmd = (
        ["ffmpeg", "-y"]
        + input_args
        + [
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    )

    n_segs = len(segment_paths)
    trimmed_frames = overlap_frames * (n_segs - 1)
    logger.info(
        "Assembly: %d segments → %s (trimming %d overlap frames at %d joins)",
        n_segs, output_path, trimmed_frames, n_segs - 1,
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Assembly failed (rc=%d):\n%s", result.returncode, result.stderr[-2000:])
    else:
        logger.info("Assembly complete → %s", output_path)


def main():
    # Imports after env is set
    from vga.state.context_factory import ContextFactory
    from vga.core.master_orchestrator import MasterOrchestrator
    from vga.core.hrg_controller import HRGController
    from vga.validation.composition_validator import CompositionValidator
    from vga.identity.identity_state_tracker import IdentityStateTracker

    from vga.agents.script_agent import ScriptAgent
    from vga.agents.scene_agent import SceneAgent
    from vga.agents.identity_design_agent import IdentityDesignAgent
    from vga.agents.scene_composition_agent import SceneCompositionAgent
    from vga.agents.base_image_agent import BaseImageAgent
    from vga.agents.image_edit_agent import ImageEditAgent
    from vga.agents.image_refinement_agent import ImageRefinementAgent
    from vga.agents.video_segment_generator import VideoSegmentGenerator
    from vga.temporal.temporal_engine import TemporalEngine
    from vga.temporal.temporal_buffer_manager import TemporalBufferManager
    from vga.temporal.motion_state_tracker import MotionStateTracker
    from vga.temporal.temporal_retry_controller import TemporalRetryController
    from vga.validation.clip_validator import CLIPValidator as CLIPVal

    # Build orchestrator
    orch = MasterOrchestrator(
        hrg_controller=HRGController(),
        composition_validator=CompositionValidator(),
        identity_tracker=IdentityStateTracker(),
    )

    ctx = ContextFactory.create_initial(job_id="test_e2e_001", scene_id="scene_001")
    logger.info("Context created: %s / %s", ctx.job_id, ctx.scene_id)

    # S-01: Script
    t0 = time.monotonic()
    logger.info("=== S-01: Script Generation ===")
    out_s01, ctx = orch.execute_stage(
        ScriptAgent(),
        {"topic": "overcoming adversity and finding hope", "theme": "resilience and faith"},
        ctx,
    )
    logger.info("S-01 done %.1fs — %d scenes", time.monotonic() - t0, len(out_s01.scenes))

    # S-02: Scene Planning (takes ScriptSchema directly)
    t0 = time.monotonic()
    logger.info("=== S-02: Scene Planning ===")
    out_s02, ctx = orch.execute_stage(SceneAgent(), out_s01, ctx)
    scene_plan_0 = out_s02[0]
    logger.info("S-02 done %.1fs — %d plans, using scene 0: %s",
                time.monotonic() - t0, len(out_s02), scene_plan_0.scene_id)

    # S-03: Identity Design
    t0 = time.monotonic()
    logger.info("=== S-03: Identity Design ===")
    out_s03, ctx = orch.execute_stage(
        IdentityDesignAgent(),
        {"script": out_s01, "character_id": "main_character"},
        ctx,
    )
    logger.info("S-03 done %.1fs", time.monotonic() - t0)

    # S-04: Scene Composition
    t0 = time.monotonic()
    logger.info("=== S-04: Scene Composition ===")
    out_s04, ctx = orch.execute_stage(
        SceneCompositionAgent(),
        {"scene_plan": scene_plan_0, "identity_design": out_s03},
        ctx,
    )
    logger.info("S-04 done %.1fs — camera=%s motion=%s",
                time.monotonic() - t0, out_s04.camera_angle, out_s04.camera_motion)

    # S-05: Base Image Generation (FLUX.2-klein, CPU offload — coexists with SVI server)
    t0 = time.monotonic()
    logger.info("=== S-05: Base Image Generation ===")
    out_s05, ctx = orch.execute_stage(
        BaseImageAgent(),
        {"identity_design": out_s03},
        ctx,
    )
    logger.info("S-05 done %.1fs — %d images best_idx=%d",
                time.monotonic() - t0, len(out_s05["images"]), out_s05["best_image_index"])

    # S-06: Image Edit / Identity Reinforcement (FLUX with LoRA, CPU offload — still OK)
    t0 = time.monotonic()
    logger.info("=== S-06: Image Edit (Identity Reinforcement) ===")
    s06_input = {**out_s05, "identity_design": out_s03}
    out_s06, ctx = orch.execute_stage(ImageEditAgent(), s06_input, ctx)
    logger.info("S-06 done %.1fs", time.monotonic() - t0)

    # ── CRITICAL: Kill SVI server before S-07 ──────────────────────────────────
    # Z-Image-Turbo needs ~14.4GB GPU VRAM. SVI server holds ~17GB.
    # Together = 31.4GB > 32GB → OOM. Kill server, free VRAM, then load Z-Image.
    logger.info("Killing SVI server to free VRAM for Z-Image-Turbo...")
    _free_vram_for_heavy_model()

    # S-07: Image Refinement + Identity Freeze (Z-Image-Turbo)
    t0 = time.monotonic()
    logger.info("=== S-07: Image Refinement + Identity Freeze ===")
    out_s07, ctx = orch.execute_stage(
        ImageRefinementAgent(),
        {"scene_expanded_image": out_s06["scene_expanded_image"], "identity_design": out_s03},
        ctx,
    )
    logger.info("S-07 done %.1fs — refined_clip=%.4f frozen=%s",
                time.monotonic() - t0, out_s07["refined_clip_score"], ctx.identity_state.is_frozen)

    # Free Z-Image VRAM before WanWrapper
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0] / 1e9
        logger.info("VRAM free before S-08: %.1f GB", free)
    except Exception:
        pass

    # S-08: Video Segment 1 (Wan2.2-I2V — also needs ~17GB, runs standalone)
    t0 = time.monotonic()
    logger.info("=== S-08: Video Segment 1 (Wan2.2 I2V) ===")
    # Prompt uses visual descriptors only — no character names.
    # Character names in prompts cause DiffSynth to render floating text instead of animating.
    s08_prompt = (
        f"cinematic scene, {out_s03.character_identity}, "
        f"{scene_plan_0.setting}, dramatic motivational atmosphere, photorealistic"
    )
    out_s08, ctx = orch.execute_stage(
        VideoSegmentGenerator(),
        {
            "refined_image": out_s07["refined_image"],
            "output_dir": "/workspace/output/test_e2e_001/scene_001",
            "prompt": s08_prompt,
        },
        ctx,
    )
    logger.info("S-08 done %.1fs — segment_1=%s clip=%.4f",
                time.monotonic() - t0, out_s08["segment_1"].file_path, out_s08["clip_score"])

    # S-09: Temporal Engine (SVI autoregressive — Segments 2..N)
    # Only run if the scene has multiple segments (>1 SegmentPlanSchema)
    t0 = time.monotonic()
    logger.info("=== S-09: Temporal Engine (SVI autoregressive) ===")
    remaining_plans = scene_plan_0.segments[1:]  # segments 2..N (skip first, handled by S-08)
    if not remaining_plans:
        logger.info("S-09 skipped — scene has only 1 segment plan (single-segment scene)")
        out_s09 = None
    else:
        temporal_engine = TemporalEngine(
            buffer_manager=TemporalBufferManager(),
            motion_tracker=MotionStateTracker(),
            retry_controller=TemporalRetryController(),
            clip_validator=CLIPVal(),
        )
        out_s09, ctx = orch.execute_stage(
            temporal_engine,
            {
                "segment_1": out_s08["segment_1"],
                "segment_plans": remaining_plans,
                "output_dir": "/workspace/output/test_e2e_001/scene_001",
            },
            ctx,
        )
        logger.info("S-09 done %.1fs — %d additional segments", time.monotonic() - t0, len(out_s09["segments"]))

    logger.info("=== S-01 through S-09 COMPLETE ===")
    logger.info("Segment 1: %s", out_s08["segment_1"].file_path)
    if out_s09:
        for seg in out_s09["segments"]:
            logger.info("Segment %s: %s", seg.segment_id, seg.file_path)

    # ── Final Assembly: concatenate all segments into one coherent video ─────────
    # SVI continuation mode produces 4-frame overlap between segments:
    #   - Segment_1 is kept in full (81 frames ≈ 5.4s)
    #   - Segments 2..N each have their first 4 frames trimmed (those frames closely
    #     replicate the end of the previous segment, providing the seamless join)
    # Assembly trims the overlap and concatenates into scene_001_final.mp4.
    _assemble_final_video(
        segment_paths=[out_s08["segment_1"].file_path]
        + ([seg.file_path for seg in out_s09["segments"]] if out_s09 else []),
        output_path="/workspace/output/test_e2e_001/scene_001/scene_001_final.mp4",
        # SVI continuation uses 5 overlap frames: the last 5 frames of each segment
        # are hard-copied to the start of the next segment (pixel-identical seam),
        # then those 5 frames are trimmed here to prevent visual duplication.
        overlap_frames=5,
        fps=15,
    )


if __name__ == "__main__":
    main()
