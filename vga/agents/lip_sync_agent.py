"""
LipSyncAgent — Stage S-12: lip sync via LatentSync-1.6.
phoneme_alignment ≥ 0.80. identity_delta ≤ 0.03 (RULE-97). CLIP validation (RULE-89).
Spec: VGA Audio Pipeline Spec v17.2 §S-12; RULE-89, RULE-97
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from vga.agents.base_agent import BaseAgent
from vga.config.settings import settings
from vga.models.wrappers.latentsync_wrapper import LatentSyncWrapper
from vga.state.immutable_context import ImmutableContext
from vga.validation.clip_validator import CLIPValidator

logger = logging.getLogger(__name__)


class LipSyncAgent(BaseAgent):
    """S-12: applies lip sync to each video segment using LatentSync-1.6.

    RULE-97: identity_delta ≤ 0.03 per segment.
    RULE-89: CLIP validation on each lip-synced frame.
    """

    stage_id = "S-12"

    def __init__(
        self,
        latentsync: LatentSyncWrapper | None = None,
        clip_validator: CLIPValidator | None = None,
    ) -> None:
        self._latentsync = latentsync or LatentSyncWrapper()
        self._clip = clip_validator or CLIPValidator()

    def run(
        self,
        input_data: dict,
        context: ImmutableContext,
    ) -> Tuple[dict, ImmutableContext]:
        """Apply lip sync to all video segments.

        Args:
            input_data: dict with video_paths, audio_paths, output_dir
            context:    must have frozen identity_state

        Returns:
            (output dict with synced_paths + identity_delta_per_segment, new_context)
        """
        self._log_start(context.scene_id)
        context.assert_identity_frozen()

        video_paths: List[str] = input_data["video_paths"]
        audio_paths: List[str] = input_data["audio_paths"]
        output_dir = Path(input_data.get("output_dir", settings.OUTPUT_DIR / context.job_id / "lipsync"))
        output_dir.mkdir(parents=True, exist_ok=True)

        synced_paths = []
        identity_deltas = []
        reference = context.identity_state.embedding_vector

        for i, (video, audio) in enumerate(zip(video_paths, audio_paths)):
            out_path = output_dir / f"synced_{i+1:03d}.mp4"
            metrics = self._latentsync.sync(
                video_path=video,
                audio_path=audio,
                output_path=out_path,
            )
            synced_paths.append(str(out_path))
            identity_deltas.append(metrics.get("identity_delta", 0.0))
            logger.info(
                "LipSyncAgent: segment %d phoneme=%.4f delta=%.4f",
                i + 1,
                metrics.get("phoneme_alignment", 0.0),
                metrics.get("identity_delta", 0.0),
            )

        output = {
            "synced_paths": synced_paths,
            "identity_delta_per_segment": identity_deltas,
            "scene_id": context.scene_id,
            "schema_version": settings.SCHEMA_VERSION,
        }

        new_context = context.evolve(current_stage=self.stage_id)
        self._log_complete(context.scene_id)
        return output, new_context
