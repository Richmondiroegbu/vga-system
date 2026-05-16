"""
TemporalBufferManager — SOLE owner of TemporalBuffer lifecycle.
Enforces RULE-86: buffer MUST contain exactly 5 frames at ALL times.
Buffer frames are always CPU-resident between segment generations.
Spec: VGA Temporal Engine Spec v17.2 §TEMPORAL ENFORCEMENT BLOCK; RULE-86, RULE-87
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

from vga.config.settings import settings
from vga.core.exceptions import AutoregressiveViolationError, TemporalBufferError

logger = logging.getLogger(__name__)

# Normalization constants — FIXED, never computed per-call (RULE-86)
_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BUFFER_SIZE = settings.TEMPORAL_BUFFER_SIZE   # must be 5


@dataclass
class TemporalBuffer:
    """Frozen container for exactly 5 video frames.

    frames: np.ndarray of shape (5, H, W, 3), dtype=float32, CPU-resident.
    timestamps: list of 5 float timestamps.
    """

    frames: np.ndarray           # shape (5, H, W, 3), float32
    timestamps: List[float]       # exactly 5 timestamps
    scene_id: str = ""
    segment_id: str = ""

    def __post_init__(self) -> None:
        TemporalBufferManager._assert_buffer_size(self)


class TemporalBufferManager:
    """Creates, updates, and encodes the rolling 5-frame TemporalBuffer.

    Device rule (MANDATORY):
    - Frames ALWAYS CPU-resident between segment generations.
    - encode() moves to GPU temporarily, returns CPU tensor immediately.
    - GPU tensors NEVER persist beyond the encode() method call.
    """

    def __init__(self) -> None:
        logger.info("TemporalBufferManager initialized — buffer size=%d", BUFFER_SIZE)

    def init(self, segment_video_path: str) -> TemporalBuffer:
        """Extract the last 5 frames from Segment_1 video and create TemporalBuffer.

        Args:
            segment_video_path: path to segment_001.mp4 (Segment_1 from Wan2.2)

        Returns:
            TemporalBuffer with exactly 5 frames (RULE-86)

        Raises:
            TemporalBufferError if fewer than 5 frames are available
        """
        frames, timestamps = self._extract_last_n_frames(segment_video_path, BUFFER_SIZE)

        if len(frames) < BUFFER_SIZE:
            raise TemporalBufferError(
                f"Segment_1 has only {len(frames)} frames — need at least {BUFFER_SIZE} "
                f"to initialize TemporalBuffer. RULE-86.",
                frame_count=len(frames),
                required=BUFFER_SIZE,
            )

        normalized = self._normalize_frames(np.stack(frames[-BUFFER_SIZE:], axis=0))
        buffer = TemporalBuffer(
            frames=normalized,
            timestamps=timestamps[-BUFFER_SIZE:],
        )
        self._assert_buffer_size(buffer)
        logger.info("TemporalBufferManager: buffer initialized from Segment_1 — %d frames", BUFFER_SIZE)
        return buffer

    def update(self, buffer: TemporalBuffer, new_segment_path: str) -> TemporalBuffer:
        """Rolling update: replace buffer with last 5 frames of new_segment.

        The old buffer instance is discarded (garbage collected).
        Returns a NEW TemporalBuffer — does not mutate the input.

        Args:
            buffer:           current TemporalBuffer (discarded after this call)
            new_segment_path: path to the newly generated video segment

        Returns:
            New TemporalBuffer with exactly 5 frames from the new segment
        """
        self._assert_buffer_size(buffer)   # verify input before update

        frames, timestamps = self._extract_last_n_frames(new_segment_path, BUFFER_SIZE)
        if len(frames) < BUFFER_SIZE:
            raise TemporalBufferError(
                f"New segment has only {len(frames)} frames — cannot update buffer. RULE-86.",
                frame_count=len(frames),
                required=BUFFER_SIZE,
            )

        normalized = self._normalize_frames(np.stack(frames[-BUFFER_SIZE:], axis=0))
        new_buffer = TemporalBuffer(
            frames=normalized,
            timestamps=timestamps[-BUFFER_SIZE:],
            scene_id=buffer.scene_id,
            segment_id=new_segment_path,
        )
        self._assert_buffer_size(new_buffer)   # verify after update
        logger.debug("TemporalBufferManager: buffer updated — %d frames", BUFFER_SIZE)
        return new_buffer

    def encode(self, buffer: TemporalBuffer) -> "np.ndarray":
        """Encode the 5-frame buffer as latent tensor for SVI conditioning.

        Device contract (MANDATORY):
        1. Transfer frames to GPU
        2. Encode as latents
        3. Return CPU tensor immediately
        GPU tensors NEVER persist beyond this method.

        Args:
            buffer: TemporalBuffer with exactly 5 frames

        Returns:
            np.ndarray of shape (5, C', H', W') — latent encoding, CPU-resident

        Raises:
            AutoregressiveViolationError if latent shape[0] != 5 (RULE-87)
        """
        self._assert_buffer_size(buffer)

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Transfer to GPU — TEMPORARY
            frames_tensor = torch.from_numpy(buffer.frames).permute(0, 3, 1, 2).float()
            frames_gpu = frames_tensor.to(device)

            # VAE-like encoding: downsample spatially for latent representation
            import torch.nn.functional as F
            H, W = frames_gpu.shape[2], frames_gpu.shape[3]
            latent_H, latent_W = H // 8, W // 8
            latents_gpu = F.interpolate(
                frames_gpu, size=(latent_H, latent_W), mode="bilinear", align_corners=False
            )
            # Apply simple scaling to simulate VAE latent space
            latents_gpu = latents_gpu * 0.18215

            # Return to CPU IMMEDIATELY — GPU tensors must not persist
            latents_cpu = latents_gpu.cpu().numpy()
            del frames_gpu, latents_gpu

        except ImportError:
            # Fallback for environments without torch (testing)
            frames = buffer.frames  # (5, H, W, 3)
            H, W = frames.shape[1] // 8, frames.shape[2] // 8
            latents_cpu = frames.transpose(0, 3, 1, 2)[:, :, :H, :W]

        # RULE-87: assert latent has 5-frame dimension
        if latents_cpu.shape[0] != BUFFER_SIZE:
            raise AutoregressiveViolationError(
                f"Encoded latent shape[0]={latents_cpu.shape[0]} but must be {BUFFER_SIZE}. "
                f"Single-frame conditioning is FORBIDDEN. RULE-87."
            )

        logger.debug("TemporalBufferManager: encoded buffer — latent shape=%s", latents_cpu.shape)
        return latents_cpu

    @staticmethod
    def _assert_buffer_size(buffer: TemporalBuffer) -> None:
        """Raise TemporalBufferError immediately if frames.shape[0] != 5. RULE-86."""
        count = buffer.frames.shape[0]
        if count != BUFFER_SIZE:
            raise TemporalBufferError(
                f"TemporalBuffer has {count} frames — MUST be exactly {BUFFER_SIZE}. RULE-86.",
                frame_count=count,
                required=BUFFER_SIZE,
            )

    @staticmethod
    def _normalize_frames(frames: np.ndarray) -> np.ndarray:
        """Apply fixed normalization to frame array. Constants never recomputed."""
        frames_f = frames.astype(np.float32) / 255.0
        frames_f = (frames_f - _NORM_MEAN) / _NORM_STD
        return frames_f.astype(np.float32)

    @staticmethod
    def _extract_last_n_frames(
        video_path: str,
        n: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Extract the last N frames from a video file as numpy arrays."""
        frames = []
        timestamps = []
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Seek to last N frames
            start_frame = max(0, total - n)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            idx = start_frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                timestamps.append(idx / fps)
                idx += 1

            cap.release()
        except Exception as exc:
            logger.error("TemporalBufferManager: frame extraction failed: %s", exc)

        return frames, timestamps
