# VGA Dependency Graph Specification
**Project:** Video Generation Automation (VGA) — Motivation System
**Version:** 17.2.0
**Status:** Authoritative Reference
**Audience:** DevOps, Engineers, Claude Code Agent

---

## 1. Overview

All v16.0 dependency categories retained. v17.0 adds:

**Temporal Engine dependencies:** SVI Pro 2 (via SVI wrapper; multi-frame latent conditioning), torchvision (optical flow estimation for MotionStateTracker), TemporalBuffer tensor management (torch only; CPU-resident between SVI calls).

**Scene Composition dependencies:** Qwen2.5-14B structured output (already in stack), `composition_prompts.py` (new prompt template), `CompositionPlanValidator` (pure Python validation; no external deps).

**Identity State dependencies:** torch (cosine similarity), CLIPValidator (already in stack), `IdentityStateTracker` (pure Python accumulator with torch ops).

**Audio Quality dependencies:** pydub (already in stack; peak level + RMS computation), torchaudio (already in stack; SNR computation), math (stdlib; log computation).

**v17.0 new API routes:** FastAPI (already in stack); new route files only — no additional external dependencies.

**Schema version advances to v6.0.** All v16.0 dependency constraints and import rules retained and extended.

---

## 2. Python Package Dependencies

### 2.1–2.12 All v16.0 Package Dependencies Retained

All v16.0 §2.1–§2.12 package dependencies retained unchanged.

### 2.13 Temporal Engine Packages (NEW v17.0)

| Package | Version | Purpose |
|---|---|---|
| `torch` | `>=2.2.0` (already required) | TemporalBuffer tensor operations; latent encoding; cosine similarity |
| `torchvision` | `>=0.17.0` | Optical flow estimation for MotionStateTracker |
| `opencv-python` | `>=4.9.0` | Fallback optical flow (cv2.calcOpticalFlowFarneback) |

Note: `torchvision` may already be installed as a dependency of `torch`. Pin version to match torch: `torchvision==0.17.1+cu121` with `torch==2.2.1+cu121`.

### 2.14 Audio Quality Packages (NEW v17.0)

| Package | Version | Purpose |
|---|---|---|
| `pydub` | `>=0.25.1` (already required) | Peak level computation; `AudioSegment.max`; `apply_gain()` |
| `torchaudio` | `>=2.2.0` (already required) | SNR computation; audio signal processing |
| `math` | stdlib | `math.log10()` for dB conversion |

No new external packages required for audio quality validation — all needed packages already in the v16.0 stack.

### 2.15 Dependency Conflict Resolution (v17.0 additions)

Additional conflict rules extending v16.0 §2.13:

```
12. torchvision vs torch:
   torchvision must match torch version exactly.
   Pin: torchvision==0.17.1+cu121 with torch==2.2.1+cu121.
   If optical flow from torchvision causes memory issues, use cv2 fallback:
     cv2.calcOpticalFlowFarneback(prev, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
   Bootstrap checks: import torchvision; assert hasattr(torchvision.models, 'optical_flow')
   Fallback: if torchvision optical_flow unavailable → import cv2; use cv2 optical flow.

13. TemporalBuffer tensor memory:
   Buffer.frames tensor MUST be CPU-resident between SVI calls (VRAM discipline).
   Only moved to GPU inside TemporalBufferManager.encode() with try/finally.
   If torch.cuda.memory_allocated() > 0.85 × total_vram BEFORE encode():
     Wait or reduce batch size; do not load more tensors.
   This is a runtime check, not a pip dependency.

14. SVI Pro 2 installation:
   SVI Pro 2 is not on PyPI. Install from source or private repository.
   requirements.lock references: svi_pro_2 @ file:///workspace/svi_pro_2
   If SVI Pro 2 unavailable: DEVIATION_LOG.md required; cannot use single-frame fallback.
   Alternative: Extended SVI inference via Wan2.2 in image2video mode (reduced quality).
```

---

## 3. Model Download Dependencies (v17.0)

All v16.0 model downloads retained. v17.0 clarifications:

| Model | HuggingFace Path | Download Size | Phase Needed | Notes |
|---|---|---|---|---|
| FLUX.2-klein 4B | `black-forest-labs/FLUX.2-klein` | ~4GB | Phase 2 | Base + identity editing |
| Consistance_Edit_LoRA | `lrzjason/Consistance_Edit_Lora` | ~0.5GB | Phase 2 (conditional) | Only stages 6A/6B/6C |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | ~4GB | Phase 2 | Refinement only |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | ~1GB | Phases 2, 3, 4 (CPU) | Cross-phase identity |
| Wan2.2-I2V-A14B-FP8 | `nalexand/Wan2.2-I2V-A14B-FP8` | ~14GB | Phase 3 (S-08) | Segment_1 only |
| SVI Pro 2 | (private/local install) | ~8–12GB | Phase 3 (S-09) | Segments 2..N |
| CosyVoice3-0.5B | `FunAudioLLM/CosyVoice3-0.5B` | ~1GB | Phase 4 | Dialogue |
| LatentSync-1.6 | `ByteDance/LatentSync-1.6` | ~2GB | Phase 4 | Lip sync |
| MMAudio | (MMAudio repo) | ~1GB | Phase 4 | Ambient audio |
| MusicGen-medium | `facebook/musicgen-medium` | ~1.5GB | Phase 4 | Background music |

**v17.0 VRAM notes for S-08 → S-09 transition:**
- Wan2.2 (~14GB) and SVI Pro 2 (~8–12GB) MUST NOT be simultaneously resident
- If total available VRAM < 14GB after cleanup: swap Wan → SVI between S-08 and S-09
- On RTX 4090 (24GB VRAM): Wan2.2 at ~14GB leaves ~10GB → SVI may fit if small variant
- Bootstrap validates: assert get_free_vram_gb() >= 14.0 before loading Wan2.2

---

## 4. Import Dependency Graph (v17.0)

```
bootstrap.py
  ↓ imports all v16.0 components (unchanged)
  ↓ imports v17.0 additions:

  → agents/scene_composition_agent.py
      → validation/composition_validator.py
      → models/schemas.py (CompositionPlanSchema)
      → config/prompts/composition_prompts.py
      → models/wrappers/qwen_wrapper.py

  → temporal/temporal_engine.py
      → temporal/temporal_buffer_manager.py
      → temporal/svi_scheduler.py
      → temporal/motion_state_tracker.py
      → temporal/temporal_retry_controller.py
      → validation/clip_validator.py
      → identity/identity_state_tracker.py
      → models/wrappers/svi_wrapper.py
      → models/schemas.py (TemporalBufferRecord, MotionStateRecord, SVIGenerationRecord)

  → identity/identity_state_tracker.py
      → validation/clip_validator.py
      → models/schemas.py (IdentityStateRecord)
      → config/settings.py (IDENTITY_CUMULATIVE_DRIFT_THRESHOLD)
      → core/exceptions.py (IdentityCumulativeDriftError)

  → validation/audio_quality_validator.py
      → pydub (external)
      → math (stdlib)
      → models/schemas.py (AudioQualityRecord)
      → config/settings.py (MIN_SNR_DB, MAX_PEAK_DBFS)

  → validation/cross_modal_alignment_validator.py
      → models/schemas.py (CrossModalAlignmentRecord, CrossModalAlignmentReport)
      → config/settings.py (TIMING_TOLERANCE_S)

  → api/routes/temporal.py
      → core/hrg_controller.py
      → models/schemas.py (TemporalBufferStatusResponse)

  → api/routes/identity.py
      → identity/identity_state_tracker.py (via orchestrator; not direct)
      → models/schemas.py (IdentityStateResponse)

  → ui/components/hrg_panels/hrg_4_composition.py
      → httpx (external; FastAPI calls only)
      → streamlit (external)

  → ui/components/temporal_engine_panel.py
      → httpx (external; FastAPI calls only)
      → streamlit (external)

agent dependency chain (v17.0 example — video_segment_generator):
  video_segment_generator.py
    → temporal/temporal_buffer_manager.py (init buffer)
    → validation/clip_validator.py (identity per segment)
    → models/wrappers/wan_wrapper.py (Segment_1 generation)
    → models/schemas.py (TemporalBufferRecord)
    → core/storage.py (video segment save)
    → validation/composition_validator.py (assert_in_context)
    → core/tracer.py (event logging)
    → runtime/system_guard.py (stage wrapping)

agent dependency chain (v17.0 — temporal_engine):
  temporal/temporal_engine.py
    → temporal/temporal_buffer_manager.py
    → temporal/svi_scheduler.py
    → temporal/motion_state_tracker.py
    → temporal/temporal_retry_controller.py
    → validation/clip_validator.py
    → identity/identity_state_tracker.py
    → models/wrappers/svi_wrapper.py
    → models/schemas.py
    → core/storage.py
    → core/tracer.py
    → state/immutable_context.py
```

---

## 5. Runtime vs Dev-time Dependency Separation

```
RUNTIME (required in production):
  All packages in §2.1–§2.14

DEV-TIME ONLY (in requirements-dev.txt):
  pytest>=8.0.0
  pre-commit>=3.7.0
  mypy>=1.9.0
  ruff>=0.4.0
  devtools (vga/devtools/) — zero runtime footprint

DEVTOOLS IMPORT RULE (unchanged from v16.0):
  No production file may import from vga.devtools.
  ArchitectureLinter enforces. Violation = RULE-67 breach.
```

---

## 6. System Startup Dependency Order (v17.0)

```
1–10. [All v16.0 steps 1–10 retained]

11. [v16.0] validation/clip_validator.py initialized (CLIP model to CPU)
12. [v16.0] models/lora_manager.py initialized
13. [v16.0] validation/timing_validator.py initialized
14. [v16.0] core/hrg_controller.py initialized (v17.0: 11 checkpoints)
15. [v16.0] api/main.py started (FastAPI on port 8000)
16. [v16.0] ui/app.py started (Streamlit on port 8501)

17. [v17.0] agents/scene_composition_agent.py initialized
18. [v17.0] validation/composition_validator.py initialized
19. [v17.0] temporal/temporal_buffer_manager.py initialized (buffer cleared)
20. [v17.0] temporal/svi_scheduler.py initialized (noise thresholds computed)
21. [v17.0] temporal/motion_state_tracker.py initialized
22. [v17.0] temporal/temporal_retry_controller.py initialized
23. [v17.0] identity/identity_state_tracker.py initialized (drift_score=0.0)
24. [v17.0] validation/audio_quality_validator.py initialized
25. [v17.0] validation/cross_modal_alignment_validator.py initialized

26. All model wrappers instantiated (models NOT loaded until their phase)
27. Bootstrap complete; job queue accepting submissions

Bootstrap assertion additions (v17.0):
  assert shutil.which("ffmpeg"), "ffmpeg binary not found"  (v16.0)
  import torchvision; assert hasattr(torchvision, 'models')  (v17.0)
  # If optical flow unavailable from torchvision: import cv2 (fallback)
  assert SVI_CFG_MIN == 5.0 and SVI_CFG_MAX == 6.0  (v17.0 — config sanity)
  assert TEMPORAL_BUFFER_SIZE == 5  (v17.0 — buffer size sanity)
  assert SCHEMA_VERSION == "v6.0"  (v17.0 — schema version check)
```


---

## v17.2 Dependency Graph Additions (NEW)

### temporal_authority_guard.py
```
Depends on:
  - vga/core/exceptions.py (ArchitectureGuardViolationError)

Used by:
  - temporal_engine.py (guard_segment_iteration)
  - temporal_buffer_manager.py (guard_buffer_update)
  - svi_wrapper.py (guard_svi_invoke)

Note: temporal_authority_guard.py MUST be generated BEFORE any file that imports it.
Build order: after exceptions.py, before temporal_engine.py (Phase 12.19.1)
```

### cross_modal_validation_unified.py
```
Depends on:
  - vga/validation/clip_validator.py (CLIPValidator)
  - vga/models/schemas.py (CrossModalValidationContract)
  - vga/config/settings.py (TIMING_TOLERANCE_S, PHONEME_ALIGNMENT_THRESHOLD, SEGMENT_CONTINUITY_MIN)
  - vga/core/tracer.py (Tracer)

Used by:
  - agents/lip_sync_agent.py (per-segment cross-modal validation)
  - agents/audio_mixing_agent.py (final cross-modal report)

Build order: after schemas.py and clip_validator.py (Phase 12.19.2)
```

### system_certification_validator.py
```
Depends on:
  - vga/config/settings.py (all threshold constants)
  - vga/core/exceptions.py (SystemCertificationFailureError)
  - vga/core/tracer.py (Tracer)
  - vga/models/schemas.py (PipelineReport)

Used by:
  - agents/quality_agent.py (final certification gate)

Build order: after exceptions.py, before quality_agent.py (Phase 12.19.3)
```
