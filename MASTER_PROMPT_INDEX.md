# VGA v17.2 — Master Prompt Index & Implementation Command Center
**Project:** Video Generation Automation (VGA) — Cinematic AI Video Production Engine  
**System Version:** 17.2.0  
**Prompt Suite Version:** 1.0.0  
**Status:** Authoritative Implementation Reference  
**Audience:** Claude Code Agent (Primary Executor)

---

## Mission Statement

> *Inspire audiences by telling stories of people who overcame adversity — restoring hope and faith.*

Every implementation decision is evaluated against this mission. Cinematic quality, temporal continuity, character identity, and emotional authenticity are non-negotiable.

---

## How to Use This Prompt Suite

This file is the **single entry point** for all VGA implementation work. Every prompt file references this index. Read this file FIRST before reading any individual prompt.

### For Claude Code — Reading Order
1. **This file** (MASTER_PROMPT_INDEX.md) — understand structure, constraints, and file map
2. **CLAUDE.md** — project-wide persistent rules that govern ALL sessions
3. **AGENT.md** — agent identity, authority model, and decision rules
4. **Target prompt file** (see §3 below) — the specific module you are implementing

### For Humans — Execution Order
1. Set up infrastructure → use `00_BOOTSTRAP_ENVIRONMENT_PROMPT.md`
2. Build in strict phase order → Phases 1 through 7 below
3. Each prompt file = one focused Claude Code session
4. Never skip a phase. Build order is enforced by dependency.

---

## §1. System Architecture Summary

VGA is a **deterministic, stateful, autoregressive, human-governed, multi-agent cinematic engine** deployed on **RunPod RTX 4090** (Ubuntu 24, Python 3.10, CUDA 12.4).

### Pipeline Overview (16 Stages, 5 Phases)
```
PHASE 1 — Narrative Intelligence (CPU/LLM)
  S-01  ScriptAgent (Qwen2.5-14B)
  S-02  ScenePlanner + SegmentPlanner
  S-03  IdentityDesignAgent
  S-04  SceneCompositionAgent → CompositionPlan [NEW v17.0]

PHASE 2 — Visual Grounding (GPU: Image)
  S-05  BaseImageAgent (FLUX.2-klein-4B, NO LoRA)
  S-06  IdentityReinforcementLoop (FLUX.2-klein + Consistance_Edit_LoRA)
  S-07  ImageRefinementAgent (Z-Image-Turbo)
  [HARD GPU CLEANUP between Phase 2 and 3]

PHASE 3 — Motion + Continuity (GPU: Video)
  S-08  VideoSegmentGenerator (Wan2.2-I2V-A14B-FP8) → Segment_1
  S-09  TemporalEngine (SVI Pro 2) → Segments 2..N [AUTOREGRESSIVE]
  S-10  ContinuityValidationAgent

PHASE 4 — Audio Realism
  S-11  DialogueAgent (CosyVoice3-0.5B)
  S-12  LipSyncAgent (LatentSync-1.6)
  S-13  AmbientAudioAgent (MMAudio)
  S-14  MusicAgent (MusicGen-medium)
  S-15  AudioMixingAgent (pydub/torchaudio) → SNR ≥ 10dB + no clipping

PHASE 5 — Finalization
  S-16  ExportAgent → AssemblyAgent + QualityAgent
```

### Critical Architecture Rules (All Must Be Enforced)
- **RULE-86**: TemporalBuffer MUST contain exactly 5 frames at all times
- **RULE-87**: Segment[n+1] MUST be conditioned on Segment[n] via TemporalBuffer — batch generation FORBIDDEN
- **RULE-88**: No image/video generation without valid CompositionPlan
- **RULE-89**: CLIP identity validation in ALL phases (image + video + lip sync)
- **RULE-90**: Validation BEFORE stage progression
- **RULE-91**: Base image generation MUST use FLUX.2-klein with NO LoRA
- **RULE-92**: CLIP identity lock threshold ≥ 0.93 everywhere
- **RULE-93**: Identity drift ≤ 0.02 per refinement step
- **RULE-95**: Same frozen `char_identity_ref` embedding across ALL phases
- **RULE-99**: Audio SNR ≥ 10dB, peaks ≤ 0 dBFS (no clipping)
- **RULE-106**: All stages execute via `execute_stage()` ONLY — direct agent calls FORBIDDEN
- **RULE-107**: SVI generation MUST be explicit per-segment loop — batch SVI FORBIDDEN
- **RULE-108**: ImmutableContext mandatory — dict-based context FORBIDDEN

---

## §2. Technology Stack

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| OS | Ubuntu 24 | — | RunPod RTX 4090 |
| Python | Python 3.10 | — | Main VGA env |
| CUDA | CUDA 12.4 | cu124 | Main env; SVI uses cu128 separately |
| PyTorch | torch≥2.5.1 | cu124 | MMAudio requires 2.5.1+ |
| Web Framework | FastAPI | ≥0.115.0 | API layer |
| UI | Streamlit | — | HRG review panels |
| LLM | Qwen2.5-14B-Instruct (unsloth 4bit) | — | Script + composition |
| Image Gen | FLUX.2-klein-4B | — | S-05, S-06 |
| Image Refine | Z-Image-Turbo | — | S-07 |
| Video Gen | Wan2.2-I2V-A14B-FP8 (nalexand) | — | S-08 |
| Temporal | SVI Pro 2 (vita-video-gen/svi-model) | v2.0 Pro | S-09, requires svi_wan22 branch + PyTorch 2.7.1+cu128 |
| Lip Sync | LatentSync-1.6 (ByteDance) | — | S-12 |
| TTS | Fun-CosyVoice3-0.5B-2512 | — | S-11 |
| Ambient Audio | MMAudio (hkchengrex) | large_44k_v2 | S-13 |
| Music | MusicGen-medium (facebook) | — | S-14 |
| Identity | CLIP ViT-L/14 (openai) | — | All phases |
| LoRA | Consistance_Edit_LoRA (lrzjason) | — | S-06 |
| SVI LoRAs | SVI_Wan2.2-I2V-A14B_{high,low}_noise_lora_v2.0_pro.safetensors | v2.0 Pro | S-09 |

### SVI Environment Isolation (CRITICAL)
```
Main VGA env:  conda env vga, Python 3.10, PyTorch 2.5.1+, CUDA 12.4 (cu124)
SVI env:       conda env svi_wan22, Python 3.10, PyTorch 2.7.1, CUDA 12.8 (cu128)
               → DiffSynth 2.0 base, svi_wan22 branch of vita-epfl/Stable-Video-Infinity
               → SVI inference called via subprocess from main VGA env
```

---

## §3. Prompt File Registry

All prompt files are located in `/prompts/` relative to the project root.

### Category A — Infrastructure & Bootstrap
| File | Purpose | Phase |
|------|---------|-------|
| `00_BOOTSTRAP_ENVIRONMENT_PROMPT.md` | RunPod env setup, model downloads, conda envs | Pre-build |
| `01_PROJECT_SKELETON_PROMPT.md` | Directory structure, pyproject.toml, configs | Phase 1 |

### Category B — Core Foundation
| File | Purpose | Phase |
|------|---------|-------|
| `02_CORE_FOUNDATION_PROMPT.md` | config/settings.py, exceptions.py, schemas.py, logger.py | Phase 2 |
| `03_IMMUTABLE_CONTEXT_SYSTEM_PROMPT.md` | ImmutableContext, ContextFactory, state management | Phase 2 |
| `04_MODEL_MANAGER_PROMPT.md` | ModelManager, AssetLoader, VRAM guard, smart reuse | Phase 2 |

### Category C — Orchestration Layer
| File | Purpose | Phase |
|------|---------|-------|
| `05_MASTER_ORCHESTRATOR_PROMPT.md` | execute_stage(), SystemGuard, HRGController (11 checkpoints) | Phase 3 |
| `06_SLA_ADAPTIVE_GATING_PROMPT.md` | SLAManager, GatingController, AdaptiveMemory, CalibrationEngine | Phase 3 |

### Category D — Validation Infrastructure
| File | Purpose | Phase |
|------|---------|-------|
| `07_VALIDATION_INFRASTRUCTURE_PROMPT.md` | CLIPValidator, CompositionValidator, AudioQualityValidator, CrossModalAlignmentValidator | Phase 3 |
| `08_IDENTITY_SYSTEM_PROMPT.md` | IdentityManager, IdentityTracker, IdentityDriftController, LightingNormalizer, TemporalIdentityValidator, IdentityReinforcementEngine, IdentityStateTracker (NEW v17.0) | Phase 3 |

### Category E — AI Model Wrappers
| File | Purpose | Phase |
|------|---------|-------|
| `09_MODEL_WRAPPERS_PROMPT.md` | All model wrappers: Qwen, FLUX, ZImage, Wan, SVI, CosyVoice, LatentSync, MMAudio, MusicGen | Phase 4 |

### Category F — Pipeline Agents (Phase 1–2)
| File | Purpose | Phase |
|------|---------|-------|
| `10_NARRATIVE_AGENTS_PROMPT.md` | ScriptAgent, ScenePlanner, SegmentPlanner, IdentityDesignAgent, SceneCompositionAgent (S-01–S-04) | Phase 5 |
| `11_IMAGE_PIPELINE_AGENTS_PROMPT.md` | BaseImageAgent, ImageEditAgent, MultiAngleAgent, ImageMergeAgent, SceneExpansionAgent, ImageRefinementAgent (S-05–S-07) | Phase 5 |

### Category G — Temporal Engine (Phase 3)
| File | Purpose | Phase |
|------|---------|-------|
| `12_TEMPORAL_ENGINE_PROMPT.md` | TemporalEngine, TemporalBufferManager, SVIScheduler, MotionStateTracker, TemporalRetryController (S-08–S-10) | Phase 5 |

### Category H — Audio Pipeline (Phase 4)
| File | Purpose | Phase |
|------|---------|-------|
| `13_AUDIO_PIPELINE_PROMPT.md` | DialogueAgent, LipSyncAgent, AmbientAudioAgent, MusicAgent, AudioMixingAgent (S-11–S-15) | Phase 5 |

### Category I — Export & Quality (Phase 5)
| File | Purpose | Phase |
|------|---------|-------|
| `14_EXPORT_QUALITY_PROMPT.md` | AssemblyAgent, ExportAgent, QualityAgent (S-16) | Phase 5 |

### Category J — API & UI
| File | Purpose | Phase |
|------|---------|-------|
| `15_FASTAPI_LAYER_PROMPT.md` | FastAPI routes, middleware, all endpoints including v17.0 routes | Phase 6 |
| `16_STREAMLIT_UI_PROMPT.md` | Streamlit UI, all 11 HRG panels, TemporalEngine status panel | Phase 6 |

### Category K — Client & Session Management
| File | Purpose | Phase |
|------|---------|-------|
| `17_SESSION_CONTROLLER_PROMPT.md` | session_controller.py — RunPod session management, bootstrap coordination | Phase 6 |
| `18_CLIENT_WATCHER_PROMPT.md` | VGA Client Watcher (AVON) — validation, download, cleanup, feedback | Phase 6 |

### Category L — Tests & DevTools
| File | Purpose | Phase |
|------|---------|-------|
| `19_TEST_SUITE_PROMPT.md` | Unit tests, integration tests, test fixtures | Phase 7 |
| `20_DEVTOOLS_PROMPT.md` | ArchitectureLinter, RuleChecker, SnapshotSystem, pre-commit hooks | Phase 7 |

---

## §4. Key Constraints for All Sessions

### Forbidden Patterns (NEVER implement these)
```
❌ Dict-based context (e.g., context["key"])          → Use ImmutableContext.evolve()
❌ Direct agent.run() calls                           → Use execute_stage() ONLY
❌ Batch SVI generation                               → Use per-segment loop ONLY
❌ Single-frame SVI conditioning (Segment 2+)         → Use 5-frame TemporalBuffer
❌ Static SVI LoRA weight                             → Use SVIScheduler (noise-aware)
❌ SVI CFG outside [5.0, 6.0]                        → Clamp strictly
❌ char_identity_ref recomputed mid-pipeline          → Freeze at S-05, immutable
❌ Image generation without CompositionPlan           → Enforce RULE-88 gate
❌ LoRA on base image generation (S-05)               → Pure FLUX.2-klein, no LoRA
❌ FLUX.1-schnell / Wav2Lip / Wan2.1 / IP-Adapter     → All removed, use replacements
❌ ModelScope / modelscope package                    → All models from HuggingFace
❌ Sequential TemporalBuffer with < 5 frames running  → Raise TemporalBufferError
```

### Required Patterns (ALWAYS implement these)
```
✅ All stages execute via execute_stage()
✅ context.evolve() called after every stage
✅ CLIP validation in image + video + lip sync phases  
✅ CompositionPlan validated before all image/video generation
✅ TemporalBuffer maintained at exactly 5 frames
✅ SNR ≥ 10dB + peaks ≤ 0 dBFS validated post-mixing
✅ HRG checkpoint called after every stage output (11 total)
✅ IdentityStateTracker updated per segment and per frame
✅ ImmutableContext with 5 dimensions: identity_state, motion_state, camera_state, lighting_state, temporal_state
✅ schema_version="v6.0" on all new artifacts
✅ All retries: max 3, with exponential backoff
```

### VRAM Sequential Contract
```
Only ONE heavy model in VRAM at any time.
Unload sequence: del model → gc.collect() → torch.cuda.empty_cache() → sleep(2)
Assert free_ratio ≥ 0.90 before loading next model.
Smart reuse: skip unload/reload if EXACT same model set already loaded.
```

---

## §5. Directory Structure (Authoritative)

```
/workspace/vga/               ← project root (RunPod)
├── vga/                      ← main source package
│   ├── agents/               ← 18 pipeline agents
│   ├── temporal/             ← TemporalEngine subsystem
│   ├── identity/             ← identity management
│   ├── validation/           ← validators
│   ├── models/               ← schemas, enums, wrappers
│   ├── core/                 ← orchestration, HRG, storage
│   ├── state/                ← ImmutableContext, ContextFactory
│   ├── runtime/              ← SLA, gating, SystemGuard, failure
│   ├── adaptive/             ← calibration, memory, optimizer
│   ├── observability/        ← tracer, metrics, audit
│   ├── api/                  ← FastAPI routes
│   ├── ui/                   ← Streamlit panels
│   ├── config/               ← settings, prompts
│   └── devtools/             ← linter, checker (dev-time only)
├── tests/
│   ├── unit/
│   └── integration/
├── prompts/                  ← THIS suite (all prompt files)
├── config/
├── scripts/
├── docs/
├── snapshots/
│   ├── v15_baseline/
│   ├── v16_candidate/
│   └── v17_candidate/
├── devtools_reports/
├── DEVIATION_LOG.md
├── .env.example
├── requirements.txt
├── requirements.lock
├── requirements-dev.txt
├── pyproject.toml
├── Makefile
├── CLAUDE.md                 ← Claude Code persistent context
├── AGENT.md                  ← Agent identity and decision rules
└── README.md
```

---

## §6. Spec Document Reference

All implementation decisions derive from these authoritative specs (located in `docs/specs/`):

| # | Document | Key Content |
|---|---------|------------|
| 01 | VGA_System_Requirements_Document_v17.2.md | All functional requirements, FR-001 through FR-983 |
| 02 | VGA_System_Architecture_Document_v17.2.md | Component architecture, stage interactions |
| 03 | VGA_Mathematical_Model_Specification_v17.2.md | CLIP scoring math, continuity formula, SVI scheduling math |
| 04 | VGA_Pipeline_Execution_Flow_Specification_v17.2.md | Stage execution flows, retry logic |
| 05 | VGA_Data_Contracts_Interface_Specification_v17.2.md | All schemas, data contracts |
| 06 | VGA_Engine_Template_Specification_v17.2.md | BaseAgent template, execution contracts |
| 07 | VGA_Codebase_Structure_Design_v17.2.md | Complete file tree |
| 08 | VGA_Dependency_Graph_Specification_v17.2.md | Module dependencies |
| 09 | VGA_File_Responsibility_Specification_v17.2.md | One file = one responsibility |
| 10 | VGA_Coding_Standards_and_Rules_v17.2.md | RULE-01 through RULE-110 |
| 11 | VGA_Development_Sequence_Build_Order_v17.2.md | Strict build order |
| 12 | VGA_Project_Skeleton_Anchor_Files_v17.2.md | Skeleton anchor files |
| 13 | VGA_Code_Generation_Specification_v17.2.md | Code generation rules |
| 14 | VGA_Client_Watcher_AutoDownload_SafeCleanup_v4.2.md | Client watcher (AVON) |
| 15 | RunPod_Model_Download_Specification_v6.5.md | Asset registry, download spec |
| 16 | VGA_Model_Stack_Setup_Guide_v7.2.md | Model installation guide |

---

## §7. Verification Commands

Run these after each build phase to verify correctness:

```bash
# Phase 2 — Core
python -c "from vga.state.immutable_context import ImmutableContext; print('ImmutableContext OK')"
python -c "from vga.config.settings import TEMPORAL_BUFFER_SIZE; assert TEMPORAL_BUFFER_SIZE == 5"

# Phase 3 — Orchestration  
python -c "from vga.core.master_orchestrator import execute_stage; print('execute_stage OK')"
python -c "from vga.validation.clip_validator import CLIPValidator; print('CLIPValidator OK')"

# Phase 5 — Agents
python -c "from vga.agents.scene_composition_agent import SceneCompositionAgent; print('SceneCompositionAgent OK')"
python -c "from vga.temporal.temporal_engine import TemporalEngine; print('TemporalEngine OK')"
python -c "from vga.temporal.temporal_buffer_manager import TemporalBufferManager; print('TemporalBufferManager OK')"

# Architecture guard
python -m vga.devtools.architecture_linter --check-all

# Unit tests
pytest tests/unit/ -v --tb=short

# Integration
pytest tests/integration/test_temporal_engine.py -v
```

---

## §8. HRG Checkpoint Map

All 11 Human Review Gate checkpoints and their stages:

| Checkpoint | Stage | What to Review | New/Updated |
|-----------|-------|---------------|-------------|
| HRG-1 | S-01 | Script content, character descriptions | Retained |
| HRG-2 | S-02 | Scene/segment plan, durations, segment counts | NEW v17.0 |
| HRG-3 | S-03 | Identity design, character visuals, environment | Renamed |
| HRG-4 | S-04 | CompositionPlan (camera, blocking, motion vectors) | NEW v17.0 |
| HRG-5 | S-05 | 6 base images, CLIP scores | Renamed |
| HRG-6 | S-06 | Identity reinforcement — angle variants, merged image | Renamed |
| HRG-7 | S-07 | Refined image, drift score | Renamed |
| HRG-8 | S-08/09 | Video segments + identity_per_segment scores | Updated |
| HRG-9 | S-11 | Dialogue audio, timing alignment | Renamed |
| HRG-10 | S-12 | Lip sync + identity delta per segment | Updated |
| HRG-11 | S-15 | Final audio — SNR, clipping status, mix levels | Updated |

---

## §9. Schema Version Contract

All artifacts produced by v17.2 agents MUST include:
```python
schema_version: str = "v6.0"
```

Legacy artifacts at v5.2 are migrated via `core/schema_migrations.py:_migrate_v5_2_to_v6_0()`.

---

## §10. Quick Reference — Model Paths (RunPod)

```
/workspace/models/qwen/          → unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit
/workspace/models/flux2/         → black-forest-labs/FLUX.2-klein-4B
/workspace/models/zimage/        → Tongyi-MAI/Z-Image-Turbo
/workspace/models/wan22/         → nalexand/Wan2.2-I2V-A14B-FP8
/workspace/models/svi/version-2.0/  → SVI LoRAs (high_noise + low_noise)
/workspace/LatentSync/checkpoints/  → ByteDance/LatentSync-1.6
/workspace/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/
/workspace/models/musicgen/      → facebook/musicgen-medium
/workspace/MMAudio/              → hkchengrex/MMAudio
/workspace/auxiliary/clip/       → openai/clip-vit-large-patch14
/workspace/loras/consistency/    → lrzjason/Consistance_Edit_Lora
/workspace/loras/svi/            → symlinks to svi/version-2.0/ LoRAs
/workspace/Stable-Video-Infinity/ → vita-epfl/Stable-Video-Infinity (svi_wan22 branch)
```

---

*This index is the entry point for ALL VGA v17.2 implementation sessions.*  
*Spec suite: v17.2 | Prompt suite: v1.0 | Last updated: 2026-05-08*
