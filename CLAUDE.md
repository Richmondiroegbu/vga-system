# CLAUDE.md — VGA v17.2 Project-Wide Persistent Context

> **Read this file at the start of every session. These rules govern ALL VGA implementation work.**

---

## Project Identity

**Project:** Video Generation Automation (VGA) — Cinematic AI Motivation System  
**Version:** 17.2.0  
**Mission:** *Inspire audiences by telling stories of people who overcame adversity — restoring hope and faith.*  
**Deployed on:** RunPod RTX 4090 · Ubuntu 24 · Python 3.10 · CUDA 12.4  
**Primary spec:** See `docs/specs/` — 16 authoritative documents. When in doubt, check spec.

---

## Build Commands

```bash
# Install dependencies (main VGA env)
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

# Linting
python -m vga.devtools.architecture_linter --check-all
pre-commit run --all-files

# Tests
pytest tests/unit/ -v --tb=short
pytest tests/integration/ -v --tb=short

# API server (development)
uvicorn vga.api.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit UI
streamlit run vga/ui/app.py --server.port 8501

# Run bootstrap (on RunPod)
python3 /workspace/bootstrap_pipeline.py

# Run session controller (from local machine)
python3 session_controller.py
```

---

## Code Style Rules

- **Python 3.10** — use `|` union types, `match` statements where cleaner
- **Pydantic v2** — all schemas use `BaseModel`; `model_validate()` not `parse_obj()`
- **Type hints everywhere** — all function signatures fully typed; no bare `Any` unless necessary
- **Frozen dataclasses** for context objects — `@dataclass(frozen=True)`
- **No mutable defaults** — use `field(default_factory=...)` in dataclasses
- **snake_case** for files, variables, functions; **PascalCase** for classes; **UPPER_SNAKE_CASE** for constants
- **Docstrings** on every class and public method — single-line for simple, multi-line for complex
- **No bare except clauses** — always catch specific exceptions; log then re-raise or handle
- **No print()** — use `vga.core.logger` structured logging everywhere
- **No global state mutations** — use ImmutableContext.evolve() pattern

---

## Architecture Non-Negotiables

These 5 rules are checked by `ArchitectureLinter` on every commit:

1. **execute_stage() contract**: All pipeline stages execute via `master_orchestrator.execute_stage()`. Direct `agent.run()` calls raise `ArchitectureGuardViolationError`.
2. **ImmutableContext**: Context is always `ImmutableContext` dataclass. Dict context raises `TypeError` at `execute_stage()` entry.
3. **TemporalBuffer = 5 frames**: `TEMPORAL_BUFFER_SIZE = 5` is a hard constant. TemporalEngine refuses to run if buffer size ≠ 5.
4. **Autoregressive generation**: SVI generates one segment at a time in an explicit for loop. Batch SVI raises `AutoregressiveViolationError`.
5. **One model in VRAM**: ModelManager enforces sequential loading. Two heavy models simultaneously = `VRAMViolationError`.

---

## File Ownership (One File = One Responsibility)

Never put responsibility X in file Y if file Y's documented responsibility is Z.  
See `docs/specs/09_VGA_File_Responsibility_Specification_v17.2.md` for the authoritative map.

**Quick reference — most-touched files:**
- `execute_stage()` lives in `vga/core/master_orchestrator.py` ONLY
- `TemporalBuffer` lifecycle lives in `vga/temporal/temporal_buffer_manager.py` ONLY  
- `CLIPValidator.score()` lives in `vga/validation/clip_validator.py` ONLY
- `ImmutableContext.evolve()` lives in `vga/state/immutable_context.py` ONLY
- `CompositionPlanValidator` lives in `vga/validation/composition_validator.py` ONLY
- `SVIScheduler` (noise-aware LoRA) lives in `vga/temporal/svi_scheduler.py` ONLY
- Settings constants live in `vga/config/settings.py` ONLY — never hardcode threshold values inline

---

## When Implementing a New Stage Agent

Follow this checklist every time:

```
□ Inherits from BaseAgent (vga/agents/base_agent.py)
□ Input validated against input schema before processing
□ Output validated against output schema before returning
□ context.evolve() called with ALL 5 state dimensions updated
□ CLIP validation called if output contains images, video frames, or lip-synced frames
□ HRG checkpoint called via hrg_controller.checkpoint() after output
□ Traced with observability.tracer.trace_event()
□ Exception types imported from vga/core/exceptions.py (not defined inline)
□ Max retries = COMPOSITION_MAX_RETRIES or equivalent constant from settings.py
□ schema_version = "v6.0" on all artifact objects written to disk
```

---

## Forbidden Imports / Patterns

```python
# NEVER
import modelscope                    # All models are from HuggingFace
from diffusers import FluxPipeline   # Old FLUX.1 pipeline class
context["key"]                       # Dict context — use ImmutableContext
agent.run(input)                     # Direct call — use execute_stage()
torch.load(path, map_location=None)  # Unsafe — use map_location="cpu" then .to(device)
```

---

## Environment Variables (Key Ones)

All loaded via `python-dotenv` from `/workspace/.env_vga` (written by bootstrap):

```
TEMPORAL_BUFFER_SIZE=5
CLIP_IDENTITY_THRESHOLD=0.93
IDENTITY_CUMULATIVE_DRIFT_THRESHOLD=0.15
MIN_SNR_DB=10.0
SVI_CFG_MIN=5.0
SVI_CFG_MAX=6.0
HRG_REVIEW_ENABLED=true
HRG_APPROVAL_TIMEOUT_SECONDS=300
IMMUTABLE_CONTEXT_ENFORCE=true
CROSS_MODAL_SYNC_THRESHOLD=0.9
SVI_REPO_BRANCH=svi_wan22
SVI_WAN22_PYTHON=/opt/conda/envs/svi_wan22/bin/python
```

---

## Deviations

Any deviation from the spec MUST be documented in `DEVIATION_LOG.md` with:
- Rule/requirement violated
- Reason for deviation
- Operator approval status

Never silently deviate from any RULE-XX or FR-XXX constraint.

---

## Prompt Suite Reference

Full implementation prompts are in `/prompts/`. Start with `MASTER_PROMPT_INDEX.md`.
