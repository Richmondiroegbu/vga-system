# Prompt 08 — Identity System

**Suite:** VGA v17.2 Implementation Prompt Suite  
**Sequence:** 08 of 20 (after `07_VALIDATION_INFRASTRUCTURE_PROMPT.md`, before `09_MODEL_WRAPPERS_PROMPT.md`)  
**Depends on:** Prompts 02 (schemas, exceptions), 03 (ImmutableContext), 07 (CLIPValidator)  
**Produces:** All 7 files in `vga/identity/`

---

## Context for the Agent

The Identity System is the **backbone of visual consistency** in VGA. Every frame, image, and video segment that passes through the pipeline must be anchored to a single, frozen `char_identity_ref` embedding that is locked into the `ImmutableContext` by `ImageRefinementAgent` at the end of stage S-07. From that point forward, every CLIPValidator call uses this frozen reference and every result is tracked by `IdentityStateTracker`.

The identity system spans **7 files** in `vga/identity/`. Five are retained from v14.0 (`identity_manager.py`, `identity_tracker.py`, `identity_drift_controller.py`, `lighting_normalizer.py`, `temporal_identity_validator.py`) plus a sixth engine file (`identity_reinforcement_engine.py`). One file is **new in v17.0**: `identity_state_tracker.py`, which introduces cross-phase cumulative drift tracking.

### Spec References
- `02_VGA_System_Architecture_Document.md` §5.20, §AP-38, §12
- `05_VGA_Data_Contracts_Interface_Specification.md` §IdentityStateRecord, §IdentityStateResponse
- `06_VGA_Engine_Template_Specification.md` §70 (IdentityStateTracker template — USE VERBATIM as base)
- `07_VGA_Codebase_Structure_Design.md` identity/ folder
- `09_VGA_File_Responsibility_Specification.md` §12.1
- `10_VGA_Coding_Standards_and_Rules.md` CGRL-94, CGRL-95, CGRL-96

---

## Files to Implement

| # | File | Status | Purpose |
|---|------|--------|---------|
| 1 | `vga/identity/identity_manager.py` | v14.0 retained | Central identity orchestration — builds identity_design from LLM output, assigns character seeds |
| 2 | `vga/identity/identity_tracker.py` | v14.0 retained | Per-stage identity embedding comparison (single-stage CLIP delta) |
| 3 | `vga/identity/identity_drift_controller.py` | v14.0 retained | Decides regeneration strategy when identity fails threshold |
| 4 | `vga/identity/lighting_normalizer.py` | v14.0 retained | Normalises lighting embeddings before CLIP identity comparison |
| 5 | `vga/identity/temporal_identity_validator.py` | v14.0 retained | Cross-frame identity similarity check (CLIP cosine sim ≥ 0.97) |
| 6 | `vga/identity/identity_reinforcement_engine.py` | v14.0 retained | Drives the S-06 IdentityReinforcementLoop retry logic |
| 7 | `vga/identity/identity_state_tracker.py` | **NEW v17.0** | Cross-phase cumulative drift tracking; raises IdentityCumulativeDriftError |

---

## Architectural Principles (AP-38)

> "Identity is not an image output. It is a persistent system state (IdentityState) that is constructed (S-05), reinforced (S-06), validated (S-07), preserved through video segments (S-09), and re-validated after lip sync (S-12). The frozen `char_identity_ref` embedding is the identity anchor across all phases."

### Non-negotiable rules:

1. **`char_identity_ref` is FROZEN after S-07** — set in `ImmutableContext.identity_state.embedding_vector`. Never recomputed downstream. Any agent reading it must read from context, not recompute.
2. **CLIPValidator + IdentityStateTracker are always paired** (CGRL-96). Every `clip_validator.score(frame, char_identity_ref)` call MUST be immediately followed by `identity_tracker.update(char_identity_ref, frame, stage_id)`.
3. **CLIP identity threshold = 0.93** at every image stage (S-05, S-06, S-07) and every video segment keyframe (S-09) and every lip-synced frame (S-12).
4. **Cross-frame identity similarity ≥ 0.97** (TemporalIdentityValidator — within a video segment).
5. **Cumulative drift threshold = 0.15** (IDENTITY_CUMULATIVE_DRIFT_THRESHOLD from settings). When exceeded, `IdentityCumulativeDriftError` is raised; the **caller** decides remediation (not IdentityStateTracker).
6. **Maximum 1 phase regeneration** (IDENTITY_MAX_PHASE_REGENERATIONS). Tracked by callers, not by this system.

---

## File 1: `vga/identity/identity_manager.py`

### Purpose
Central identity orchestration for stage S-03 (IdentityDesignAgent). Receives raw `identity_design.json` from Qwen, validates the `reference_strategy` field, assigns deterministic character seeds, and ensures the design is complete before being stored in the job context.

### What to implement

```python
"""
identity_manager.py
Central identity orchestration — stage S-03 support.
Enforces: RULE-89, CGRL-94, CGRL-95
"""
```

**Class:** `IdentityManager`

**Constructor params:**
- `prompt_builder` — builds identity_prompts for Qwen
- `tracer: Tracer` — observability
- `storage` — job storage access

**Key methods:**

```python
def build_identity_design(
    self,
    script_output: ScriptSchema,
    scene_plan: SegmentPlanSchema,
    trace_id: str
) -> IdentityDesignSchema:
    """
    Orchestrates identity design construction.
    - Calls prompt_builder.build_identity_prompt(script_output, scene_plan)
    - Validates IdentityDesignSchema including reference_strategy field
    - Assigns deterministic seeds per character via _assign_character_seeds()
    - Returns validated IdentityDesignSchema
    """

def _assign_character_seeds(self, character_identity: dict) -> dict:
    """
    Generates deterministic integer seed per character name using
    hash(character_name) % (2**31). Seeds are stored in character_identity dict.
    This ensures seed locking across retries and sessions.
    """

def validate_reference_strategy(self, identity_design: IdentityDesignSchema) -> None:
    """
    Validates reference_strategy is present and non-empty.
    Raises IdentityDesignValidationError if missing.
    """
```

**Forbidden:**
- Computing `char_identity_ref` (that is done only by `ImageRefinementAgent` / CLIPValidator after S-07)
- Calling CLIPValidator (identity_manager operates pre-GPU; no embeddings here)

**Imports required:**
```python
from vga.models.schemas import IdentityDesignSchema, ScriptSchema, SegmentPlanSchema
from vga.core.exceptions import IdentityDesignValidationError
from vga.core.tracer import Tracer
```

---

## File 2: `vga/identity/identity_tracker.py`

### Purpose
Single-stage CLIP embedding delta computation. Called by `CLIPValidator` (via its consumer agents) to measure per-stage identity drift *before* handing off to `IdentityStateTracker`. Unlike `IdentityStateTracker`, this class measures only the **current frame vs. reference** — it does NOT accumulate across stages.

### What to implement

```python
"""
identity_tracker.py
Per-stage identity embedding comparison. Single-stage CLIP delta.
Enforces: CGRL-94 (CLIPValidator must be called at every phase boundary)
"""
```

**Class:** `IdentityTracker`

**Constructor params:**
- `clip_encoder` — shared CLIP encoder (CPU; no VRAM allocation here)
- `tracer: Tracer`

**Key methods:**

```python
def compute_delta(
    self,
    char_identity_ref: torch.Tensor,  # frozen embedding from ImmutableContext
    frame,                             # PIL Image or torch.Tensor (H×W×3)
    stage_id: str
) -> float:
    """
    Computes cosine distance between char_identity_ref and frame embedding.
    delta = 1.0 - cosine_similarity(ref, frame_embedding)
    Returns float delta in range [0.0, 1.0].
    Does NOT accumulate; does NOT raise errors.
    Logs: identity_tracker_delta event.
    """

def is_within_threshold(self, delta: float, threshold: float = 0.07) -> bool:
    """
    Returns True if delta is within acceptable single-stage bound.
    Default threshold 0.07 (1.0 - 0.93 CLIP score).
    """
```

**Note:** `IdentityTracker` is a pure computation utility. It never modifies state. All state mutation is done by `IdentityStateTracker`.

---

## File 3: `vga/identity/identity_drift_controller.py`

### Purpose
Decides the regeneration strategy when identity checks fail (CLIP < 0.93 or temporal cosine sim < 0.97). Called by agents **after** CLIPValidator fails and **before** retry. Provides parameterised retry adjustments.

### What to implement

```python
"""
identity_drift_controller.py
Decides regeneration strategy when identity fails threshold.
Enforces: CGRL-95, CGRL-96
"""
```

**Class:** `IdentityDriftController`

**Constructor params:**
- `tracer: Tracer`
- `max_regenerations: int = IDENTITY_MAX_PHASE_REGENERATIONS`

**Key methods:**

```python
def get_retry_strategy(
    self,
    attempt: int,
    stage_id: str,
    clip_score: float,
    threshold: float = 0.93
) -> dict:
    """
    Returns a strategy dict with reinforcement adjustments for the next attempt.
    
    Attempt 0→1:
      - Increase LoRA weight by 0.05 (capped at 0.65)
      - Add "highly detailed face, sharp features" to positive prompt
      - Lower CFG slightly (SVI only): -0.2 from current
    
    Attempt 1→2:
      - Add char_identity_ref re-injection signal flag
      - Stronger identity anchoring tokens in prompt
    
    Returns: {
        "lora_weight_delta": float,
        "prompt_suffix": str,
        "cfg_delta": float,
        "reinject_identity": bool,
        "attempt": int
    }
    """

def should_abort(self, attempt: int) -> bool:
    """Returns True if attempt >= max_regenerations. Caller raises IdentityDriftError."""

def _strengthen_identity_prompt(self, attempt: int) -> str:
    """Returns increasingly strong identity anchoring suffix per attempt."""
```

**Forbidden:**
- Raising exceptions (controller only advises; agents raise)
- Accessing ImmutableContext directly
- Calling CLIPValidator

---

## File 4: `vga/identity/lighting_normalizer.py`

### Purpose
Normalises lighting conditions across frames before CLIP identity comparison. Without lighting normalization, CLIP scores may drop below 0.93 in scenes with dramatic lighting changes (e.g., scene 1 = bright daylight, scene 3 = candlelit). This is purely a pre-processing utility used inside CLIPValidator and TemporalIdentityValidator.

### What to implement

```python
"""
lighting_normalizer.py
Normalizes lighting embeddings before CLIP identity comparison.
Prevents false identity-drift alerts due to lighting changes.
"""
```

**Class:** `LightingNormalizer`

**Constructor params:**
- `tracer: Tracer`

**Key methods:**

```python
def normalize(self, frame: torch.Tensor) -> torch.Tensor:
    """
    Applies luminance normalization to frame tensor.
    Steps:
      1. Convert RGB → LAB colour space
      2. Normalize L-channel to mean=0.5, std=0.2
      3. Convert back to RGB
      4. Return normalized tensor (same dtype/device as input)
    """

def normalize_pair(
    self,
    ref: torch.Tensor,
    frame: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes both reference and frame to same lighting baseline.
    Returns (normalized_ref, normalized_frame).
    Use when comparing frames from different scenes.
    """

def estimate_luminance(self, frame: torch.Tensor) -> float:
    """
    Returns mean luminance of frame as float in [0.0, 1.0].
    Used by TemporalIdentityValidator to flag scenes with 
    extreme lighting variation.
    """
```

**Implementation notes:**
- Use `kornia` (already in requirements) for LAB conversion: `kornia.color.rgb_to_lab()` and `kornia.color.lab_to_rgb()`
- Keep operations on CPU (no VRAM needed for normalization)
- Log luminance estimates per frame with tracer for debugging drift issues

---

## File 5: `vga/identity/temporal_identity_validator.py`

### Purpose
Validates cross-frame identity similarity **within** a single video segment (across consecutive frames). While `IdentityTracker` compares each frame to `char_identity_ref`, `TemporalIdentityValidator` checks frame-to-frame continuity. Threshold: cosine similarity ≥ 0.97 between consecutive frames' CLIP embeddings.

### What to implement

```python
"""
temporal_identity_validator.py
Cross-frame identity similarity check within a video segment.
Enforces: RULE-89 (identity validated in video stage)
Threshold: CLIP cosine similarity ≥ 0.97 between consecutive frames
"""
```

**Class:** `TemporalIdentityValidator`

**Constructor params:**
- `clip_encoder` — shared CLIP encoder
- `lighting_normalizer: LightingNormalizer`
- `tracer: Tracer`
- `threshold: float = 0.97`

**Key methods:**

```python
def validate_segment(
    self,
    frames: List[torch.Tensor],  # list of video segment frames
    char_identity_ref: torch.Tensor,
    stage_id: str,
    scene_id: str
) -> TemporalIdentityReport:
    """
    Validates cross-frame identity similarity for all consecutive frame pairs.
    
    For each consecutive pair (frame_i, frame_i+1):
      1. Normalize lighting on both via lighting_normalizer.normalize_pair()
      2. Encode both with clip_encoder
      3. Compute cosine similarity
      4. If sim < threshold → append to failures list
    
    Also validates each frame vs char_identity_ref (CLIP ≥ 0.93).
    
    Returns TemporalIdentityReport with:
      - overall_passed: bool
      - frame_similarities: List[float]
      - ref_similarities: List[float]   # each frame vs char_identity_ref
      - failures: List[dict]            # {frame_pair, similarity, delta}
      - stage_id: str
      - scene_id: str
    """

def validate_keyframe(
    self,
    keyframe: torch.Tensor,
    char_identity_ref: torch.Tensor,
    stage_id: str
) -> float:
    """
    Validates single keyframe vs char_identity_ref.
    Returns CLIP cosine similarity score.
    Does NOT raise — caller decides action.
    """
```

**Data class to define in this file:**
```python
@dataclass
class TemporalIdentityReport:
    overall_passed: bool
    frame_similarities: List[float]
    ref_similarities: List[float]
    failures: List[dict]
    stage_id: str
    scene_id: str
    timestamp: str
```

---

## File 6: `vga/identity/identity_reinforcement_engine.py`

### Purpose
Drives the **S-06 IdentityReinforcementLoop** — the three sub-stages (6A initial reinforcement, 6B composition binding, 6C final stabilisation) that iteratively refine the base image from S-05 until CLIP ≥ 0.93 is achieved. This engine wraps `ImageEditAgent` with retry logic and LoRA scheduling. It is the only component allowed to modify LoRA weights during image editing.

### What to implement

```python
"""
identity_reinforcement_engine.py
Drives Stage S-06: IdentityReinforcementLoop (6A, 6B, 6C).
Enforces: RULE-86 (noise-aware LoRA scheduling), CGRL-94, CGRL-96
LoRA weights for S-06: 6A → 0.5–0.6, 6B → 0.6, 6C → 0.5–0.6
LoRA is FORBIDDEN in S-05 (BaseImageAgent).
"""
```

**Class:** `IdentityReinforcementEngine`

**Constructor params:**
- `image_edit_agent` — `ImageEditAgent` instance (pre-loaded)
- `clip_validator: CLIPValidator`
- `identity_tracker: IdentityTracker`
- `identity_state_tracker: IdentityStateTracker`
- `drift_controller: IdentityDriftController`
- `lora_manager` — `LoRAManager` instance
- `tracer: Tracer`
- `max_attempts: int = 3`

**Key methods:**

```python
def run_stage_6a(
    self,
    base_image,
    composition_plan: CompositionPlanSchema,
    identity_design: IdentityDesignSchema,
    char_identity_ref: torch.Tensor,
    context: ImmutableContext,
    trace_id: str
) -> tuple[Any, float]:
    """
    Sub-stage 6A: Initial identity reinforcement.
    LoRA weight: 0.5 (noise_level < 0.3) → 0.6 (noise_level ≥ 0.3).
    Retries up to max_attempts with drift_controller.get_retry_strategy().
    
    On each attempt:
      1. Build edit prompt from identity_design + composition_plan
      2. Run image_edit_agent.edit(base_image, prompt, lora_weight=scheduled_weight)
      3. clip_score = clip_validator.score(edited_image, char_identity_ref)
      4. identity_tracker.compute_delta(char_identity_ref, edited_image, "S-06A")
      5. identity_state_tracker.update(char_identity_ref, edited_image, "S-06A")  ← CGRL-96
      6. If clip_score >= 0.93: return (edited_image, clip_score)
      7. Else: get_retry_strategy(attempt, "S-06A", clip_score)
    
    Raises IdentityDriftError if all attempts fail.
    Returns (best_image, best_clip_score).
    """

def run_stage_6b(
    self,
    stage_6a_image,
    composition_plan: CompositionPlanSchema,
    char_identity_ref: torch.Tensor,
    context: ImmutableContext,
    trace_id: str
) -> tuple[Any, float]:
    """
    Sub-stage 6B: Composition binding.
    LoRA weight: fixed 0.6 (CompositionPlan binds identity to environment).
    Same retry logic as 6A.
    """

def run_stage_6c(
    self,
    stage_6b_image,
    identity_design: IdentityDesignSchema,
    char_identity_ref: torch.Tensor,
    context: ImmutableContext,
    trace_id: str
) -> tuple[Any, float]:
    """
    Sub-stage 6C: Final stabilisation.
    LoRA weight: 0.5–0.6 (noise-aware scheduling).
    Produces identity-stabilised master image.
    """

def _schedule_lora_weight(self, noise_level: float) -> float:
    """
    Noise-aware LoRA scheduling (RULE-86).
    noise_level < 0.3 → weight = 0.5
    noise_level >= 0.3 → weight = 0.6
    """
```

**Critical constraint:** LoRA is **FORBIDDEN** in S-05 (BaseImageAgent / FLUX.2-klein-4B). IdentityReinforcementEngine only operates for S-06A/6B/6C.

---

## File 7: `vga/identity/identity_state_tracker.py` ← **NEW v17.0 — PRIMARY DELIVERABLE**

### Purpose
Track **cumulative** identity drift across ALL pipeline phases (image stages S-05/S-06/S-07, video segments S-09, lip-synced segments S-12). This is a new architectural primitive introduced in v17.0. It raises `IdentityCumulativeDriftError` when drift accumulates beyond `IDENTITY_CUMULATIVE_DRIFT_THRESHOLD = 0.15`.

### Implementation (use §70 template from spec as direct base)

```python
"""
identity_state_tracker.py
Tracks cumulative identity drift across all pipeline phases.
Enforces: RULE-89 (identity validated in image, video, lip sync stages)
          RULE-95 (same identity reference across all phases)
NEW v17.0 — See spec §5.20, §AP-38, §12.1 of File Responsibility Spec
"""
import torch
import uuid
from datetime import datetime

from vga.models.schemas import IdentityStateRecord
from vga.core.exceptions import IdentityCumulativeDriftError
from vga.config.settings import (
    IDENTITY_CUMULATIVE_DRIFT_THRESHOLD,
    SCHEMA_VERSION
)
```

**Class:** `IdentityStateTracker`

**Constructor params:**
- `clip_encoder` — shared CLIP encoder (same instance used by CLIPValidator)
- `tracer: Tracer`
- `storage` — job storage (for `append_identity_state_record()`)

**Full implementation spec (implement exactly per §70 template):**

```python
class IdentityStateTracker:
    """
    Maintains IdentityState across all pipeline phases.
    Called after every CLIPValidator check in any phase.
    Raises IdentityCumulativeDriftError if cumulative drift exceeds threshold.
    
    CRITICAL DESIGN RULES:
    - This class ONLY raises IdentityCumulativeDriftError.
    - It NEVER decides what to do when drift exceeds threshold.
    - The CALLER (temporal_engine.py, lip_sync_agent.py, etc.) decides action.
    - reset() is called by callers after successful phase regeneration.
    """

    def __init__(self, clip_encoder, tracer, storage):
        self.clip_encoder = clip_encoder
        self.tracer = tracer
        self.storage = storage
        self._drift_score: float = 0.0
        self._history: list[float] = []

    def update(
        self,
        char_identity_ref: torch.Tensor,
        new_frame,                          # PIL Image or torch.Tensor
        stage_id: str,
        scene_id: str = ""
    ) -> dict:
        """
        Update IdentityState with new frame's CLIP embedding.
        
        Steps:
          1. Encode new_frame with clip_encoder → e_new
          2. Compute cosine similarity between char_identity_ref and e_new
          3. delta = 1.0 - cosine_similarity (normalized drift; 0=perfect)
          4. Accumulate: self._drift_score += delta
          5. Append delta to self._history
          6. Build IdentityStateRecord and write to storage
          7. Trace 'identity_state_update' event
          8. If drift_score > IDENTITY_CUMULATIVE_DRIFT_THRESHOLD:
               raise IdentityCumulativeDriftError(stage_id, drift_score, threshold)
          9. Return state dict for context.evolve()
        
        Returns:
          {
            "embedding_vector": char_identity_ref,  # NEVER changes
            "drift_score": self._drift_score,
            "history": list(self._history)
          }
        
        Raises:
          IdentityCumulativeDriftError — when cumulative drift > threshold
        """

    def reset(self) -> None:
        """
        Reset drift accumulator after successful phase regeneration.
        Called by: temporal_engine.py, lip_sync_agent.py (after successful retry).
        """
        self._drift_score = 0.0
        self._history = []

    @property
    def current_drift_score(self) -> float:
        """Read-only access to current cumulative drift."""
        return self._drift_score

    @property
    def drift_history(self) -> list[float]:
        """Read-only copy of per-stage drift history."""
        return list(self._history)
```

**`IdentityStateRecord` fields** (from schemas.py — reference only, do not redefine):
```
record_id: str          # uuid4
stage_id: str           # "S-05" | "S-06A" | "S-06B" | "S-06C" | "S-07" | "S-09_seg_N" | "S-12"
scene_id: str
delta: float            # per-stage drift
drift_score: float      # cumulative drift after this update
drift_history: List[float]
threshold_exceeded: bool
timestamp: str          # utcnow().isoformat()
schema_version: str     # SCHEMA_VERSION constant
```

**Exception to raise** (must be defined in `vga/core/exceptions.py`):
```python
class IdentityCumulativeDriftError(VGABaseError):
    """Raised when cumulative identity drift exceeds IDENTITY_CUMULATIVE_DRIFT_THRESHOLD."""
    def __init__(self, stage_id: str, cumulative_drift: float, threshold: float):
        self.stage_id = stage_id
        self.cumulative_drift = cumulative_drift
        self.threshold = threshold
        super().__init__(
            f"Identity cumulative drift {cumulative_drift:.4f} exceeded "
            f"threshold {threshold} at stage {stage_id}"
        )
```

---

## `__init__.py` for `vga/identity/`

Create `vga/identity/__init__.py` with clean public exports:

```python
"""
vga/identity — Identity system package.
Public API: IdentityManager, IdentityTracker, IdentityDriftController,
            LightingNormalizer, TemporalIdentityValidator,
            IdentityReinforcementEngine, IdentityStateTracker
"""
from vga.identity.identity_manager import IdentityManager
from vga.identity.identity_tracker import IdentityTracker
from vga.identity.identity_drift_controller import IdentityDriftController
from vga.identity.lighting_normalizer import LightingNormalizer
from vga.identity.temporal_identity_validator import TemporalIdentityValidator
from vga.identity.identity_reinforcement_engine import IdentityReinforcementEngine
from vga.identity.identity_state_tracker import IdentityStateTracker

__all__ = [
    "IdentityManager",
    "IdentityTracker",
    "IdentityDriftController",
    "LightingNormalizer",
    "TemporalIdentityValidator",
    "IdentityReinforcementEngine",
    "IdentityStateTracker",
]
```

---

## Cross-Cutting Wiring Instructions

After implementing all 7 files, the agent must verify these integration points are correct:

### 1. CLIPValidator ↔ IdentityStateTracker pairing (CGRL-96)

In every agent that calls `clip_validator.score()` for identity checking, the call pattern MUST be:
```python
# CORRECT — CGRL-96 mandatory pairing
clip_score = clip_validator.score(frame, char_identity_ref)
identity_state_tracker.update(char_identity_ref, frame, stage_id)  # ← always immediately after
```

Never:
```python
# FORBIDDEN — CGRL-96 violation
clip_score = clip_validator.score(frame, char_identity_ref)
# ... no identity_state_tracker.update() call
```

### 2. char_identity_ref hash guard (CGRL-95)

In `CLIPValidator.__init__()` (file: `vga/validation/clip_validator.py`), add after char_identity_ref is first set:
```python
self._frozen_ref_hash = hash(char_identity_ref.numpy().tobytes())

# In every subsequent call:
current_hash = hash(char_identity_ref.numpy().tobytes())
if current_hash != self._frozen_ref_hash:
    raise IdentityReferenceCorruptionError("char_identity_ref mutated — CRITICAL")
```

### 3. Phase boundaries where identity validation is mandatory (CGRL-94)

| Stage | Validation Call |
|-------|----------------|
| S-05 (each generated image) | `clip_validator.score(image, char_identity_ref) ≥ 0.93` |
| S-06A/6B/6C (each edited image) | `clip_validator.score(image, char_identity_ref) ≥ 0.93` |
| S-07 (refined image; FREEZE point) | `clip_validator.score(image, char_identity_ref) ≥ 0.93` |
| S-09 (each segment keyframe) | `clip_validator.score(keyframe, char_identity_ref) ≥ 0.93` |
| S-12 (each lip-synced frame) | `clip_validator.score(frame, char_identity_ref) ≥ 0.93` |

All of these MUST be followed by `identity_state_tracker.update()`.

### 4. ImmutableContext evolution for identity

After S-07 freezes `char_identity_ref`:
```python
char_identity_ref = clip_encoder(best_refined_image)
context = context.evolve({
    "identity_state": IdentityState(
        embedding_vector=char_identity_ref,
        drift_score=0.0,
        history=[]
    )
})
```

After every subsequent `identity_state_tracker.update()`:
```python
state_dict = identity_state_tracker.update(char_identity_ref, frame, stage_id)
context = context.evolve({"identity_state": IdentityState(**state_dict)})
```

---

## Settings Constants Required

Verify these constants exist in `vga/config/settings.py` (Prompt 02 responsibility — add if missing):

```python
# Identity State (v17.0)
IDENTITY_CUMULATIVE_DRIFT_THRESHOLD: float = 0.15
IDENTITY_MAX_PHASE_REGENERATIONS: int = 1
CLIP_IDENTITY_THRESHOLD: float = 0.93
TEMPORAL_IDENTITY_SIMILARITY_THRESHOLD: float = 0.97
IDENTITY_LORA_WEIGHT_LOW: float = 0.5      # S-06A/6C when noise < 0.3
IDENTITY_LORA_WEIGHT_HIGH: float = 0.6     # S-06A/6C when noise >= 0.3; S-06B fixed
IDENTITY_LORA_WEIGHT_COMPOSITION: float = 0.6  # S-06B always
```

---

## Unit Tests Required

File: `tests/unit/test_identity_state_tracker.py`

```python
# Tests to implement:

def test_update_accumulates_drift():
    """drift_score increases with each update call."""

def test_paired_with_clip_validator():
    """Every CLIPValidator call is followed by IdentityStateTracker.update."""

def test_raises_on_threshold_exceeded():
    """IdentityCumulativeDriftError raised when drift_score > 0.15."""

def test_reset_clears_drift():
    """reset() sets drift_score=0.0 and history=[]."""

def test_embedding_vector_never_changes():
    """update() return dict always contains original char_identity_ref."""

def test_identity_state_record_written_to_storage():
    """storage.append_identity_state_record called on every update."""
```

File: `tests/chaos/test_identity_reference_immutability.py`

```python
def test_char_identity_ref_mutation_raises():
    """CLIPValidator raises IdentityReferenceCorruptionError if ref changes."""

def test_downstream_agents_use_frozen_ref():
    """S-09 and S-12 use same hash as set at S-07."""
```

---

## Implementation Sequence

Implement files in this order to satisfy dependencies:

```
1. lighting_normalizer.py          ← no internal deps
2. identity_tracker.py             ← needs clip_encoder only
3. identity_state_tracker.py       ← needs clip_encoder, schemas, exceptions
4. temporal_identity_validator.py  ← needs LightingNormalizer, clip_encoder
5. identity_drift_controller.py    ← no internal deps
6. identity_manager.py             ← needs schemas, exceptions
7. identity_reinforcement_engine.py ← needs all of the above
8. __init__.py                     ← export all
```

---

## Forbidden Patterns (ABSOLUTE)

```python
# FORBIDDEN-1: Recomputing char_identity_ref downstream
char_identity_ref = clip_encoder(segment_n.keyframe)  # ← NEVER after S-07

# FORBIDDEN-2: Skipping IdentityStateTracker after CLIPValidator
clip_score = clip_validator.score(frame, ref)
# no identity_state_tracker.update() ← CGRL-96 violation

# FORBIDDEN-3: IdentityStateTracker deciding remediation
if threshold_exceeded:
    self._trigger_regeneration()  # ← FORBIDDEN; raise only

# FORBIDDEN-4: LoRA in BaseImageAgent (S-05)
base_image_agent.generate(prompt, lora_weight=0.5)  # ← S-05 is FLUX only, no LoRA

# FORBIDDEN-5: identity_manager.py calling CLIPValidator
self.clip_encoder.encode(image)  # ← identity_manager is CPU/LLM phase; no CLIP calls

# FORBIDDEN-6: IdentityDesignSchema missing reference_strategy
{"character_identity": {...}, "environment": "..."}  # ← must have reference_strategy field
```

---

## Verification Commands

```bash
# After implementation:
python -m pytest tests/unit/test_identity_state_tracker.py -v
python -m pytest tests/chaos/test_identity_reference_immutability.py -v

# Architecture lint:
python -m vga.devtools.architecture_linter --check-rule CGRL-94 CGRL-95 CGRL-96

# Verify no recomputation of char_identity_ref:
grep -rn "clip_encoder\|clip_encoder\.encode" vga/agents/ vga/temporal/ \
  | grep -v "clip_validator.py\|image_refinement_agent.py"
# Should return EMPTY — only CLIPValidator and ImageRefinementAgent may call clip_encoder
```

---

*End of Prompt 08 — Identity System*
