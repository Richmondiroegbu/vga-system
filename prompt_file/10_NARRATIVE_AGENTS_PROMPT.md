# Prompt 10: Narrative Intelligence Agents (S-01 through S-04)
**Category:** Pipeline Agents — Phase 1  
**Files:**
- `vga/agents/base_agent.py`
- `vga/agents/script_agent.py`
- `vga/agents/scene_agent.py`
- `vga/agents/segment_agent.py`
- `vga/agents/identity_design_agent.py`
- `vga/agents/scene_composition_agent.py` [NEW v17.0 — S-04]
- `vga/config/prompts/script_prompts.py`
- `vga/config/prompts/identity_prompts.py`
- `vga/config/prompts/composition_prompts.py` [NEW v17.0]
**Spec:** `01_VGA_SRD_v17.2.md` §1.4, §3.66

## BaseAgent
```python
class BaseAgent(ABC):
    """
    Abstract base for all VGA pipeline agents.
    
    CRITICAL: Never call agent.run() directly — always use execute_stage().
    Subclasses implement run() which is invoked ONLY by execute_stage().
    """
    stage_id: str  # must be set by subclass
    
    @abstractmethod
    def run(
        self,
        input_data: Any,
        context: ImmutableContext,
    ) -> tuple[Any, ImmutableContext]:
        """
        Execute stage logic.
        MUST call context.evolve() before returning.
        MUST NOT mutate input_data or context in place.
        """
        ...
    
    def _validate_input(self, input_data) -> None:
        """Validate input against expected schema."""
        ...
    
    def _validate_output(self, output) -> None:
        """Validate output against expected schema."""
        ...
```

## SceneCompositionAgent (S-04) [NEW v17.0 — CRITICAL]
```python
class SceneCompositionAgent(BaseAgent):
    """
    Stage S-04: Translate scene narrative into CompositionPlan.
    RULE-88: CompositionPlan is MANDATORY before any image generation.
    
    Input: dialogue, emotion, motion_intent, characters, environment
    Output: CompositionPlanSchema (ALL 6 fields required)
    
    Flow:
    1. Build structured generation prompt
    2. Call qwen_wrapper.generate_structured(prompt, CompositionPlanSchema)
    3. Retry up to COMPOSITION_MAX_RETRIES on schema validation failure
    4. Call CompositionPlanValidator.validate(plan)
    5. Write composition_plan_{scene_id}.json to /workspace/composition/{job_id}/{scene_id}/
    6. Log composition_plan_created event
    7. Evolve context with composition_plan
    8. HRG-4 checkpoint (CompositionPlan review)
    """
    stage_id = "S-04"
    
    def run(self, input_data, context: ImmutableContext):
        # Build Qwen prompt for structured composition generation
        prompt = self._build_composition_prompt(input_data)
        
        # Retry loop with schema validation
        for attempt in range(settings.COMPOSITION_MAX_RETRIES):
            try:
                plan = self.qwen_wrapper.generate_structured(prompt, CompositionPlanSchema)
                validated = self.composition_validator.validate(plan.model_dump())
                break
            except (ValidationError, CompositionPlanValidationError):
                if attempt == settings.COMPOSITION_MAX_RETRIES - 1:
                    raise CompositionPlanValidationError(
                        f"SceneCompositionAgent failed after {settings.COMPOSITION_MAX_RETRIES} retries"
                    )
        
        # Write to disk (OR-033)
        self._write_plan(validated, context)
        
        # Log trace event
        self.tracer.trace_event("composition_plan_created", scene_id=context.scene_id)
        
        # Evolve context with CompositionPlan + camera/lighting state
        new_context = context.evolve(
            composition_plan=validated,
            camera_state=CameraState(
                angle=validated.camera_angle,
                motion=validated.camera_motion,
            ),
            lighting_state=LightingState(style=validated.lighting_style),
        )
        
        return validated, new_context
    
    def _build_composition_prompt(self, input_data) -> str:
        """Build Qwen structured generation prompt from scene data."""
        from vga.config.prompts.composition_prompts import COMPOSITION_SYSTEM_PROMPT, build_composition_user_prompt
        return build_composition_user_prompt(input_data)
```

## ScriptAgent (S-01)
- Calls `qwen_wrapper.generate_structured(SCRIPT_SYSTEM_PROMPT + user_prompt, ScriptSchema)`
- Validates against `ScriptSchema` (v6.0)
- HRG-1 checkpoint after successful generation

## ScenePlanner + SegmentPlanner (S-02)
- Parses ScriptSchema → ScenePlanSchema
- Enforces: scene 10–30s, segment 3–5s
- Initializes ImmutableContext via `ContextFactory.create_initial()`
- HRG-2 checkpoint (NEW v17.0)

## IdentityDesignAgent (S-03)
- Generates character_identity, environment_description, reference_strategy
- `reference_strategy` is MANDATORY (not optional)
- HRG-3 checkpoint

## composition_prompts.py [NEW v17.0]
```python
COMPOSITION_SYSTEM_PROMPT = """
You are a professional cinematographer and film director.
Given scene narrative data, produce a precise CompositionPlan with EXACTLY these 6 fields:
- camera_angle: choose from [extreme close-up, close-up, medium close-up, medium shot, ...]
- camera_motion: describe camera movement (e.g., "slow dolly forward", "static", "pan left")
- character_positions: list of {character_id, position, facing} for each character
- focus_subject: which character or element is the focal point
- lighting_style: (e.g., "low-key dramatic", "soft natural", "harsh overhead")
- motion_vector: overall motion direction (e.g., "forward_slow", "stationary")

Mission: Inspire audiences with visual storytelling of human triumph over adversity.
Respond ONLY with valid JSON matching CompositionPlanSchema. No preamble.
"""
```

## Acceptance Criteria
- [ ] `SceneCompositionAgent.run()` raises `CompositionPlanValidationError` after COMPOSITION_MAX_RETRIES failures
- [ ] `SceneCompositionAgent` writes `composition_plan_{scene_id}.json` to disk
- [ ] Context from S-04 has `composition_plan` field populated (not None)
- [ ] Context from S-04 has `camera_state` and `lighting_state` updated
- [ ] HRG-2 checkpoint fires after S-02 completion
- [ ] HRG-4 checkpoint fires after S-04 completion
