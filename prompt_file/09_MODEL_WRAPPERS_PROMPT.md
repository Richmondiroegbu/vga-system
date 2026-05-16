# Prompt 09: AI Model Wrappers
**Category:** Model Wrappers  
**Files:**
- `vga/models/wrappers/qwen_wrapper.py`
- `vga/models/wrappers/flux_wrapper.py`
- `vga/models/wrappers/z_image_wrapper.py`
- `vga/models/wrappers/wan_wrapper.py`
- `vga/models/wrappers/svi_wrapper.py`
- `vga/models/wrappers/cosyvoice_wrapper.py`
- `vga/models/wrappers/latentsync_wrapper.py`
- `vga/models/wrappers/mmaudio_wrapper.py`
- `vga/models/wrappers/musicgen_wrapper.py`
- `vga/models/wrappers/qwen_wrapper.py`
**Spec:** `VGA_Model_Stack_Setup_Guide_v7.2.md`, `RunPod_Model_Download_Spec_v6.5.md` §2

## Critical Rules for All Wrappers

1. Each wrapper loads ONLY its designated model — never multiple
2. All wrappers call `ModelManager.load(model_key)` before inference (not direct loading)
3. All wrappers call `ModelManager.unload_all()` after inference (sequential contract)
4. No wrapper holds a persistent reference to a loaded model across calls
5. Use `enable_model_cpu_offload()` for FLUX and Wan2.2 to save VRAM

## Key Wrapper Implementations

### qwen_wrapper.py
```python
class QwenWrapper:
    """
    Wraps unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit.
    Used at S-01 (ScriptAgent) and S-04 (SceneCompositionAgent).
    Provides generate_structured() with schema binding and 3-attempt retry.
    """
    
    def generate_structured(
        self,
        prompt: str,
        output_schema: type[BaseModel],
        max_retries: int = 3,
    ) -> BaseModel:
        """
        Generate structured output with Pydantic schema validation.
        Retry up to 3 times on validation failure.
        Use JSON mode or schema-guided generation.
        """
        ...
    
    def generate_text(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate plain text response."""
        ...
```

### flux_wrapper.py
```python
class FluxWrapper:
    """
    Wraps black-forest-labs/FLUX.2-klein-4B.
    Used at S-05 (base images — NO LoRA) and S-06 (with Consistance_Edit_LoRA).
    
    CRITICAL: S-05 MUST use NO LoRA (RULE-91). 
    Assert lora_manager.assert_unloaded() before S-05 call.
    """
    
    def generate_base_images(
        self,
        prompt: str,
        composition_plan: CompositionPlanSchema,
        count: int = 6,
        seed: int = 42,
    ) -> list:
        """
        Generate base images. RULE-91: NO LoRA. RULE-88: CompositionPlan required.
        Use CompositionPlan fields in prompt construction.
        """
        # Assert NO LoRA loaded
        assert not self.lora_manager.any_loaded(), "No LoRA allowed at S-05 (RULE-91)"
        # Include composition plan in prompt
        full_prompt = self._build_composition_prompt(prompt, composition_plan)
        ...
    
    def generate_with_consistency_lora(
        self,
        reference_image,
        prompt: str,
        lora_weight: float = 0.6,
    ) -> "PIL.Image":
        """S-06: With Consistance_Edit_LoRA. Weight range [0.4, 0.7]."""
        assert settings.FLUX_IDENTITY_LORA_WEIGHT_MIN <= lora_weight <= settings.FLUX_IDENTITY_LORA_WEIGHT_MAX
        ...
```

### wan_wrapper.py (updated v17.0)
```python
class WanWrapper:
    """
    Wraps nalexand/Wan2.2-I2V-A14B-FP8.
    S-08: Generates Segment_1 from refined image (init_image).
    Accepts CompositionPlan motion params (NEW v17.0).
    """
    
    def generate(
        self,
        init_image,
        prompt: str,
        motion_vector: str,      # from CompositionPlan (RULE-88, NEW v17.0)
        camera_motion: str,      # from CompositionPlan (RULE-88, NEW v17.0)
        num_frames: int = 81,
        fps: int = 15,
    ) -> "VideoSegment":
        """
        Generate Segment_1 from init_image using Wan2.2.
        CompositionPlan motion_vector and camera_motion MUST be passed (RULE-88, FR-981).
        """
        ...
```

### cosyvoice_wrapper.py
```python
class CosyVoiceWrapper:
    """
    Wraps FunAudioLLM/Fun-CosyVoice3-0.5B-2512.
    S-11: Generates segment-aligned speech.
    Timing error ≤ 0.10s (RULE-96).
    """
    
    def synthesize(
        self,
        text: str,
        target_duration_s: float,
        voice_ref_audio: str = None,
    ) -> tuple[str, float]:
        """
        Generate speech audio. Returns (audio_path, actual_duration_s).
        Validates timing_error = abs(actual_duration - target_duration) ≤ 0.10s.
        """
        # Call via: sys.path.append('/workspace/CosyVoice/third_party/Matcha-TTS')
        # from cosyvoice.cli.cosyvoice import AutoModel
        ...
```

### latentsync_wrapper.py
```python
class LatentSyncWrapper:
    """
    Wraps ByteDance/LatentSync-1.6.
    S-12: Lip sync alignment.
    phoneme_alignment ≥ 0.80; identity_delta ≤ 0.03 (RULE-97).
    Called via subprocess using /workspace/LatentSync inference scripts.
    """
    
    def sync(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ) -> dict:
        """
        Lip sync video to audio. Returns metrics dict.
        Validate: phoneme_alignment ≥ 0.80, identity_delta ≤ 0.03.
        """
        # Run: bash /workspace/LatentSync/inference.sh
        # Or direct Python via subprocess
        ...
```

## Acceptance Criteria
- [ ] `FluxWrapper.generate_base_images()` fails if any LoRA is loaded
- [ ] `WanWrapper.generate()` requires motion_vector and camera_motion params
- [ ] `CosyVoiceWrapper.synthesize()` validates timing error ≤ 0.10s
- [ ] `LatentSyncWrapper.sync()` validates phoneme_alignment ≥ 0.80
- [ ] Each wrapper calls `ModelManager.load()` before and `unload_all()` after
