# Prompt 11: Image Pipeline Agents (S-05 through S-07)
**Category:** Pipeline Agents — Phase 2  
**Files:**
- `vga/agents/base_image_agent.py` (S-05, updated v17.0)
- `vga/agents/image_edit_agent.py` (S-06, sub-stages 6A/6B/6C)
- `vga/agents/multi_angle_agent.py` (S-06A)
- `vga/agents/image_merge_agent.py` (S-06B)
- `vga/agents/scene_expansion_agent.py` (S-06C)
- `vga/agents/image_refinement_agent.py` (S-07)
**Spec:** `01_VGA_SRD_v17.2.md` §1.4 (S-05 through S-07), §5.26 (Identity freeze at S-05)

## BaseImageAgent (S-05) — NO LoRA, CompositionPlan required
```python
class BaseImageAgent(BaseAgent):
    """
    S-05: Generate 6 base images using FLUX.2-klein-4B with NO LoRA.
    RULE-88: CompositionPlan MUST be input (assert before generation).
    RULE-91: NO LoRA during base generation (pure FLUX.2-klein).
    CLIP ≥ 0.93 per image.
    Identity freeze: char_identity_ref computed and frozen here (RULE-95).
    HRG-5 checkpoint.
    """
    
    def run(self, input_data, context: ImmutableContext):
        # Assert CompositionPlan (RULE-88)
        context.assert_composition_plan()
        
        # Assert NO LoRA (RULE-91)
        assert not self.lora_manager.any_loaded(), "NO LoRA at S-05 (RULE-91)"
        
        # Generate 6 base images with diversity (angle/lighting/pose)
        images = self.flux_wrapper.generate_base_images(
            prompt=input_data.character_description,
            composition_plan=context.composition_plan,
            count=settings.BASE_IMAGE_COUNT,  # 6
        )
        
        # CLIP validation for each image
        for img in images:
            score = self.clip_validator.validate_score(img, min_threshold=settings.CLIP_IDENTITY_THRESHOLD)
        
        # Select best image and compute char_identity_ref (FREEZE HERE)
        best_image = self._select_best_clip_image(images)
        identity_embedding = self.clip_validator.compute_embedding(best_image)
        
        # FREEZE identity reference (RULE-95 — never recomputed downstream)
        frozen_identity = context.identity_state.freeze(identity_embedding)
        
        # Evolve context with frozen identity
        new_context = context.evolve(
            identity_state=frozen_identity,
        )
        
        # HRG-5
        self.hrg_controller.checkpoint(HRGCheckpoint.HRG_5_BASE_IMAGES, new_context, images)
        
        return images, new_context
```

## IdentityReinforcementLoop (S-06) — Sub-stages 6A, 6B, 6C
```python
class ImageEditAgent(BaseAgent):
    """
    S-06: Three-pass identity reinforcement.
    All sub-stages use FLUX.2-klein + Consistance_Edit_LoRA.
    CLIP ≥ 0.93 enforced AFTER EACH sub-stage.
    CompositionPlan fully bound in 6C.
    """
    
    def run(self, input_data, context: ImmutableContext):
        context.assert_composition_plan()
        context.assert_identity_frozen()
        
        # 6A: Multi-Angle Expansion (min 5–8 angle variants)
        angle_variants = self.multi_angle_agent.run(input_data.best_base_image, context)
        self._validate_clip_all(angle_variants, context)
        
        # 6B: Merge/Edit (identity-stabilized master image)
        master_image = self.image_merge_agent.run(angle_variants, context)
        self._validate_clip(master_image, context)
        
        # 6C: Scene Expansion (bind identity to CompositionPlan environment)
        scene_image = self.scene_expansion_agent.run(master_image, context)
        self._validate_clip(scene_image, context)
        
        # Update identity drift (cumulative tracking)
        drift = self._compute_drift(scene_image, context)
        new_identity = context.identity_state.update_drift(drift)
        new_context = context.evolve(identity_state=new_identity)
        
        self.hrg_controller.checkpoint(HRGCheckpoint.HRG_6_IDENTITY_REINFORCEMENT, new_context, scene_image)
        return scene_image, new_context
```

## ImageRefinementAgent (S-07) — Z-Image-Turbo
```python
class ImageRefinementAgent(BaseAgent):
    """
    S-07: Refine best image using Z-Image-Turbo.
    denoise ∈ [0.05, 0.15]; cfg = 5.0.
    drift ≤ 0.02 (RULE-93); CLIP ≥ 0.93 (RULE-92).
    char_identity_ref is FROZEN here — assert no recomputation (RULE-95).
    """
    
    def run(self, input_data, context: ImmutableContext):
        context.assert_identity_frozen()  # Must be frozen from S-05
        
        refined = self.zimage_wrapper.refine(
            image=input_data.scene_image,
            denoise=settings.ZIMAGE_DENOISE_MIN,  # start conservative
            cfg=settings.ZIMAGE_CFG,
        )
        
        # Validate drift (RULE-93)
        drift = self._compute_clip_drift(refined, context)
        if drift > settings.CLIP_DRIFT_THRESHOLD:
            raise CLIPValidationError(f"Refinement drift {drift:.4f} > {settings.CLIP_DRIFT_THRESHOLD}")
        
        # CLIP validation
        clip_score = self.clip_validator.validate_or_raise(
            refined, context.identity_state.embedding_vector, "S-07"
        )
        
        self.hrg_controller.checkpoint(HRGCheckpoint.HRG_7_REFINED_IMAGE, context, refined)
        return refined, context
```

## Acceptance Criteria
- [ ] `BaseImageAgent` with any LoRA loaded raises assertion error
- [ ] `BaseImageAgent` freezes identity embedding in `context.identity_state`
- [ ] `ImageEditAgent` runs all 3 sub-stages (6A, 6B, 6C) with CLIP validation after each
- [ ] `ImageRefinementAgent` with drift > 0.02 raises `CLIPValidationError`
- [ ] No agent recomputes `char_identity_ref` — all use `context.identity_state.embedding_vector`
