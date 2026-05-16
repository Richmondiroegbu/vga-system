# Prompt 19: Test Suite
**Category:** Tests  
**Files:**
- `tests/unit/test_temporal_buffer_manager.py`
- `tests/unit/test_svi_scheduler.py`
- `tests/unit/test_scene_composition_agent.py`
- `tests/unit/test_immutable_context.py`
- `tests/unit/test_audio_quality_validator.py`
- `tests/unit/test_clip_validator.py`
- `tests/integration/test_temporal_engine.py`
- `tests/integration/test_composition_to_image.py`
- `tests/conftest.py`

## Key Tests

### test_temporal_buffer_manager.py
```python
def test_init_rejects_short_segment():
    """TemporalBufferError raised if segment has < 5 frames."""
    seg = MockVideoSegment(frame_count=3)
    with pytest.raises(TemporalBufferError):
        TemporalBufferManager.init(seg, "scene_1")

def test_buffer_always_5_frames(valid_segment):
    buf = TemporalBufferManager.init(valid_segment, "scene_1")
    assert buf.frames.shape[0] == 5

def test_update_maintains_5_frames(valid_segment, another_segment):
    buf = TemporalBufferManager.init(valid_segment, "scene_1")
    buf2 = TemporalBufferManager.update(buf, another_segment)
    assert buf2.frames.shape[0] == 5

def test_encode_returns_cpu_tensor(valid_buffer):
    latents = TemporalBufferManager.encode(valid_buffer)
    assert not latents.is_cuda
    assert latents.shape[0] == 5
```

### test_svi_scheduler.py
```python
def test_cfg_out_of_range_raises():
    with pytest.raises(SVICFGViolationError):
        SVIScheduler(total_steps=50, cfg=6.5)

def test_high_noise_phase_weight():
    sched = SVIScheduler(total_steps=50, cfg=5.5)
    # Step 0 = start of diffusion = high noise
    weight = sched.get_lora_weight(timestep_index=0)
    assert weight == 0.6

def test_low_noise_phase_weight():
    sched = SVIScheduler(total_steps=50, cfg=5.5)
    weight = sched.get_lora_weight(timestep_index=48)
    assert weight == 0.4

def test_static_weight_forbidden():
    # Applying same weight at all steps should be detected as violation
    # (implementation-specific check)
    ...
```

### test_immutable_context.py
```python
def test_context_is_frozen():
    ctx = ImmutableContext()
    with pytest.raises(FrozenInstanceError):
        ctx.job_id = "new_id"

def test_evolve_creates_new_instance():
    ctx = ImmutableContext(job_id="a")
    new_ctx = ctx.evolve(job_id="b")
    assert ctx.job_id == "a"
    assert new_ctx.job_id == "b"

def test_dict_context_rejected():
    from vga.state.context_factory import ContextFactory
    with pytest.raises(ImmutableContextViolationError):
        ContextFactory.validate({"job_id": "test"})

def test_identity_freeze_prevents_double_freeze():
    import numpy as np
    state = IdentityState()
    frozen = state.freeze(np.zeros(512))
    with pytest.raises(IdentityReferenceCorruptionError):
        frozen.freeze(np.ones(512))

def test_assert_composition_plan_raises_when_missing():
    ctx = ImmutableContext(composition_plan=None)
    with pytest.raises(CompositionPlanValidationError):
        ctx.assert_composition_plan()
```

### conftest.py fixtures
```python
@pytest.fixture
def valid_buffer():
    """5-frame TemporalBuffer for testing."""
    frames = torch.rand(5, 3, 480, 832)
    return TemporalBuffer(
        frames=frames,
        timestamps=(0.0, 0.066, 0.133, 0.2, 0.266),
        scene_id="scene_test",
        segment_index=1,
    )

@pytest.fixture  
def valid_context():
    """Well-formed ImmutableContext for testing."""
    from vga.state.context_factory import ContextFactory
    return ContextFactory.create_initial("job_test", "scene_test")
```

## Acceptance Criteria
- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] TemporalBuffer tests cover: init, update, encode, assertion
- [ ] SVIScheduler tests cover: CFG validation, phase-aware weights
- [ ] ImmutableContext tests cover: frozen, evolve, dict rejection, identity freeze
- [ ] AudioQualityValidator tests cover: SNR pass/fail, clipping detection
