# Prompt 04: Model Manager & Asset Loader
**Category:** Core Foundation  
**Files:**
- `vga/core/model_manager.py`
- `vga/core/vram_guard.py`
**Spec:** `RunPod_Model_Download_Specification_v6.5.md` §13 (AssetLoader), §1.4 (Sequential VRAM Contract)

## Requirements

### ModelManager
The ONLY component that loads/unloads AI models into VRAM. Enforces:
- **Sequential contract**: exactly one heavy model in VRAM at a time
- **Smart reuse**: skip unload/reload if exact same model set already loaded
- **VRAM guard**: assert free_ratio ≥ 0.90 before each load
- **Unload sequence**: `del model → gc.collect() → torch.cuda.empty_cache() → sleep(2)`

```python
class ModelManager:
    """Singleton. Only instance controls all model lifecycle."""
    
    def load(self, model_key: str, **kwargs) -> object:
        """Load model into VRAM. Enforce sequential contract."""
        self.vram_guard.assert_headroom(model_key)  # raises if insufficient
        if self._currently_loaded == model_key:
            return self._current_model  # smart reuse
        self.unload_all()
        model = self._load_model(model_key, **kwargs)
        self._currently_loaded = model_key
        return model
    
    def unload_all(self) -> None:
        """Mandatory unload sequence. RULE-XX."""
        import gc, torch
        del self._current_model
        self._current_model = None
        self._currently_loaded = None
        gc.collect()
        torch.cuda.empty_cache()
        import time; time.sleep(2)
        # Assert free ratio ≥ 0.90
        self.vram_guard.assert_free_ratio(settings.VRAM_FREE_RATIO_MIN)
```

## Acceptance Criteria
- [ ] Loading model A then model B calls unload_all() between them
- [ ] Loading model A twice in a row skips the second load (smart reuse)
- [ ] `vram_guard.assert_headroom()` raises `VRAMViolationError` if insufficient VRAM
