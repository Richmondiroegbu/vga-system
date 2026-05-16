# Prompt 06: SLA Manager, Adaptive Memory, Gating Controller
**Category:** Orchestration Layer  
**Files:**
- `vga/runtime/sla_manager.py`
- `vga/runtime/gating_controller.py`
- `vga/adaptive/adaptive_memory.py`
- `vga/adaptive/calibration_engine.py`
**Spec:** `01_VGA_SRD_v17.2.md` §16 (SLA Guarantees), v15.0 architecture pillars

## Requirements

### SLAManager (v15.0 — retained)
- Track KPI thresholds per stage from settings.py
- `check_kpi(stage_id, elapsed_s)` — log WARNING if within 80% of SLA, log ERROR if exceeded
- `record_stage_time(stage_id, elapsed_s)` — persist for adaptive calibration
- Report SLA violations in PipelineReport

### GatingController (v15.0 — retained)
- Mode: STRICT / BALANCED / FAST (from `GatingMode` enum)
- Default: STRICT (all validations enforced)
- Never allow bypassing CLIP or CompositionPlan gates regardless of mode

### AdaptiveMemory + CalibrationEngine (v15.0 — retained)
- Track historical stage performance
- Exponential smoothing: α=0.90 for KPI updates
- Expose `calibrate_sla(stage_id)` based on historical data

## Acceptance Criteria
- [ ] `SLAManager.check_kpi("S-04", 20.0)` logs error (SLA_COMPOSITION_MAX_S=15.0)
- [ ] GatingController in FAST mode still enforces CLIP ≥ 0.93 and CompositionPlan gate
