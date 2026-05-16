# Prompt 14: Export & Quality Agents (S-16)
**Category:** Export & Quality  
**Files:**
- `vga/agents/assembly_agent.py`
- `vga/agents/export_agent.py`
- `vga/agents/quality_agent.py`
**Spec:** `01_VGA_SRD_v17.2.md` §1.4 (Phase 5)

## AssemblyAgent
```python
class AssemblyAgent(BaseAgent):
    """
    Merges video segments + mixed audio using ffmpeg.
    Output: /workspace/output/{job_id}/{scene_id}/final.mp4
    """
    def run(self, input_data, context):
        # ffmpeg concat + audio merge
        output_path = f"/workspace/output/{context.job_id}/{context.scene_id}/final.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", input_data.segment_list_file,
            "-i", input_data.mixed_audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-shortest", output_path,
        ]
        subprocess.run(cmd, check=True)
        return output_path, context
```

## QualityAgent — PipelineReport generation
```python
class QualityAgent(BaseAgent):
    """
    Produces PipelineReport with v17.0 fields:
    - All 11 HRG checkpoint decisions
    - SLA compliance per stage
    - Identity drift history
    - Temporal buffer state log
    - Audio quality records
    - Adaptive calibration snapshots
    """
    def run(self, input_data, context):
        report = PipelineReport(
            job_id=context.job_id,
            schema_version="v6.0",
            stages_completed=list(context.completed_stages),
            hrg_decisions=self._collect_hrg_decisions(context),
            identity_drift_history=list(context.identity_state.history),
            audio_quality_records=input_data.audio_quality_records,
            # ... all v17.0 fields
        )
        report_path = f"/workspace/output/{context.job_id}/pipeline_report.json"
        report.save(report_path)
        return report, context
```

## Acceptance Criteria
- [ ] `AssemblyAgent` produces valid MP4 via ffmpeg concat
- [ ] `QualityAgent` writes `pipeline_report.json` with schema_version="v6.0"
- [ ] PipelineReport includes all 11 HRG checkpoint decisions (OR-038)
- [ ] PipelineReport includes identity_drift_history
