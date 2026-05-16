# Prompt 18: VGA Client Watcher (AVON — Autonomous Validation & Optimization Node)
**Category:** Client  
**File:** `vga_client_watcher.py` (local machine, separate from main VGA project)  
**Spec:** `14_VGA_Client_Watcher_AutoDownload_SafeCleanup_v4.2.md`

## Identity (v4.2)
The client is an **Autonomous Validation & Optimization Node (AVON)** — not just a downloader. It:
- Validates schemas, not just file existence
- Scores quality with weighted multi-dimensional formula
- Reasons probabilistically about borderline results
- Adapts generation parameters based on historical outcomes
- Emits actionable feedback reports to server
- Cleans up server ONLY after ALL validations pass

## Core Components to Implement

### WatcherSystem
```python
class VGAClientWatcher:
    """
    Main watcher loop. Polls GET /jobs/{job_id} every POLL_INTERVAL seconds.
    Detects: status == "completed" → trigger full validation pipeline.
    """
    def run(self, job_id: str):
        while True:
            status = self.poll_job_status(job_id)
            if status == "completed":
                self.run_validation_pipeline(job_id)
                break
            time.sleep(POLL_INTERVAL)
```

### ValidationPipeline
```python
def run_validation_pipeline(self, job_id: str):
    """
    Sequential validation (cleanup ONLY if ALL pass):
    1. Version check (system_version + schema_version)
    2. Collect artifacts (video + pipeline_report + identity + audio + composition)
    3. Schema validation (strict Pydantic validation of all JSON artifacts)
    4. File verification (size, checksum, duration, bitrate, codec, resolution, frames)
    5. System validation (identity drift, SNR, temporal continuity, composition)
    6. Pipeline audit (stage coverage, retry counts, SLA compliance)
    7. Quality scoring (weighted formula)
    8. Probabilistic confidence scoring
    9. Cross-validation (hidden instability detection)
    10. Decision evaluation
    11. Feedback report (POST /client_report)
    12. Adaptation (update parameters from outcome)
    13. Memory store (persist run history)
    14. Metrics emission
    15. Cleanup (ONLY if quality_score ≥ 0.75 AND confidence ≥ 0.70)
    """
```

### Quality Scoring (weighted)
```python
def compute_quality_score(self, validation_results: dict) -> tuple[float, float]:
    """
    Returns: (quality_score, confidence)
    quality_score = weighted combination of:
      - identity_score (weight: 0.35)  
      - audio_score (weight: 0.25)
      - temporal_score (weight: 0.25)
      - composition_score (weight: 0.15)
    """
```

## Acceptance Criteria
- [ ] Client never deletes server files unless quality_score ≥ 0.75 AND confidence ≥ 0.70
- [ ] Schema validation uses Pydantic (not just JSON parsing)
- [ ] Quality scoring computes weighted score across all 4 dimensions
- [ ] Feedback report POSTed to /client_report after every run
- [ ] Historical run memory enables parameter adaptation
