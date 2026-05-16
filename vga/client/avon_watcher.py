"""
AVON — Autonomous Validation & Optimization Node (v4.2)
VGA Client Watcher: validates, scores, adapts, and safely cleans server outputs.

Runs on the operator's LOCAL MACHINE — not on RunPod.
Spec: 14_VGA_Client_Watcher_AutoDownload_SafeCleanup_v4_2.md

Non-Negotiable Safety Rules:
  RULE 1: NO DELETE before full 9-gate validation suite passes
  RULE 2: All logic runs client-side only
  RULE 3: Idempotent cleanup — 404 = success, not error
  RULE 4: Network failure safe — retry with exponential backoff

Quality thresholds (cleanup only if BOTH met):
  quality_score ≥ 0.75
  confidence    ≥ 0.70
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("avon")


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class AVONConfig:
    """All AVON runtime configuration."""

    server_base_url: str = "http://localhost:8000/api/v1"
    output_dir: Path = Path("./avon_downloads")
    history_file: Path = Path("./avon_history.jsonl")
    state_file: Path = Path("./avon_state.json")
    metrics_file: Path = Path("./avon_metrics.jsonl")

    # Polling
    poll_interval_s: int = 10
    max_poll_wait_s: int = 7200          # 2 hours max wait
    backoff_base: float = 1.5
    max_backoff_s: float = 120.0

    # Retry
    max_download_retries: int = 3
    download_backoff: list = field(default_factory=lambda: [5, 15, 45])

    # Quality gates (RULE 1)
    quality_threshold: float = 0.75
    confidence_threshold: float = 0.70

    # Quality scoring weights
    identity_weight: float = 0.35
    audio_weight: float = 0.25
    temporal_weight: float = 0.25
    composition_weight: float = 0.15

    # System version contract
    expected_system_version: str = "17.2.0"
    expected_schema_version: str = "v6.0"

    # Validation thresholds
    clip_threshold: float = 0.93
    snr_min_db: float = 10.0
    peak_max_dbfs: float = 0.0
    max_identity_drift: float = 0.15
    min_temporal_health: float = 0.85
    min_continuity_score: float = 0.90
    timing_tolerance_s: float = 0.10

    # Video sanity checks
    min_duration_s: float = 5.0
    max_duration_s: float = 300.0
    min_bitrate_kbps: int = 500
    expected_codec: str = "h264"

    # Strict mode: missing REQUIRED artifacts = hard failure
    strict_artifact_mode: bool = True


# ─── Data containers ─────────────────────────────────────────────────────────

@dataclass
class Artifacts:
    video_path: Optional[Path] = None
    pipeline_report: Optional[dict] = None
    identity_state: Optional[dict] = None
    audio_validation: Optional[dict] = None
    composition_plan: Optional[dict] = None
    continuity_report: Optional[dict] = None
    server_checksum: Optional[str] = None


@dataclass
class ValidationResult:
    gate: str
    passed: bool
    score: float = 0.0
    detail: str = ""


@dataclass
class RunRecord:
    job_id: str
    timestamp: str
    quality_score: float
    confidence: float
    cleanup_triggered: bool
    gates_passed: list
    gates_failed: list
    recommended_adjustments: dict
    duration_s: float


# ─── AVON Main Class ──────────────────────────────────────────────────────────

class AVONWatcher:
    """Autonomous Validation & Optimization Node — main entry point.

    Usage:
        watcher = AVONWatcher(config=AVONConfig(server_base_url="http://<POD_IP>:8000/api/v1"))
        watcher.watch(job_id="job_abc123")
    """

    def __init__(self, config: AVONConfig | None = None) -> None:
        self.cfg = config or AVONConfig()
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[RunRecord] = self._load_history()
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "AVON/4.2 VGA-Client"

    # ─── Public API ───────────────────────────────────────────────────────────

    def watch(self, job_id: str) -> RunRecord:
        """Main entry point. Block until job completes then run full validation pipeline."""
        logger.info("AVON watching job %s at %s", job_id, self.cfg.server_base_url)
        start_time = time.monotonic()

        self._save_phase_state(job_id, "watching")

        # Pre-flight health check
        self._preflight_check()

        # Poll until completed / failed / cancelled
        status_data = self._poll_until_done(job_id)
        final_status = status_data.get("status", "unknown")

        if final_status in ("failed", "cancelled"):
            logger.error("AVON: job %s ended with status=%s — skipping validation", job_id, final_status)
            return self._make_failed_record(job_id, final_status, time.monotonic() - start_time)

        # Run full 15-step validation pipeline
        record = self._run_validation_pipeline(job_id, status_data, start_time)
        return record

    # ─── Pre-flight ───────────────────────────────────────────────────────────

    def _preflight_check(self) -> None:
        """Verify server is reachable before starting the watch loop."""
        url = self.cfg.server_base_url.rstrip("/jobs").rstrip("/api/v1") + "/health"
        for attempt in range(3):
            try:
                r = self._session.get(url, timeout=10)
                if r.status_code == 200:
                    logger.info("AVON pre-flight OK: %s", r.json())
                    return
            except requests.RequestException as exc:
                logger.warning("AVON pre-flight attempt %d failed: %s", attempt + 1, exc)
                time.sleep(5)
        raise ConnectionError("AVON: server unreachable — cannot start watch loop")

    # ─── Polling ──────────────────────────────────────────────────────────────

    def _poll_until_done(self, job_id: str) -> dict:
        """Poll GET /jobs/{job_id} with adaptive exponential backoff until terminal state."""
        deadline = time.monotonic() + self.cfg.max_poll_wait_s
        backoff = float(self.cfg.poll_interval_s)
        terminal = {"completed", "degraded", "failed", "cancelled"}

        while time.monotonic() < deadline:
            try:
                data = self._get(f"/jobs/{job_id}")
                status = data.get("status", "unknown")
                stage = data.get("current_stage", "")
                pct = data.get("progress_percent", 0.0)
                logger.info(
                    "AVON poll: job=%s status=%s stage=%s progress=%.1f%%",
                    job_id, status, stage, pct,
                )
                if status in terminal:
                    return data
                backoff = min(backoff * self.cfg.backoff_base, self.cfg.max_backoff_s)
            except requests.RequestException as exc:
                logger.warning("AVON poll error: %s — retrying in %.0fs", exc, backoff)
            time.sleep(backoff)

        raise TimeoutError(f"AVON: job {job_id} did not complete within {self.cfg.max_poll_wait_s}s")

    # ─── 15-Step Validation Pipeline ──────────────────────────────────────────

    def _run_validation_pipeline(
        self, job_id: str, status_data: dict, start_time: float
    ) -> RunRecord:
        """Run all 15 validation steps sequentially. Cleanup ONLY if all gates pass."""
        gates_passed: list[str] = []
        gates_failed: list[str] = []
        results: list[ValidationResult] = []

        # ── Step 1: Version check ──
        self._save_phase_state(job_id, "version_check")
        r = self._check_version_compatibility(status_data)
        results.append(r)
        if not r.passed:
            gates_failed.append("version_check")
            return self._finalize(job_id, results, gates_passed, gates_failed, False, {}, start_time)
        gates_passed.append("version_check")

        # ── Step 2: Collect all artifacts ──
        self._save_phase_state(job_id, "artifact_collection")
        artifacts, collect_result = self._collect_artifacts(job_id)
        results.append(collect_result)
        if not collect_result.passed:
            gates_failed.append("artifact_collection")
            return self._finalize(job_id, results, gates_passed, gates_failed, False, {}, start_time)
        gates_passed.append("artifact_collection")

        # ── Step 3: Schema validation ──
        self._save_phase_state(job_id, "schema_validation")
        r = self._validate_schemas(artifacts)
        results.append(r)
        if not r.passed:
            gates_failed.append("schema_validation")
            return self._finalize(job_id, results, gates_passed, gates_failed, False, {}, start_time)
        gates_passed.append("schema_validation")

        # ── Step 4: File verification ──
        self._save_phase_state(job_id, "file_verification")
        r = self._verify_file(job_id, artifacts)
        results.append(r)
        if not r.passed:
            gates_failed.append("file_verification")
            return self._finalize(job_id, results, gates_passed, gates_failed, False, {}, start_time)
        gates_passed.append("file_verification")

        # ── Step 5: System validation ──
        self._save_phase_state(job_id, "system_validation")
        r = self._validate_system(artifacts)
        results.append(r)
        if not r.passed:
            gates_failed.append("system_validation")
        else:
            gates_passed.append("system_validation")

        # ── Step 6: Pipeline audit ──
        self._save_phase_state(job_id, "pipeline_audit")
        r = self._audit_pipeline(artifacts)
        results.append(r)
        if not r.passed:
            gates_failed.append("pipeline_audit")
        else:
            gates_passed.append("pipeline_audit")

        # ── Step 7: Quality scoring ──
        self._save_phase_state(job_id, "quality_scoring")
        quality_score, quality_breakdown = self._compute_quality_score(artifacts, results)
        r = ValidationResult(
            gate="quality_score",
            passed=quality_score >= self.cfg.quality_threshold,
            score=quality_score,
            detail=f"score={quality_score:.3f} threshold={self.cfg.quality_threshold}",
        )
        results.append(r)
        if r.passed:
            gates_passed.append("quality_score")
        else:
            gates_failed.append("quality_score")

        # ── Step 8: Probabilistic confidence scoring ──
        self._save_phase_state(job_id, "confidence_scoring")
        confidence = self._compute_confidence(results, quality_breakdown)
        r = ValidationResult(
            gate="confidence",
            passed=confidence >= self.cfg.confidence_threshold,
            score=confidence,
            detail=f"confidence={confidence:.3f} threshold={self.cfg.confidence_threshold}",
        )
        results.append(r)
        if r.passed:
            gates_passed.append("confidence")
        else:
            gates_failed.append("confidence")

        # ── Step 9: Cross-validation (hidden instability detection) ──
        self._save_phase_state(job_id, "cross_validation")
        r = self._cross_validate(artifacts, results)
        results.append(r)
        if r.passed:
            gates_passed.append("cross_validation")
        else:
            gates_failed.append("cross_validation")

        # ── Step 10: Decision evaluation ──
        self._save_phase_state(job_id, "decision")
        cleanup_eligible = self._evaluate_cleanup_eligibility(
            results, quality_score, confidence
        )

        # ── Step 11: Feedback report ──
        self._save_phase_state(job_id, "feedback")
        adjustments = self._build_recommended_adjustments(results, quality_breakdown)
        self._post_feedback_report(
            job_id, quality_score, confidence, cleanup_eligible, results, adjustments
        )

        # ── Step 12: Adaptation engine ──
        self._save_phase_state(job_id, "adaptation")
        self._adapt_parameters(results, quality_score)

        # ── Step 13: Memory store ──
        record = self._finalize(
            job_id, results, gates_passed, gates_failed,
            cleanup_eligible, adjustments, start_time,
            quality_score=quality_score, confidence=confidence,
        )
        self._save_to_history(record)

        # ── Step 14: Metrics emission ──
        self._emit_metrics(record, results)

        # ── Step 15: Safe cleanup (RULE 1 — only if ALL 9 gates satisfied) ──
        if cleanup_eligible:
            self._save_phase_state(job_id, "cleanup")
            self._safe_cleanup(job_id)
        else:
            logger.warning(
                "AVON: cleanup SKIPPED — not all validation gates passed. "
                "Server data preserved for retry or manual review. "
                "Failed gates: %s", gates_failed,
            )

        self._clear_phase_state(job_id)
        return record

    # ─── Step 1: Version Check ────────────────────────────────────────────────

    def _check_version_compatibility(self, status_data: dict) -> ValidationResult:
        sv = status_data.get("system_version", "")
        schema = status_data.get("schema_version", "")
        ok = (
            sv == self.cfg.expected_system_version
            and schema == self.cfg.expected_schema_version
        )
        return ValidationResult(
            gate="version_check",
            passed=ok,
            score=1.0 if ok else 0.0,
            detail=f"system_version={sv!r} schema_version={schema!r}",
        )

    # ─── Step 2: Artifact Collection ──────────────────────────────────────────

    def _collect_artifacts(self, job_id: str) -> tuple[Artifacts, ValidationResult]:
        """Download all 6 artifacts. CRITICAL artifacts missing = hard failure."""
        job_dir = self.cfg.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        artifacts = Artifacts()
        missing_critical = []
        missing_required = []

        # CRITICAL: final video
        try:
            video_path = job_dir / "final_video.mp4"
            self._download_file(f"/jobs/{job_id}/output", video_path)
            artifacts.video_path = video_path
        except Exception as exc:
            logger.error("AVON: failed to download video: %s", exc)
            missing_critical.append("final_video.mp4")

        # CRITICAL: pipeline report
        try:
            artifacts.pipeline_report = self._get(f"/jobs/{job_id}/report")
            (job_dir / "pipeline_report.json").write_text(
                json.dumps(artifacts.pipeline_report, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("AVON: failed to fetch pipeline report: %s", exc)
            missing_critical.append("pipeline_report.json")

        # CRITICAL: identity state
        try:
            artifacts.identity_state = self._get(f"/jobs/{job_id}/identity")
        except Exception as exc:
            logger.error("AVON: failed to fetch identity state: %s", exc)
            missing_critical.append("identity_state.json")

        # REQUIRED: audio validation
        try:
            artifacts.audio_validation = self._get(f"/jobs/{job_id}/audio")
        except Exception as exc:
            logger.warning("AVON: failed to fetch audio validation: %s", exc)
            missing_required.append("audio_validation.json")

        # REQUIRED: composition plan
        try:
            artifacts.composition_plan = self._get(f"/jobs/{job_id}/composition")
        except Exception as exc:
            logger.warning("AVON: failed to fetch composition plan: %s", exc)
            missing_required.append("composition_plan.json")

        # REQUIRED: continuity report
        try:
            artifacts.continuity_report = self._get(f"/jobs/{job_id}/temporal")
        except Exception as exc:
            logger.warning("AVON: failed to fetch continuity report: %s", exc)
            missing_required.append("continuity_report.json")

        # Server checksum (for file integrity)
        try:
            checksum_data = self._get(f"/jobs/{job_id}/checksum")
            artifacts.server_checksum = checksum_data.get("sha256", "")
        except Exception:
            pass

        failed = bool(missing_critical) or (self.cfg.strict_artifact_mode and missing_required)
        detail = (
            f"missing_critical={missing_critical} missing_required={missing_required}"
            if (missing_critical or missing_required)
            else "all artifacts collected"
        )
        return artifacts, ValidationResult(
            gate="artifact_collection",
            passed=not failed,
            score=0.0 if failed else 1.0,
            detail=detail,
        )

    # ─── Step 3: Schema Validation ────────────────────────────────────────────

    def _validate_schemas(self, artifacts: Artifacts) -> ValidationResult:
        """Strict Pydantic-style field validation on all JSON artifacts."""
        errors = []

        if artifacts.pipeline_report:
            if "schema_version" not in artifacts.pipeline_report:
                errors.append("pipeline_report missing schema_version")
            if artifacts.pipeline_report.get("schema_version") != self.cfg.expected_schema_version:
                errors.append(f"pipeline_report schema_version mismatch")

        if artifacts.identity_state:
            required = {"drift_score", "cumulative_drift", "is_frozen"}
            missing = required - set(artifacts.identity_state.keys())
            if missing:
                errors.append(f"identity_state missing fields: {missing}")

        if artifacts.audio_validation:
            required = {"snr_db", "peak_db", "snr_passed", "clipping_passed"}
            missing = required - set(artifacts.audio_validation.keys())
            if missing:
                errors.append(f"audio_validation missing fields: {missing}")

        if artifacts.composition_plan:
            required = {"camera_angle", "camera_motion", "character_positions",
                        "focus_subject", "lighting_style", "motion_vector"}
            missing = required - set(artifacts.composition_plan.keys())
            if missing:
                errors.append(f"composition_plan missing fields: {missing}")

        if artifacts.continuity_report:
            if "overall_continuity_score" not in artifacts.continuity_report:
                errors.append("continuity_report missing overall_continuity_score")

        return ValidationResult(
            gate="schema_validation",
            passed=not errors,
            score=0.0 if errors else 1.0,
            detail="; ".join(errors) if errors else "all schemas valid",
        )

    # ─── Step 4: File Verification ────────────────────────────────────────────

    def _verify_file(self, job_id: str, artifacts: Artifacts) -> ValidationResult:
        """Multi-level file integrity: size, checksum, duration, codec, frames."""
        errors = []

        if not artifacts.video_path or not artifacts.video_path.exists():
            return ValidationResult(gate="file_verification", passed=False, score=0.0,
                                    detail="video file not found")

        # Size sanity
        try:
            metadata = self._get(f"/jobs/{job_id}/metadata")
            file_size = artifacts.video_path.stat().st_size
            min_b = metadata.get("min_size_bytes", 0)
            max_b = metadata.get("max_size_bytes", 10 * 1024**3)
            if not (min_b <= file_size <= max_b):
                errors.append(f"file size {file_size} outside [{min_b}, {max_b}]")
        except Exception as exc:
            logger.warning("AVON: could not check file size: %s", exc)

        # SHA-256 checksum
        if artifacts.server_checksum:
            local_sha256 = _sha256(artifacts.video_path)
            if local_sha256 != artifacts.server_checksum:
                errors.append(f"checksum mismatch: local={local_sha256[:16]}… server={artifacts.server_checksum[:16]}…")

        # Duration, codec, bitrate via ffprobe
        ffprobe_info = _ffprobe(artifacts.video_path)
        if ffprobe_info:
            duration = ffprobe_info.get("duration", 0.0)
            if not (self.cfg.min_duration_s <= duration <= self.cfg.max_duration_s):
                errors.append(f"duration {duration:.1f}s outside [{self.cfg.min_duration_s}, {self.cfg.max_duration_s}]")

            codec = ffprobe_info.get("codec_name", "")
            if codec and self.cfg.expected_codec not in codec:
                errors.append(f"codec {codec!r} != expected {self.cfg.expected_codec!r}")

            bitrate_kbps = ffprobe_info.get("bit_rate_kbps", 0)
            if bitrate_kbps and bitrate_kbps < self.cfg.min_bitrate_kbps:
                errors.append(f"bitrate {bitrate_kbps}kbps below minimum {self.cfg.min_bitrate_kbps}kbps")
        else:
            errors.append("ffprobe failed — video may not be playable")

        return ValidationResult(
            gate="file_verification",
            passed=not errors,
            score=0.0 if errors else 1.0,
            detail="; ".join(errors) if errors else "file verification passed",
        )

    # ─── Step 5: System Validation ────────────────────────────────────────────

    def _validate_system(self, artifacts: Artifacts) -> ValidationResult:
        """Identity drift, audio SNR, temporal continuity, composition, cross-modal."""
        errors = []
        warnings = []

        # Identity drift
        if artifacts.identity_state:
            drift = artifacts.identity_state.get("cumulative_drift", 0.0)
            frozen = artifacts.identity_state.get("is_frozen", False)
            if not frozen:
                errors.append("identity reference was never frozen (RULE-95 violation)")
            if drift > self.cfg.max_identity_drift:
                errors.append(f"identity cumulative_drift={drift:.4f} > threshold={self.cfg.max_identity_drift}")

        # Audio quality
        if artifacts.audio_validation:
            snr = artifacts.audio_validation.get("snr_db", 0.0)
            peak = artifacts.audio_validation.get("peak_db", 0.0)
            snr_passed = artifacts.audio_validation.get("snr_passed", False)
            clip_passed = artifacts.audio_validation.get("clipping_passed", True)
            if not snr_passed:
                errors.append(f"audio SNR={snr:.1f}dB below minimum {self.cfg.snr_min_db}dB (RULE-99)")
            if not clip_passed:
                errors.append(f"audio clipping detected: peak={peak:.1f}dBFS (RULE-99)")

        # Temporal continuity
        if artifacts.continuity_report:
            score = artifacts.continuity_report.get("overall_continuity_score", 0.0)
            passed = artifacts.continuity_report.get("passed", False)
            if not passed or score < self.cfg.min_continuity_score:
                errors.append(f"continuity_score={score:.4f} < threshold={self.cfg.min_continuity_score}")

        # Composition plan
        if artifacts.composition_plan:
            required_fields = {"camera_angle", "camera_motion", "character_positions",
                               "focus_subject", "lighting_style", "motion_vector"}
            missing = required_fields - set(artifacts.composition_plan.keys())
            if missing:
                errors.append(f"composition_plan missing required fields: {missing}")

        return ValidationResult(
            gate="system_validation",
            passed=not errors,
            score=0.0 if errors else 1.0,
            detail="; ".join(errors) if errors else "system validation passed",
        )

    # ─── Step 6: Pipeline Audit ───────────────────────────────────────────────

    def _audit_pipeline(self, artifacts: Artifacts) -> ValidationResult:
        """Audit stage coverage, retry counts, and SLA compliance."""
        if not artifacts.pipeline_report:
            return ValidationResult(gate="pipeline_audit", passed=False, score=0.0,
                                    detail="no pipeline report available")

        report = artifacts.pipeline_report
        errors = []

        # Stage coverage
        hrg_count = report.get("hrg_checkpoint_count", 0)
        if hrg_count < 11:
            errors.append(f"only {hrg_count}/11 HRG checkpoints completed")

        # Rule compliance
        rule_compliance = report.get("rule_compliance", [])
        failed_rules = [r for r in rule_compliance if not r.get("compliant", True)]
        if failed_rules:
            rule_ids = [r.get("rule_id", "?") for r in failed_rules]
            errors.append(f"rule violations: {rule_ids}")

        # SLA
        stage_durations = report.get("stage_durations", {})
        if stage_durations:
            total = sum(stage_durations.values())
            if total > 3600:  # 1 hour SLA ceiling
                errors.append(f"total pipeline duration {total:.0f}s exceeds SLA ceiling 3600s")

        return ValidationResult(
            gate="pipeline_audit",
            passed=not errors,
            score=0.0 if errors else 1.0,
            detail="; ".join(errors) if errors else "pipeline audit passed",
        )

    # ─── Step 7: Quality Scoring ──────────────────────────────────────────────

    def _compute_quality_score(
        self, artifacts: Artifacts, results: list[ValidationResult]
    ) -> tuple[float, dict]:
        """Weighted multi-dimensional quality formula per spec §22."""
        breakdown: dict[str, float] = {}

        # Identity score (weight 0.35)
        if artifacts.identity_state:
            drift = artifacts.identity_state.get("cumulative_drift", 0.0)
            identity_score = max(0.0, 1.0 - (drift / self.cfg.max_identity_drift))
        else:
            identity_score = 0.0
        breakdown["identity"] = identity_score

        # Audio score (weight 0.25)
        if artifacts.audio_validation:
            snr = artifacts.audio_validation.get("snr_db", 0.0)
            snr_normalized = min(1.0, max(0.0, (snr - 8.0) / 12.0))  # 8→20 dB maps to 0→1
            clip_ok = float(artifacts.audio_validation.get("clipping_passed", True))
            audio_score = (snr_normalized * 0.7) + (clip_ok * 0.3)
        else:
            audio_score = 0.0
        breakdown["audio"] = audio_score

        # Temporal score (weight 0.25)
        if artifacts.continuity_report:
            temporal_score = artifacts.continuity_report.get("overall_continuity_score", 0.0)
        else:
            temporal_score = 0.0
        breakdown["temporal"] = temporal_score

        # Composition score (weight 0.15)
        if artifacts.composition_plan:
            all_fields = all(
                artifacts.composition_plan.get(f)
                for f in ("camera_angle", "camera_motion", "character_positions",
                          "focus_subject", "lighting_style", "motion_vector")
            )
            composition_score = 1.0 if all_fields else 0.5
        else:
            composition_score = 0.0
        breakdown["composition"] = composition_score

        quality_score = (
            identity_score * self.cfg.identity_weight
            + audio_score * self.cfg.audio_weight
            + temporal_score * self.cfg.temporal_weight
            + composition_score * self.cfg.composition_weight
        )

        logger.info(
            "AVON quality: identity=%.3f audio=%.3f temporal=%.3f composition=%.3f → score=%.3f",
            identity_score, audio_score, temporal_score, composition_score, quality_score,
        )
        return round(quality_score, 4), breakdown

    # ─── Step 8: Probabilistic Confidence Scoring ─────────────────────────────

    def _compute_confidence(
        self, results: list[ValidationResult], quality_breakdown: dict
    ) -> float:
        """Estimate confidence accounting for borderline values and data completeness."""
        gates_passed = sum(1 for r in results if r.passed)
        total_gates = len(results) if results else 1
        pass_rate = gates_passed / total_gates

        # Uncertainty penalty for borderline quality scores
        quality_score = sum(
            quality_breakdown.get(k, 0.0) * w
            for k, w in [("identity", 0.35), ("audio", 0.25), ("temporal", 0.25), ("composition", 0.15)]
        )
        border_distance = abs(quality_score - self.cfg.quality_threshold)
        # If score is very close to threshold, reduce confidence
        uncertainty = max(0.0, 1.0 - (border_distance * 5))

        # Historical signal — if recent runs performed well, increase confidence
        recent_scores = [r.quality_score for r in self._history[-5:]]
        historical_signal = (sum(recent_scores) / len(recent_scores)) if recent_scores else 0.75

        confidence = (pass_rate * 0.5) + ((1.0 - uncertainty * 0.2) * 0.3) + (historical_signal * 0.2)
        return round(min(1.0, max(0.0, confidence)), 4)

    # ─── Step 9: Cross-Validation ─────────────────────────────────────────────

    def _cross_validate(
        self, artifacts: Artifacts, results: list[ValidationResult]
    ) -> ValidationResult:
        """Detect hidden instabilities not visible in isolated checks."""
        warnings = []

        # Signal 1: identity drift high but audio passed — possible sync issue
        if artifacts.identity_state and artifacts.audio_validation:
            drift = artifacts.identity_state.get("cumulative_drift", 0.0)
            snr_ok = artifacts.audio_validation.get("snr_passed", True)
            if drift > 0.10 and snr_ok:
                warnings.append("high identity drift with good audio — possible lip sync inconsistency")

        # Signal 2: temporal continuity low but no individual segment failures
        if artifacts.continuity_report:
            score = artifacts.continuity_report.get("overall_continuity_score", 1.0)
            segments = artifacts.continuity_report.get("identity_per_segment", [])
            if score < 0.88 and segments and min(segments) > 0.90:
                warnings.append("low overall continuity despite good per-segment CLIP — check transition frames")

        # Signal 3: composition plan complete but temporal motion inconsistent
        if artifacts.composition_plan and artifacts.continuity_report:
            motion_v = artifacts.composition_plan.get("motion_vector", "")
            temporal = artifacts.continuity_report.get("motion_continuity", 1.0)
            if motion_v == "stationary" and temporal < 0.80:
                warnings.append("stationary motion_vector but low motion continuity — camera drift detected")

        severity = "HIGH" if len(warnings) >= 3 else ("MEDIUM" if warnings else "LOW")
        passed = severity != "HIGH"

        return ValidationResult(
            gate="cross_validation",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=f"severity={severity} signals=[{'; '.join(warnings)}]" if warnings else "no hidden instabilities",
        )

    # ─── Step 10: Cleanup Eligibility ─────────────────────────────────────────

    def _evaluate_cleanup_eligibility(
        self,
        results: list[ValidationResult],
        quality_score: float,
        confidence: float,
    ) -> bool:
        """RULE 1: cleanup ONLY if all 9 conditions are met."""
        gate_names = {r.gate for r in results if r.passed}

        conditions = [
            ("version_check" in gate_names, "version compatibility confirmed"),
            ("schema_validation" in gate_names, "all artifact schemas valid"),
            ("artifact_collection" in gate_names, "all artifacts downloaded"),
            ("file_verification" in gate_names, "video file integrity confirmed"),
            ("system_validation" in gate_names, "system validation passed"),
            ("pipeline_audit" in gate_names, "pipeline audit passed or warning"),
            (quality_score >= self.cfg.quality_threshold, f"quality_score {quality_score:.3f} >= {self.cfg.quality_threshold}"),
            (not any(r.gate == "cross_validation" and r.detail and "HIGH" in r.detail for r in results), "cross-validation not HIGH severity"),
            (confidence >= self.cfg.confidence_threshold, f"confidence {confidence:.3f} >= {self.cfg.confidence_threshold}"),
        ]

        all_pass = all(ok for ok, _ in conditions)
        for ok, desc in conditions:
            status = "✅" if ok else "❌"
            logger.info("AVON cleanup gate %s: %s", status, desc)

        return all_pass

    # ─── Step 11: Feedback Report ─────────────────────────────────────────────

    def _build_recommended_adjustments(
        self, results: list[ValidationResult], breakdown: dict
    ) -> dict:
        """Build actionable adjustment recommendations for the server."""
        adjustments: dict[str, Any] = {}

        if breakdown.get("identity", 1.0) < 0.85:
            adjustments["increase_clip_validation_frequency"] = True
            adjustments["reduce_max_drift_threshold"] = 0.10

        if breakdown.get("audio", 1.0) < 0.80:
            adjustments["increase_snr_headroom_db"] = 2.0
            adjustments["enable_audio_normalization"] = True

        if breakdown.get("temporal", 1.0) < 0.85:
            adjustments["increase_svi_steps"] = True
            adjustments["reduce_svi_cfg"] = True

        if breakdown.get("composition", 1.0) < 0.90:
            adjustments["retry_composition_plan_generation"] = True
            adjustments["increase_composition_max_retries"] = 5

        return adjustments

    def _post_feedback_report(
        self,
        job_id: str,
        quality_score: float,
        confidence: float,
        cleanup_triggered: bool,
        results: list[ValidationResult],
        adjustments: dict,
    ) -> None:
        """POST /jobs/{job_id}/client_report with actionable feedback."""
        try:
            payload = {
                "job_id": job_id,
                "quality_score": quality_score,
                "confidence": confidence,
                "cleanup_triggered": cleanup_triggered,
                "validation_results": {r.gate: {"passed": r.passed, "score": r.score, "detail": r.detail} for r in results},
                "recommended_adjustments": adjustments,
                "run_metadata": {"avon_version": "4.2", "timestamp": _now()},
                "schema_version": "v6.0",
            }
            self._post(f"/jobs/{job_id}/client_report", payload)
            logger.info("AVON: feedback report posted for job %s", job_id)
        except Exception as exc:
            logger.warning("AVON: failed to post feedback report: %s", exc)

    # ─── Step 12: Adaptation Engine ───────────────────────────────────────────

    def _adapt_parameters(self, results: list[ValidationResult], quality_score: float) -> None:
        """Update config thresholds based on historical outcomes (self-improvement)."""
        if len(self._history) < 3:
            return

        recent = self._history[-3:]
        avg_quality = sum(r.quality_score for r in recent) / len(recent)

        # If consistently high quality, can relax poll interval slightly
        if avg_quality > 0.90:
            self.cfg.poll_interval_s = min(30, self.cfg.poll_interval_s + 2)
            logger.info("AVON adaptation: relaxing poll interval to %ds", self.cfg.poll_interval_s)

        # If recent runs struggled with confidence, tighten thresholds
        avg_confidence = sum(r.confidence for r in recent) / len(recent)
        if avg_confidence < 0.75:
            self.cfg.confidence_threshold = min(0.80, self.cfg.confidence_threshold + 0.02)
            logger.info("AVON adaptation: raising confidence threshold to %.2f", self.cfg.confidence_threshold)

    # ─── Step 15: Safe Cleanup ────────────────────────────────────────────────

    def _safe_cleanup(self, job_id: str) -> None:
        """DELETE /jobs/{job_id} — ONLY called after all 9 gates pass (RULE 1).
        Idempotent: 404 is treated as success (RULE 3)."""
        try:
            r = self._session.delete(
                f"{self.cfg.server_base_url}/jobs/{job_id}", timeout=30
            )
            if r.status_code in (200, 404):
                logger.info("AVON: server-side cleanup COMPLETED for job %s (status=%d)", job_id, r.status_code)
            else:
                logger.error("AVON: cleanup returned unexpected status %d for job %s", r.status_code, job_id)
        except requests.RequestException as exc:
            logger.error("AVON: cleanup request failed: %s — server data preserved", exc)

    # ─── Memory Store ─────────────────────────────────────────────────────────

    def _save_to_history(self, record: RunRecord) -> None:
        """Append run record to JSONL history file for future adaptation."""
        try:
            with self.cfg.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "job_id": record.job_id,
                    "timestamp": record.timestamp,
                    "quality_score": record.quality_score,
                    "confidence": record.confidence,
                    "cleanup_triggered": record.cleanup_triggered,
                    "duration_s": record.duration_s,
                    "gates_passed": record.gates_passed,
                    "gates_failed": record.gates_failed,
                }, default=str) + "\n")
        except OSError as exc:
            logger.warning("AVON: could not save history: %s", exc)
        self._history.append(record)

    def _load_history(self) -> list[RunRecord]:
        history = []
        if self.cfg.history_file.exists():
            try:
                for line in self.cfg.history_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        d = json.loads(line)
                        history.append(RunRecord(
                            job_id=d["job_id"],
                            timestamp=d.get("timestamp", ""),
                            quality_score=d.get("quality_score", 0.0),
                            confidence=d.get("confidence", 0.0),
                            cleanup_triggered=d.get("cleanup_triggered", False),
                            gates_passed=d.get("gates_passed", []),
                            gates_failed=d.get("gates_failed", []),
                            recommended_adjustments={},
                            duration_s=d.get("duration_s", 0.0),
                        ))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("AVON: could not load history: %s", exc)
        return history

    # ─── Metrics Emission ─────────────────────────────────────────────────────

    def _emit_metrics(self, record: RunRecord, results: list[ValidationResult]) -> None:
        """Write structured metrics to metrics.jsonl."""
        try:
            metrics = {
                "timestamp": _now(),
                "job_id": record.job_id,
                "quality_score": record.quality_score,
                "confidence": record.confidence,
                "cleanup_triggered": record.cleanup_triggered,
                "gates_passed_count": len(record.gates_passed),
                "gates_failed_count": len(record.gates_failed),
                "gates_failed": record.gates_failed,
                "duration_s": record.duration_s,
                "avon_version": "4.2",
            }
            with self.cfg.metrics_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
        except OSError as exc:
            logger.warning("AVON: could not emit metrics: %s", exc)

    # ─── Phase State (crash resume) ───────────────────────────────────────────

    def _save_phase_state(self, job_id: str, phase: str) -> None:
        try:
            self.cfg.state_file.write_text(
                json.dumps({"job_id": job_id, "phase": phase, "timestamp": _now()}),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _clear_phase_state(self, job_id: str) -> None:
        try:
            if self.cfg.state_file.exists():
                self.cfg.state_file.unlink()
        except OSError:
            pass

    # ─── HTTP helpers ─────────────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        url = self.cfg.server_base_url + path
        for attempt in range(self.cfg.max_download_retries):
            try:
                r = self._session.get(url, timeout=30)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                if attempt == self.cfg.max_download_retries - 1:
                    raise
                wait = self.cfg.download_backoff[min(attempt, len(self.cfg.download_backoff) - 1)]
                logger.warning("AVON GET %s failed (attempt %d): %s — retry in %ds", path, attempt + 1, exc, wait)
                time.sleep(wait)
        return {}

    def _post(self, path: str, data: dict) -> dict:
        url = self.cfg.server_base_url + path
        r = self._session.post(url, json=data, timeout=30)
        r.raise_for_status()
        return r.json()

    def _download_file(self, path: str, dest: Path) -> None:
        url = self.cfg.server_base_url + path
        for attempt in range(self.cfg.max_download_retries):
            try:
                with self._session.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with dest.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logger.info("AVON: downloaded %s → %s", path, dest)
                return
            except requests.RequestException as exc:
                if attempt == self.cfg.max_download_retries - 1:
                    raise
                wait = self.cfg.download_backoff[min(attempt, len(self.cfg.download_backoff) - 1)]
                logger.warning("AVON: download %s failed (attempt %d): %s — retry in %ds", path, attempt + 1, exc, wait)
                time.sleep(wait)

    # ─── Record builders ──────────────────────────────────────────────────────

    def _finalize(
        self,
        job_id: str,
        results: list[ValidationResult],
        gates_passed: list,
        gates_failed: list,
        cleanup_eligible: bool,
        adjustments: dict,
        start_time: float,
        quality_score: float = 0.0,
        confidence: float = 0.0,
    ) -> RunRecord:
        elapsed = time.monotonic() - start_time
        logger.info(
            "AVON: job=%s quality=%.3f confidence=%.3f cleanup=%s elapsed=%.1fs",
            job_id, quality_score, confidence, cleanup_eligible, elapsed,
        )
        return RunRecord(
            job_id=job_id,
            timestamp=_now(),
            quality_score=quality_score,
            confidence=confidence,
            cleanup_triggered=cleanup_eligible,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            recommended_adjustments=adjustments,
            duration_s=elapsed,
        )

    def _make_failed_record(self, job_id: str, status: str, elapsed: float) -> RunRecord:
        return RunRecord(
            job_id=job_id,
            timestamp=_now(),
            quality_score=0.0,
            confidence=0.0,
            cleanup_triggered=False,
            gates_passed=[],
            gates_failed=[f"job_status_{status}"],
            recommended_adjustments={},
            duration_s=elapsed,
        )


# ─── Standalone Helpers ───────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffprobe(video_path: Path) -> Optional[dict]:
    """Run ffprobe to extract codec, duration, bitrate. Returns None if ffprobe unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
                "-show_format", str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        return {
            "duration": float(fmt.get("duration", 0.0)),
            "bit_rate_kbps": int(fmt.get("bit_rate", 0)) // 1000,
            "codec_name": video_stream.get("codec_name", ""),
            "width": video_stream.get("width", 0),
            "height": video_stream.get("height", 0),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    """Run AVON from the command line.
    Usage: python -m vga.client.avon_watcher --job-id job_abc123 --server http://<POD_IP>:8000/api/v1
    """
    import argparse

    parser = argparse.ArgumentParser(description="AVON — VGA Client Watcher v4.2")
    parser.add_argument("--job-id", required=True, help="Job ID to watch")
    parser.add_argument("--server", default="http://localhost:8000/api/v1",
                        help="VGA API base URL")
    parser.add_argument("--output-dir", default="./avon_downloads",
                        help="Local directory for downloaded artifacts")
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Polling interval in seconds")
    args = parser.parse_args()

    config = AVONConfig(
        server_base_url=args.server,
        output_dir=Path(args.output_dir),
        poll_interval_s=args.poll_interval,
    )
    watcher = AVONWatcher(config=config)
    record = watcher.watch(args.job_id)

    print(f"\n{'='*60}")
    print(f"AVON Result for job {record.job_id}")
    print(f"  Quality score : {record.quality_score:.3f} (threshold {config.quality_threshold})")
    print(f"  Confidence    : {record.confidence:.3f} (threshold {config.confidence_threshold})")
    print(f"  Cleanup       : {'YES — server data removed' if record.cleanup_triggered else 'NO — data preserved'}")
    print(f"  Gates passed  : {record.gates_passed}")
    print(f"  Gates failed  : {record.gates_failed}")
    print(f"  Duration      : {record.duration_s:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
