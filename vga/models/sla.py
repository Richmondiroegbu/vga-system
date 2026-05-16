"""
SLA models — per-stage SLA records for PipelineReport.
Spec: VGA Codebase Structure Design v17.2 §models/sla.py
"""
from __future__ import annotations

from pydantic import BaseModel


class SLARecord(BaseModel):
    """SLA compliance record for a single pipeline stage."""
    stage_id: str
    elapsed_s: float
    budget_s: float
    within_budget: bool
    utilization_pct: float   # elapsed / budget * 100
    schema_version: str = "v6.0"


class SLASummary(BaseModel):
    """Aggregated SLA summary for the full pipeline run."""
    total_elapsed_s: float
    stages_within_budget: int
    stages_over_budget: int
    records: list[SLARecord] = []
    schema_version: str = "v6.0"

    @property
    def compliance_rate(self) -> float:
        total = self.stages_within_budget + self.stages_over_budget
        return self.stages_within_budget / total if total > 0 else 1.0
