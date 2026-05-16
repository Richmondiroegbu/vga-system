"""
Integration test for HRG 11-checkpoint flow.
Verifies all checkpoints exist and decision writing/reading works.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vga.core.hrg_controller import HRGController
from vga.models.enums import HRGCheckpoint


@pytest.fixture(autouse=True)
def use_tmp_hrg_dir(tmp_path, monkeypatch):
    """Redirect HRG controller to use tmp_path for testing."""
    monkeypatch.setattr(
        "vga.core.hrg_controller.settings",
        type("MockSettings", (), {
            "HRG_DIR": tmp_path,
            "HRG_REVIEW_ENABLED": False,   # auto-approve for testing
            "HRG_APPROVAL_TIMEOUT_SECONDS": 5,
            "HRG_CHECKPOINT_COUNT": 11,
        })(),
    )


def test_all_11_hrg_checkpoints_defined():
    """HRGCheckpoint enum must have exactly 11 values."""
    checkpoints = list(HRGCheckpoint)
    assert len(checkpoints) == 11, f"Expected 11 HRG checkpoints, got {len(checkpoints)}"


def test_hrg_checkpoint_values_1_through_11():
    """Checkpoints must be HRG-1 through HRG-11."""
    values = {cp.value for cp in HRGCheckpoint}
    for i in range(1, 12):
        assert f"HRG-{i}" in values, f"HRG-{i} missing from HRGCheckpoint enum"


def test_hrg_controller_auto_approves_when_disabled(tmp_path):
    """With HRG disabled, checkpoint() auto-approves without blocking."""
    controller = HRGController()
    # Should not raise or block
    controller.checkpoint(
        HRGCheckpoint.HRG_1_SCRIPT,
        {"script": "test"},
        scene_id="sc_001",
        job_id="job_test",
    )
    outcomes = controller.get_outcomes()
    assert "HRG-1" in outcomes


def test_hrg_state_manager_records_11_checkpoints(tmp_path):
    """HRGStateManager can record and retrieve all 11 checkpoint decisions."""
    from vga.core.hrg_state_manager import HRGStateManager
    manager = HRGStateManager.__new__(HRGStateManager)
    manager._state_dir = tmp_path / "hrg_state"
    manager._state_dir.mkdir()

    for i in range(1, 12):
        cp = f"HRG-{i}"
        manager.record_decision("job_test", "sc_001", cp, "approved")

    for i in range(1, 12):
        decision = manager.get_decision("job_test", "sc_001", f"HRG-{i}")
        assert decision == "approved", f"HRG-{i} decision not retrieved correctly"

    all_decisions = manager.get_all_decisions("job_test", "sc_001")
    assert len(all_decisions) == 11
