"""
SchemaMigrations — migrate legacy artifact schemas to v6.0.
Handles v5.2 → v6.0 migration for all VGA artifact types.
Spec: VGA Codebase Structure Design v17.2 §core/schema_migrations.py; §9 Schema Version Contract
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_CURRENT_VERSION = "v6.0"
_LEGACY_VERSION = "v5.2"


def _migrate_v5_2_to_v6_0(artifact: dict) -> dict:
    """Migrate a v5.2 artifact dict to v6.0 schema.

    Changes from v5.2 → v6.0:
    - schema_version field added
    - identity_state gains is_frozen field (default False for legacy)
    - temporal_state gains buffer_initialized field
    - HRG checkpoint count updated from 9 to 11

    Args:
        artifact: dict representation of a v5.2 artifact

    Returns:
        Updated dict conforming to v6.0 schema
    """
    if artifact.get("schema_version") == _CURRENT_VERSION:
        return artifact   # already current

    migrated = dict(artifact)
    migrated["schema_version"] = _CURRENT_VERSION

    # Identity state migration
    if "identity_state" in migrated and isinstance(migrated["identity_state"], dict):
        identity = migrated["identity_state"]
        identity.setdefault("is_frozen", False)
        identity.setdefault("cumulative_drift", identity.get("drift_score", 0.0))
        identity.setdefault("history", [])

    # Temporal state migration
    if "temporal_state" in migrated and isinstance(migrated["temporal_state"], dict):
        temporal = migrated["temporal_state"]
        temporal.setdefault("buffer_initialized", False)
        temporal.setdefault("total_segments", 0)

    # HRG count migration
    if "hrg_checkpoint_count" in migrated:
        migrated["hrg_checkpoint_count"] = max(migrated["hrg_checkpoint_count"], 11)

    # Composition plan (new in v6.0)
    migrated.setdefault("composition_plan_summary", None)
    migrated.setdefault("temporal_engine_health", None)

    logger.info("Schema migrated: v5.2 → v6.0")
    return migrated


def migrate_artifact(artifact: dict) -> dict:
    """Public entry point — migrates any artifact to the current schema version."""
    version = artifact.get("schema_version", _LEGACY_VERSION)

    if version == _CURRENT_VERSION:
        return artifact

    if version == _LEGACY_VERSION:
        return _migrate_v5_2_to_v6_0(artifact)

    logger.warning(
        "SchemaMigrations: unknown schema version %r — returning artifact unchanged", version
    )
    return artifact
