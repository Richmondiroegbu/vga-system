"""
CharacterRegistry — manages character identities across pipeline jobs.
Maps character_id → identity design + frozen embedding reference.
Spec: VGA Codebase Structure Design v17.2 §core/character_registry.py
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CharacterRegistry:
    """In-memory registry of character identities per job."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, object]] = {}

    def register(self, job_id: str, character_id: str, identity_design: dict) -> None:
        """Register a character identity design for a job."""
        self._registry.setdefault(job_id, {})[character_id] = {
            "identity_design": identity_design,
            "embedding_vector": None,
            "is_frozen": False,
        }
        logger.debug("CharacterRegistry: registered %s for job %s", character_id, job_id)

    def set_embedding(self, job_id: str, character_id: str, embedding: List[float]) -> None:
        """Store the frozen identity embedding for a character."""
        if job_id in self._registry and character_id in self._registry[job_id]:
            self._registry[job_id][character_id]["embedding_vector"] = embedding
            self._registry[job_id][character_id]["is_frozen"] = True
            logger.info(
                "CharacterRegistry: embedding frozen for %s job=%s", character_id, job_id
            )

    def get_embedding(self, job_id: str, character_id: str) -> Optional[List[float]]:
        """Retrieve frozen embedding. Returns None if not yet set."""
        entry = self._registry.get(job_id, {}).get(character_id, {})
        return entry.get("embedding_vector")

    def get_identity_design(self, job_id: str, character_id: str) -> Optional[dict]:
        """Retrieve identity design for a character."""
        entry = self._registry.get(job_id, {}).get(character_id, {})
        return entry.get("identity_design")

    def list_characters(self, job_id: str) -> List[str]:
        """List all character IDs registered for a job."""
        return list(self._registry.get(job_id, {}).keys())

    def clear_job(self, job_id: str) -> None:
        """Remove all character records for a completed job."""
        self._registry.pop(job_id, None)
