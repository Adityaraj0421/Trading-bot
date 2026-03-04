"""
Phase 9 decision log routes.

Endpoints:
  GET /phase9/decisions  — Last N entries from the Phase 9 JSONL audit log.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from api.data_store import DataStore
from config import Config

_log = logging.getLogger(__name__)

# Maximum lines to read from the JSONL tail to avoid reading the whole file
_MAX_TAIL_LINES = 2000


def _tail_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    """Read the last *n* valid JSON lines from a JSONL file.

    Reads up to ``_MAX_TAIL_LINES`` lines from the end of the file, parses
    each as JSON (silently skipping malformed lines), and returns the last *n*
    successfully parsed entries in chronological order (oldest first).

    Args:
        path: Absolute path to the JSONL file.
        n: Number of entries to return.

    Returns:
        List of parsed dicts, up to *n* entries, in ascending time order.
    """
    if not path.exists():
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("Phase9 log read error: %s", exc)
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Examine only the tail to bound memory usage
    lines = lines[-_MAX_TAIL_LINES:]

    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records[-n:]


def create_router(store: DataStore) -> APIRouter:  # noqa: ARG001
    """Create the Phase 9 router.

    Args:
        store: Shared DataStore (not used by this router but kept consistent
               with the project's ``create_router(store)`` convention).

    Returns:
        Configured ``APIRouter`` with prefix ``/phase9``.
    """
    router = APIRouter(prefix="/phase9", tags=["phase9"])

    @router.get("/decisions")
    def get_decisions(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        """Return the last *limit* Phase 9 decision log entries.

        Reads from the JSONL file at ``Config.PHASE9_DECISION_LOG_PATH``.
        Returns an empty list when the log path is not configured or the file
        does not yet exist.

        Args:
            limit: Number of most-recent entries to return (1–500).

        Returns:
            Dict with ``decisions`` (list of log entries) and ``count`` (int).
        """
        log_path = Config.PHASE9_DECISION_LOG_PATH
        if not log_path:
            return {"decisions": [], "count": 0, "log_path": None}

        path = Path(log_path)
        decisions = _tail_jsonl(path, limit)
        # Return newest-first so the dashboard table shows latest at the top
        decisions_reversed = list(reversed(decisions))
        return {
            "decisions": decisions_reversed,
            "count": len(decisions_reversed),
            "log_path": str(path),
        }

    return router
