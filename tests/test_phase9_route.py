"""
Tests for api/routes/phase9.py — Phase 9 decision log endpoint.

Covers:
  - _tail_jsonl() unit tests (missing file, empty, malformed lines, limit,
    ordering, memory guard)
  - GET /phase9/decisions endpoint tests (no log path, missing file,
    entries returned newest-first, limit parameter, query bounds)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.routes.phase9 import _tail_jsonl
from api.server import create_app
from config import Config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """TestClient wired to the app, with auth header when required."""
    app = create_app()
    c = TestClient(app)
    if Config.API_AUTH_KEY:
        c.headers["X-API-Key"] = Config.API_AUTH_KEY
    return c


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Helper: write a list of dicts to a JSONL file."""
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Unit tests — _tail_jsonl()
# ---------------------------------------------------------------------------


class TestTailJsonl:
    def test_missing_file_returns_empty(self, tmp_path):
        result = _tail_jsonl(tmp_path / "nonexistent.jsonl", 10)
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text("", encoding="utf-8")
        result = _tail_jsonl(path, 10)
        assert result == []

    def test_whitespace_only_file_returns_empty(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text("   \n\n  \n", encoding="utf-8")
        result = _tail_jsonl(path, 10)
        assert result == []

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text(
            '{"ts": "2025-01-01", "action": "trade"}\n'
            "NOT JSON\n"
            "{broken\n"
            '{"ts": "2025-01-02", "action": "skip"}\n',
            encoding="utf-8",
        )
        result = _tail_jsonl(path, 10)
        assert len(result) == 2
        assert result[0]["ts"] == "2025-01-01"
        assert result[1]["ts"] == "2025-01-02"

    def test_limit_enforced(self, tmp_path):
        records = [{"i": i} for i in range(20)]
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, records)
        result = _tail_jsonl(path, 5)
        assert len(result) == 5
        # Returns the LAST 5 records (most-recent chronologically)
        assert [r["i"] for r in result] == [15, 16, 17, 18, 19]

    def test_limit_larger_than_file_returns_all(self, tmp_path):
        records = [{"i": i} for i in range(3)]
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, records)
        result = _tail_jsonl(path, 100)
        assert len(result) == 3

    def test_chronological_order_preserved(self, tmp_path):
        """_tail_jsonl returns records oldest-first (chronological)."""
        records = [{"ts": f"2025-01-0{i+1}"} for i in range(5)]
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, records)
        result = _tail_jsonl(path, 3)
        assert result[0]["ts"] == "2025-01-03"
        assert result[1]["ts"] == "2025-01-04"
        assert result[2]["ts"] == "2025-01-05"

    def test_limit_one_returns_last_record(self, tmp_path):
        records = [{"i": i} for i in range(10)]
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, records)
        result = _tail_jsonl(path, 1)
        assert result == [{"i": 9}]

    def test_all_malformed_returns_empty(self, tmp_path):
        path = tmp_path / "log.jsonl"
        path.write_text("bad\nworse\n{{invalid}}\n", encoding="utf-8")
        result = _tail_jsonl(path, 10)
        assert result == []

    def test_memory_guard_limits_lines_read(self, tmp_path):
        """When file has >2000 lines, only the last 2000 are examined."""
        # Write 2010 records; _MAX_TAIL_LINES=2000 → only last 2000 used.
        records = [{"i": i} for i in range(2010)]
        path = tmp_path / "log.jsonl"
        _write_jsonl(path, records)
        # With limit=2000, we should get the last 2000 of the last 2000 tail
        # → records 10..2009 (the first 10 are dropped by the tail guard).
        result = _tail_jsonl(path, 2000)
        assert len(result) == 2000
        assert result[0]["i"] == 10   # first of the 2000-line tail
        assert result[-1]["i"] == 2009


# ---------------------------------------------------------------------------
# Endpoint tests — GET /phase9/decisions
# ---------------------------------------------------------------------------


class TestGetDecisionsEndpoint:
    def test_no_log_path_configured_returns_empty(self, client):
        """When PHASE9_DECISION_LOG_PATH is empty, endpoint returns null log_path."""
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", ""):
            resp = client.get("/phase9/decisions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["decisions"] == []
        assert body["count"] == 0
        assert body["log_path"] is None

    def test_file_not_found_returns_empty(self, client, tmp_path):
        """When log path is set but file doesn't exist, return empty decisions."""
        missing = tmp_path / "missing.jsonl"
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(missing)):
            resp = client.get("/phase9/decisions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["decisions"] == []
        assert body["count"] == 0

    def test_returns_decisions_newest_first(self, client, tmp_path):
        """Endpoint returns entries in reverse chronological order (newest first)."""
        log = tmp_path / "phase9.jsonl"
        records = [
            {"ts": "2025-01-01T00:00:00", "decision": {"action": "skip"}, "i": 0},
            {"ts": "2025-01-02T00:00:00", "decision": {"action": "trade"}, "i": 1},
            {"ts": "2025-01-03T00:00:00", "decision": {"action": "skip"}, "i": 2},
        ]
        _write_jsonl(log, records)

        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        # Newest-first → i=2, i=1, i=0
        assert body["decisions"][0]["i"] == 2
        assert body["decisions"][1]["i"] == 1
        assert body["decisions"][2]["i"] == 0

    def test_limit_parameter_respected(self, client, tmp_path):
        """?limit=N returns only N entries."""
        log = tmp_path / "phase9.jsonl"
        _write_jsonl(log, [{"i": i} for i in range(20)])

        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions?limit=5")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 5
        assert len(body["decisions"]) == 5
        # Newest-first: last 5 in the file reversed → i=19,18,17,16,15
        assert body["decisions"][0]["i"] == 19
        assert body["decisions"][-1]["i"] == 15

    def test_limit_default_is_50(self, client, tmp_path):
        """Default limit is 50 when not specified."""
        log = tmp_path / "phase9.jsonl"
        _write_jsonl(log, [{"i": i} for i in range(60)])

        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions")

        body = resp.json()
        assert body["count"] == 50

    def test_limit_below_1_rejected(self, client, tmp_path):
        """limit=0 should return 422 (Query ge=1 constraint)."""
        log = tmp_path / "phase9.jsonl"
        _write_jsonl(log, [{"i": 0}])
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions?limit=0")
        assert resp.status_code == 422

    def test_limit_above_500_rejected(self, client, tmp_path):
        """limit=501 should return 422 (Query le=500 constraint)."""
        log = tmp_path / "phase9.jsonl"
        _write_jsonl(log, [{"i": 0}])
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions?limit=501")
        assert resp.status_code == 422

    def test_limit_at_boundary_500_accepted(self, client, tmp_path):
        """limit=500 is within bounds and should succeed."""
        log = tmp_path / "phase9.jsonl"
        _write_jsonl(log, [{"i": i} for i in range(5)])
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions?limit=500")
        assert resp.status_code == 200

    def test_response_includes_log_path(self, client, tmp_path):
        """Response body includes log_path string when configured."""
        log = tmp_path / "phase9.jsonl"
        _write_jsonl(log, [{"i": 0}])
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions")
        body = resp.json()
        assert body["log_path"] == str(log)

    def test_empty_log_file_returns_zero_decisions(self, client, tmp_path):
        """An empty JSONL file returns count=0, decisions=[]."""
        log = tmp_path / "phase9.jsonl"
        log.write_text("", encoding="utf-8")
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions")
        body = resp.json()
        assert body["decisions"] == []
        assert body["count"] == 0

    def test_malformed_lines_skipped_in_endpoint(self, client, tmp_path):
        """Malformed JSON lines in the log file are silently skipped."""
        log = tmp_path / "phase9.jsonl"
        log.write_text(
            '{"ts": "2025-01-01", "decision": {"action": "skip"}}\n'
            "not-json\n"
            '{"ts": "2025-01-02", "decision": {"action": "trade"}}\n',
            encoding="utf-8",
        )
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions")
        body = resp.json()
        assert body["count"] == 2
        # Newest-first: Jan 2 first
        assert body["decisions"][0]["ts"] == "2025-01-02"
        assert body["decisions"][1]["ts"] == "2025-01-01"

    def test_limit_one_returns_only_last_entry(self, client, tmp_path):
        """limit=1 returns the single most recent entry."""
        log = tmp_path / "phase9.jsonl"
        _write_jsonl(log, [{"i": 0}, {"i": 1}, {"i": 2}])
        with patch.object(Config, "PHASE9_DECISION_LOG_PATH", str(log)):
            resp = client.get("/phase9/decisions?limit=1")
        body = resp.json()
        assert body["count"] == 1
        assert body["decisions"][0]["i"] == 2
