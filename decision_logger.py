"""DecisionLogger — structured JSON log of every evaluate() call.

Every call to evaluate(context, triggers) → Decision is persisted here with:
  - The full ContextState (serialised to dict)
  - All TriggerSignals that were evaluated (serialised to dicts)
  - The resulting Decision (action, reason, direction, route, score)
  - A UTC timestamp

This log is the primary truth-building mechanism during P4/P5 paper trading —
more valuable than backtests because it shows exactly why each trade was
rejected or accepted under real market conditions.

Two outputs:
  1. Python logging (INFO level) — always active.
  2. JSONL append file — active when a log_path is provided.
     Each line is a self-contained JSON object (newline-delimited JSON).
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from decision import ContextState, Decision, TriggerSignal

_log = logging.getLogger(__name__)


def _serialise_context(ctx: ContextState) -> dict:
    """Convert a ContextState to a plain JSON-serialisable dict."""
    return {
        "context_id": ctx.context_id,
        "swing_bias": ctx.swing_bias,
        "allowed_directions": ctx.allowed_directions,
        "volatility_regime": ctx.volatility_regime,
        "funding_pressure": ctx.funding_pressure,
        "whale_flow": ctx.whale_flow,
        "oi_trend": ctx.oi_trend,
        "key_levels": ctx.key_levels,
        "risk_mode": ctx.risk_mode,
        "confidence": round(ctx.confidence, 4),
        "tradeable": ctx.tradeable,
        "valid_until": ctx.valid_until.isoformat(),
        "updated_at": ctx.updated_at.isoformat(),
    }


def _serialise_trigger(t: TriggerSignal) -> dict:
    """Convert a TriggerSignal to a plain JSON-serialisable dict."""
    return {
        "trigger_id": t.trigger_id,
        "source": t.source,
        "direction": t.direction,
        "strength": round(t.strength, 4),
        "urgency": t.urgency,
        "symbol_scope": t.symbol_scope,
        "reason": t.reason,
        "expires_at": t.expires_at.isoformat(),
        "raw_data": t.raw_data,
    }


def _serialise_decision(d: Decision) -> dict:
    """Convert a Decision to a plain JSON-serialisable dict."""
    return {
        "action": d.action,
        "reason": d.reason,
        "direction": d.direction,
        "route": d.route,
        "score": round(d.score, 4) if d.score is not None else None,
    }


class DecisionLogger:
    """Logs every evaluate() call as structured JSON for post-mortem analysis.

    Thread-safe: a single lock protects the file handle. Suitable for use
    from the agent thread.

    Args:
        log_path: Path to the JSONL output file. If None or empty string,
            only Python logging output is produced (no file written).
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._log_path: Path | None = Path(log_path) if log_path else None
        self._lock = threading.Lock()
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            _log.info("DecisionLogger: writing to %s", self._log_path)

    def log(
        self,
        context: ContextState,
        triggers: list[TriggerSignal],
        decision: Decision,
        *,
        symbol: str | None = None,
    ) -> None:
        """Record one evaluate() call.

        Args:
            context: The ContextState passed to evaluate().
            triggers: The trigger list passed to evaluate().
            decision: The Decision returned by evaluate().
            symbol: Optional trading pair (e.g. ``"BTC/USDT"``). Added to the
                record when provided.
        """
        record: dict = {
            "ts": datetime.now(UTC).isoformat(),
            "decision": _serialise_decision(decision),
            "context": _serialise_context(context),
            "triggers": [_serialise_trigger(t) for t in triggers],
        }
        if symbol is not None:
            record["symbol"] = symbol

        # Python log (always)
        _log.info(
            "decision=%s reason=%s dir=%s route=%s score=%s triggers=%d",
            decision.action,
            decision.reason,
            decision.direction,
            decision.route,
            f"{decision.score:.3f}" if decision.score is not None else "n/a",
            len(triggers),
        )

        # JSONL file (when configured)
        if self._log_path is not None:
            line = json.dumps(record, separators=(",", ":")) + "\n"
            with self._lock, self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def recent(self, n: int = 50) -> list[dict]:
        """Return the last n decision records from the JSONL file.

        Args:
            n: Maximum number of records to return (newest last).

        Returns:
            List of decision record dicts. Empty if no log file configured
            or the file does not yet exist.
        """
        if self._log_path is None or not self._log_path.exists():
            return []
        with self._lock:
            lines = self._log_path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    records.append(json.loads(line))
        return records
