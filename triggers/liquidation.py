"""LiquidationTrigger — detects liquidation cascades and fires perp event signals.

Large liquidation volumes create momentum in the opposite direction:
  - Long liquidations → bearish cascade → short signal
  - Short liquidations → short squeeze → long signal
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_SIGNAL_TTL_MINUTES = 15
_MIN_LIQ_USD = 10_000_000  # $10M minimum to be significant


class LiquidationTrigger:
    """Fires high-urgency TriggerSignals on significant liquidation events.

    Args:
        symbol: Trading pair. BTC cascades set ``symbol_scope`` to ``"market"``
            since BTC liquidations affect the whole market.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._base = symbol.split("/")[0]

    def evaluate(self, liq_data: dict[str, Any] | None) -> list[TriggerSignal]:
        """Evaluate liquidation data and return cascade triggers.

        Args:
            liq_data: Dict with keys ``liq_volume_usd`` (float) and
                ``direction`` (``"long"`` = longs being liquidated = bearish;
                ``"short"`` = bullish). ``None`` returns an empty list.

        Returns:
            List of high-urgency TriggerSignals; empty if below threshold.
        """
        if liq_data is None:
            return []

        volume = liq_data.get("liq_volume_usd", 0)
        direction = liq_data.get("direction", "")

        if volume < _MIN_LIQ_USD or direction not in ("long", "short"):
            return []

        # Long liquidations = bearish signal; short liquidations = bullish
        signal_direction = "short" if direction == "long" else "long"
        strength = min(0.6 + (volume / 100_000_000) * 0.3, 0.95)

        # BTC liquidation cascades are market-wide
        scope = "market" if self._base == "BTC" else self._base

        return [
            TriggerSignal(
                trigger_id=str(uuid.uuid4()),
                source="liquidation",
                direction=signal_direction,
                strength=strength,
                urgency="high",
                symbol_scope=scope,
                reason=f"${volume / 1e6:.0f}M {direction} liquidation cascade",
                expires_at=datetime.now(UTC) + timedelta(minutes=_SIGNAL_TTL_MINUTES),
                raw_data={"liq_volume_usd": volume, "liquidated_direction": direction},
            )
        ]
