"""FundingExtremeTrigger — fires when funding rate reaches contrarian extremes.

Extreme funding = crowded trade = mean-reversion setup:
  - Rate > 0.10%/8h (long crowded) → fade the crowd → short signal
  - Rate < -0.05%/8h (short crowded) → fade the crowd → long signal
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_SIGNAL_TTL_MINUTES = 60
_LONG_EXTREME = 0.0010    # 0.10% per 8h — longs are crowded
_SHORT_EXTREME = -0.0005  # -0.05% per 8h — shorts are crowded


class FundingExtremeTrigger:
    """Fires high-urgency TriggerSignals when funding reaches contrarian extremes.

    Args:
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]

    def evaluate(self, funding_rate: float | None) -> list[TriggerSignal]:
        """Evaluate current funding rate and return extreme signals.

        Args:
            funding_rate: Current 8h funding rate as a decimal (e.g. ``0.0001``
                for 0.01%). ``None`` returns an empty list.

        Returns:
            List of high-urgency TriggerSignals; empty if not at extreme.
        """
        if funding_rate is None:
            return []

        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)

        if funding_rate >= _LONG_EXTREME:
            strength = min(0.55 + (funding_rate - _LONG_EXTREME) * 100, 0.90)
            return [
                TriggerSignal(
                    trigger_id=str(uuid.uuid4()),
                    source="funding_extreme",
                    direction="short",
                    strength=strength,
                    urgency="high",
                    symbol_scope=self._symbol_scope,
                    reason=(
                        f"Funding extreme long {funding_rate * 100:.3f}%/8h"
                        " — fade crowd"
                    ),
                    expires_at=expiry,
                    raw_data={"funding_rate": funding_rate},
                )
            ]

        if funding_rate <= _SHORT_EXTREME:
            strength = min(0.55 + abs(funding_rate - _SHORT_EXTREME) * 100, 0.90)
            return [
                TriggerSignal(
                    trigger_id=str(uuid.uuid4()),
                    source="funding_extreme",
                    direction="long",
                    strength=strength,
                    urgency="high",
                    symbol_scope=self._symbol_scope,
                    reason=(
                        f"Funding extreme short {funding_rate * 100:.3f}%/8h"
                        " — fade crowd"
                    ),
                    expires_at=expiry,
                    raw_data={"funding_rate": funding_rate},
                )
            ]

        return []
