"""ContextEngine — produces ContextState every 15 minutes.

Orchestrates the four context analyzers and combines their outputs
into a single versioned ContextState. The RiskSupervisor can push
risk_mode changes via set_risk_mode().
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from context.funding import FundingAnalyzer
from context.oi_trend import OITrendAnalyzer
from context.session import SessionAnalyzer
from context.swing import SwingAnalyzer
from context.volatility import VolatilityAnalyzer
from context.whale_flow import WhaleFlowAnalyzer
from data_snapshot import DataSnapshot
from decision import ContextState

_log = logging.getLogger(__name__)

_CONTEXT_WINDOW_MINUTES = 15
_TRADEABLE_CONFIDENCE_FLOOR = 0.30

_NEUTRAL_FALLBACK: dict = {
    "swing_bias": "neutral",
    "allowed_directions": [],
    "key_levels": {"support": 0.0, "resistance": 0.0, "poc": 0.0},
    "confidence": 0.0,
}


class ContextEngine:
    """Builds ContextState from a DataSnapshot and real-time market data.

    Call build() every 15 minutes (or whenever a new DataSnapshot is available).
    The RiskSupervisor calls set_risk_mode() to inject defensive/cautious states.
    """

    def __init__(self) -> None:
        self._swing = SwingAnalyzer()
        self._funding = FundingAnalyzer()
        self._whale = WhaleFlowAnalyzer()
        self._oi = OITrendAnalyzer()
        self._vol = VolatilityAnalyzer()
        self._session = SessionAnalyzer()
        self._risk_mode: str = "normal"

    def set_risk_mode(self, mode: str) -> None:
        """Called by RiskSupervisor to override the risk_mode field.

        Args:
            mode: One of "normal", "cautious", "defensive".
        """
        self._risk_mode = mode

    def build(
        self,
        snapshot: DataSnapshot,
        funding_rate: float | None,
        net_whale_flow: float | None,
        oi_change_pct: float | None,
        price_change_pct: float | None,
        _now: datetime | None = None,
    ) -> ContextState:
        """Build a ContextState from current market data.

        Args:
            snapshot: Multi-timeframe DataSnapshot (4h used for swing analysis).
            funding_rate: Current 8h perpetual funding rate (decimal).
            net_whale_flow: Net whale USD flow (positive = accumulation).
            oi_change_pct: % change in open interest over last 4h.
            price_change_pct: % change in price over last 4h.
            _now: Override current UTC time (used in tests). Defaults to datetime.now(UTC).

        Returns:
            A new ContextState valid for the next 15 minutes.
        """
        now = _now if _now is not None else datetime.now(UTC)
        context_id = now.strftime("%Y-%m-%dT%H:%MZ")

        # Use 4h data for swing; fall back to 1h if 4h unavailable
        df_swing = snapshot.df_4h if snapshot.df_4h is not None else snapshot.df_1h

        swing = (
            self._swing.analyze(df_swing)
            if df_swing is not None
            else dict(_NEUTRAL_FALLBACK)
        )
        funding = self._funding.analyze(funding_rate)
        whale = self._whale.analyze(net_whale_flow)
        oi = self._oi.analyze(oi_change_pct, price_change_pct)
        vol = self._vol.analyze(snapshot.df_1h)
        session = self._session.analyze(now=now)

        # Scale swing confidence by session liquidity multiplier
        confidence = round(swing["confidence"] * session["confidence_multiplier"], 3)

        allowed_directions = swing["allowed_directions"]
        tradeable = bool(len(allowed_directions) > 0 and confidence >= _TRADEABLE_CONFIDENCE_FLOOR)

        return ContextState(
            context_id=context_id,
            swing_bias=swing["swing_bias"],
            allowed_directions=allowed_directions,
            volatility_regime=vol["volatility_regime"],
            funding_pressure=funding["funding_pressure"],
            whale_flow=whale["whale_flow"],
            oi_trend=oi["oi_trend"],
            key_levels=swing["key_levels"],
            risk_mode=self._risk_mode,
            confidence=confidence,
            tradeable=tradeable,
            valid_until=now + timedelta(minutes=_CONTEXT_WINDOW_MINUTES),
            updated_at=now,
        )
