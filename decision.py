"""
Decision Layer — Core schemas for Context + Trigger architecture.

Three dataclasses define the contract between all system components:
  ContextState  — structural market state, produced by ContextEngine every 15min
  TriggerSignal — entry opportunity, produced by TriggerEngine per-candle/event
  Decision      — trade or reject verdict, produced by evaluate()

These schemas are the system spine. No downstream component mutates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

# --- Literal type aliases (prevents string creep) ---

SwingBias = Literal["bullish", "bearish", "neutral"]
VolatilityRegime = Literal["low", "normal", "elevated", "extreme"]
FundingPressure = Literal[
    "long_crowded_mild", "long_crowded_extreme",
    "short_crowded_mild", "short_crowded_extreme",
    "neutral",
]
WhaleFlow = Literal["accumulating", "distributing", "neutral"]
OITrend = Literal["expanding_up", "expanding_down", "contracting", "neutral"]
RiskMode = Literal["normal", "cautious", "defensive"]

_VALID_SWING_BIAS: set[str] = {"bullish", "bearish", "neutral"}
_VALID_VOL_REGIME: set[str] = {"low", "normal", "elevated", "extreme"}
_VALID_FUNDING: set[str] = {
    "long_crowded_mild", "long_crowded_extreme",
    "short_crowded_mild", "short_crowded_extreme",
    "neutral",
}
_VALID_WHALE: set[str] = {"accumulating", "distributing", "neutral"}
_VALID_OI: set[str] = {"expanding_up", "expanding_down", "contracting", "neutral"}
_VALID_RISK: set[str] = {"normal", "cautious", "defensive"}


@dataclass
class ContextState:
    """Structural market state snapshot produced by ContextEngine every 15 minutes.

    Immutable within its validity window (valid_until). If context changes,
    a new ContextState with a new context_id is created — this one is never mutated.

    Attributes:
        context_id: ISO-8601 string identifying this context window.
        swing_bias: 4h price structure direction.
        allowed_directions: Directions the Decision Layer may trade. Empty list = no trades.
        volatility_regime: Current volatility classification.
        funding_pressure: Perpetual funding rate signal.
        whale_flow: Net whale accumulation/distribution signal.
        oi_trend: Open interest trend direction.
        key_levels: Support, resistance, and point-of-control prices.
        risk_mode: Current operating mode from RiskSupervisor.
        confidence: 0.0-1.0 overall context clarity (informational; used in scoring).
        tradeable: Derived by ContextEngine. Never set by downstream components.
        valid_until: Datetime after which this ContextState is stale.
        updated_at: When this ContextState was produced.
    """

    context_id: str
    swing_bias: SwingBias
    allowed_directions: list[str]
    volatility_regime: VolatilityRegime
    funding_pressure: FundingPressure
    whale_flow: WhaleFlow
    oi_trend: OITrend
    key_levels: dict
    risk_mode: RiskMode
    confidence: float
    tradeable: bool
    valid_until: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.swing_bias not in _VALID_SWING_BIAS:
            raise ValueError(f"Invalid swing_bias: {self.swing_bias!r}")
        if self.volatility_regime not in _VALID_VOL_REGIME:
            raise ValueError(f"Invalid volatility_regime: {self.volatility_regime!r}")
        if self.funding_pressure not in _VALID_FUNDING:
            raise ValueError(f"Invalid funding_pressure: {self.funding_pressure!r}")
        if self.whale_flow not in _VALID_WHALE:
            raise ValueError(f"Invalid whale_flow: {self.whale_flow!r}")
        if self.oi_trend not in _VALID_OI:
            raise ValueError(f"Invalid oi_trend: {self.oi_trend!r}")
        if self.risk_mode not in _VALID_RISK:
            raise ValueError(f"Invalid risk_mode: {self.risk_mode!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")


@dataclass
class TriggerSignal:
    """Entry opportunity produced by TriggerEngine.

    Triggers are observations only — they know nothing about position size,
    instrument choice, or whether a trade will happen.

    Attributes:
        trigger_id: UUID for audit trail.
        source: Which trigger produced this signal.
        direction: "long" or "short".
        strength: 0.0-1.0 signal strength.
        urgency: "normal" (spot) or "high" (event-driven, routes to perp).
        symbol_scope: Symbol this trigger applies to, or "market" for system-wide.
        reason: Human-readable explanation for logging/debugging.
        expires_at: Trigger is stale after this datetime.
        raw_data: Source-specific forensic data for post-mortem analysis.
    """

    trigger_id: str
    source: str
    direction: str
    strength: float
    urgency: str
    symbol_scope: str
    reason: str
    expires_at: datetime
    raw_data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {self.direction!r}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be 0.0-1.0, got {self.strength}")
        if self.urgency not in ("normal", "high"):
            raise ValueError(f"urgency must be 'normal' or 'high', got {self.urgency!r}")

    def is_expired(self) -> bool:
        """Return True if this trigger has passed its expiry time."""
        return datetime.now(UTC) > self.expires_at


@dataclass(frozen=True)
class Decision:
    """Trade or reject verdict produced by evaluate().

    Pure data — no methods, no logic. Loggable and replayable.
    Execution layers never interpret intent beyond action/direction/route.

    Attributes:
        action: "trade" or "reject".
        reason: Human-readable explanation (always set, even for trades).
        direction: "long" | "short" | None (None when action="reject").
        route: "spot" | "perp" | None (None when action="reject").
        score: Combined context+trigger score (None when action="reject").
    """

    action: Literal["trade", "reject"]
    reason: str
    direction: str | None = None
    route: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if self.action == "trade" and (self.direction is None or self.route is None):
            raise ValueError("trade Decision must have direction and route")


# --- Score threshold (module-level constant, not buried in evaluate()) ---
SCORE_THRESHOLD: float = 0.50


def evaluate(context: ContextState, triggers: list[TriggerSignal]) -> Decision:
    """Evaluate context + triggers and produce a trade or reject decision.

    Decision authority hierarchy:
      1. Gate (hard): context must allow trading and have valid directions.
      2. Consensus: 2+ triggers must agree on the same direction.
      3. Score: context.confidence x mean(trigger strengths) >= SCORE_THRESHOLD.
      4. Route: event triggers ("high" urgency) -> perp; others -> spot.
         Event routing is blocked in "defensive" risk_mode.

    Args:
        context: Current ContextState from ContextEngine.
        triggers: List of TriggerSignal from TriggerEngine (may include expired ones).

    Returns:
        Decision with action="trade" or action="reject" plus audit reason.
    """
    # Step 1: Gate — hard stop if context is not tradeable
    if not context.tradeable:
        return Decision(action="reject", reason="context_not_tradeable")
    if not context.allowed_directions:
        return Decision(action="reject", reason="no_allowed_directions")

    # Step 1.5: Funding extreme gate — block trading the crowded side
    effective_allowed = list(context.allowed_directions)
    if context.funding_pressure == "long_crowded_extreme":
        effective_allowed = [d for d in effective_allowed if d != "long"]
    if context.funding_pressure == "short_crowded_extreme":
        effective_allowed = [d for d in effective_allowed if d != "short"]
    if not effective_allowed:
        return Decision(action="reject", reason="funding_extreme_blocks_direction")

    # Step 2: Filter to allowed directions + non-expired triggers
    valid = [
        t for t in triggers
        if t.direction in effective_allowed and not t.is_expired()
    ]

    if not valid:
        return Decision(action="reject", reason="no_valid_triggers")

    # Step 3: Directional consensus — 2+ triggers must agree on same direction
    by_dir: dict[str, list[TriggerSignal]] = {}
    for t in valid:
        by_dir.setdefault(t.direction, []).append(t)

    best_dir, agreeing = max(
        by_dir.items(),
        key=lambda g: (len(g[1]), sum(t.strength for t in g[1])),
    )
    if len(agreeing) < 2:
        return Decision(action="reject", reason="insufficient_directional_agreement")

    # Step 4: Score
    avg_strength = sum(t.strength for t in agreeing) / len(agreeing)
    score = context.confidence * avg_strength
    if score < SCORE_THRESHOLD:
        return Decision(action="reject", reason=f"score_below_threshold:{score:.2f}")

    # Step 5: Route — event triggers go to perp, but defensive mode blocks them
    if any(t.urgency == "high" for t in agreeing):
        if context.risk_mode == "defensive":
            return Decision(action="reject", reason="event_blocked_by_risk_mode")
        route = "perp"
    else:
        route = "spot"

    return Decision(action="trade", direction=best_dir, route=route, score=score, reason="ok")
