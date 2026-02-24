"""
Liquidation Cascade Predictor v1.0
====================================
Predicts and exploits liquidation cascades — chain reactions where
forced liquidations push price into more liquidation levels.

Features:
  - Builds liquidation heatmap from open interest + leverage data
  - Computes cascade risk score (0-100)
  - Trading signals: pre-cascade shorts, post-cascade reversal longs
  - Emergency position close when cascade risk > 90%

Key insight: Oct 2025 cascade wiped $9.89B in 14 hours. Those who predicted
the cascade direction and positioned accordingly made fortunes.

Uses Binance public API (no auth required):
  - Open interest data
  - Long/short ratios
  - Funding rates
  - Taker buy/sell volume
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests

_log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

BINANCE_FUTURES = "https://fapi.binance.com"


@dataclass
class LiquidationLevel:
    """Estimated liquidation cluster at a price level."""

    price: float
    side: str  # "long" or "short"
    estimated_value: float  # USD value of liquidations
    leverage: float  # Average leverage at this level
    distance_pct: float  # Distance from current price


@dataclass
class CascadeRisk:
    """Cascade risk assessment for a symbol."""

    symbol: str
    risk_score: float  # 0-100
    direction: str  # "down" (long cascade) or "up" (short squeeze)
    nearest_cluster_pct: float  # Distance to nearest liquidation cluster
    oi_concentration: float  # How concentrated OI is (higher = riskier)
    funding_pressure: float  # Funding rate pressure
    ls_imbalance: float  # Long/short ratio imbalance
    taker_aggression: float  # Net taker buy-sell ratio
    long_levels: list = field(default_factory=list)
    short_levels: list = field(default_factory=list)
    signal: str = "neutral"  # "pre_cascade_short", "post_cascade_long", etc.
    signal_confidence: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize cascade risk assessment to dict.

        Returns:
            Dictionary containing scalar risk metrics suitable for JSON
            serialisation and dashboard display.  Level lists are replaced
            with counts (``n_long_levels``, ``n_short_levels``) to keep the
            payload compact.
        """
        return {
            "symbol": self.symbol,
            "risk_score": round(self.risk_score, 1),
            "direction": self.direction,
            "signal": self.signal,
            "signal_confidence": round(self.signal_confidence, 3),
            "nearest_cluster_pct": round(self.nearest_cluster_pct, 2),
            "oi_concentration": round(self.oi_concentration, 3),
            "funding_pressure": round(self.funding_pressure, 4),
            "ls_imbalance": round(self.ls_imbalance, 3),
            "taker_aggression": round(self.taker_aggression, 3),
            "n_long_levels": len(self.long_levels),
            "n_short_levels": len(self.short_levels),
        }


class CascadePredictor:
    """Predicts liquidation cascades and generates trading signals.

    Uses publicly available Binance futures data to estimate where
    forced liquidations will cluster and when cascades are likely.
    Results are cached for ``CACHE_TTL`` seconds to avoid excessive
    API calls.
    """

    CACHE_TTL = 120  # 2 min cache
    CASCADE_HIGH_RISK = 80  # Score threshold for pre-cascade signals
    CASCADE_EXTREME = 90  # Score threshold for emergency exit

    # Leverage distribution assumptions (based on research)
    LEVERAGE_DIST = [2, 3, 5, 10, 20, 25, 50, 75, 100, 125]

    def __init__(self, symbols: list[str] | None = None) -> None:
        """Initialise the predictor for a set of futures symbols.

        Args:
            symbols: List of Binance futures symbol strings (e.g.
                ``['BTCUSDT', 'ETHUSDT']``).  Defaults to
                ``['BTCUSDT', 'ETHUSDT']`` when ``None``.
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._cache: dict[str, CascadeRisk] = {}
        self._cache_ts: float = 0
        self._oi_history: dict[str, deque] = {s: deque(maxlen=50) for s in self.symbols}
        self._price_history: dict[str, deque] = {s: deque(maxlen=50) for s in self.symbols}

    def get_signal(self) -> dict[str, Any]:
        """Standard intelligence provider interface.

        Returns a cached result when the cache is still fresh; otherwise
        re-analyses all tracked symbols and refreshes the cache.

        Returns:
            Dictionary with keys: ``source``, ``signal``
            (``'bullish'``, ``'bearish'``, or ``'neutral'``), ``strength``
            (0.0–1.0), and ``data`` containing per-symbol cascade risk dicts
            plus ``max_risk_symbol``, ``max_risk_score``, and
            ``cascade_direction``.
        """
        now = time.time()
        if now - self._cache_ts < self.CACHE_TTL and self._cache:
            return self._aggregate_signal()

        risks = {}
        for symbol in self.symbols:
            try:
                risk = self._analyze_symbol(symbol)
                risks[symbol] = risk
            except Exception as e:
                _log.warning(f"Cascade analysis error for {symbol}: {e}")

        self._cache = risks
        self._cache_ts = now
        return self._aggregate_signal()

    def get_emergency_exit_symbols(self) -> list[str]:
        """Return list of symbols where cascade risk is extreme.

        Returns:
            List of symbol strings whose ``risk_score`` exceeds
            ``CASCADE_EXTREME`` (90).  Empty list when no extreme-risk
            symbols are tracked.
        """
        return [symbol for symbol, risk in self._cache.items() if risk.risk_score >= self.CASCADE_EXTREME]

    def get_status(self) -> dict[str, Any]:
        """Return cascade predictor status for dashboard display.

        Returns:
            Dictionary with keys: ``symbols`` (list of tracked symbols),
            ``risks`` (dict mapping symbol → serialised ``CascadeRisk``),
            and ``emergency_symbols`` (list of extreme-risk symbols).
        """
        return {
            "symbols": self.symbols,
            "risks": {s: r.to_dict() for s, r in self._cache.items()},
            "emergency_symbols": self.get_emergency_exit_symbols(),
        }

    def _analyze_symbol(self, symbol: str) -> CascadeRisk:
        """Full cascade risk analysis for one symbol.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            A ``CascadeRisk`` dataclass populated with all computed metrics
            and a trading signal.
        """
        # Fetch data
        oi_data = self._fetch_open_interest(symbol)
        ls_data = self._fetch_long_short_ratio(symbol)
        funding = self._fetch_funding_rate(symbol)
        taker = self._fetch_taker_ratio(symbol)
        price = self._fetch_current_price(symbol)

        if not price:
            return CascadeRisk(
                symbol=symbol,
                risk_score=0,
                direction="neutral",
                nearest_cluster_pct=100,
                oi_concentration=0,
                funding_pressure=0,
                ls_imbalance=0,
                taker_aggression=0,
            )

        # Build liquidation heatmap
        long_levels, short_levels = self._estimate_liquidation_levels(price, oi_data, ls_data)

        # Compute risk components
        oi_concentration = self._compute_oi_concentration(oi_data, symbol)
        funding_pressure = self._compute_funding_pressure(funding)
        ls_imbalance = self._compute_ls_imbalance(ls_data)
        taker_aggression = self._compute_taker_aggression(taker)
        nearest_pct = self._nearest_cluster_distance(price, long_levels, short_levels)

        # Cascade direction: which side has more to liquidate near current price?
        long_risk = sum(lvl.estimated_value for lvl in long_levels if lvl.distance_pct < 5)
        short_risk = sum(lvl.estimated_value for lvl in short_levels if lvl.distance_pct < 5)
        direction = "down" if long_risk > short_risk else "up"

        # Composite risk score (0-100)
        risk_score = self._compute_risk_score(
            oi_concentration, funding_pressure, ls_imbalance, taker_aggression, nearest_pct
        )

        # Generate trading signal
        signal, confidence = self._generate_signal(
            risk_score, direction, oi_concentration, funding_pressure, nearest_pct
        )

        return CascadeRisk(
            symbol=symbol,
            risk_score=risk_score,
            direction=direction,
            nearest_cluster_pct=nearest_pct,
            oi_concentration=oi_concentration,
            funding_pressure=funding_pressure,
            ls_imbalance=ls_imbalance,
            taker_aggression=taker_aggression,
            long_levels=long_levels,
            short_levels=short_levels,
            signal=signal,
            signal_confidence=confidence,
            timestamp=time.time(),
        )

    def _fetch_open_interest(self, symbol: str) -> list[dict]:
        """Fetch OI history from Binance.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            List of OI history dicts from the Binance API, or an empty
            list on failure.
        """
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES}/futures/data/openInterestHist",
                params={"symbol": symbol, "period": "1h", "limit": 24},
                headers=_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
            _log.debug("OI fetch %s returned status %d", symbol, resp.status_code)
        except requests.RequestException as e:
            _log.warning("OI fetch failed for %s: %s", symbol, e)
        except (ValueError, KeyError) as e:
            _log.error("OI parse error for %s: %s", symbol, e)
        return []

    def _fetch_long_short_ratio(self, symbol: str) -> list[dict]:
        """Fetch global long/short account ratio.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            List of long/short ratio history dicts, or an empty list on
            failure.
        """
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES}/futures/data/globalLongShortAccountRatio",
                params={"symbol": symbol, "period": "1h", "limit": 12},
                headers=_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
            _log.debug("L/S ratio fetch %s returned status %d", symbol, resp.status_code)
        except requests.RequestException as e:
            _log.warning("L/S ratio fetch failed for %s: %s", symbol, e)
        except (ValueError, KeyError) as e:
            _log.error("L/S ratio parse error for %s: %s", symbol, e)
        return []

    def _fetch_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            Current funding rate as a float (e.g. 0.0001 = 0.01%).
            Returns 0.0 on failure.
        """
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES}/fapi/v1/premiumIndex",
                params={"symbol": symbol},
                headers=_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                return float(resp.json().get("lastFundingRate", 0))
            _log.debug("Funding rate fetch %s returned status %d", symbol, resp.status_code)
        except requests.RequestException as e:
            _log.warning("Funding rate fetch failed for %s: %s", symbol, e)
        except (ValueError, TypeError) as e:
            _log.error("Funding rate parse error for %s: %s", symbol, e)
        return 0.0

    def _fetch_taker_ratio(self, symbol: str) -> list[dict]:
        """Fetch taker buy/sell volume ratio.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            List of taker ratio history dicts, or an empty list on failure.
        """
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES}/futures/data/takerlongshortRatio",
                params={"symbol": symbol, "period": "1h", "limit": 6},
                headers=_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
            _log.debug("Taker ratio fetch %s returned status %d", symbol, resp.status_code)
        except requests.RequestException as e:
            _log.warning("Taker ratio fetch failed for %s: %s", symbol, e)
        except (ValueError, KeyError) as e:
            _log.error("Taker ratio parse error for %s: %s", symbol, e)
        return []

    def _fetch_current_price(self, symbol: str) -> float:
        """Fetch current price.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            Current mark price as a float, or 0.0 on failure.
        """
        try:
            resp = requests.get(
                f"{BINANCE_FUTURES}/fapi/v1/ticker/price",
                params={"symbol": symbol},
                headers=_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                price = float(resp.json().get("price", 0))
                if symbol not in self._price_history:
                    self._price_history[symbol] = deque(maxlen=50)
                self._price_history[symbol].append(price)
                return price
            _log.debug("Price fetch %s returned status %d", symbol, resp.status_code)
        except requests.RequestException as e:
            _log.warning("Price fetch failed for %s: %s", symbol, e)
        except (ValueError, TypeError) as e:
            _log.error("Price parse error for %s: %s", symbol, e)
        return 0.0

    def _estimate_liquidation_levels(
        self, price: float, oi_data: list, ls_data: list
    ) -> tuple[list[LiquidationLevel], list[LiquidationLevel]]:
        """Estimate liquidation price levels based on OI and leverage.

        Long liquidations occur at: ``entry_price * (1 - 1/leverage)``
        Short liquidations occur at: ``entry_price * (1 + 1/leverage)``

        Args:
            price: Current market price.
            oi_data: Open interest history list from the Binance API.
            ls_data: Long/short ratio history list from the Binance API.

        Returns:
            A tuple of ``(long_levels, short_levels)`` where each element is
            a list of ``LiquidationLevel`` objects sorted by proximity to
            current price.
        """
        if not oi_data:
            return [], []

        latest_oi = float(oi_data[-1].get("sumOpenInterestValue", 0)) if oi_data else 0

        # Estimate long/short split
        long_pct = 0.5
        if ls_data:
            latest_ls = float(ls_data[-1].get("longAccount", 0.5))
            long_pct = latest_ls

        long_oi = latest_oi * long_pct
        short_oi = latest_oi * (1 - long_pct)

        long_levels = []
        short_levels = []

        # Estimate liquidation clusters at various leverage levels
        for leverage in self.LEVERAGE_DIST:
            # Long liquidation price (price drops trigger these)
            liq_price_long = price * (1 - 1.0 / leverage)
            distance_long = (price - liq_price_long) / price * 100

            # Assume OI is distributed across leverage levels
            # Higher leverage = less OI but closer liquidation
            weight = 1.0 / leverage  # Less money at higher leverage
            estimated_value_long = long_oi * weight * 0.2  # Rough distribution

            if distance_long > 0 and distance_long < 50:
                long_levels.append(
                    LiquidationLevel(
                        price=round(liq_price_long, 2),
                        side="long",
                        estimated_value=estimated_value_long,
                        leverage=leverage,
                        distance_pct=round(distance_long, 2),
                    )
                )

            # Short liquidation price (price rises trigger these)
            liq_price_short = price * (1 + 1.0 / leverage)
            distance_short = (liq_price_short - price) / price * 100
            estimated_value_short = short_oi * weight * 0.2

            if distance_short > 0 and distance_short < 50:
                short_levels.append(
                    LiquidationLevel(
                        price=round(liq_price_short, 2),
                        side="short",
                        estimated_value=estimated_value_short,
                        leverage=leverage,
                        distance_pct=round(distance_short, 2),
                    )
                )

        return long_levels, short_levels

    def _compute_oi_concentration(self, oi_data: list, symbol: str) -> float:
        """Compute OI concentration risk.

        High OI + recent increase = risky.

        Args:
            oi_data: Open interest history list from the Binance API.
            symbol: Symbol string (unused in computation, kept for logging).

        Returns:
            Float in [0, 1] where 1.0 = maximum concentration risk
            (>20% OI growth vs recent average).
        """
        if len(oi_data) < 2:
            return 0.0

        oi_values = [float(d.get("sumOpenInterestValue", 0)) for d in oi_data]
        latest = oi_values[-1]
        avg = np.mean(oi_values[:-1]) if len(oi_values) > 1 else latest

        if avg <= 0:
            return 0.0

        # OI growth rate
        growth = (latest - avg) / avg
        # Normalize to 0-1 (>20% growth = score 1.0)
        return min(1.0, max(0.0, growth / 0.2))

    def _compute_funding_pressure(self, funding_rate: float) -> float:
        """Compute signed funding pressure from the funding rate.

        High absolute funding = high leverage = high cascade risk.

        Args:
            funding_rate: Raw funding rate value (e.g. 0.0001 = 0.01%).

        Returns:
            Signed float in [-1, 1].  Positive = longs paying (long cascade
            risk); negative = shorts paying (short squeeze risk).
        """
        # Normalize: 0.1% funding = extreme pressure
        return max(-1.0, min(1.0, funding_rate / 0.001))

    def _compute_ls_imbalance(self, ls_data: list) -> float:
        """Compute long/short ratio imbalance.

        Extreme imbalance = cascade risk.

        Args:
            ls_data: Long/short ratio history list from the Binance API.

        Returns:
            Float in [0, 1] where 1.0 = extreme imbalance.
        """
        if not ls_data:
            return 0.0

        latest = float(ls_data[-1].get("longAccount", 0.5))
        imbalance = abs(latest - 0.5) * 2  # Normalize to 0-1
        return min(1.0, imbalance)

    def _compute_taker_aggression(self, taker_data: list) -> float:
        """Compute net taker aggression from taker buy/sell data.

        High sell aggression + long OI = cascade incoming.

        Args:
            taker_data: Taker ratio history list from the Binance API.

        Returns:
            Float in [-1, 1].  -1 = strong sell aggression,
            +1 = strong buy aggression.
        """
        if not taker_data:
            return 0.0

        ratios = [float(d.get("buySellRatio", 1.0)) for d in taker_data]
        avg_ratio = np.mean(ratios)
        return max(-1.0, min(1.0, avg_ratio - 1.0))

    def _nearest_cluster_distance(
        self, price: float, long_levels: list[LiquidationLevel], short_levels: list[LiquidationLevel]
    ) -> float:
        """Compute distance to nearest liquidation cluster.

        Args:
            price: Current market price (unused directly; distances are
                pre-computed in ``LiquidationLevel.distance_pct``).
            long_levels: List of estimated long liquidation levels.
            short_levels: List of estimated short liquidation levels.

        Returns:
            Minimum distance (%) to the nearest cluster across all levels.
            Returns 100.0 when no levels are available.
        """
        distances = [lvl.distance_pct for lvl in long_levels] + [lvl.distance_pct for lvl in short_levels]
        return min(distances) if distances else 100.0

    def _compute_risk_score(
        self, oi_conc: float, funding: float, ls_imb: float, taker: float, nearest_pct: float
    ) -> float:
        """Compute composite cascade risk score (0–100).

        High score = cascade is imminent.

        Args:
            oi_conc: OI concentration score in [0, 1].
            funding: Funding pressure in [-1, 1].
            ls_imb: Long/short imbalance in [0, 1].
            taker: Net taker aggression in [-1, 1].
            nearest_pct: Distance (%) to nearest liquidation cluster.

        Returns:
            Composite risk score clamped to [0, 100].
        """
        # Proximity risk: closer clusters = higher risk
        proximity_risk = max(0, (10 - nearest_pct) / 10) * 100

        # OI concentration risk
        oi_risk = oi_conc * 100

        # Funding pressure risk
        funding_risk = abs(funding) * 100

        # Imbalance risk
        imbalance_risk = ls_imb * 100

        # Aggression risk (selling into long OI = very dangerous)
        aggression_risk = max(0, -taker) * 100  # Negative taker = sell aggression

        # Weighted composite
        score = (
            proximity_risk * 0.30
            + oi_risk * 0.25
            + funding_risk * 0.20
            + imbalance_risk * 0.15
            + aggression_risk * 0.10
        )

        return min(100, max(0, score))

    def _generate_signal(
        self, risk_score: float, direction: str, oi_conc: float, funding: float, nearest_pct: float
    ) -> tuple[str, float]:
        """Generate trading signal from cascade analysis.

        Args:
            risk_score: Composite risk score in [0, 100].
            direction: Cascade direction, ``'down'`` or ``'up'``.
            oi_conc: OI concentration score in [0, 1].
            funding: Funding pressure in [-1, 1].
            nearest_pct: Distance (%) to nearest liquidation cluster.

        Returns:
            A tuple of ``(signal_name, confidence)`` where ``signal_name``
            is one of ``'pre_cascade_short'``, ``'pre_cascade_long'``,
            ``'cascade_warning_bearish'``, ``'cascade_warning_bullish'``,
            ``'post_cascade_reversal'``, or ``'neutral'``.
        """
        if risk_score >= self.CASCADE_EXTREME:
            # Emergency: very high cascade risk
            if direction == "down":
                return "pre_cascade_short", min(0.9, risk_score / 100)
            else:
                return "pre_cascade_long", min(0.9, risk_score / 100)

        elif risk_score >= self.CASCADE_HIGH_RISK:
            # High risk: position for cascade
            if direction == "down":
                return "cascade_warning_bearish", min(0.7, risk_score / 120)
            else:
                return "cascade_warning_bullish", min(0.7, risk_score / 120)

        elif risk_score < 20 and oi_conc < 0.3:
            # Post-cascade: OI flushed, look for reversal
            return "post_cascade_reversal", 0.4

        return "neutral", 0.0

    def _aggregate_signal(self) -> dict[str, Any]:
        """Aggregate cascade signals across all monitored symbols.

        Returns:
            Standard intelligence provider signal dict with keys:
            ``source``, ``signal``, ``strength``, ``data``.  The ``data``
            sub-dict contains ``max_risk_symbol``, ``max_risk_score``,
            ``cascade_direction``, ``cascade_signal``, and a ``symbols``
            mapping of per-symbol risk serialisations.
        """
        if not self._cache:
            return {
                "source": "CascadePredictor",
                "signal": "neutral",
                "strength": 0.0,
                "data": {},
            }

        max_risk = max(self._cache.values(), key=lambda r: r.risk_score)

        if max_risk.risk_score >= self.CASCADE_HIGH_RISK:
            if max_risk.direction == "down":
                signal = "bearish"
            else:
                signal = "bullish"
            strength = min(1.0, max_risk.risk_score / 100)
        elif max_risk.signal == "post_cascade_reversal":
            signal = "bullish"
            strength = 0.3
        else:
            signal = "neutral"
            strength = 0.0

        return {
            "source": "CascadePredictor",
            "signal": signal,
            "strength": round(strength, 3),
            "data": {
                "max_risk_symbol": max_risk.symbol,
                "max_risk_score": round(max_risk.risk_score, 1),
                "cascade_direction": max_risk.direction,
                "cascade_signal": max_risk.signal,
                "symbols": {s: r.to_dict() for s, r in self._cache.items()},
            },
        }
