"""
Order Book Depth Analysis v2.0 — Advanced Order Flow Intelligence.
==================================================================
Uses CCXT fetch_order_book() from the existing exchange connection.

v2.0 enhancements:
  - VPIN (Volume-Synchronized Probability of Informed Trading)
  - Cumulative Volume Delta (CVD) with divergence detection
  - Spoofing / fake wall detection (wall persistence tracking)
  - Absorption detection (large order filled without price move)
  - All original features preserved (multi-level imbalance, walls, VWMP)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from config import Config

_log = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────


@dataclass
class VolumeBar:
    """A single volume-synchronized bar for VPIN calculation."""

    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    start_price: float = 0.0
    end_price: float = 0.0
    timestamp: float = 0.0


@dataclass
class WallSnapshot:
    """Snapshot of a detected wall for persistence tracking."""

    price: float
    volume: float
    side: str  # "bid" or "ask"
    first_seen: float = 0.0
    last_seen: float = 0.0
    times_seen: int = 0
    volume_history: list = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def is_persistent(self) -> bool:
        return self.times_seen >= 3 and self.age_seconds >= 60


# ── Main Analyzer ─────────────────────────────────────────────────


class OrderBookAnalyzer:
    """
    Advanced order flow intelligence with VPIN, CVD, spoofing detection,
    and absorption analysis layered on top of traditional depth signals.
    """

    CACHE_TTL = 30  # 30s cache

    # VPIN parameters
    VPIN_BUCKET_SIZE = 50  # Volume bars per bucket (units of base asset)
    VPIN_N_BUCKETS = 50  # Rolling window of buckets for VPIN
    VPIN_TOXIC_THRESHOLD = 0.7  # VPIN > this = toxic flow

    # CVD parameters
    CVD_LOOKBACK = 200  # Bars of CVD history
    CVD_DIVERGENCE_BARS = 20  # Bars to check for divergence

    # Spoofing detection
    WALL_TRACK_WINDOW = 300  # 5 min of wall tracking
    SPOOF_THRESHOLD = 3  # Must see wall >= N times then vanish
    PRICE_PROXIMITY_PCT = 0.3  # Wall vanishes when price within 0.3%

    # Absorption detection
    ABSORPTION_VOL_MULT = 3.0  # Volume at level >= 3x average
    ABSORPTION_PRICE_MOVE = 0.05  # Price moves < 0.05% despite absorption

    def __init__(self, exchange: Any = None) -> None:
        self.exchange = exchange
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._spread_history: deque = deque(maxlen=100)

        # VPIN state
        self._volume_bars: deque = deque(maxlen=self.VPIN_N_BUCKETS)
        self._current_bar = VolumeBar()
        self._accumulated_volume = 0.0

        # CVD state
        self._cvd_history: deque = deque(maxlen=self.CVD_LOOKBACK)
        self._price_history: deque = deque(maxlen=self.CVD_LOOKBACK)
        self._cumulative_delta: float = 0.0

        # Wall tracking for spoofing detection
        self._tracked_walls: dict = {}  # key: (side, rounded_price) → WallSnapshot
        self._last_wall_cleanup: float = 0

        # Absorption state
        self._absorbed_levels: deque = deque(maxlen=50)

    # ── Public Interface ──────────────────────────────────────────

    def get_signal(self) -> dict[str, Any]:
        """Analyze order book with advanced flow intelligence."""
        if not Config.ENABLE_ORDERBOOK or self.exchange is None:
            return {"source": "orderbook", "signal": "neutral", "strength": 0.0, "data": {}}

        try:
            analysis = self._analyze_book()
            return {
                "source": "orderbook",
                "signal": analysis["signal"],
                "strength": analysis["strength"],
                "data": analysis,
            }
        except Exception as e:
            _log.warning("Order book analysis failed: %s", e)
            return {"source": "orderbook", "signal": "neutral", "strength": 0.0, "data": {"error": str(e)}}

    def get_vpin(self) -> float:
        """Return current VPIN value (0.0 to 1.0)."""
        return self._compute_vpin()

    def get_cvd(self) -> dict[str, Any]:
        """Return CVD state and divergence info."""
        return {
            "cumulative_delta": round(self._cumulative_delta, 4),
            "divergence": self._detect_cvd_divergence(),
            "history_length": len(self._cvd_history),
        }

    def get_spoofing_alerts(self) -> list[dict]:
        """Return detected spoofing/fake wall events."""
        return self._detect_spoofing()

    def get_absorption_events(self) -> list[dict]:
        """Return recent absorption events."""
        return list(self._absorbed_levels)

    # ── Core Analysis ─────────────────────────────────────────────

    def _analyze_book(self) -> dict[str, Any]:
        """Full order book analysis with flow intelligence."""
        now = time.time()
        if now - self._cache_ts < self.CACHE_TTL and self._cache:
            return self._cache

        book = self.exchange.fetch_order_book(Config.TRADING_PAIR, limit=100)
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            return {"signal": "neutral", "strength": 0.0}

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = spread / mid_price * 100

        self._spread_history.append({"ts": now, "spread_pct": spread_pct})

        # ── Original Analysis ─────────────────────────────────────
        imbalances = self._multi_level_imbalance(bids, asks)
        vwmp, vwmp_skew = self._compute_vwmp(bids, asks, mid_price)
        bid_walls = self._detect_walls(bids, mid_price, side="bid")
        ask_walls = self._detect_walls(asks, mid_price, side="ask")

        # ── New: Flow Intelligence ────────────────────────────────
        # Update VPIN with latest book state
        self._update_volume_bars(bids, asks, mid_price, now)
        vpin = self._compute_vpin()

        # Update CVD
        self._update_cvd(bids, asks, mid_price, now)
        cvd_divergence = self._detect_cvd_divergence()

        # Track walls for spoofing
        self._update_wall_tracking(bid_walls, ask_walls, mid_price, now)
        spoof_alerts = self._detect_spoofing()

        # Check absorption
        absorption = self._detect_absorption(bids, asks, mid_price, now)

        # ── Composite Signal ──────────────────────────────────────
        score = 0.0

        # Factor 1: Top-20 imbalance (20% weight)
        top20_imb = imbalances.get("top_20", {}).get("imbalance", 0)
        score += top20_imb * 0.2

        # Factor 2: VWMP skew (10% weight)
        if vwmp_skew > 0.05:
            score += 0.1
        elif vwmp_skew < -0.05:
            score -= 0.1

        # Factor 3: Wall analysis (10% weight)
        score += self._wall_score(bid_walls, ask_walls) * 0.1

        # Factor 4: VPIN flow toxicity (20% weight) — NEW
        # High VPIN = informed traders active = trend continuation likely
        if vpin > self.VPIN_TOXIC_THRESHOLD:
            # Direction from CVD: positive delta = informed buying
            if self._cumulative_delta > 0:
                score += 0.2 * min(vpin, 1.0)
            else:
                score -= 0.2 * min(vpin, 1.0)

        # Factor 5: CVD divergence (20% weight) — NEW
        if cvd_divergence == "bullish_divergence":
            score += 0.2  # Hidden accumulation
        elif cvd_divergence == "bearish_divergence":
            score -= 0.2  # Hidden distribution

        # Factor 6: Spoofing adjustment (10% weight) — NEW
        for alert in spoof_alerts:
            if alert["side"] == "bid":
                score -= 0.1  # Bid wall was fake → bearish
            elif alert["side"] == "ask":
                score += 0.1  # Ask wall was fake → bullish

        # Factor 7: Absorption signal (10% weight) — NEW
        if absorption:
            latest_abs = absorption[-1]
            if latest_abs["side"] == "bid":
                score += 0.1  # Bid absorption = accumulation = bullish
            else:
                score -= 0.1  # Ask absorption = distribution = bearish

        # Spread quality adjustment
        if len(self._spread_history) >= 5:
            recent_spreads = [h["spread_pct"] for h in list(self._spread_history)[-5:]]
            avg_spread = sum(recent_spreads) / len(recent_spreads)
            if spread_pct > avg_spread * 1.5:
                score *= 0.7

        # Convert to signal
        if score > 0.1:
            signal = "bullish"
            strength = min(abs(score), 0.6)
        elif score < -0.1:
            signal = "bearish"
            strength = min(abs(score), 0.6)
        else:
            signal = "neutral"
            strength = 0.0

        result = {
            "signal": signal,
            "strength": round(strength, 3),
            "mid_price": round(mid_price, 2),
            "spread_pct": round(spread_pct, 4),
            "vwmp": round(vwmp, 2),
            "vwmp_skew_pct": round(vwmp_skew, 4),
            "imbalances": imbalances,
            "bid_walls": bid_walls[:3],
            "ask_walls": ask_walls[:3],
            "composite_score": round(score, 4),
            # New v2.0 fields
            "vpin": round(vpin, 4),
            "vpin_toxic": vpin > self.VPIN_TOXIC_THRESHOLD,
            "cvd": round(self._cumulative_delta, 4),
            "cvd_divergence": cvd_divergence,
            "spoof_alerts": spoof_alerts,
            "absorption_events": absorption,
        }

        self._cache = result
        self._cache_ts = now
        return result

    # ── Multi-Level Imbalance ─────────────────────────────────────

    def _multi_level_imbalance(self, bids: list, asks: list) -> dict:
        """Compute bid/ask imbalance at multiple depth levels."""
        levels = [5, 10, 20, 50]
        imbalances = {}
        for n in levels:
            bid_vol = sum(b[1] for b in bids[:n]) if len(bids) >= n else sum(b[1] for b in bids)
            ask_vol = sum(a[1] for a in asks[:n]) if len(asks) >= n else sum(a[1] for a in asks)
            total = bid_vol + ask_vol
            if total > 0:
                imbalances[f"top_{n}"] = {
                    "bid_volume": round(bid_vol, 4),
                    "ask_volume": round(ask_vol, 4),
                    "imbalance": round((bid_vol - ask_vol) / total, 4),
                }
        return imbalances

    # ── VWMP ──────────────────────────────────────────────────────

    def _compute_vwmp(self, bids: list, asks: list, mid_price: float) -> tuple[float, float]:
        """Compute volume-weighted mid price and its skew."""
        bid_vw = sum(b[0] * b[1] for b in bids[:20])
        bid_vol = sum(b[1] for b in bids[:20])
        ask_vw = sum(a[0] * a[1] for a in asks[:20])
        ask_vol = sum(a[1] for a in asks[:20])

        total_vol = bid_vol + ask_vol
        vwmp = (bid_vw + ask_vw) / total_vol if total_vol > 0 else mid_price
        skew = (vwmp - mid_price) / mid_price * 100
        return vwmp, skew

    # ── Wall Detection ────────────────────────────────────────────

    def _detect_walls(
        self, orders: list, mid_price: float, side: str, threshold_mult: float = 5.0, price_range_pct: float = 2.0
    ) -> list[dict]:
        """Detect large order walls near mid price."""
        if len(orders) < 10:
            return []

        max_dist = mid_price * price_range_pct / 100
        nearby = [o for o in orders if abs(o[0] - mid_price) <= max_dist]

        if len(nearby) < 5:
            return []

        avg_vol = sum(o[1] for o in nearby) / len(nearby)
        threshold = avg_vol * threshold_mult

        walls = []
        for price, volume in nearby:
            if volume >= threshold:
                dist_pct = abs(price - mid_price) / mid_price * 100
                walls.append(
                    {
                        "price": round(price, 2),
                        "volume": round(volume, 4),
                        "distance_pct": round(dist_pct, 3),
                        "multiple_of_avg": round(volume / avg_vol, 1),
                        "side": side,
                    }
                )

        walls.sort(key=lambda w: w["volume"], reverse=True)
        return walls[:5]

    def _wall_score(self, bid_walls: list, ask_walls: list) -> float:
        """Compute directional score from wall analysis."""
        if bid_walls and not ask_walls:
            return 1.0
        elif ask_walls and not bid_walls:
            return -1.0
        elif bid_walls and ask_walls:
            bid_vol = sum(w["volume"] for w in bid_walls)
            ask_vol = sum(w["volume"] for w in ask_walls)
            if bid_vol > ask_vol * 1.5:
                return 0.5
            elif ask_vol > bid_vol * 1.5:
                return -0.5
        return 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW v2.0: VPIN (Volume-Synchronized Probability of Informed Trading)
    # ══════════════════════════════════════════════════════════════

    def _update_volume_bars(self, bids: list, asks: list, mid_price: float, now: float) -> None:
        """
        Update VPIN volume bars from order book snapshot.

        VPIN uses volume-synchronized bars rather than time bars.
        We estimate buy/sell volume from order book pressure since
        we don't have actual trade-by-trade data via REST API.
        """
        if not bids or not asks:
            return

        # Estimate buy vs sell pressure from book imbalance
        bid_vol = sum(b[1] for b in bids[:10])
        ask_vol = sum(a[1] for a in asks[:10])
        total = bid_vol + ask_vol

        if total <= 0:
            return

        # Approximate trade flow using book imbalance as proxy
        buy_fraction = bid_vol / total
        # Counter-intuitive: heavy bid side means more sell aggression is needed
        # to move price, so we use ask-side as buy proxy
        # But in practice, bid-heavy book = buyers willing to pay = bullish flow
        estimated_volume = total * 0.01  # Scale factor for REST-based estimation

        buy_vol = estimated_volume * buy_fraction
        sell_vol = estimated_volume * (1 - buy_fraction)

        self._current_bar.buy_volume += buy_vol
        self._current_bar.sell_volume += sell_vol
        self._current_bar.total_volume += estimated_volume
        self._current_bar.end_price = mid_price
        if self._current_bar.start_price == 0:
            self._current_bar.start_price = mid_price
            self._current_bar.timestamp = now

        # When bucket fills, push to history
        if self._current_bar.total_volume >= self.VPIN_BUCKET_SIZE:
            self._volume_bars.append(self._current_bar)
            self._current_bar = VolumeBar()

    def _compute_vpin(self) -> float:
        """
        Compute VPIN from volume bars.

        VPIN = (1/n) × Σ |V_buy - V_sell| / (V_buy + V_sell)

        Higher VPIN means more order flow imbalance, suggesting
        informed traders are active (one-sided flow).
        """
        bars = list(self._volume_bars)
        if len(bars) < 5:
            return 0.0

        total_imbalance = 0.0
        total_volume = 0.0

        for bar in bars:
            if bar.total_volume > 0:
                total_imbalance += abs(bar.buy_volume - bar.sell_volume)
                total_volume += bar.total_volume

        if total_volume <= 0:
            return 0.0

        vpin = total_imbalance / total_volume
        return min(vpin, 1.0)

    # ══════════════════════════════════════════════════════════════
    # NEW v2.0: Cumulative Volume Delta (CVD)
    # ══════════════════════════════════════════════════════════════

    def _update_cvd(self, bids: list, asks: list, mid_price: float, now: float) -> None:
        """
        Update cumulative volume delta.

        CVD = running sum of (aggressive_buy_volume - aggressive_sell_volume)
        Rising CVD = net buying pressure
        Falling CVD = net selling pressure
        """
        if not bids or not asks:
            return

        # Estimate aggressive buy vs sell from top-of-book
        # Aggressive buys hit the ask, aggressive sells hit the bid
        # Proxy: compare volume clustering on each side
        bid_tight = sum(b[1] for b in bids[:3])  # Tight bid volume
        ask_tight = sum(a[1] for a in asks[:3])  # Tight ask volume
        bid_deep = sum(b[1] for b in bids[3:15])  # Deep bid volume
        ask_deep = sum(a[1] for a in asks[3:15])  # Deep ask volume

        # Ratio of tight to deep gives aggression estimate
        # Heavy tight-ask vs deep-ask means aggressive selling (hitting bids)
        bid_aggression = bid_tight / (bid_deep + 1e-10)
        ask_aggression = ask_tight / (ask_deep + 1e-10)

        # Delta for this snapshot
        delta = (bid_aggression - ask_aggression) * 0.1  # Scaled
        self._cumulative_delta += delta

        self._cvd_history.append(round(self._cumulative_delta, 6))
        self._price_history.append(mid_price)

    def _detect_cvd_divergence(self) -> str:
        """
        Detect divergence between CVD and price.

        Bullish divergence: Price falling but CVD rising (hidden accumulation)
        Bearish divergence: Price rising but CVD falling (hidden distribution)
        """
        if len(self._cvd_history) < self.CVD_DIVERGENCE_BARS:
            return "none"

        recent_cvd = list(self._cvd_history)[-self.CVD_DIVERGENCE_BARS :]
        recent_price = list(self._price_history)[-self.CVD_DIVERGENCE_BARS :]

        if len(recent_cvd) < 2 or len(recent_price) < 2:
            return "none"

        # Simple trend: compare first half avg vs second half avg
        half = len(recent_cvd) // 2
        cvd_early = sum(recent_cvd[:half]) / half
        cvd_late = sum(recent_cvd[half:]) / (len(recent_cvd) - half)
        price_early = sum(recent_price[:half]) / half
        price_late = sum(recent_price[half:]) / (len(recent_price) - half)

        cvd_trend = cvd_late - cvd_early
        price_trend = (price_late - price_early) / price_early if price_early > 0 else 0

        # Divergence: trends move in opposite directions
        if cvd_trend > 0.01 and price_trend < -0.001:
            return "bullish_divergence"
        elif cvd_trend < -0.01 and price_trend > 0.001:
            return "bearish_divergence"
        elif cvd_trend > 0.01 and price_trend > 0.001:
            return "bullish_confirmation"
        elif cvd_trend < -0.01 and price_trend < -0.001:
            return "bearish_confirmation"

        return "none"

    # ══════════════════════════════════════════════════════════════
    # NEW v2.0: Spoofing / Fake Wall Detection
    # ══════════════════════════════════════════════════════════════

    def _update_wall_tracking(self, bid_walls: list, ask_walls: list, mid_price: float, now: float) -> None:
        """
        Track wall persistence across snapshots.

        Spoofing signature:
        1. Large wall appears at level L
        2. Wall persists for several snapshots (building trust)
        3. Price approaches wall
        4. Wall vanishes before being filled
        """
        # Cleanup old tracked walls
        if now - self._last_wall_cleanup > 60:
            stale = [k for k, v in self._tracked_walls.items() if now - v.last_seen > self.WALL_TRACK_WINDOW]
            for k in stale:
                del self._tracked_walls[k]
            self._last_wall_cleanup = now

        # Update tracking for current walls
        all_walls = bid_walls + ask_walls
        seen_keys = set()

        for wall in all_walls:
            price_key = round(wall["price"], 0)  # Round to nearest dollar
            key = (wall["side"], price_key)
            seen_keys.add(key)

            if key in self._tracked_walls:
                tracked = self._tracked_walls[key]
                tracked.last_seen = now
                tracked.times_seen += 1
                tracked.volume = wall["volume"]
                tracked.volume_history.append(wall["volume"])
                if len(tracked.volume_history) > 20:
                    tracked.volume_history = tracked.volume_history[-20:]
            else:
                self._tracked_walls[key] = WallSnapshot(
                    price=wall["price"],
                    volume=wall["volume"],
                    side=wall["side"],
                    first_seen=now,
                    last_seen=now,
                    times_seen=1,
                    volume_history=[wall["volume"]],
                )

    def _detect_spoofing(self) -> list[dict]:
        """
        Detect potential spoofing events.

        A spoof is flagged when:
        - Wall was seen >= SPOOF_THRESHOLD times (established itself)
        - Wall has now disappeared
        - Price was approaching the wall level
        """
        now = time.time()
        alerts = []

        for key, wall in list(self._tracked_walls.items()):
            # Wall must have been persistent but now gone
            age = now - wall.last_seen
            if age < 30 or age > self.WALL_TRACK_WINDOW:
                continue
            if wall.times_seen < self.SPOOF_THRESHOLD:
                continue

            # Check if wall was near current price when it vanished
            if self._price_history:
                current_price = self._price_history[-1]
                dist_pct = abs(wall.price - current_price) / current_price * 100
                if dist_pct < self.PRICE_PROXIMITY_PCT * 3:
                    alerts.append(
                        {
                            "type": "spoof_detected",
                            "side": wall.side,
                            "price": round(wall.price, 2),
                            "peak_volume": round(max(wall.volume_history), 4),
                            "times_seen": wall.times_seen,
                            "age_seconds": round(wall.age_seconds, 0),
                            "vanished_seconds_ago": round(age, 0),
                        }
                    )
                    # Remove after alerting
                    del self._tracked_walls[key]

        return alerts

    # ══════════════════════════════════════════════════════════════
    # NEW v2.0: Absorption Detection
    # ══════════════════════════════════════════════════════════════

    def _detect_absorption(self, bids: list, asks: list, mid_price: float, now: float) -> list[dict]:
        """
        Detect absorption events.

        Absorption = large volume traded at a price level without
        the price moving through that level. This means a major player
        is accumulating (bid absorption) or distributing (ask absorption).

        Signature:
        - A wall-sized order sits at a level
        - Volume at that level is >> average (orders being filled into it)
        - Price stays at or near that level (wall absorbs the flow)
        """
        events = []

        if not bids or not asks or not self._price_history:
            return events

        # Check if price has been stable despite heavy volume at a level
        if len(self._price_history) >= 5:
            recent_prices = list(self._price_history)[-5:]
            price_range = max(recent_prices) - min(recent_prices)
            price_move_pct = price_range / mid_price * 100

            if price_move_pct < self.ABSORPTION_PRICE_MOVE:
                # Price is barely moving — check for heavy volume at top levels
                bid_top_vol = bids[0][1] if bids else 0
                ask_top_vol = asks[0][1] if asks else 0

                # Average volume across the book
                all_vols = [b[1] for b in bids[:20]] + [a[1] for a in asks[:20]]
                avg_vol = sum(all_vols) / len(all_vols) if all_vols else 1

                if bid_top_vol > avg_vol * self.ABSORPTION_VOL_MULT:
                    event = {
                        "type": "absorption",
                        "side": "bid",
                        "price": round(bids[0][0], 2),
                        "volume": round(bid_top_vol, 4),
                        "multiple_of_avg": round(bid_top_vol / avg_vol, 1),
                        "price_stability_pct": round(price_move_pct, 4),
                        "timestamp": now,
                        "interpretation": "accumulation",
                    }
                    events.append(event)
                    self._absorbed_levels.append(event)

                if ask_top_vol > avg_vol * self.ABSORPTION_VOL_MULT:
                    event = {
                        "type": "absorption",
                        "side": "ask",
                        "price": round(asks[0][0], 2),
                        "volume": round(ask_top_vol, 4),
                        "multiple_of_avg": round(ask_top_vol / avg_vol, 1),
                        "price_stability_pct": round(price_move_pct, 4),
                        "timestamp": now,
                        "interpretation": "distribution",
                    }
                    events.append(event)
                    self._absorbed_levels.append(event)

        return events
