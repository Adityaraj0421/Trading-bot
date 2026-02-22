"""
On-Chain Analytics v2.0 — Enhanced with ML Feature Extraction.
================================================================
Sources:
  - Blockchain.com free API (hash rate, difficulty, tx volume, mempool)
  - Mempool.space API (fee estimates, mempool stats)
All free, no API keys needed.

v2.0 enhancements:
  - ML-compatible feature extraction (normalized 0-1 features)
  - Exchange flow proxy features (mempool-based)
  - Active address momentum (7d/30d rolling)
  - Fee pressure as volatility predictor
  - Historical metric storage for trend computation
  - All original signal generation preserved

Produces bullish/bearish/neutral signal with strength 0.0 - 0.5.
"""

import time
import logging
import numpy as np
import requests
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from config import Config

_log = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────

@dataclass
class OnChainFeatures:
    """ML-compatible on-chain feature vector."""
    hash_rate_trend: float = 0.0       # -1 to +1, momentum of mining power
    mempool_pressure: float = 0.5      # 0 = empty, 1 = congested
    fee_pressure: float = 0.5          # 0 = low fees, 1 = extreme fees
    tx_volume_momentum: float = 0.0    # -1 to +1, tx count trend
    network_activity: float = 0.5      # 0 = quiet, 1 = very active
    miner_revenue_trend: float = 0.0   # Proxy for miner selling pressure
    fee_volatility: float = 0.0        # Rapid fee changes = incoming vol
    mempool_clearing_rate: float = 0.5 # How fast mempool clears

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model input."""
        return np.array([
            self.hash_rate_trend,
            self.mempool_pressure,
            self.fee_pressure,
            self.tx_volume_momentum,
            self.network_activity,
            self.miner_revenue_trend,
            self.fee_volatility,
            self.mempool_clearing_rate,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        """Return ordered feature names for model registration."""
        return [
            "onchain_hash_rate_trend",
            "onchain_mempool_pressure",
            "onchain_fee_pressure",
            "onchain_tx_volume_momentum",
            "onchain_network_activity",
            "onchain_miner_revenue_trend",
            "onchain_fee_volatility",
            "onchain_mempool_clearing_rate",
        ]


# ── On-Chain Analyzer ─────────────────────────────────────────────

class OnChainAnalyzer:
    """
    v2.0: Fetches Bitcoin on-chain metrics for both signal generation
    and ML feature extraction.
    """

    BLOCKCHAIN_API = "https://api.blockchain.info"
    MEMPOOL_API = "https://mempool.space/api"
    CACHE_TTL = 300  # 5 min cache

    # History for trend computation
    MAX_HISTORY = 50  # ~4 hours of 5-min readings

    def __init__(self) -> None:
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._prev_metrics: dict = {}

        # v2.0: Historical metrics for ML features
        self._hash_rate_history: deque = deque(maxlen=self.MAX_HISTORY)
        self._tx_count_history: deque = deque(maxlen=self.MAX_HISTORY)
        self._fee_history: deque = deque(maxlen=self.MAX_HISTORY)
        self._mempool_history: deque = deque(maxlen=self.MAX_HISTORY)
        self._last_features: OnChainFeatures = OnChainFeatures()

    # ── Public Interface ──────────────────────────────────────────

    def get_signal(self) -> dict[str, Any]:
        """Original signal generation interface."""
        if not Config.ENABLE_ONCHAIN:
            return {"source": "onchain", "signal": "neutral",
                    "strength": 0.0, "data": {}}

        try:
            metrics = self._fetch_all_metrics()
            signal, strength, analysis = self._analyze(metrics)

            # v2.0: Update ML features
            self._update_features(metrics)

            self._prev_metrics = metrics
            return {
                "source": "onchain",
                "signal": signal,
                "strength": round(strength, 3),
                "data": {**metrics, "analysis": analysis},
            }
        except Exception as e:
            _log.warning("OnChain analysis failed: %s", e)
            return {"source": "onchain", "signal": "neutral",
                    "strength": 0.0, "data": {"error": str(e)}}

    def get_ml_features(self) -> OnChainFeatures:
        """
        v2.0: Return normalized on-chain features for ML model.

        Features are updated each time get_signal() is called.
        Safe to call without network fetch — returns last computed features.
        """
        return self._last_features

    def get_feature_array(self) -> np.ndarray:
        """v2.0: Return features as numpy array for model input."""
        return self._last_features.to_array()

    def get_feature_names(self) -> list[str]:
        """v2.0: Return feature names for model registration."""
        return OnChainFeatures.feature_names()

    # ── v2.0: ML Feature Extraction ───────────────────────────────

    def _update_features(self, metrics: dict) -> None:
        """Compute normalized ML features from raw metrics."""

        # 1. Hash Rate Trend (-1 to +1)
        hr = metrics.get("hash_rate", 0)
        if isinstance(hr, (int, float)) and hr > 0:
            self._hash_rate_history.append(hr)
        hash_trend = self._compute_momentum(self._hash_rate_history)

        # 2. Mempool Pressure (0 to 1)
        unconfirmed = metrics.get("unconfirmed_count", 0)
        if isinstance(unconfirmed, (int, float)):
            self._mempool_history.append(unconfirmed)
        # Normalize: 0 at 0, 1 at 200K+ unconfirmed
        mempool_pressure = min(unconfirmed / 200000, 1.0) if isinstance(unconfirmed, (int, float)) else 0.5

        # 3. Fee Pressure (0 to 1)
        fastest_fee = metrics.get("fee_fastest", 0)
        if isinstance(fastest_fee, (int, float)):
            self._fee_history.append(fastest_fee)
        # Normalize: 0 at 1 sat/vB, 1 at 300+ sat/vB (log scale)
        if isinstance(fastest_fee, (int, float)) and fastest_fee > 0:
            fee_pressure = min(np.log1p(fastest_fee) / np.log1p(300), 1.0)
        else:
            fee_pressure = 0.5

        # 4. TX Volume Momentum (-1 to +1)
        tx_count = metrics.get("tx_count_24h", 0)
        if isinstance(tx_count, (int, float)) and tx_count > 0:
            self._tx_count_history.append(tx_count)
        tx_momentum = self._compute_momentum(self._tx_count_history)

        # 5. Network Activity (0 to 1)
        # Composite: tx count + mempool activity
        if isinstance(tx_count, (int, float)) and tx_count > 0:
            # BTC typically does 250K-400K txns/day
            tx_activity = min(tx_count / 400000, 1.0)
        else:
            tx_activity = 0.5
        mempool_tx = metrics.get("mempool_tx_count", 0)
        if isinstance(mempool_tx, (int, float)) and mempool_tx > 0:
            mempool_activity = min(mempool_tx / 100000, 1.0)
        else:
            mempool_activity = 0.5
        network_activity = 0.6 * tx_activity + 0.4 * mempool_activity

        # 6. Miner Revenue Trend (proxy)
        # Use hash rate trend as proxy — rising hash rate means miners
        # are profitable, less likely to sell
        miner_revenue_trend = hash_trend * 0.7

        # 7. Fee Volatility
        fee_vol = self._compute_volatility(self._fee_history)

        # 8. Mempool Clearing Rate
        # How fast is the mempool draining? Fast = network healthy
        if len(self._mempool_history) >= 3:
            recent = list(self._mempool_history)[-3:]
            if recent[0] > 0:
                clearing = 1 - (recent[-1] / recent[0])
                clearing_rate = max(0, min(clearing + 0.5, 1.0))
            else:
                clearing_rate = 0.5
        else:
            clearing_rate = 0.5

        self._last_features = OnChainFeatures(
            hash_rate_trend=round(hash_trend, 4),
            mempool_pressure=round(mempool_pressure, 4),
            fee_pressure=round(fee_pressure, 4),
            tx_volume_momentum=round(tx_momentum, 4),
            network_activity=round(network_activity, 4),
            miner_revenue_trend=round(miner_revenue_trend, 4),
            fee_volatility=round(fee_vol, 4),
            mempool_clearing_rate=round(clearing_rate, 4),
        )

    def _compute_momentum(self, history: deque, short: int = 5,
                          long: int = 20) -> float:
        """
        Compute momentum as (short MA - long MA) / long MA.
        Returns value in [-1, 1].
        """
        data = list(history)
        if len(data) < 3:
            return 0.0

        short_data = data[-min(short, len(data)):]
        long_data = data[-min(long, len(data)):]

        short_avg = sum(short_data) / len(short_data)
        long_avg = sum(long_data) / len(long_data)

        if long_avg <= 0:
            return 0.0

        momentum = (short_avg - long_avg) / long_avg
        return max(-1.0, min(1.0, momentum * 10))  # Scale and clamp

    def _compute_volatility(self, history: deque) -> float:
        """Compute normalized volatility of a metric."""
        data = list(history)
        if len(data) < 5:
            return 0.0

        recent = data[-10:]
        if max(recent) <= 0:
            return 0.0

        mean = sum(recent) / len(recent)
        if mean <= 0:
            return 0.0

        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        cv = (variance ** 0.5) / mean  # Coefficient of variation
        return min(cv, 1.0)

    # ── Original Analysis (preserved) ─────────────────────────────

    def _fetch_all_metrics(self) -> dict[str, Any]:
        """Fetch all on-chain metrics, using cache if fresh."""
        now = time.time()
        if now - self._cache_ts < self.CACHE_TTL and self._cache:
            return self._cache

        metrics = {}

        metrics["hash_rate"] = self._fetch_json(
            f"{self.BLOCKCHAIN_API}/q/hashrate", parse_text=True)
        metrics["difficulty"] = self._fetch_json(
            f"{self.BLOCKCHAIN_API}/q/getdifficulty", parse_text=True)
        metrics["tx_count_24h"] = self._fetch_json(
            f"{self.BLOCKCHAIN_API}/q/24hrtransactioncount", parse_text=True)
        metrics["unconfirmed_count"] = self._fetch_json(
            f"{self.BLOCKCHAIN_API}/q/unconfirmedcount", parse_text=True)

        mempool_stats = self._fetch_json(f"{self.MEMPOOL_API}/mempool")
        if isinstance(mempool_stats, dict):
            metrics["mempool_size_bytes"] = mempool_stats.get("vsize", 0)
            metrics["mempool_tx_count"] = mempool_stats.get("count", 0)

        fee_estimates = self._fetch_json(
            f"{self.MEMPOOL_API}/v1/fees/recommended")
        if isinstance(fee_estimates, dict):
            metrics["fee_fastest"] = fee_estimates.get("fastestFee", 0)
            metrics["fee_half_hour"] = fee_estimates.get("halfHourFee", 0)
            metrics["fee_hour"] = fee_estimates.get("hourFee", 0)
            metrics["fee_economy"] = fee_estimates.get("economyFee", 0)

        block_height = self._fetch_json(
            f"{self.BLOCKCHAIN_API}/q/getblockcount", parse_text=True)
        metrics["block_height"] = block_height

        self._cache = metrics
        self._cache_ts = now
        return metrics

    def _analyze(self, metrics: dict) -> tuple[str, float, dict[str, Any]]:
        """Analyze on-chain metrics and produce a signal."""
        scores = []
        analysis = {}

        # Hash rate score
        hr = metrics.get("hash_rate", 0)
        prev_hr = self._prev_metrics.get("hash_rate", 0)
        if hr > 0 and prev_hr > 0:
            hr_change = (hr - prev_hr) / prev_hr
            if hr_change > 0.01:
                scores.append(0.3)
                analysis["hash_rate"] = "rising (bullish - miners investing)"
            elif hr_change < -0.02:
                scores.append(-0.3)
                analysis["hash_rate"] = "falling (bearish - miners leaving)"
            else:
                scores.append(0.05)
                analysis["hash_rate"] = "stable"
        elif hr > 0:
            scores.append(0.1)
            analysis["hash_rate"] = "first reading"

        # Mempool congestion score
        unconfirmed = metrics.get("unconfirmed_count", 0)
        if isinstance(unconfirmed, (int, float)):
            if unconfirmed > 150000:
                scores.append(-0.25)
                analysis["mempool"] = f"congested ({unconfirmed:,} unconfirmed)"
            elif unconfirmed > 80000:
                scores.append(-0.1)
                analysis["mempool"] = f"busy ({unconfirmed:,} unconfirmed)"
            elif unconfirmed > 20000:
                scores.append(0.1)
                analysis["mempool"] = f"healthy ({unconfirmed:,} unconfirmed)"
            else:
                scores.append(0.0)
                analysis["mempool"] = f"quiet ({unconfirmed:,} unconfirmed)"

        # Fee level score
        fastest_fee = metrics.get("fee_fastest", 0)
        if isinstance(fastest_fee, (int, float)) and fastest_fee > 0:
            if fastest_fee > 200:
                scores.append(-0.2)
                analysis["fees"] = f"very high ({fastest_fee} sat/vB) - possible peak"
            elif fastest_fee > 80:
                scores.append(-0.05)
                analysis["fees"] = f"elevated ({fastest_fee} sat/vB)"
            elif fastest_fee > 10:
                scores.append(0.1)
                analysis["fees"] = f"normal ({fastest_fee} sat/vB)"
            else:
                scores.append(0.0)
                analysis["fees"] = f"low ({fastest_fee} sat/vB)"

        # TX volume trend
        tx_count = metrics.get("tx_count_24h", 0)
        prev_tx = self._prev_metrics.get("tx_count_24h", 0)
        if isinstance(tx_count, (int, float)) and tx_count > 0:
            if isinstance(prev_tx, (int, float)) and prev_tx > 0:
                tx_change = (tx_count - prev_tx) / prev_tx
                if tx_change > 0.05:
                    scores.append(0.2)
                    analysis["tx_volume"] = f"rising ({tx_count:,} txns/24h, +{tx_change:.1%})"
                elif tx_change < -0.1:
                    scores.append(-0.15)
                    analysis["tx_volume"] = f"falling ({tx_count:,} txns/24h, {tx_change:.1%})"
                else:
                    scores.append(0.05)
                    analysis["tx_volume"] = f"stable ({tx_count:,} txns/24h)"
            else:
                analysis["tx_volume"] = f"{tx_count:,} txns/24h (first reading)"

        # Aggregate
        if not scores:
            return "neutral", 0.0, analysis

        net_score = sum(scores) / len(scores)

        if net_score > 0.08:
            signal = "bullish"
            strength = min(abs(net_score) * 1.5, 0.4)
        elif net_score < -0.08:
            signal = "bearish"
            strength = min(abs(net_score) * 1.5, 0.4)
        else:
            signal = "neutral"
            strength = 0.0

        analysis["net_score"] = round(net_score, 4)
        return signal, strength, analysis

    # ── HTTP Helper ───────────────────────────────────────────────

    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }

    def _fetch_json(self, url: str, parse_text: bool = False, timeout: int = 8) -> Any:
        """Fetch a URL and return parsed JSON or numeric value."""
        try:
            resp = requests.get(url, headers=self._HEADERS, timeout=timeout)
            if resp.status_code == 429:
                _log.debug("Rate limited: %s", url)
                return 0 if parse_text else {}
            if not resp.ok:
                return 0 if parse_text else {}
            if parse_text:
                try:
                    return float(resp.text.strip())
                except (ValueError, TypeError):
                    return 0
            return resp.json()
        except Exception as e:
            _log.debug("Fetch failed %s: %s", url, e)
            return 0 if parse_text else {}
