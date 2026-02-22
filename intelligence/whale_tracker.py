"""
Whale Transaction Tracker — large BTC movements.
Sources:
  - Blockchain.com unconfirmed TX API (free, no key)
  - Mempool.space recent blocks API (free, no key)

Tracks large transactions (>100 BTC) and exchange flow patterns.
Many whale transactions in a short window often signal upcoming price moves.

Signal logic:
  - Many large outflows from exchanges = bullish (accumulation)
  - Many large inflows to exchanges = bearish (preparing to sell)
  - General high whale activity = slightly bearish (uncertainty)
"""

import time
import logging
import requests
from collections import deque
from typing import Any
from config import Config

_log = logging.getLogger(__name__)

# Known exchange addresses (subset — cold wallets of major exchanges)
# These are publicly known BTC addresses
KNOWN_EXCHANGE_ADDRESSES = {
    # Binance
    "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",
    "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3",
    "3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb",
    # Coinbase
    "3Kzh9qAqVWQhEsfQz7zEQL1EuSx5tyNLNS",
    "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
    # Bitfinex
    "3D2oetdNuZUqQHPJmcMDDHYoqkyNVsFk9r",
    "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
    # Kraken
    "3AfP7AUiPcCkVPnGnBXEqJoFpM3cce4sYN",
}


class WhaleTracker:
    """Monitors large Bitcoin transactions with exchange flow detection."""

    THRESHOLD_BTC = 100  # Transactions > 100 BTC
    MEMPOOL_API = "https://mempool.space/api"
    BLOCKCHAIN_API = "https://blockchain.info"
    CACHE_TTL = 120  # 2 min cache (whale activity is time-sensitive)

    # v7.0: Browser-like headers to avoid bot blocking
    _HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }

    def __init__(self) -> None:
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._history: deque = deque(maxlen=100)  # Track whale activity over time

    def get_signal(self) -> dict[str, Any]:
        """Check for recent whale activity and produce signal."""
        if not Config.ENABLE_WHALE_TRACKING:
            return {"source": "whale_tracker", "signal": "neutral", "strength": 0.0, "data": {}}

        try:
            whale_data = self._detect_whales()
            signal, strength = self._analyze(whale_data)

            # Track history
            self._history.append({
                "ts": time.time(),
                "whale_count": whale_data["whale_tx_count"],
                "total_btc": whale_data["total_whale_btc"],
                "signal": signal,
            })

            return {
                "source": "whale_tracker",
                "signal": signal,
                "strength": round(strength, 3),
                "data": whale_data,
            }
        except Exception as e:
            _log.warning("Whale tracking failed: %s", e)
            return {"source": "whale_tracker", "signal": "neutral", "strength": 0.0, "data": {"error": str(e)}}

    def _detect_whales(self) -> dict[str, Any]:
        """Detect large transactions from multiple sources."""
        now = time.time()
        if now - self._cache_ts < self.CACHE_TTL and self._cache:
            return self._cache

        large_txs = []
        exchange_inflows = 0
        exchange_outflows = 0
        total_whale_btc = 0.0

        # Source 1: Unconfirmed transactions (mempool)
        try:
            resp = requests.get(
                f"{self.BLOCKCHAIN_API}/unconfirmed-transactions?format=json",
                headers=self._HEADERS,
                timeout=10,
            )
            if resp.ok:
                txs = resp.json().get("txs", [])
                for tx in txs[:200]:  # Check up to 200
                    total_out = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8
                    if total_out >= self.THRESHOLD_BTC:
                        tx_info = {
                            "hash": tx.get("hash", "")[:16],
                            "btc_amount": round(total_out, 2),
                            "output_count": len(tx.get("out", [])),
                            "input_count": len(tx.get("inputs", [])),
                        }

                        # Check if any outputs go to known exchange addresses
                        for out in tx.get("out", []):
                            addr = out.get("addr", "")
                            if addr in KNOWN_EXCHANGE_ADDRESSES:
                                tx_info["exchange_flow"] = "inflow"
                                exchange_inflows += total_out
                                break

                        # Check if inputs come from known exchange addresses
                        for inp in tx.get("inputs", []):
                            prev_out = inp.get("prev_out", {})
                            addr = prev_out.get("addr", "")
                            if addr in KNOWN_EXCHANGE_ADDRESSES:
                                tx_info["exchange_flow"] = "outflow"
                                exchange_outflows += total_out
                                break

                        large_txs.append(tx_info)
                        total_whale_btc += total_out
        except Exception as e:
            _log.debug("Unconfirmed TX fetch failed: %s", e)

        # Source 2: Recent blocks for confirmed large TXs
        try:
            resp = requests.get(f"{self.MEMPOOL_API}/blocks", headers=self._HEADERS, timeout=8)
            if resp.ok:
                blocks = resp.json()[:3]  # Last 3 blocks
                for block in blocks:
                    block_hash = block.get("id", "")
                    if block_hash:
                        block_resp = requests.get(
                            f"{self.MEMPOOL_API}/block/{block_hash}/txs/0",
                            headers=self._HEADERS,
                            timeout=8,
                        )
                        if block_resp.ok:
                            block_txs = block_resp.json()
                            for tx in block_txs[:50]:
                                total_out = sum(
                                    v.get("value", 0) for v in tx.get("vout", [])
                                ) / 1e8
                                if total_out >= self.THRESHOLD_BTC * 5:  # Higher threshold for confirmed
                                    large_txs.append({
                                        "hash": tx.get("txid", "")[:16],
                                        "btc_amount": round(total_out, 2),
                                        "confirmed": True,
                                        "block": block.get("height", 0),
                                    })
                                    total_whale_btc += total_out
        except Exception as e:
            _log.debug("Block TX fetch failed: %s", e)

        result = {
            "whale_tx_count": len(large_txs),
            "total_whale_btc": round(total_whale_btc, 2),
            "exchange_inflows_btc": round(exchange_inflows, 2),
            "exchange_outflows_btc": round(exchange_outflows, 2),
            "large_transactions": large_txs[:10],  # Top 10
            "threshold_btc": self.THRESHOLD_BTC,
        }

        self._cache = result
        self._cache_ts = now
        return result

    def _analyze(self, data: dict) -> tuple[str, float]:
        """Analyze whale data to produce signal and strength.

        Key signals:
          - Exchange inflows > outflows = bearish (selling pressure)
          - Exchange outflows > inflows = bullish (accumulation)
          - Very high general whale activity = slightly bearish (uncertainty)
        """
        whale_count = data.get("whale_tx_count", 0)
        inflows = data.get("exchange_inflows_btc", 0)
        outflows = data.get("exchange_outflows_btc", 0)
        total_btc = data.get("total_whale_btc", 0)

        score = 0.0

        # Exchange flow analysis (strongest signal)
        if inflows > 0 or outflows > 0:
            net_flow = outflows - inflows  # Positive = more leaving exchanges = bullish
            flow_total = inflows + outflows
            if flow_total > 0:
                flow_ratio = net_flow / flow_total
                score += flow_ratio * 0.3  # -0.3 to +0.3

        # General activity level
        if whale_count >= 10:
            score -= 0.15  # Very high activity = uncertainty/selling
        elif whale_count >= 5:
            score -= 0.05
        elif whale_count >= 2:
            score += 0.0  # Some activity is normal

        # Historical trend (are whales becoming more active?)
        if len(self._history) >= 3:
            recent_avg = sum(h["whale_count"] for h in list(self._history)[-3:]) / 3
            older_avg = sum(h["whale_count"] for h in list(self._history)[:3]) / 3 if len(self._history) >= 6 else recent_avg
            if older_avg > 0 and recent_avg > older_avg * 1.5:
                score -= 0.1  # Accelerating whale activity

        # Convert to signal
        if score > 0.1:
            return "bullish", min(abs(score), 0.4)
        elif score < -0.1:
            return "bearish", min(abs(score), 0.4)
        else:
            return "neutral", 0.0
