"""
LLM-Powered Sentiment Intelligence Provider v1.0
==================================================
Uses LLM (Claude/GPT) to analyze crypto news headlines with full contextual
understanding, detecting nuance that keyword matching misses.

Features:
  - Fetches from CryptoPanic, Reddit, CoinGecko
  - Batches headlines to LLM for contextual polarity scoring
  - Exponentially-weighted rolling sentiment (half-life = 2h)
  - Social volume spike detection (mention surge = incoming volatility)
  - Falls back to enhanced keyword scoring when no LLM API key available

Signal format (same as other intelligence providers):
  {"source": "LLMSentiment", "signal": "bullish/bearish/neutral",
   "strength": 0.0-1.0, "data": {...}}
"""

import json
import logging
import time
from collections import deque
from typing import Any

import requests

from config import Config

_log = logging.getLogger(__name__)

# Try importing LLM clients
_HAS_ANTHROPIC = False
_HAS_OPENAI = False
try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    pass

try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    pass


# Enhanced keyword scoring as fallback
_BULLISH_PATTERNS = {
    "all-time high": 2.0,
    "ath": 1.5,
    "etf approved": 2.5,
    "institutional buy": 2.0,
    "adoption": 1.2,
    "partnership": 1.0,
    "upgrade": 0.8,
    "bullish": 1.5,
    "breakout": 1.3,
    "rally": 1.5,
    "surge": 1.5,
    "soars": 1.5,
    "accumulate": 1.2,
    "inflows": 1.0,
    "treasury": 1.0,
    "reserve": 1.0,
    "halving": 1.0,
    "recovery": 1.0,
    "green": 0.5,
    "gains": 1.0,
}

_BEARISH_PATTERNS = {
    "crash": 2.0,
    "collapse": 2.0,
    "bankrupt": 2.5,
    "ban": 2.0,
    "hack": 1.8,
    "exploit": 1.5,
    "rug pull": 2.0,
    "fraud": 1.5,
    "liquidation": 1.3,
    "sell-off": 1.5,
    "plunge": 1.5,
    "dump": 1.5,
    "bearish": 1.5,
    "fear": 1.0,
    "crackdown": 1.5,
    "investigation": 1.0,
    "outflows": 1.0,
    "warning": 0.8,
    "delisted": 1.2,
    "bubble": 1.0,
}


class LLMSentimentProvider:
    """
    LLM-powered real-time sentiment analysis for crypto markets.
    Contextually scores headlines instead of naive keyword matching.
    """

    CACHE_TTL = 300  # 5 min cache for API calls
    LLM_BATCH_SIZE = 20  # Headlines per LLM call
    VOLUME_SPIKE_MULT = 2.5  # Social mention volume spike threshold
    SENTIMENT_HALF_LIFE = 7200  # 2 hours in seconds

    REDDIT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    def __init__(self) -> None:
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._sentiment_history: deque = deque(maxlen=200)
        self._volume_history: deque = deque(maxlen=100)
        self._last_signal = None

        # Determine LLM backend
        self._llm_api_key = Config.__dict__.get("ANTHROPIC_API_KEY", "") or ""
        self._openai_api_key = Config.__dict__.get("OPENAI_API_KEY", "") or ""
        self._llm_backend = "none"
        if _HAS_ANTHROPIC and self._llm_api_key:
            self._llm_backend = "anthropic"
        elif _HAS_OPENAI and self._openai_api_key:
            self._llm_backend = "openai"

        _log.info(f"LLM Sentiment: backend={self._llm_backend}")

    def get_signal(self) -> dict[str, Any]:
        """
        Standard intelligence provider interface.
        Returns sentiment signal for the aggregator.
        """
        now = time.time()
        if now - self._cache_ts < self.CACHE_TTL and self._last_signal:
            return self._last_signal

        try:
            headlines = self._fetch_headlines()
            if not headlines:
                return self._neutral_signal("no_headlines")

            # Score headlines
            scores = self._score_headlines(headlines)

            # Record sentiment data point
            avg_score = sum(s["polarity"] for s in scores) / len(scores) if scores else 0
            self._sentiment_history.append(
                {
                    "timestamp": now,
                    "polarity": avg_score,
                    "n_headlines": len(headlines),
                }
            )

            # Record volume data point
            self._volume_history.append(
                {
                    "timestamp": now,
                    "count": len(headlines),
                }
            )

            # Compute exponentially-weighted sentiment
            ew_sentiment = self._compute_ew_sentiment()

            # Detect social volume spike
            volume_spike = self._detect_volume_spike()

            # Convert to signal
            signal = self._to_signal(ew_sentiment, volume_spike, scores)
            self._last_signal = signal
            self._cache_ts = now
            return signal

        except Exception as e:
            _log.warning(f"LLM Sentiment error: {e}")
            return self._neutral_signal(f"error: {str(e)[:50]}")

    def _fetch_headlines(self) -> list[dict]:
        """Fetch recent crypto headlines from multiple free sources."""
        headlines = []

        # Reddit r/cryptocurrency
        try:
            resp = requests.get(
                "https://www.reddit.com/r/cryptocurrency/hot.json?limit=25", headers=self.REDDIT_HEADERS, timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                for post in data.get("data", {}).get("children", []):
                    d = post["data"]
                    headlines.append(
                        {
                            "title": d.get("title", ""),
                            "source": "reddit_crypto",
                            "score": d.get("score", 0),
                            "created": d.get("created_utc", 0),
                            "comments": d.get("num_comments", 0),
                        }
                    )
        except Exception as e:
            _log.debug("Reddit r/cryptocurrency fetch failed: %s", e)

        # Reddit r/bitcoin
        try:
            resp = requests.get(
                "https://www.reddit.com/r/bitcoin/hot.json?limit=15", headers=self.REDDIT_HEADERS, timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                for post in data.get("data", {}).get("children", []):
                    d = post["data"]
                    headlines.append(
                        {
                            "title": d.get("title", ""),
                            "source": "reddit_bitcoin",
                            "score": d.get("score", 0),
                            "created": d.get("created_utc", 0),
                            "comments": d.get("num_comments", 0),
                        }
                    )
        except Exception as e:
            _log.debug("Reddit r/bitcoin fetch failed: %s", e)

        # CryptoPanic (if API key available)
        cryptopanic_key = Config.CRYPTOPANIC_API_KEY
        if cryptopanic_key:
            try:
                resp = requests.get(
                    f"https://cryptopanic.com/api/v1/posts/"
                    f"?auth_token={cryptopanic_key}&currencies=BTC,ETH"
                    f"&kind=news&filter=important",
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("results", [])[:20]:
                        headlines.append(
                            {
                                "title": item.get("title", ""),
                                "source": "cryptopanic",
                                "score": item.get("votes", {}).get("positive", 0),
                                "created": time.time(),
                                "comments": item.get("votes", {}).get("comments", 0),
                            }
                        )
            except Exception as e:
                _log.debug("CryptoPanic fetch failed: %s", e)

        # Filter to recent headlines (last 6 hours)
        cutoff = time.time() - 6 * 3600
        headlines = [h for h in headlines if h.get("created", 0) > cutoff or h.get("created", 0) == 0]

        return headlines[:50]  # Cap at 50

    def _score_headlines(self, headlines: list[dict]) -> list[dict]:
        """Score headlines using LLM or keyword fallback."""
        titles = [h["title"] for h in headlines if h.get("title")]
        if not titles:
            return []

        if self._llm_backend != "none":
            return self._score_with_llm(titles)

        return self._score_with_keywords(titles)

    def _score_with_llm(self, titles: list[str]) -> list[dict]:
        """Score headlines using LLM for contextual understanding."""
        # Batch titles into groups
        results = []
        for i in range(0, len(titles), self.LLM_BATCH_SIZE):
            batch = titles[i : i + self.LLM_BATCH_SIZE]
            batch_text = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(batch))

            prompt = (
                "Score each crypto headline for market sentiment. "
                "For each, give: polarity (-1.0 bearish to +1.0 bullish), "
                "confidence (0.0-1.0), urgency (low/medium/high).\n"
                "Understand context: 'crashes through resistance' is BULLISH.\n"
                "Return ONLY a JSON array of objects with keys: "
                "index, polarity, confidence, urgency.\n\n"
                f"Headlines:\n{batch_text}"
            )

            try:
                if self._llm_backend == "anthropic":
                    scores = self._call_anthropic(prompt)
                else:
                    scores = self._call_openai(prompt)

                if scores:
                    results.extend(scores)
                else:
                    # Fallback for this batch
                    results.extend(self._score_with_keywords(batch))
            except Exception as e:
                _log.warning(f"LLM scoring failed: {e}")
                results.extend(self._score_with_keywords(batch))

        return results

    def _call_anthropic(self, prompt: str) -> list[dict]:
        """Call Claude API for headline scoring."""
        client = anthropic.Anthropic(api_key=self._llm_api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return self._parse_llm_response(text)

    def _call_openai(self, prompt: str) -> list[dict]:
        """Call OpenAI API for headline scoring."""
        client = openai.OpenAI(api_key=self._openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        text = response.choices[0].message.content
        return self._parse_llm_response(text)

    def _parse_llm_response(self, text: str) -> list[dict]:
        """Parse JSON array from LLM response."""
        # Find JSON array in response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end <= start:
            return []

        try:
            data = json.loads(text[start:end])
            results = []
            for item in data:
                results.append(
                    {
                        "polarity": max(-1.0, min(1.0, float(item.get("polarity", 0)))),
                        "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                        "urgency": item.get("urgency", "medium"),
                    }
                )
            return results
        except (json.JSONDecodeError, ValueError):
            return []

    def _score_with_keywords(self, titles: list[str]) -> list[dict]:
        """Enhanced keyword-based scoring as fallback."""
        results = []
        for title in titles:
            lower = title.lower()
            bull_score = sum(weight for keyword, weight in _BULLISH_PATTERNS.items() if keyword in lower)
            bear_score = sum(weight for keyword, weight in _BEARISH_PATTERNS.items() if keyword in lower)

            if bull_score + bear_score == 0:
                polarity = 0.0
                confidence = 0.2
            else:
                polarity = (bull_score - bear_score) / max(bull_score + bear_score, 1)
                confidence = min(0.7, (bull_score + bear_score) / 5.0)

            results.append(
                {
                    "polarity": round(polarity, 3),
                    "confidence": round(confidence, 3),
                    "urgency": "medium",
                }
            )
        return results

    def _compute_ew_sentiment(self) -> float:
        """
        Compute exponentially-weighted sentiment score.
        Recent headlines weighted more (half-life = 2 hours).
        """
        if not self._sentiment_history:
            return 0.0

        now = time.time()
        decay = 0.693 / self.SENTIMENT_HALF_LIFE  # ln(2) / half_life
        weighted_sum = 0.0
        weight_sum = 0.0

        for entry in self._sentiment_history:
            age = now - entry["timestamp"]
            weight = pow(2.718, -decay * age)  # e^(-lambda * t)
            weighted_sum += entry["polarity"] * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def _detect_volume_spike(self) -> dict:
        """
        Detect sudden surge in social mention volume.
        A spike = incoming volatility regardless of direction.
        """
        if len(self._volume_history) < 5:
            return {"is_spike": False, "ratio": 1.0}

        recent = [v["count"] for v in list(self._volume_history)[-3:]]
        baseline = [v["count"] for v in list(self._volume_history)[:-3]]

        if not baseline:
            return {"is_spike": False, "ratio": 1.0}

        avg_recent = sum(recent) / len(recent)
        avg_baseline = sum(baseline) / len(baseline)

        ratio = avg_recent / max(avg_baseline, 1)
        is_spike = ratio > self.VOLUME_SPIKE_MULT

        return {
            "is_spike": is_spike,
            "ratio": round(ratio, 2),
            "avg_recent": round(avg_recent, 1),
            "avg_baseline": round(avg_baseline, 1),
        }

    def _to_signal(self, ew_sentiment: float, volume_spike: dict, scores: list[dict]) -> dict[str, Any]:
        """Convert analysis to standard intelligence signal format."""
        # Determine signal direction
        if ew_sentiment > 0.15:
            signal = "bullish"
        elif ew_sentiment < -0.15:
            signal = "bearish"
        else:
            signal = "neutral"

        # Strength: magnitude of sentiment * average confidence
        avg_conf = sum(s.get("confidence", 0.5) for s in scores) / len(scores) if scores else 0.3
        strength = min(1.0, abs(ew_sentiment) * avg_conf * 2)

        # Volume spike modifies strength (volatility warning)
        if volume_spike["is_spike"]:
            strength = min(1.0, strength * 1.3)

        # Count urgency
        urgent_count = sum(1 for s in scores if s.get("urgency") == "high")

        return {
            "source": "LLMSentiment",
            "signal": signal,
            "strength": round(strength, 3),
            "data": {
                "ew_sentiment": round(ew_sentiment, 4),
                "n_headlines": len(scores),
                "avg_confidence": round(avg_conf, 3),
                "volume_spike": volume_spike,
                "urgent_headlines": urgent_count,
                "llm_backend": self._llm_backend,
                "top_bullish": sum(1 for s in scores if s.get("polarity", 0) > 0.3),
                "top_bearish": sum(1 for s in scores if s.get("polarity", 0) < -0.3),
            },
        }

    def _neutral_signal(self, reason: str = "") -> dict[str, Any]:
        """Return a neutral signal with an optional reason."""
        return {
            "source": "LLMSentiment",
            "signal": "neutral",
            "strength": 0.0,
            "data": {"reason": reason},
        }
