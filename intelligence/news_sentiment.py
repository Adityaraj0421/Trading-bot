"""
News Sentiment Analyzer — crypto headline sentiment from multiple sources.
Sources:
  - Reddit r/cryptocurrency (free JSON API, no key)
  - Reddit r/bitcoin (free JSON API, no key)
  - CoinGecko trending coins (free, no key)
  - CryptoPanic API (free tier, optional key for more data)

Uses keyword matching with weighted scoring and recency bias.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import requests

from config import Config

_log = logging.getLogger(__name__)


# Weighted keyword lists (keyword, weight)
BULLISH_KEYWORDS = [
    ("surge", 1.5),
    ("rally", 1.5),
    ("bull", 1.0),
    ("breakout", 1.3),
    ("adoption", 1.2),
    ("partnership", 0.8),
    ("institutional", 1.3),
    ("etf approved", 2.0),
    ("etf", 1.0),
    ("all-time high", 2.0),
    ("ath", 1.5),
    ("upgrade", 0.8),
    ("bullish", 1.5),
    ("soars", 1.5),
    ("gains", 1.0),
    ("accumulate", 1.2),
    ("buy", 0.5),
    ("moon", 0.8),
    ("halving", 1.0),
    ("recovery", 1.0),
    ("approval", 1.2),
    ("green", 0.5),
    ("pump", 0.8),
    ("listing", 0.7),
    ("launch", 0.6),
    ("treasury", 1.0),
    ("reserve", 1.0),
    ("inflows", 1.0),
]

BEARISH_KEYWORDS = [
    ("crash", 2.0),
    ("dump", 1.5),
    ("bear", 1.0),
    ("hack", 1.5),
    ("regulation", 0.8),
    ("ban", 2.0),
    ("lawsuit", 1.2),
    ("sec", 0.8),
    ("fraud", 1.5),
    ("bankrupt", 2.0),
    ("collapse", 2.0),
    ("sell-off", 1.5),
    ("plunge", 1.5),
    ("bearish", 1.5),
    ("fear", 1.0),
    ("liquidation", 1.3),
    ("ponzi", 1.5),
    ("scam", 1.5),
    ("rug pull", 2.0),
    ("exploit", 1.3),
    ("outflows", 1.0),
    ("red", 0.5),
    ("tank", 1.0),
    ("bubble", 1.0),
    ("crackdown", 1.5),
    ("investigation", 1.0),
    ("warning", 0.8),
    ("delisted", 1.2),
    ("withdraw", 0.7),
    ("selling", 0.8),
]


class NewsSentimentAnalyzer:
    """Analyzes crypto news headlines for sentiment from multiple sources.

    Fetches posts from Reddit, CoinGecko trending, and optionally
    CryptoPanic, then applies weighted keyword scoring with engagement
    multipliers to produce a bullish/bearish/neutral signal.
    """

    CACHE_TTL = 180  # 3 min cache
    REDDIT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    CRYPTOPANIC_API = "https://cryptopanic.com/api/v1"

    def __init__(self) -> None:
        """Initialise the analyzer with an empty cache and history."""
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._history: deque = deque(maxlen=50)  # Track sentiment over time

    def get_signal(self) -> dict[str, Any]:
        """Analyze news sentiment from all sources.

        Fetches headlines from all configured sources, applies weighted
        keyword sentiment analysis, and applies a trend multiplier based
        on recent signal history.

        Returns:
            Dictionary with keys: ``source`` (``'news_sentiment'``),
            ``signal`` (``'bullish'``, ``'bearish'``, or ``'neutral'``),
            ``strength`` (0.0–0.5), ``data`` containing ``signal``,
            ``strength``, ``headline_count``, ``matched_count``,
            ``bullish_score``, ``bearish_score``, ``sentiment_ratio``, and
            ``top_headlines``.

        Note:
            Returns a neutral signal immediately when
            ``Config.ENABLE_NEWS_NLP`` is ``False``.
        """
        if not Config.ENABLE_NEWS_NLP:
            return {"source": "news_sentiment", "signal": "neutral", "strength": 0.0, "data": {}}

        try:
            headlines = self._fetch_all_headlines()
            if not headlines:
                return {
                    "source": "news_sentiment",
                    "signal": "neutral",
                    "strength": 0.0,
                    "data": {"error": "no_headlines"},
                }

            analysis = self._analyze_headlines(headlines)
            signal = analysis["signal"]
            strength = analysis["strength"]

            # Track history for trend analysis
            self._history.append(
                {
                    "ts": time.time(),
                    "signal": signal,
                    "ratio": analysis["sentiment_ratio"],
                }
            )

            # Trend amplification: if sentiment is consistently moving in one direction
            trend_mult = self._compute_trend_multiplier()

            return {
                "source": "news_sentiment",
                "signal": signal,
                "strength": round(min(strength * trend_mult, 0.5), 3),
                "data": analysis,
            }
        except Exception as e:
            _log.warning("News sentiment failed: %s", e)
            return {"source": "news_sentiment", "signal": "neutral", "strength": 0.0, "data": {"error": str(e)}}

    def _fetch_all_headlines(self) -> list[dict]:
        """Fetch headlines from all sources.

        Uses a ``CACHE_TTL`` second cache to avoid redundant requests.

        Returns:
            Combined list of headline dicts from Reddit, CoinGecko, and
            optionally CryptoPanic.
        """
        now = time.time()
        if now - self._cache_ts < self.CACHE_TTL and self._cache.get("headlines"):
            return self._cache["headlines"]

        headlines = []

        # Source 1: Reddit r/cryptocurrency
        crypto_headlines = self._fetch_reddit("cryptocurrency", limit=30)
        headlines.extend(crypto_headlines)

        # Source 2: Reddit r/bitcoin
        btc_headlines = self._fetch_reddit("bitcoin", limit=20)
        headlines.extend(btc_headlines)

        # Source 3: CoinGecko trending (sentiment proxy)
        trending = self._fetch_coingecko_trending()
        headlines.extend(trending)

        # Source 4: CryptoPanic (if API key available)
        cryptopanic_key = getattr(Config, "CRYPTOPANIC_API_KEY", "")
        if cryptopanic_key:
            cp_headlines = self._fetch_cryptopanic(cryptopanic_key)
            headlines.extend(cp_headlines)

        self._cache = {"headlines": headlines}
        self._cache_ts = now
        return headlines

    def _fetch_reddit(self, subreddit: str, limit: int = 25) -> list[dict]:
        """Fetch top posts from a subreddit.

        Tries www.reddit.com first, then old.reddit.com as fallback.

        Args:
            subreddit: Subreddit name without the ``r/`` prefix, e.g.
                ``'cryptocurrency'``.
            limit: Maximum number of posts to fetch (default 25).

        Returns:
            List of post dicts with ``title``, ``score``, ``source``,
            ``upvote_ratio``, and ``num_comments`` keys.  Returns an empty
            list when all URL attempts fail.
        """
        urls = [
            f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}",
            f"https://old.reddit.com/r/{subreddit}/hot.json?limit={limit}",
        ]
        for url in urls:
            try:
                resp = requests.get(url, headers=self.REDDIT_HEADERS, timeout=8)
                if resp.status_code == 429:
                    _log.debug("Reddit rate limited on %s", url)
                    continue
                if not resp.ok:
                    continue
                data = resp.json()
                posts = data.get("data", {}).get("children", [])
                return [
                    {
                        "title": p["data"]["title"],
                        "score": p["data"].get("score", 0),
                        "source": f"r/{subreddit}",
                        "upvote_ratio": p["data"].get("upvote_ratio", 0.5),
                        "num_comments": p["data"].get("num_comments", 0),
                    }
                    for p in posts
                    if "data" in p and "title" in p["data"]
                ]
            except Exception as e:
                _log.debug("Reddit fetch failed for %s: %s", url, e)
                continue
        return []

    def _fetch_coingecko_trending(self) -> list[dict]:
        """Fetch trending coins from CoinGecko as sentiment proxy.

        Returns:
            List of pseudo-headline dicts generated from the top 7
            trending coins, each containing ``title``, ``score``,
            ``source``, ``upvote_ratio``, and ``num_comments`` keys.
            Returns an empty list on failure.
        """
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=10,
            )
            if not resp.ok:
                return []
            data = resp.json()
            coins = data.get("coins", [])
            # Convert trending coins to pseudo-headlines
            headlines = []
            for coin in coins[:7]:
                item = coin.get("item", {})
                name = item.get("name", "Unknown")
                symbol = item.get("symbol", "")
                price_change = item.get("data", {}).get("price_change_percentage_24h", {})
                if isinstance(price_change, dict):
                    change_usd = price_change.get("usd", 0)
                elif isinstance(price_change, (int, float)):
                    change_usd = price_change
                else:
                    change_usd = 0

                direction = "surging" if change_usd > 5 else "dropping" if change_usd < -5 else "trending"
                headlines.append(
                    {
                        "title": f"{name} ({symbol}) {direction} - 24h change {change_usd:.1f}%",
                        "score": abs(change_usd) * 10,
                        "source": "coingecko_trending",
                        "upvote_ratio": 0.5,
                        "num_comments": 0,
                    }
                )
            return headlines
        except Exception as e:
            _log.debug("CoinGecko trending fetch failed: %s", e)
            return []

    def _fetch_cryptopanic(self, api_key: str) -> list[dict]:
        """Fetch from CryptoPanic API (free tier: 5 req/min).

        Args:
            api_key: CryptoPanic API authentication token.

        Returns:
            List of headline dicts from CryptoPanic with ``title``,
            ``score``, ``source``, ``upvote_ratio``, ``num_comments``, and
            ``votes`` keys.  Returns an empty list on failure.
        """
        try:
            resp = requests.get(
                f"{self.CRYPTOPANIC_API}/posts/?auth_token={api_key}&currencies=BTC,ETH&filter=hot",
                timeout=10,
            )
            if not resp.ok:
                return []
            data = resp.json()
            results = data.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "score": {"positive": 100, "negative": -100, "important": 50}.get(r.get("kind", ""), 0),
                    "source": "cryptopanic",
                    "upvote_ratio": 0.5,
                    "num_comments": r.get("comments_count", 0),
                    "votes": r.get("votes", {}),
                }
                for r in results[:20]
            ]
        except Exception as e:
            _log.debug("CryptoPanic fetch failed: %s", e)
            return []

    def _analyze_headlines(self, headlines: list[dict]) -> dict[str, Any]:
        """Weighted sentiment analysis on headlines.

        Each headline is scored based on:
          1. Keyword matches with weights
          2. Reddit score (popular = more weight)
          3. Engagement (comments = conviction)

        Args:
            headlines: List of headline dicts from ``_fetch_all_headlines()``.

        Returns:
            Dictionary with keys: ``signal``, ``strength``,
            ``headline_count``, ``matched_count``, ``bullish_score``,
            ``bearish_score``, ``sentiment_ratio``, and ``top_headlines``
            (up to 5 most polarised matched headlines).
        """
        bullish_score = 0.0
        bearish_score = 0.0
        matched_headlines = []

        for h in headlines:
            title_lower = h["title"].lower()
            engagement_mult = 1.0

            # Engagement multiplier: highly upvoted/commented posts matter more
            score = h.get("score", 0)
            comments = h.get("num_comments", 0)
            if score > 1000:
                engagement_mult = 1.5
            elif score > 100:
                engagement_mult = 1.2
            if comments > 200:
                engagement_mult *= 1.3

            # Keyword matching
            h_bull = 0.0
            h_bear = 0.0
            for kw, weight in BULLISH_KEYWORDS:
                if kw in title_lower:
                    h_bull += weight * engagement_mult
            for kw, weight in BEARISH_KEYWORDS:
                if kw in title_lower:
                    h_bear += weight * engagement_mult

            if h_bull > 0 or h_bear > 0:
                matched_headlines.append(
                    {
                        "title": h["title"][:80],
                        "source": h.get("source", "unknown"),
                        "bullish": round(h_bull, 2),
                        "bearish": round(h_bear, 2),
                    }
                )

            bullish_score += h_bull
            bearish_score += h_bear

        total = bullish_score + bearish_score
        if total == 0:
            return {
                "signal": "neutral",
                "strength": 0.0,
                "headline_count": len(headlines),
                "matched_count": 0,
                "bullish_score": 0,
                "bearish_score": 0,
                "sentiment_ratio": 0.0,
                "top_headlines": [],
            }

        ratio = (bullish_score - bearish_score) / total  # -1 to +1

        if ratio > 0.15:
            signal = "bullish"
            strength = min(abs(ratio) * 0.4, 0.4)
        elif ratio < -0.15:
            signal = "bearish"
            strength = min(abs(ratio) * 0.4, 0.4)
        else:
            signal = "neutral"
            strength = 0.0

        return {
            "signal": signal,
            "strength": round(strength, 3),
            "headline_count": len(headlines),
            "matched_count": len(matched_headlines),
            "bullish_score": round(bullish_score, 2),
            "bearish_score": round(bearish_score, 2),
            "sentiment_ratio": round(ratio, 3),
            "top_headlines": sorted(
                matched_headlines,
                key=lambda x: abs(x["bullish"] - x["bearish"]),
                reverse=True,
            )[:5],
        }

    def _compute_trend_multiplier(self) -> float:
        """Compute trend multiplier from recent signal history.

        If sentiment has been consistently directional over the last 5
        readings, amplify the signal strength.

        Returns:
            Multiplier float: 1.3 (strong trend, 4+ same-direction),
            1.15 (moderate trend, 3+ same-direction), or 1.0 (no trend).
        """
        if len(self._history) < 3:
            return 1.0

        recent = list(self._history)[-5:]
        bullish_count = sum(1 for h in recent if h["signal"] == "bullish")
        bearish_count = sum(1 for h in recent if h["signal"] == "bearish")

        if bullish_count >= 4 or bearish_count >= 4:
            return 1.3  # Strong trend
        elif bullish_count >= 3 or bearish_count >= 3:
            return 1.15
        return 1.0
