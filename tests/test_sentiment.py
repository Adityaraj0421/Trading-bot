"""
Unit tests for sentiment.py — SentimentLevel, SentimentAnalyzer.

Tests pure analysis methods directly; mocks requests.get for API tests.
"""

from unittest.mock import MagicMock, patch

import pytest

from demo_data import generate_ohlcv
from sentiment import SentimentAnalyzer, SentimentLevel, SentimentState


@pytest.fixture()
def analyzer():
    return SentimentAnalyzer()


@pytest.fixture()
def sample_df():
    return generate_ohlcv(periods=100, seed=42)


# ---------------------------------------------------------------------------
# SentimentLevel enum
# ---------------------------------------------------------------------------


class TestSentimentLevel:
    def test_all_levels_exist(self):
        assert len(SentimentLevel) == 5
        assert SentimentLevel.EXTREME_FEAR.value == "extreme_fear"
        assert SentimentLevel.EXTREME_GREED.value == "extreme_greed"


# ---------------------------------------------------------------------------
# _classify_fg — boundary tests
# ---------------------------------------------------------------------------


class TestClassifyFG:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (10, SentimentLevel.EXTREME_FEAR),
            (25, SentimentLevel.EXTREME_FEAR),  # <=25
            (26, SentimentLevel.FEAR),
            (30, SentimentLevel.FEAR),
            (45, SentimentLevel.FEAR),  # <=45
            (46, SentimentLevel.NEUTRAL),
            (50, SentimentLevel.NEUTRAL),
            (55, SentimentLevel.NEUTRAL),  # <=55
            (56, SentimentLevel.GREED),
            (70, SentimentLevel.GREED),
            (75, SentimentLevel.GREED),  # <=75
            (76, SentimentLevel.EXTREME_GREED),
            (90, SentimentLevel.EXTREME_GREED),
        ],
    )
    def test_classification(self, analyzer, value, expected):
        assert analyzer._classify_fg(value) == expected


# ---------------------------------------------------------------------------
# _volume_sentiment
# ---------------------------------------------------------------------------


class TestVolumeSentiment:
    def test_returns_float(self, analyzer, sample_df):
        result = analyzer._volume_sentiment(sample_df)
        assert isinstance(result, float)

    def test_range_bounded(self, analyzer, sample_df):
        result = analyzer._volume_sentiment(sample_df)
        assert -1 <= result <= 1

    def test_short_df_returns_zero(self, analyzer):
        df = generate_ohlcv(periods=10, seed=42)
        assert analyzer._volume_sentiment(df) == 0.0

    def test_zero_volume_returns_zero(self, analyzer):
        df = generate_ohlcv(periods=30, seed=42)
        df["volume"] = 0.0
        assert analyzer._volume_sentiment(df) == 0.0


# ---------------------------------------------------------------------------
# _price_momentum
# ---------------------------------------------------------------------------


class TestPriceMomentum:
    def test_returns_float(self, analyzer, sample_df):
        result = analyzer._price_momentum(sample_df)
        assert isinstance(result, float)

    def test_range_bounded(self, analyzer, sample_df):
        result = analyzer._price_momentum(sample_df)
        assert -1 <= result <= 1

    def test_with_precomputed_indicators(self, analyzer, sample_df):
        df_ind = sample_df.copy()
        df_ind["returns_5"] = sample_df["close"].pct_change(5)
        df_ind["returns_10"] = sample_df["close"].pct_change(10)
        result = analyzer._price_momentum(sample_df, df_ind=df_ind)
        assert isinstance(result, float)

    def test_short_df_without_indicators_returns_zero(self, analyzer):
        df = generate_ohlcv(periods=30, seed=42)
        assert analyzer._price_momentum(df) == 0.0


# ---------------------------------------------------------------------------
# _contrarian_signal
# ---------------------------------------------------------------------------


class TestContrarianSignal:
    def test_extreme_fear_negative_composite_buy(self, analyzer):
        assert analyzer._contrarian_signal(15, -0.5) == "BUY"

    def test_extreme_greed_positive_composite_sell(self, analyzer):
        assert analyzer._contrarian_signal(85, 0.5) == "SELL"

    def test_neutral_returns_neutral(self, analyzer):
        assert analyzer._contrarian_signal(50, 0.0) == "NEUTRAL"

    def test_fear_but_positive_composite_neutral(self, analyzer):
        assert analyzer._contrarian_signal(15, 0.5) == "NEUTRAL"

    def test_greed_but_negative_composite_neutral(self, analyzer):
        assert analyzer._contrarian_signal(85, -0.5) == "NEUTRAL"


# ---------------------------------------------------------------------------
# analyze — mocked API
# ---------------------------------------------------------------------------


class TestAnalyze:
    def _mock_api_response(self, value=50):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"value": str(value)}]}
        return mock_resp

    @patch("sentiment.requests.get")
    def test_returns_sentiment_state(self, mock_get, analyzer, sample_df):
        mock_get.return_value = self._mock_api_response(50)
        result = analyzer.analyze(sample_df)
        assert isinstance(result, SentimentState)

    @patch("sentiment.requests.get")
    def test_sentiment_state_fields(self, mock_get, analyzer, sample_df):
        mock_get.return_value = self._mock_api_response(50)
        result = analyzer.analyze(sample_df)
        assert isinstance(result.fear_greed_index, int)
        assert isinstance(result.fear_greed_label, SentimentLevel)
        assert isinstance(result.volume_sentiment, float)
        assert isinstance(result.price_momentum_score, float)
        assert isinstance(result.composite_score, float)
        assert result.contrarian_signal in ("BUY", "SELL", "NEUTRAL")
        assert result.source in ("api", "cached", "fallback")

    @patch("sentiment.requests.get")
    def test_api_failure_uses_fallback(self, mock_get, analyzer, sample_df):
        import requests as _requests

        mock_get.side_effect = _requests.RequestException("network error")
        result = analyzer.analyze(sample_df)
        assert result.source == "fallback"
        assert result.fear_greed_index == 50  # default last_fg_index


# ---------------------------------------------------------------------------
# _fetch_fear_greed cache
# ---------------------------------------------------------------------------


class TestFetchCache:
    @patch("sentiment.requests.get")
    def test_cache_hit_within_ttl(self, mock_get, analyzer):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"value": "65"}]}
        mock_get.return_value = mock_resp

        # First call — hits API
        analyzer._fetch_fear_greed()
        assert mock_get.call_count == 1

        # Second call — should use cache (within TTL)
        analyzer._fetch_fear_greed()
        assert mock_get.call_count == 1  # No additional API call

    @patch("sentiment.requests.get")
    def test_cache_miss_after_ttl(self, mock_get, analyzer):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"value": "65"}]}
        mock_get.return_value = mock_resp

        # First call
        analyzer._fetch_fear_greed()
        assert mock_get.call_count == 1

        # Expire cache by setting cache time far in the past
        analyzer._fg_cache_time = 0
        analyzer._fetch_fear_greed()
        assert mock_get.call_count == 2  # New API call
