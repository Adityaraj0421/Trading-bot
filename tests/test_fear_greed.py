"""Tests for FearGreedProvider intelligence provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intelligence.fear_greed import FearGreedProvider


def _mock_response(value: str, classification: str, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response for the FNG API."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = {
        "data": [{"value": value, "value_classification": classification}]
    }
    return mock


class TestFearGreedProvider:
    """Unit tests for FearGreedProvider."""

    def test_extreme_fear_returns_bullish(self) -> None:
        """Value 10 (extreme fear) should return bullish with high strength."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("10", "Extreme Fear")):
            result = provider.get_signal()
        assert result["signal"] == "bullish"
        assert result["strength"] > 0.6
        assert result["source"] == "fear_greed"

    def test_extreme_greed_returns_bearish(self) -> None:
        """Value 90 (extreme greed) should return bearish with high strength."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("90", "Extreme Greed")):
            result = provider.get_signal()
        assert result["signal"] == "bearish"
        assert result["strength"] > 0.6

    def test_neutral_value_returns_neutral(self) -> None:
        """Value 50 (neutral) should return neutral with strength 0.0."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("50", "Neutral")):
            result = provider.get_signal()
        assert result["signal"] == "neutral"
        assert result["strength"] == 0.0

    def test_mild_fear_returns_bullish_lower_strength(self) -> None:
        """Value 35 (fear) should return bullish with moderate strength."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("35", "Fear")):
            result = provider.get_signal()
        assert result["signal"] == "bullish"
        assert 0.1 < result["strength"] < 0.5

    def test_mild_greed_returns_bearish_lower_strength(self) -> None:
        """Value 65 (greed) should return bearish with moderate strength."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("65", "Greed")):
            result = provider.get_signal()
        assert result["signal"] == "bearish"
        assert 0.1 < result["strength"] < 0.5

    def test_boundary_value_0_max_strength(self) -> None:
        """Value 0 (extreme fear) should return strength == 0.8."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("0", "Extreme Fear")):
            result = provider.get_signal()
        assert result["signal"] == "bullish"
        assert result["strength"] == pytest.approx(0.8)

    def test_boundary_value_100_max_strength(self) -> None:
        """Value 100 (extreme greed) should return strength == 0.8."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("100", "Extreme Greed")):
            result = provider.get_signal()
        assert result["signal"] == "bearish"
        assert result["strength"] == pytest.approx(0.8)

    def test_cache_is_used_on_second_call(self) -> None:
        """Second call within TTL should not make a new HTTP request."""
        provider = FearGreedProvider()
        mock_get = MagicMock(return_value=_mock_response("25", "Fear"))
        with patch("intelligence.fear_greed.requests.get", mock_get):
            provider.get_signal()
            provider.get_signal()
        mock_get.assert_called_once()

    def test_http_500_returns_neutral(self) -> None:
        """HTTP 500 should return neutral signal."""
        provider = FearGreedProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("intelligence.fear_greed.requests.get", return_value=mock_resp):
            result = provider.get_signal()
        assert result["signal"] == "neutral"
        assert result["strength"] == 0.0
        assert "error" in result["data"]

    def test_http_429_returns_neutral(self) -> None:
        """HTTP 429 (rate limit) should return neutral signal."""
        provider = FearGreedProvider()
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with patch("intelligence.fear_greed.requests.get", return_value=mock_resp):
            result = provider.get_signal()
        assert result["signal"] == "neutral"
        assert result["data"]["error"] == "rate_limited"

    def test_network_exception_returns_neutral(self) -> None:
        """Connection error should return neutral signal."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", side_effect=ConnectionError("no connection")):
            result = provider.get_signal()
        assert result["signal"] == "neutral"
        assert result["strength"] == 0.0

    def test_source_key_is_fear_greed(self) -> None:
        """Source key must be 'fear_greed' in all responses."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("72", "Greed")):
            result = provider.get_signal()
        assert result["source"] == "fear_greed"

    def test_data_contains_value_as_int_and_classification(self) -> None:
        """Data dict must contain value as int and classification as str."""
        provider = FearGreedProvider()
        with patch("intelligence.fear_greed.requests.get", return_value=_mock_response("72", "Greed")):
            result = provider.get_signal()
        assert isinstance(result["data"]["value"], int)
        assert result["data"]["value"] == 72
        assert isinstance(result["data"]["classification"], str)
        assert result["data"]["classification"] == "Greed"
