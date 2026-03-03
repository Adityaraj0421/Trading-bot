# tests/test_context_session.py
from datetime import UTC, datetime, timedelta

from context.session import SessionAnalyzer


def dt(weekday: int, hour: int) -> datetime:
    """Build a UTC datetime on the given ISO weekday (0=Mon, 6=Sun) and hour."""
    base_monday = datetime(2026, 3, 2, hour, 0, tzinfo=UTC)
    return base_monday + timedelta(days=weekday)


class TestSessionAnalyzer:
    def test_us_session_full_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(0, 15))  # Monday 15:00 UTC
        assert result["session"] == "US"
        assert result["confidence_multiplier"] == 1.00

    def test_eu_session_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(0, 10))  # Monday 10:00 UTC
        assert result["session"] == "EU"
        assert result["confidence_multiplier"] == 0.90

    def test_asia_session_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(0, 4))  # Monday 04:00 UTC
        assert result["session"] == "Asia"
        assert result["confidence_multiplier"] == 0.75

    def test_weekend_saturday_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(5, 14))  # Saturday 14:00 UTC
        assert result["session"] == "weekend"
        assert result["confidence_multiplier"] == 0.60

    def test_weekend_sunday_confidence(self):
        result = SessionAnalyzer().analyze(now=dt(6, 20))  # Sunday 20:00 UTC
        assert result["session"] == "weekend"
        assert result["confidence_multiplier"] == 0.60

    def test_us_eu_overlap_returns_us(self):
        # 14:00 UTC is in both EU (07-15) and US (13-22) → US wins (higher multiplier)
        result = SessionAnalyzer().analyze(now=dt(1, 14))
        assert result["session"] == "US"
        assert result["confidence_multiplier"] == 1.00

    def test_dead_zone_returns_asia(self):
        # 23:00 UTC is outside US/EU/Asia windows
        result = SessionAnalyzer().analyze(now=dt(2, 23))
        assert result["session"] == "Asia"
        assert result["confidence_multiplier"] == 0.75
