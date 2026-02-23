"""
Unit tests for pair_scorer.py — PairScorer.score_pairs() and select_top_pairs().

Validates scoring logic, sort order, top-N selection, fallback behaviour,
and edge cases (empty data, insufficient bars, zero division guards).
"""

import pandas as pd
import pytest

from config import Config
from pair_scorer import PairScore, PairScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ohlcv(
    rows: int = 30,
    base_price: float = 50000.0,
    vol_mult: float = 1.0,
    atr_factor: float = 1.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    Args:
        rows: Number of bars to generate.
        base_price: Constant close price.
        vol_mult: Multiplier for last-bar volume (> 1 = volume spike).
        atr_factor: Multiplier for high-low spread (> 1 = wider ATR).
    """
    spread = base_price * 0.01 * atr_factor  # 1% of price × atr_factor
    data = {
        "open": [base_price] * rows,
        "high": [base_price + spread] * rows,
        "low": [base_price - spread] * rows,
        "close": [base_price] * rows,
        "volume": [1000.0] * rows,
    }
    df = pd.DataFrame(data)
    # Apply volume multiplier on last bar (used by PairScorer volume_ratio)
    df.loc[df.index[-1], "volume"] = 1000.0 * vol_mult
    return df


# ---------------------------------------------------------------------------
# PairScore dataclass
# ---------------------------------------------------------------------------


class TestPairScore:
    def test_fields(self):
        ps = PairScore(symbol="BTC/USDT", score=0.5, atr_pct=0.02, volume_ratio=1.5)
        assert ps.symbol == "BTC/USDT"
        assert ps.score == 0.5
        assert ps.atr_pct == 0.02
        assert ps.volume_ratio == 1.5


# ---------------------------------------------------------------------------
# PairScorer.score_pairs()
# ---------------------------------------------------------------------------


class TestScorePairs:
    @pytest.fixture()
    def scorer(self):
        return PairScorer()

    def test_returns_list_of_pair_scores(self, scorer):
        price_data = {"BTC/USDT": make_ohlcv()}
        result = scorer.score_pairs(price_data)
        assert isinstance(result, list)
        assert all(isinstance(s, PairScore) for s in result)

    def test_score_is_atr_pct_times_volume_ratio(self, scorer):
        """score = atr_pct × volume_ratio (positive)."""
        price_data = {"BTC/USDT": make_ohlcv(base_price=50000.0, vol_mult=2.0, atr_factor=1.0)}
        result = scorer.score_pairs(price_data)
        assert len(result) == 1
        ps = result[0]
        assert ps.score == pytest.approx(ps.atr_pct * ps.volume_ratio)
        assert ps.score > 0

    def test_sorted_descending_by_score(self, scorer):
        """Results must be sorted highest score first."""
        price_data = {
            "LOW/USDT": make_ohlcv(atr_factor=0.5, vol_mult=0.5),  # small score
            "HIGH/USDT": make_ohlcv(atr_factor=3.0, vol_mult=3.0),  # large score
            "MID/USDT": make_ohlcv(atr_factor=1.5, vol_mult=1.5),
        }
        result = scorer.score_pairs(price_data)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_skips_empty_dataframe(self, scorer):
        """Empty DataFrame for a symbol should be silently skipped."""
        price_data = {
            "BTC/USDT": make_ohlcv(),
            "EMPTY/USDT": pd.DataFrame(),
        }
        result = scorer.score_pairs(price_data)
        symbols = [r.symbol for r in result]
        assert "EMPTY/USDT" not in symbols
        assert "BTC/USDT" in symbols

    def test_skips_insufficient_bars(self, scorer):
        """DataFrame with < 20 bars (volume SMA min) should be skipped."""
        price_data = {
            "BTC/USDT": make_ohlcv(),
            "SHORT/USDT": make_ohlcv(rows=15),  # < 20
        }
        result = scorer.score_pairs(price_data)
        symbols = [r.symbol for r in result]
        assert "SHORT/USDT" not in symbols

    def test_skips_none_dataframe(self, scorer):
        """None value in price_data should be silently skipped."""
        price_data = {"BTC/USDT": make_ohlcv(), "NONE/USDT": None}
        result = scorer.score_pairs(price_data)
        symbols = [r.symbol for r in result]
        assert "NONE/USDT" not in symbols

    def test_empty_price_data_returns_empty_list(self, scorer):
        result = scorer.score_pairs({})
        assert result == []

    def test_all_pairs_insufficient_data_returns_empty(self, scorer):
        price_data = {
            "A/USDT": make_ohlcv(rows=10),
            "B/USDT": make_ohlcv(rows=5),
        }
        result = scorer.score_pairs(price_data)
        assert result == []

    def test_get_last_scores_updates_after_call(self, scorer):
        """get_last_scores() reflects the most recent score_pairs() result."""
        assert scorer.get_last_scores() == []
        price_data = {"BTC/USDT": make_ohlcv()}
        scorer.score_pairs(price_data)
        assert len(scorer.get_last_scores()) == 1

    def test_volume_ratio_near_one_for_constant_volume(self, scorer):
        """Constant volume → volume_ratio ≈ 1.0 (last bar ≈ 20-bar average)."""
        price_data = {"BTC/USDT": make_ohlcv(vol_mult=1.0)}
        result = scorer.score_pairs(price_data)
        assert result[0].volume_ratio == pytest.approx(1.0, rel=0.05)

    def test_volume_spike_increases_volume_ratio(self, scorer):
        """2× volume spike on last bar → volume_ratio ≈ 2.0."""
        price_data = {"BTC/USDT": make_ohlcv(vol_mult=2.0)}
        result = scorer.score_pairs(price_data)
        # With constant volume[:-1]=1000 and last=2000, ratio ≈ 2000/1050 ≈ 1.9
        assert result[0].volume_ratio > 1.5


# ---------------------------------------------------------------------------
# PairScorer.select_top_pairs()
# ---------------------------------------------------------------------------


class TestSelectTopPairs:
    @pytest.fixture()
    def scorer(self):
        return PairScorer()

    def test_returns_list_of_strings(self, scorer):
        price_data = {"BTC/USDT": make_ohlcv(), "ETH/USDT": make_ohlcv()}
        result = scorer.select_top_pairs(price_data)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_respects_pair_selector_top_n(self, scorer):
        """Returns at most PAIR_SELECTOR_TOP_N symbols."""
        price_data = {sym: make_ohlcv() for sym in ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]}
        result = scorer.select_top_pairs(price_data)
        assert len(result) <= Config.PAIR_SELECTOR_TOP_N

    def test_returns_highest_scoring_pairs(self, scorer):
        """Top pair should have the highest score."""
        price_data = {
            "LOW/USDT": make_ohlcv(atr_factor=0.5, vol_mult=0.5),
            "HIGH/USDT": make_ohlcv(atr_factor=5.0, vol_mult=5.0),
            "MID/USDT": make_ohlcv(atr_factor=1.5, vol_mult=1.5),
        }
        result = scorer.select_top_pairs(price_data)
        assert result[0] == "HIGH/USDT"

    def test_fallback_to_trading_pairs_on_empty_data(self, scorer):
        """Empty price_data → falls back to Config.TRADING_PAIRS."""
        result = scorer.select_top_pairs({})
        assert result == list(Config.TRADING_PAIRS)

    def test_fallback_to_trading_pairs_when_all_data_insufficient(self, scorer):
        """All pairs have < 20 bars → no scores → falls back to Config.TRADING_PAIRS."""
        price_data = {sym: make_ohlcv(rows=10) for sym in ["A/USDT", "B/USDT"]}
        result = scorer.select_top_pairs(price_data)
        assert result == list(Config.TRADING_PAIRS)

    def test_select_does_not_exceed_available_pairs(self, scorer):
        """If fewer pairs pass scoring than PAIR_SELECTOR_TOP_N, returns all scored pairs."""
        price_data = {"ONLY/USDT": make_ohlcv()}
        result = scorer.select_top_pairs(price_data)
        assert len(result) == 1
        assert result[0] == "ONLY/USDT"
