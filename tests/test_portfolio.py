"""
Unit tests for portfolio.py — PairAllocation, PortfolioManager.

Tests correlation computation, capital allocation with volatility/correlation
adjustments, portfolio risk metrics (Herfindahl, corr risk), and rebalancing.
"""

import pytest
import numpy as np
import pandas as pd
from portfolio import PairAllocation, PortfolioManager


@pytest.fixture()
def portfolio():
    return PortfolioManager(pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"])


def _make_price_series(n=50, start=100, seed=42):
    """Generate a price Series with n data points."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.02, n)
    prices = start * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.date_range("2024-01-01", periods=n, freq="h"))


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestPortfolioInit:
    def test_custom_pairs(self):
        pm = PortfolioManager(pairs=["BTC/USDT", "ETH/USDT"])
        assert pm.pairs == ["BTC/USDT", "ETH/USDT"]

    def test_equal_weights(self, portfolio):
        for w in portfolio.weights.values():
            assert w == pytest.approx(1 / 3)

    def test_empty_correlations(self, portfolio):
        assert portfolio.correlations.empty


# ---------------------------------------------------------------------------
# update_prices
# ---------------------------------------------------------------------------

class TestUpdatePrices:
    def test_stores_returns(self, portfolio):
        prices = _make_price_series(n=50)
        portfolio.update_prices("BTC/USDT", prices)
        assert "BTC/USDT" in portfolio.pair_returns
        assert len(portfolio.pair_returns["BTC/USDT"]) > 0

    def test_stores_volatility(self, portfolio):
        prices = _make_price_series(n=50)
        portfolio.update_prices("BTC/USDT", prices)
        assert portfolio.pair_volatility["BTC/USDT"] > 0

    def test_short_series_default_vol(self, portfolio):
        prices = _make_price_series(n=5)
        portfolio.update_prices("BTC/USDT", prices)
        assert portfolio.pair_volatility["BTC/USDT"] == 0.02


# ---------------------------------------------------------------------------
# compute_correlations
# ---------------------------------------------------------------------------

class TestComputeCorrelations:
    def test_single_pair_returns_empty(self, portfolio):
        portfolio.update_prices("BTC/USDT", _make_price_series(50, seed=1))
        corr = portfolio.compute_correlations()
        assert corr.empty

    def test_two_pairs_returns_matrix(self, portfolio):
        portfolio.update_prices("BTC/USDT", _make_price_series(50, seed=1))
        portfolio.update_prices("ETH/USDT", _make_price_series(50, seed=2))
        corr = portfolio.compute_correlations()
        assert corr.shape == (2, 2)

    def test_correlation_diagonal_is_one(self, portfolio):
        portfolio.update_prices("BTC/USDT", _make_price_series(50, seed=1))
        portfolio.update_prices("ETH/USDT", _make_price_series(50, seed=2))
        corr = portfolio.compute_correlations()
        for pair in corr.index:
            assert corr.loc[pair, pair] == pytest.approx(1.0)

    def test_correlation_range(self, portfolio):
        portfolio.update_prices("BTC/USDT", _make_price_series(50, seed=1))
        portfolio.update_prices("ETH/USDT", _make_price_series(50, seed=2))
        corr = portfolio.compute_correlations()
        assert (corr >= -1).all().all()
        assert (corr <= 1).all().all()

    def test_insufficient_data_returns_empty(self, portfolio):
        portfolio.update_prices("BTC/USDT", _make_price_series(10, seed=1))
        portfolio.update_prices("ETH/USDT", _make_price_series(10, seed=2))
        corr = portfolio.compute_correlations()
        assert corr.empty


# ---------------------------------------------------------------------------
# get_allocation
# ---------------------------------------------------------------------------

class TestGetAllocation:
    def test_returns_pair_allocation(self, portfolio):
        alloc = portfolio.get_allocation("BTC/USDT")
        assert isinstance(alloc, PairAllocation)

    def test_weight_positive(self, portfolio):
        alloc = portfolio.get_allocation("BTC/USDT")
        assert alloc.weight > 0

    def test_no_positions_no_penalty(self, portfolio):
        alloc = portfolio.get_allocation("BTC/USDT")
        assert alloc.correlation_penalty == 0

    def test_is_tradeable_by_default(self, portfolio):
        alloc = portfolio.get_allocation("BTC/USDT")
        assert alloc.is_tradeable is True

    def test_high_correlation_reduces_allocation(self, portfolio):
        # Set up perfectly correlated pairs
        prices = _make_price_series(50, seed=1)
        portfolio.update_prices("BTC/USDT", prices)
        portfolio.update_prices("ETH/USDT", prices)  # Same data = corr=1.0
        portfolio.compute_correlations()

        alloc_no_pos = portfolio.get_allocation("ETH/USDT")
        alloc_with_pos = portfolio.get_allocation("ETH/USDT", existing_positions=["BTC/USDT"])
        assert alloc_with_pos.correlation_penalty > 0
        assert alloc_with_pos.weight < alloc_no_pos.weight

    def test_lower_vol_gets_higher_allocation(self, portfolio):
        # BTC: low vol, ETH: high vol
        portfolio.pair_volatility["BTC/USDT"] = 0.01
        portfolio.pair_volatility["ETH/USDT"] = 0.05
        alloc_btc = portfolio.get_allocation("BTC/USDT")
        alloc_eth = portfolio.get_allocation("ETH/USDT")
        assert alloc_btc.weight > alloc_eth.weight


# ---------------------------------------------------------------------------
# get_portfolio_risk
# ---------------------------------------------------------------------------

class TestGetPortfolioRisk:
    def test_empty_positions(self, portfolio):
        result = portfolio.get_portfolio_risk([])
        assert result["total_exposure"] == 0
        assert result["concentration"] == 0

    def test_single_position_concentration(self, portfolio):
        pos = {"symbol": "BTC/USDT", "notional_value": 100}
        # Make mock-like with attribute access
        class MockPos:
            symbol = "BTC/USDT"
            notional_value = 100
        result = portfolio.get_portfolio_risk([MockPos()])
        assert result["concentration"] == pytest.approx(1.0)

    def test_two_equal_positions_concentration(self, portfolio):
        class MockPos:
            def __init__(self, sym, val):
                self.symbol = sym
                self.notional_value = val
        positions = [MockPos("BTC/USDT", 100), MockPos("ETH/USDT", 100)]
        result = portfolio.get_portfolio_risk(positions)
        assert result["concentration"] == pytest.approx(0.5)

    def test_corr_risk_low_default(self, portfolio):
        class MockPos:
            symbol = "BTC/USDT"
            notional_value = 100
        result = portfolio.get_portfolio_risk([MockPos()])
        assert result["corr_risk"] == "low"

    def test_corr_risk_high_when_correlated(self, portfolio):
        # Set up correlation matrix with high correlation
        prices = _make_price_series(50, seed=1)
        portfolio.update_prices("BTC/USDT", prices)
        portfolio.update_prices("ETH/USDT", prices)  # Perfect correlation
        portfolio.compute_correlations()

        class MockPos:
            def __init__(self, sym, val):
                self.symbol = sym
                self.notional_value = val
        positions = [MockPos("BTC/USDT", 100), MockPos("ETH/USDT", 100)]
        result = portfolio.get_portfolio_risk(positions)
        assert result["corr_risk"] == "high"

    def test_pair_exposure_dict(self, portfolio):
        class MockPos:
            def __init__(self, sym, val):
                self.symbol = sym
                self.notional_value = val
        positions = [MockPos("BTC/USDT", 100), MockPos("ETH/USDT", 200)]
        result = portfolio.get_portfolio_risk(positions)
        assert result["pair_exposure"]["BTC/USDT"] == 100
        assert result["pair_exposure"]["ETH/USDT"] == 200


# ---------------------------------------------------------------------------
# rebalance_weights
# ---------------------------------------------------------------------------

class TestRebalanceWeights:
    def test_no_volatility_data_noop(self, portfolio):
        original = portfolio.weights.copy()
        portfolio.rebalance_weights()
        assert portfolio.weights == original

    def test_inverse_vol_weighting(self, portfolio):
        portfolio.pair_volatility = {
            "BTC/USDT": 0.01,   # Low vol → high weight
            "ETH/USDT": 0.03,   # Med vol
            "SOL/USDT": 0.06,   # High vol → low weight
        }
        portfolio.rebalance_weights()
        assert portfolio.weights["BTC/USDT"] > portfolio.weights["ETH/USDT"]
        assert portfolio.weights["ETH/USDT"] > portfolio.weights["SOL/USDT"]

    def test_weights_sum_to_one(self, portfolio):
        portfolio.pair_volatility = {
            "BTC/USDT": 0.02,
            "ETH/USDT": 0.03,
            "SOL/USDT": 0.05,
        }
        portfolio.rebalance_weights()
        total = sum(portfolio.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_zero_vol_handled(self, portfolio):
        portfolio.pair_volatility = {
            "BTC/USDT": 0.0,    # Zero vol → gets capped inverse
            "ETH/USDT": 0.02,
            "SOL/USDT": 0.04,
        }
        portfolio.rebalance_weights()
        # Should not crash, zero vol gets 1/0 → capped at 50.0
        assert portfolio.weights["BTC/USDT"] > 0
        total = sum(portfolio.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)
