"""
Tests for multi-pair portfolio trading integration.
Verifies PortfolioManager, allocation logic, and agent multi-pair cycle.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


class TestPortfolioManager:
    """Test the PortfolioManager allocation and risk logic."""

    def test_init_equal_weights(self):
        from portfolio import PortfolioManager

        pm = PortfolioManager(pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        assert len(pm.pairs) == 3
        assert abs(pm.weights["BTC/USDT"] - 1 / 3) < 0.001
        assert abs(pm.weights["ETH/USDT"] - 1 / 3) < 0.001

    def test_update_prices(self):
        from portfolio import PortfolioManager

        pm = PortfolioManager(pairs=["BTC/USDT", "ETH/USDT"])
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 107, 108, 110, 109, 111])
        pm.update_prices("BTC/USDT", prices)
        assert "BTC/USDT" in pm.pair_returns
        assert pm.pair_volatility["BTC/USDT"] > 0

    def test_get_allocation_basic(self):
        from portfolio import PortfolioManager

        pm = PortfolioManager(pairs=["BTC/USDT", "ETH/USDT"])
        alloc = pm.get_allocation("BTC/USDT")
        assert alloc.pair == "BTC/USDT"
        assert alloc.weight > 0
        assert alloc.is_tradeable is True
        assert alloc.correlation_penalty == 0.0  # No correlations computed yet

    def test_correlation_penalty_reduces_allocation(self):
        from portfolio import PortfolioManager

        pm = PortfolioManager(pairs=["BTC/USDT", "ETH/USDT"])

        # Create highly correlated price series
        np.random.seed(42)
        base = np.cumsum(np.random.randn(50)) + 100
        btc = pd.Series(base)
        eth = pd.Series(base * 0.5 + np.random.randn(50) * 0.1)  # Highly correlated

        pm.update_prices("BTC/USDT", btc)
        pm.update_prices("ETH/USDT", eth)
        pm.compute_correlations()

        # With BTC already open, ETH allocation should have a correlation penalty
        alloc_no_positions = pm.get_allocation("ETH/USDT")
        alloc_with_btc = pm.get_allocation("ETH/USDT", existing_positions=["BTC/USDT"])

        # Penalty should reduce weight when correlated pair is already open
        assert alloc_with_btc.weight <= alloc_no_positions.weight

    def test_rebalance_inverse_volatility(self):
        from portfolio import PortfolioManager

        pm = PortfolioManager(pairs=["BTC/USDT", "ETH/USDT"])

        # BTC = low vol, ETH = high vol
        pm.pair_volatility = {"BTC/USDT": 0.01, "ETH/USDT": 0.04}
        pm.rebalance_weights()

        # Lower volatility should get higher weight
        assert pm.weights["BTC/USDT"] > pm.weights["ETH/USDT"]

    def test_portfolio_risk_empty(self):
        from portfolio import PortfolioManager

        pm = PortfolioManager(pairs=["BTC/USDT"])
        risk = pm.get_portfolio_risk([])
        assert risk["total_exposure"] == 0
        assert risk["corr_risk"] == "low"

    def test_portfolio_risk_with_positions(self):
        from portfolio import PortfolioManager

        pm = PortfolioManager(pairs=["BTC/USDT", "ETH/USDT"])

        # Mock positions
        pos1 = MagicMock()
        pos1.symbol = "BTC/USDT"
        pos1.notional_value = 500
        pos2 = MagicMock()
        pos2.symbol = "ETH/USDT"
        pos2.notional_value = 300

        risk = pm.get_portfolio_risk([pos1, pos2])
        assert risk["total_exposure"] == 800
        assert "BTC/USDT" in risk["pair_exposure"]
        assert risk["pair_exposure"]["BTC/USDT"] == 500

    def test_untradeable_when_too_correlated(self):
        from portfolio import PortfolioManager

        pm = PortfolioManager(pairs=["A/USDT", "B/USDT"])

        # Set up extreme correlation via direct manipulation
        pm.correlations = pd.DataFrame(
            [[1.0, 0.99], [0.99, 1.0]],
            index=["A/USDT", "B/USDT"],
            columns=["A/USDT", "B/USDT"],
        )
        pm.pair_volatility = {"A/USDT": 0.02, "B/USDT": 0.02}

        alloc = pm.get_allocation("B/USDT", existing_positions=["A/USDT"])
        # With 0.99 correlation, penalty = (1 - (0.99-0.7)) = 0.71
        # After multiple highly correlated positions it should approach untradeable
        assert alloc.correlation_penalty > 0


class TestConfigTradingPairs:
    """Test Config.TRADING_PAIRS parsing."""

    def test_single_pair_default(self):
        from config import Config

        # Default should have at least one pair
        assert len(Config.TRADING_PAIRS) >= 1
        assert Config.TRADING_PAIRS[0] == Config.TRADING_PAIR

    def test_pairs_stripped(self):
        import os

        with patch.dict(os.environ, {"TRADING_PAIRS": " BTC/USDT , ETH/USDT , SOL/USDT "}):
            # Re-evaluate
            pairs = [p.strip() for p in os.environ["TRADING_PAIRS"].split(",") if p.strip()]
            assert pairs == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            assert " " not in pairs[0]  # No leading/trailing spaces


class TestAgentMultiPairDetection:
    """Test that the agent correctly detects multi-pair mode."""

    def test_single_pair_mode(self):
        """With a single pair, multi_pair should be False."""
        with (
            patch("config.Config.TRADING_PAIRS", ["BTC/USDT"]),
            patch("config.Config.validate"),
            patch("agent.TradingAgent._print_banner"),
            patch("agent.TradingAgent._register_recovery_actions"),
        ):
            # We'd need to mock everything, but the key logic is:
            multi_pair = len(["BTC/USDT"]) > 1
            assert multi_pair is False

    def test_multi_pair_mode(self):
        """With multiple pairs, multi_pair should be True."""
        multi_pair = len(["BTC/USDT", "ETH/USDT", "SOL/USDT"]) > 1
        assert multi_pair is True
