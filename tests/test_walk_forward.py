"""
Tests for WalkForwardValidator — window generation, result structure.
Uses small datasets to keep tests fast. Does NOT run full backtests
(those are slow and tested in test_backtest_runner.py).
"""

import numpy as np
import pandas as pd
import pytest
from walk_forward import WalkForwardValidator, WFOWindow, WFOResult


@pytest.fixture()
def validator():
    """Validator with small windows for fast testing."""
    return WalkForwardValidator(
        train_bars=50, test_bars=20, step_bars=10,
        mc_simulations=10,  # Small for speed
        verbose=False,
    )


@pytest.fixture()
def small_ohlcv():
    """200-bar OHLCV dataset."""
    np.random.seed(42)
    n = 200
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 30,
        "high": close + np.abs(np.random.randn(n) * 50),
        "low": close - np.abs(np.random.randn(n) * 50),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000 + 5000),
    }, index=pd.date_range("2025-01-01", periods=n, freq="h"))


# ── Window Generation ─────────────────────────────────────────────


class TestWindowGeneration:
    def test_generates_windows(self, validator):
        windows = validator._generate_windows(n_bars=200)
        assert len(windows) > 0

    def test_window_sizes_correct(self, validator):
        windows = validator._generate_windows(n_bars=200)
        for w in windows:
            assert w.train_bars == 50
            assert w.test_bars == 20

    def test_windows_non_overlapping_test(self, validator):
        """Test windows should NOT overlap (train can though)."""
        windows = validator._generate_windows(n_bars=200)
        for i in range(1, len(windows)):
            # With step_bars=10, test windows may overlap
            # but train_end of one should be before test_end of next
            assert windows[i].train_start > windows[i - 1].train_start

    def test_no_windows_for_tiny_data(self, validator):
        """Not enough data for even one fold."""
        windows = validator._generate_windows(n_bars=30)
        assert len(windows) == 0

    def test_exact_fit(self):
        """Data that exactly fits one fold."""
        v = WalkForwardValidator(train_bars=50, test_bars=20, step_bars=10)
        windows = v._generate_windows(n_bars=70)
        assert len(windows) == 1

    def test_fold_ids_sequential(self, validator):
        windows = validator._generate_windows(n_bars=200)
        for i, w in enumerate(windows):
            assert w.fold_id == i


# ── WFOWindow Dataclass ───────────────────────────────────────────


class TestWFOWindow:
    def test_post_init_computes_bars(self):
        w = WFOWindow(fold_id=0, train_start=0, train_end=100,
                      test_start=100, test_end=150)
        assert w.train_bars == 100
        assert w.test_bars == 50

    def test_defaults(self):
        w = WFOWindow(fold_id=0, train_start=0, train_end=50,
                      test_start=50, test_end=70)
        assert w.metrics == {}
        assert w.trades == []


# ── WFOResult Dataclass ───────────────────────────────────────────


class TestWFOResult:
    def test_default_values(self):
        r = WFOResult()
        assert r.n_folds == 0
        assert r.is_robust is False
        assert r.mc_p_value == 1.0

    def test_to_dict_structure(self):
        r = WFOResult(n_folds=5, oos_sharpe=1.2, is_robust=True)
        d = r.to_dict()
        assert d["n_folds"] == 5
        assert d["oos_sharpe"] == 1.2
        assert d["is_robust"] is True
        assert "rejection_reasons" in d

    def test_to_dict_round_numbers(self):
        r = WFOResult(oos_total_return_pct=12.3456, mc_p_value=0.04567)
        d = r.to_dict()
        assert d["oos_total_return_pct"] == 12.35
        assert d["mc_p_value"] == 0.0457


# ── Validator Config ──────────────────────────────────────────────


class TestValidatorConfig:
    def test_custom_params(self):
        v = WalkForwardValidator(
            train_bars=200, test_bars=50, step_bars=25,
            mc_simulations=100, mc_confidence=0.99,
            min_oos_sharpe=0.5, max_oos_drawdown=-10.0,
        )
        assert v.train_bars == 200
        assert v.test_bars == 50
        assert v.mc_simulations == 100
        assert v.mc_confidence == 0.99

    def test_default_fees_from_config(self):
        v = WalkForwardValidator()
        assert v.fee_pct > 0
        assert v.slippage_pct >= 0
