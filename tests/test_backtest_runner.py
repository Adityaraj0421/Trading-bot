"""
Unit tests for backtest_runner.py — BacktestRunner orchestrating scenario/pair/timeframe backtests.

Mocks DataFetcher and Backtester in the backtest_runner module namespace.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest_runner import BacktestRunner
from demo_data import generate_ohlcv


def _mock_metrics():
    return {
        "total_return_pct": 5.0,
        "sharpe_ratio": 1.5,
        "max_drawdown_pct": -3.0,
        "total_trades": 12,
        "win_rate": 0.6,
    }


@pytest.fixture()
def mock_fetcher():
    fetcher = MagicMock()
    fetcher.fetch_ohlcv.return_value = generate_ohlcv(periods=200, seed=42)
    return fetcher


@pytest.fixture()
def runner(monkeypatch, mock_fetcher):
    """Patch DataFetcher so BacktestRunner uses our mock."""
    monkeypatch.setattr(
        "backtest_runner.DataFetcher",
        lambda: mock_fetcher,
    )
    # Patch Backtester so bt.run() returns metrics
    mock_bt_instance = MagicMock()
    mock_bt_instance.run.return_value = _mock_metrics()
    mock_bt_class = MagicMock(return_value=mock_bt_instance)
    monkeypatch.setattr("backtest_runner.Backtester", mock_bt_class)

    r = BacktestRunner()
    r.fetcher = mock_fetcher
    return r


# ---------------------------------------------------------------------------
# run_scenario
# ---------------------------------------------------------------------------


class TestRunScenario:
    def test_returns_result_dict(self, runner, monkeypatch):
        mock_bt = MagicMock()
        mock_bt.run.return_value = _mock_metrics()
        monkeypatch.setattr("backtest_runner.Backtester", MagicMock(return_value=mock_bt))
        result = runner.run_scenario("bull_run")
        assert "type" in result
        assert "scenario" in result
        assert "pair" in result
        assert "timeframe" in result
        assert "periods" in result
        assert "run_at" in result
        assert "metrics" in result

    def test_type_is_scenario(self, runner, monkeypatch):
        mock_bt = MagicMock()
        mock_bt.run.return_value = _mock_metrics()
        monkeypatch.setattr("backtest_runner.Backtester", MagicMock(return_value=mock_bt))
        result = runner.run_scenario("bull_run")
        assert result["type"] == "scenario"

    def test_pair_is_synthetic(self, runner, monkeypatch):
        mock_bt = MagicMock()
        mock_bt.run.return_value = _mock_metrics()
        monkeypatch.setattr("backtest_runner.Backtester", MagicMock(return_value=mock_bt))
        result = runner.run_scenario("bear_market")
        assert result["pair"] == "SYNTHETIC"

    def test_appends_to_results(self, runner, monkeypatch):
        mock_bt = MagicMock()
        mock_bt.run.return_value = _mock_metrics()
        monkeypatch.setattr("backtest_runner.Backtester", MagicMock(return_value=mock_bt))
        assert len(runner.results) == 0
        runner.run_scenario("bull_run")
        assert len(runner.results) == 1


# ---------------------------------------------------------------------------
# run_multi_pair
# ---------------------------------------------------------------------------


class TestRunMultiPair:
    def test_returns_results_per_pair(self, runner):
        results = runner.run_multi_pair(pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        assert len(results) == 3

    def test_insufficient_data_error(self, runner):
        runner.fetcher.fetch_ohlcv.return_value = pd.DataFrame()
        results = runner.run_multi_pair(pairs=["BTC/USDT"])
        assert results[0]["error"] == "insufficient_data"

    def test_exception_as_error(self, runner):
        runner.fetcher.fetch_ohlcv.side_effect = Exception("boom")
        results = runner.run_multi_pair(pairs=["BTC/USDT"])
        assert "error" in results[0]

    def test_successful_structure(self, runner):
        results = runner.run_multi_pair(pairs=["BTC/USDT"])
        r = results[0]
        assert r["type"] == "multi_pair"
        assert r["pair"] == "BTC/USDT"
        assert "metrics" in r

    def test_appended_to_runner(self, runner):
        runner.run_multi_pair(pairs=["BTC/USDT", "ETH/USDT"])
        assert len(runner.results) >= 2


# ---------------------------------------------------------------------------
# run_all_scenarios
# ---------------------------------------------------------------------------


class TestRunAllScenarios:
    def test_runs_all_scenarios(self, runner, monkeypatch):
        mock_bt = MagicMock()
        mock_bt.run.return_value = _mock_metrics()
        monkeypatch.setattr("backtest_runner.Backtester", MagicMock(return_value=mock_bt))
        results = runner.run_all_scenarios()
        assert len(results) == 6  # list_scenarios returns 6

    def test_each_has_metrics(self, runner, monkeypatch):
        mock_bt = MagicMock()
        mock_bt.run.return_value = _mock_metrics()
        monkeypatch.setattr("backtest_runner.Backtester", MagicMock(return_value=mock_bt))
        results = runner.run_all_scenarios()
        for r in results:
            assert "metrics" in r

    def test_appended_to_runner(self, runner, monkeypatch):
        mock_bt = MagicMock()
        mock_bt.run.return_value = _mock_metrics()
        monkeypatch.setattr("backtest_runner.Backtester", MagicMock(return_value=mock_bt))
        runner.run_all_scenarios()
        assert len(runner.results) == 6


# ---------------------------------------------------------------------------
# run_multi_timeframe
# ---------------------------------------------------------------------------


class TestRunMultiTimeframe:
    def test_returns_per_timeframe(self, runner):
        results = runner.run_multi_timeframe(timeframes=["15m", "1h", "4h"])
        assert len(results) == 3

    def test_insufficient_data_skipped(self, runner):
        # Return short df → gets skipped silently (no error entry)
        runner.fetcher.fetch_ohlcv.return_value = generate_ohlcv(periods=50, seed=42)
        results = runner.run_multi_timeframe(timeframes=["1h"])
        assert len(results) == 0

    def test_exception_as_error(self, runner):
        runner.fetcher.fetch_ohlcv.side_effect = Exception("boom")
        results = runner.run_multi_timeframe(timeframes=["1h"])
        assert "error" in results[0]
        assert "timeframe" in results[0]


# ---------------------------------------------------------------------------
# get_all_results
# ---------------------------------------------------------------------------


class TestGetAllResults:
    def test_empty_initially(self, runner):
        assert runner.get_all_results() == []

    def test_accumulates_results(self, runner, monkeypatch):
        mock_bt = MagicMock()
        mock_bt.run.return_value = _mock_metrics()
        monkeypatch.setattr("backtest_runner.Backtester", MagicMock(return_value=mock_bt))
        runner.run_scenario("bull_run")
        runner.run_multi_pair(pairs=["BTC/USDT"])
        assert len(runner.get_all_results()) >= 2
