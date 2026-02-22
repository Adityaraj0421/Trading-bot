"""Tests for the risk simulation module."""

import numpy as np
import pytest

from risk_simulation.monte_carlo import MonteCarloResult, MonteCarloSimulator
from risk_simulation.scenarios import HISTORICAL_SCENARIOS, StressTestRunner
from risk_simulation.var_calculator import VaRCalculator
from risk_simulation.visualizer import RiskVisualizer


class TestMonteCarloSimulator:
    def test_basic_run(self):
        sim = MonteCarloSimulator(n_simulations=100, n_days=30)
        returns = list(np.random.normal(0.001, 0.02, 50))
        result = sim.run(returns, initial_equity=1000.0)
        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 100
        assert result.n_days == 30
        assert result.initial_equity == 1000.0

    def test_var_is_negative(self):
        """VaR should be negative (it represents a loss)."""
        sim = MonteCarloSimulator(n_simulations=1000, n_days=30)
        returns = list(np.random.normal(-0.001, 0.02, 100))
        result = sim.run(returns, initial_equity=1000.0)
        assert result.var_95 < 0
        assert result.var_99 < 0

    def test_cvar_worse_than_var(self):
        """CVaR should be worse (more negative) than VaR."""
        sim = MonteCarloSimulator(n_simulations=1000, n_days=30)
        returns = list(np.random.normal(0, 0.03, 100))
        result = sim.run(returns, initial_equity=1000.0)
        assert result.cvar_95 <= result.var_95

    def test_insufficient_data(self):
        sim = MonteCarloSimulator(n_simulations=100, n_days=30)
        result = sim.run([], initial_equity=1000.0)
        assert result.n_simulations == 0
        assert result.median_final_equity == 1000.0

    def test_result_to_dict(self):
        sim = MonteCarloSimulator(n_simulations=100, n_days=30)
        returns = list(np.random.normal(0.001, 0.02, 50))
        result = sim.run(returns, initial_equity=1000.0)
        d = result.to_dict()
        assert "var_95" in d
        assert "interpretation" in d
        assert "risk_level" in d["interpretation"]

    def test_paths_summary_generated(self):
        sim = MonteCarloSimulator(n_simulations=100, n_days=50)
        returns = list(np.random.normal(0.001, 0.02, 50))
        result = sim.run(returns, initial_equity=1000.0)
        assert len(result.paths_summary) > 0
        assert "day" in result.paths_summary[0]
        assert "p50" in result.paths_summary[0]

    def test_probability_of_ruin_range(self):
        sim = MonteCarloSimulator(n_simulations=100, n_days=30)
        returns = list(np.random.normal(0.001, 0.01, 50))
        result = sim.run(returns, initial_equity=1000.0)
        assert 0 <= result.probability_of_ruin <= 1


class TestStressTestRunner:
    def test_run_all_scenarios(self):
        runner = StressTestRunner()
        results = runner.run_stress_test(1000.0)
        assert len(results) == len(HISTORICAL_SCENARIOS)
        for r in results:
            assert "name" in r
            assert "projected_loss" in r
            assert "survived" in r

    def test_covid_crash(self):
        runner = StressTestRunner()
        results = runner.run_stress_test(1000.0, [HISTORICAL_SCENARIOS[0]])
        assert len(results) == 1
        assert results[0]["projected_loss"] > 0  # Should lose money

    def test_equity_path_length(self):
        runner = StressTestRunner()
        results = runner.run_stress_test(1000.0)
        for r in results:
            # equity_path has initial + each day
            assert len(r["equity_path"]) > 1

    def test_list_scenarios(self):
        runner = StressTestRunner()
        scenarios = runner.list_scenarios()
        assert len(scenarios) == 4
        assert "name" in scenarios[0]


class TestVaRCalculator:
    def test_historical_var(self):
        returns = list(np.random.normal(0, 0.02, 100))
        result = VaRCalculator.historical_var(returns, 0.95, 1000.0)
        assert "var" in result
        assert "cvar" in result
        assert result["method"] == "historical"

    def test_parametric_var(self):
        pytest.importorskip("scipy", reason="scipy not available")
        returns = list(np.random.normal(0, 0.02, 100))
        result = VaRCalculator.parametric_var(returns, 0.95, 1000.0)
        assert "var" in result
        assert result["method"] == "parametric"

    def test_insufficient_data(self):
        result = VaRCalculator.historical_var([], 0.95, 1000.0)
        assert result["var"] == 0
        assert "error" in result


class TestRiskVisualizer:
    def test_stress_test_comparison(self):
        stress_results = [{"name": "Test", "projected_loss_pct": 10.0, "survived": True}]
        chart_data = RiskVisualizer.stress_test_comparison(stress_results)
        assert len(chart_data) == 1
        assert chart_data[0]["scenario"] == "Test"

    def test_fan_chart_data_empty(self):
        result = MonteCarloResult(
            n_simulations=0,
            n_days=0,
            initial_equity=1000,
            var_95=0,
            var_99=0,
            cvar_95=0,
            cvar_99=0,
            max_drawdown_median=0,
            max_drawdown_95th=0,
            probability_of_ruin=0,
            median_final_equity=1000,
            percentile_5=1000,
            percentile_25=1000,
            percentile_75=1000,
            percentile_95=1000,
            mean_final_equity=1000,
            paths_summary=[],
        )
        data = RiskVisualizer.fan_chart_data(result)
        assert data == []

    def test_drawdown_histogram(self):
        drawdowns = list(np.random.uniform(0.01, 0.3, 100))
        hist = RiskVisualizer.drawdown_histogram(drawdowns, bins=10)
        assert len(hist) == 10
        assert "count" in hist[0]
