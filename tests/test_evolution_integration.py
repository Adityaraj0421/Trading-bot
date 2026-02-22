"""
Integration tests for strategy evolution → live strategy hot-reload,
and auto-optimizer → backtester integration.
"""

import numpy as np

from auto_optimizer import AutoOptimizer
from decision_engine import DecisionEngine
from strategies import (
    BreakoutStrategy,
    GridStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    ScalpingStrategy,
    SentimentDrivenStrategy,
    StrategyEngine,
)
from strategy_evolver import StrategyEvolver

# ---------------------------------------------------------------------------
# Strategy parameterization
# ---------------------------------------------------------------------------


class TestStrategyParamInjection:
    """Verify strategies accept and use evolved parameters."""

    def test_momentum_accepts_params(self):
        params = {"rsi_oversold": 25, "rsi_overbought": 65, "confidence_base": 0.7}
        s = MomentumStrategy(params=params)
        assert s.rsi_oversold == 25
        assert s.rsi_overbought == 65
        assert s.confidence_base == 0.7

    def test_momentum_defaults_without_params(self):
        s = MomentumStrategy()
        assert s.rsi_oversold == 30
        assert s.rsi_overbought == 70

    def test_mean_reversion_accepts_params(self):
        s = MeanReversionStrategy(params={"rsi_low": 20, "rsi_high": 80})
        assert s.rsi_low == 20
        assert s.rsi_high == 80

    def test_breakout_accepts_params(self):
        s = BreakoutStrategy(params={"lookback": 30, "volume_mult": 2.0})
        assert s.lookback == 30
        assert s.volume_mult == 2.0

    def test_grid_accepts_params(self):
        s = GridStrategy(params={"grid_size_pct": 2.0, "num_levels": 7})
        assert s.grid_spacing == 0.02  # 2.0 / 100
        assert s.grid_levels == 7

    def test_scalping_accepts_params(self):
        s = ScalpingStrategy(params={"volume_spike": 3.0})
        assert s.volume_spike == 3.0

    def test_sentiment_accepts_params(self):
        s = SentimentDrivenStrategy(params={"fear_threshold": 20, "greed_threshold": 80, "composite_threshold": 0.4})
        assert s.fear_threshold == 20
        assert s.greed_threshold == 80
        assert s.composite_threshold == 0.4


# ---------------------------------------------------------------------------
# StrategyEngine evolved params hot-reload
# ---------------------------------------------------------------------------


class TestStrategyEngineEvolution:
    def test_engine_accepts_evolved_params(self):
        params = {
            "Momentum": {"rsi_oversold": 25, "rsi_overbought": 65},
            "MeanReversion": {"rsi_low": 20, "rsi_high": 80},
        }
        engine = StrategyEngine(evolved_params=params)
        assert engine.strategies["Momentum"].rsi_oversold == 25
        assert engine.strategies["MeanReversion"].rsi_low == 20
        # Unparameterized strategies use defaults
        assert engine.strategies["Breakout"].lookback == 20

    def test_apply_evolved_params_hot_reload(self):
        engine = StrategyEngine()
        assert engine.strategies["Momentum"].rsi_oversold == 30  # default

        engine.apply_evolved_params(
            {
                "Momentum": {"rsi_oversold": 22, "rsi_overbought": 72},
            }
        )
        assert engine.strategies["Momentum"].rsi_oversold == 22
        assert engine.strategies["Momentum"].rsi_overbought == 72

    def test_apply_evolved_params_ignores_unknown(self):
        engine = StrategyEngine()
        engine.apply_evolved_params({"NonExistent": {"foo": 1}})
        # v9.0: 9 strategies (was 6) — added VWAP, OBVDivergence, EMACrossover
        assert len(engine.strategies) == 9  # unchanged


# ---------------------------------------------------------------------------
# Evolution → Engine pipeline
# ---------------------------------------------------------------------------


class TestEvolutionPipeline:
    def test_evolver_best_params_feed_into_engine(self):
        """Simulate a full evolution → hot-reload cycle."""
        evolver = StrategyEvolver(population_size=5)
        evolver.initialize_population(["Momentum", "MeanReversion"])

        # Evaluate fitness for all genomes
        for name in evolver.populations:
            for g in evolver.populations[name]:
                metrics = {
                    "sharpe_ratio": np.random.uniform(0.5, 2.0),
                    "total_trades": np.random.randint(5, 30),
                    "max_drawdown_pct": -np.random.uniform(1, 10),
                    "win_rate": np.random.uniform(0.4, 0.7),
                }
                evolver.evaluate_fitness(g, metrics)

        # Evolve
        for name in evolver.populations:
            evolver.evolve(name)

        # Get best params
        best = evolver.get_all_best()
        evolved_params = {name: data["parameters"] for name, data in best.items() if data.get("parameters")}

        # Feed into engine
        engine = StrategyEngine(evolved_params=evolved_params)

        # Verify params were applied
        for name in evolved_params:
            strat = engine.strategies[name]
            for key, val in evolved_params[name].items():
                if hasattr(strat, key):
                    assert getattr(strat, key) == val


# ---------------------------------------------------------------------------
# Auto-Optimizer
# ---------------------------------------------------------------------------


class TestAutoOptimizerIntegration:
    def test_optimizer_suggest_and_record(self):
        optimizer = AutoOptimizer(max_trials=20)
        params = optimizer.suggest_params()
        assert "stop_loss_pct" in params
        assert "confidence_threshold" in params

        metrics = {
            "total_return_pct": 5.0,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": -3.0,
            "total_trades": 15,
        }
        optimizer.record_result(params, metrics)
        assert optimizer.best_result is not None
        assert optimizer.best_result.sharpe == 1.5

    def test_optimizer_run_round(self):
        optimizer = AutoOptimizer()
        suggestions = optimizer.run_optimization_round(n_trials=3)
        assert len(suggestions) == 3
        assert all("stop_loss_pct" in s for s in suggestions)

    def test_optimizer_apply_best_to_config(self):
        optimizer = AutoOptimizer()
        params = optimizer.suggest_params()
        optimizer.record_result(
            params,
            {
                "total_return_pct": 10.0,
                "sharpe_ratio": 2.0,
                "max_drawdown_pct": -2.0,
                "total_trades": 20,
            },
        )
        config_update = optimizer.apply_best_to_config()
        assert config_update is not None
        assert "STOP_LOSS_PCT" in config_update
        assert "TAKE_PROFIT_PCT" in config_update


# ---------------------------------------------------------------------------
# DecisionEngine evolved params passthrough
# ---------------------------------------------------------------------------


class TestDecisionEngineEvolution:
    def test_instructions_include_evolved_params(self):
        engine = DecisionEngine(initial_capital=10000)
        # Simulate evolved params being set
        engine._latest_evolved_params = {
            "Momentum": {"rsi_oversold": 22},
        }
        instructions = engine.orchestrate(1, 10000, 0)
        assert "evolved_params" in instructions
        assert instructions["evolved_params"]["Momentum"]["rsi_oversold"] == 22

    def test_instructions_include_optimized_config(self):
        engine = DecisionEngine(initial_capital=10000)
        # Record a trial result so optimizer has a best
        engine.optimizer.record_result(
            {
                "stop_loss_pct": 2.5,
                "take_profit_pct": 5.0,
                "trailing_stop_pct": 1.5,
                "confidence_threshold": 0.55,
                "lookback_bars": 200,
                "max_hold_bars": 80,
                "position_size_pct": 8.0,
                "max_open_positions": 3,
                "retrain_hours": 4.0,
            },
            {"total_return_pct": 8.0, "sharpe_ratio": 1.8, "max_drawdown_pct": -3.0, "total_trades": 18},
        )
        instructions = engine.orchestrate(1, 10000, 0)
        assert "optimized_config" in instructions
        assert instructions["optimized_config"]["STOP_LOSS_PCT"] == 2.5
