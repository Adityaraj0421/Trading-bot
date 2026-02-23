"""
Unit tests for strategy_evolver.py — Genome dataclass, StrategyEvolver GA.

Tests genome serialization, population initialization, fitness evaluation,
evolution cycle, crossover/mutation bounds, and state persistence.
"""

import copy
import random

import numpy as np
import pytest

from strategy_evolver import (
    DEFAULT_PARAMS,
    PARAM_BOUNDS,
    Genome,
    StrategyEvolver,
)


@pytest.fixture()
def evolver():
    return StrategyEvolver(population_size=10, mutation_rate=0.2, elite_fraction=0.3)


def make_metrics(sharpe=1.0, trades=10, drawdown=3.0, win_rate=0.55):
    return {
        "sharpe_ratio": sharpe,
        "total_trades": trades,
        "max_drawdown_pct": -drawdown,  # negative by convention
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# Genome dataclass
# ---------------------------------------------------------------------------


class TestGenome:
    def test_defaults(self):
        g = Genome(strategy_name="Momentum", parameters={"a": 1})
        assert g.fitness_score == 0.0
        assert g.generation == 0
        assert g.trade_count == 0
        assert g.sharpe == 0.0
        assert g.max_drawdown == 0.0

    def test_to_dict_keys(self):
        g = Genome(strategy_name="Momentum", parameters={"a": 1})
        d = g.to_dict()
        assert set(d.keys()) == {
            "strategy_name",
            "parameters",
            "fitness_score",
            "generation",
            "trade_count",
            "sharpe",
            "max_drawdown",
        }

    def test_from_dict_round_trip(self):
        g = Genome(strategy_name="Breakout", parameters={"x": 5}, fitness_score=2.5)
        restored = Genome.from_dict(g.to_dict())
        assert restored.strategy_name == g.strategy_name
        assert restored.parameters == g.parameters
        assert restored.fitness_score == g.fitness_score

    def test_preserves_parameters(self):
        params = {"rsi_oversold": 30, "macd_threshold": 0.1}
        g = Genome(strategy_name="Momentum", parameters=params)
        assert g.parameters == params

    def test_from_dict_with_fitness(self):
        d = {
            "strategy_name": "Grid",
            "parameters": {"grid_size_pct": 1.5},
            "fitness_score": 5.0,
            "generation": 3,
            "trade_count": 20,
            "sharpe": 1.8,
            "max_drawdown": 4.0,
        }
        g = Genome.from_dict(d)
        assert g.fitness_score == 5.0
        assert g.generation == 3


# ---------------------------------------------------------------------------
# PARAM_BOUNDS / DEFAULT_PARAMS
# ---------------------------------------------------------------------------


class TestParamBounds:
    def test_all_strategies_have_bounds(self):
        # v9.1: 7 strategy param sets (added Ichimoku)
        assert len(PARAM_BOUNDS) == 7
        for name in ["Momentum", "MeanReversion", "Breakout", "Grid", "Scalping", "Sentiment", "Ichimoku"]:
            assert name in PARAM_BOUNDS

    def test_defaults_within_bounds(self):
        for name, bounds in PARAM_BOUNDS.items():
            defaults = DEFAULT_PARAMS[name]
            for key, (lo, hi, _dtype) in bounds.items():
                val = defaults[key]
                assert lo <= val <= hi, f"{name}.{key}: {val} not in [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# initialize_population
# ---------------------------------------------------------------------------


class TestInitializePopulation:
    def test_correct_population_size(self, evolver):
        evolver.initialize_population(["Momentum"])
        assert len(evolver.populations["Momentum"]) == 10

    def test_first_genome_is_defaults(self, evolver):
        evolver.initialize_population(["Momentum"])
        first = evolver.populations["Momentum"][0]
        assert first.parameters == DEFAULT_PARAMS["Momentum"]

    def test_rest_differ_from_defaults(self, evolver):
        random.seed(42)
        evolver.initialize_population(["Momentum"])
        pop = evolver.populations["Momentum"]
        defaults = DEFAULT_PARAMS["Momentum"]
        # At least one non-default genome should differ
        any_different = any(g.parameters != defaults for g in pop[1:])
        assert any_different

    def test_unknown_strategy_skipped(self, evolver):
        evolver.initialize_population(["Momentum", "FakeStrategy"])
        assert "Momentum" in evolver.populations
        assert "FakeStrategy" not in evolver.populations

    def test_sets_initialized_flag(self, evolver):
        assert evolver._initialized is False
        evolver.initialize_population(["Momentum"])
        assert evolver._initialized is True


# ---------------------------------------------------------------------------
# evaluate_fitness
# ---------------------------------------------------------------------------


class TestEvaluateFitness:
    def test_basic_fitness(self, evolver):
        g = Genome(strategy_name="Momentum", parameters={})
        evolver.evaluate_fitness(g, make_metrics(sharpe=2.0, trades=16, drawdown=3.0, win_rate=0.4))
        # fitness = 2.0 * sqrt(16) = 8.0, no win-rate bonus, no drawdown penalty
        assert g.fitness_score == pytest.approx(8.0, abs=0.01)

    def test_win_rate_bonus(self, evolver):
        g = Genome(strategy_name="Momentum", parameters={})
        evolver.evaluate_fitness(g, make_metrics(sharpe=2.0, trades=16, drawdown=3.0, win_rate=0.7))
        # base = 8.0, win_rate bonus: *= (1 + 0.2) = 9.6
        assert g.fitness_score == pytest.approx(9.6, abs=0.01)

    def test_no_bonus_below_50_win_rate(self, evolver):
        g = Genome(strategy_name="Momentum", parameters={})
        evolver.evaluate_fitness(g, make_metrics(sharpe=2.0, trades=16, drawdown=3.0, win_rate=0.4))
        assert g.fitness_score == pytest.approx(8.0, abs=0.01)

    def test_drawdown_penalty(self, evolver):
        g = Genome(strategy_name="Momentum", parameters={})
        evolver.evaluate_fitness(g, make_metrics(sharpe=2.0, trades=16, drawdown=15, win_rate=0.4))
        # base = 8.0, drawdown penalty: *= max(0.1, 1 - (15-5)/20) = 0.5
        assert g.fitness_score == pytest.approx(4.0, abs=0.01)

    def test_extreme_drawdown_floors_at_01(self, evolver):
        g = Genome(strategy_name="Momentum", parameters={})
        evolver.evaluate_fitness(g, make_metrics(sharpe=2.0, trades=16, drawdown=30, win_rate=0.4))
        # base = 8.0, max(0.1, 1-(30-5)/20) = max(0.1, -0.25) = 0.1
        assert g.fitness_score == pytest.approx(0.8, abs=0.01)

    def test_low_trade_penalty(self, evolver):
        g = Genome(strategy_name="Momentum", parameters={})
        evolver.evaluate_fitness(g, make_metrics(sharpe=2.0, trades=3, drawdown=3.0, win_rate=0.4))
        # base = 2.0 * sqrt(3) ≈ 3.464, * 0.5 = 1.732
        expected = 2.0 * np.sqrt(3) * 0.5
        assert g.fitness_score == pytest.approx(expected, abs=0.01)

    def test_sets_genome_fields(self, evolver):
        g = Genome(strategy_name="Momentum", parameters={})
        evolver.evaluate_fitness(g, make_metrics(sharpe=1.5, trades=20, drawdown=8.0, win_rate=0.55))
        assert g.trade_count == 20
        assert g.sharpe == 1.5
        assert g.max_drawdown == 8.0
        assert g.fitness_score > 0


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------


class TestEvolve:
    def _setup_pop(self, evolver, strategy="Momentum"):
        random.seed(42)
        evolver.initialize_population([strategy])
        # Give each genome some fitness
        for i, g in enumerate(evolver.populations[strategy]):
            evolver.evaluate_fitness(
                g,
                make_metrics(
                    sharpe=float(i),
                    trades=10 + i,
                    drawdown=2.0,
                    win_rate=0.55,
                ),
            )
        return strategy

    def test_preserves_population_size(self, evolver):
        name = self._setup_pop(evolver)
        new_pop = evolver.evolve(name)
        assert len(new_pop) == evolver.population_size

    def test_elites_survive(self, evolver):
        name = self._setup_pop(evolver)
        old_pop = sorted(evolver.populations[name], key=lambda g: g.fitness_score, reverse=True)
        top_fitness = old_pop[0].fitness_score
        new_pop = evolver.evolve(name)
        new_fitnesses = [g.fitness_score for g in new_pop]
        assert top_fitness in new_fitnesses

    def test_generation_increments(self, evolver):
        name = self._setup_pop(evolver)
        old_gen = evolver.generation
        evolver.evolve(name)
        assert evolver.generation == old_gen + 1

    def test_small_population_unchanged(self, evolver):
        evolver.populations["Tiny"] = [
            Genome(strategy_name="Tiny", parameters={"x": 1}),
            Genome(strategy_name="Tiny", parameters={"x": 2}),
        ]
        result = evolver.evolve("Tiny")
        assert len(result) == 2

    def test_history_appended(self, evolver):
        name = self._setup_pop(evolver)
        old_len = len(evolver.evolution_history)
        evolver.evolve(name)
        assert len(evolver.evolution_history) == old_len + 1


# ---------------------------------------------------------------------------
# _crossover / _mutate
# ---------------------------------------------------------------------------


class TestCrossoverMutation:
    def test_crossover_valid_params(self, evolver):
        random.seed(42)
        bounds = PARAM_BOUNDS["Momentum"]
        parent_a = Genome(strategy_name="Momentum", parameters=DEFAULT_PARAMS["Momentum"])
        parent_b = Genome(strategy_name="Momentum", parameters=DEFAULT_PARAMS["Momentum"])
        child = evolver._crossover(parent_a, parent_b, bounds, "Momentum")
        assert set(child.parameters.keys()) == set(bounds.keys())

    def test_mutation_within_bounds(self, evolver):
        random.seed(42)
        bounds = PARAM_BOUNDS["Momentum"]
        g = Genome(strategy_name="Momentum", parameters=copy.deepcopy(DEFAULT_PARAMS["Momentum"]))
        # Force high mutation rate to trigger changes
        evolver.mutation_rate = 1.0
        mutated = evolver._mutate(g, bounds)
        for key, (lo, hi, _dtype) in bounds.items():
            assert lo <= mutated.parameters[key] <= hi, f"{key} out of bounds"

    def test_mutation_respects_dtype(self, evolver):
        random.seed(42)
        bounds = PARAM_BOUNDS["Momentum"]
        g = Genome(strategy_name="Momentum", parameters=copy.deepcopy(DEFAULT_PARAMS["Momentum"]))
        evolver.mutation_rate = 1.0
        mutated = evolver._mutate(g, bounds)
        for key, (_lo, _hi, dtype) in bounds.items():
            if dtype is int:
                assert isinstance(mutated.parameters[key], int), f"{key} should be int"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_structure(self, evolver):
        random.seed(42)
        evolver.initialize_population(["Momentum"])
        d = evolver.to_dict()
        assert "generation" in d
        assert "best_genomes" in d
        assert "evolution_history" in d

    def test_from_dict_round_trip(self, evolver):
        random.seed(42)
        evolver.initialize_population(["Momentum"])
        evolver.generation = 5
        evolver.best_genomes["Momentum"].fitness_score = 3.0
        saved = evolver.to_dict()

        new_evolver = StrategyEvolver()
        new_evolver.from_dict(saved)
        assert new_evolver.generation == 5
        assert new_evolver.best_genomes["Momentum"].fitness_score == 3.0
        assert new_evolver._initialized is True

    def test_get_status_structure(self, evolver):
        random.seed(42)
        evolver.initialize_population(["Momentum"])
        status = evolver.get_status()
        assert "generation" in status
        assert "initialized" in status
        assert "strategies" in status
        assert "recent_evolution" in status
