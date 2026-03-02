"""
Strategy Evolution Module
==========================
Genetic algorithm that evolves strategy parameters over time.
Tests parameter variations via backtesting and breeds the
best-performing combinations.
"""

from __future__ import annotations

import copy
import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class Genome:
    """A single set of strategy parameters with fitness tracking.

    Attributes:
        strategy_name: Name of the strategy this genome belongs to
            (e.g. "Momentum", "Ichimoku").
        parameters: Mapping of parameter name to current value.
        fitness_score: Composite fitness value computed by
            ``StrategyEvolver.evaluate_fitness()``.
        generation: Generation index at which this genome was created.
        trade_count: Number of trades produced during the most recent
            backtest evaluation.
        sharpe: Sharpe ratio from the most recent backtest evaluation.
        max_drawdown: Absolute maximum drawdown (%) from the most recent
            backtest evaluation.
    """

    strategy_name: str
    parameters: dict[str, Any]
    fitness_score: float = 0.0
    generation: int = 0
    trade_count: int = 0
    sharpe: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the genome to a JSON-compatible dict.

        Returns:
            Dict with keys ``strategy_name``, ``parameters``,
            ``fitness_score``, ``generation``, ``trade_count``,
            ``sharpe``, and ``max_drawdown``.
        """
        return {
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "fitness_score": self.fitness_score,
            "generation": self.generation,
            "trade_count": self.trade_count,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Genome:
        """Create a Genome from a previously serialized dict.

        Args:
            data: Dict as produced by ``to_dict()``.

        Returns:
            Reconstructed ``Genome`` instance.
        """
        return cls(**data)


# Parameter bounds for each strategy
PARAM_BOUNDS = {
    "Momentum": {
        "rsi_oversold": (15, 40, int),
        "rsi_overbought": (60, 85, int),
        "macd_threshold": (0.0, 0.5, float),
        "adx_threshold": (15, 35, int),
        "confidence_base": (0.4, 0.8, float),
    },
    "MeanReversion": {
        "bb_period": (10, 40, int),
        "bb_std": (1.0, 3.0, float),
        "rsi_low": (15, 35, int),
        "rsi_high": (65, 85, int),
        "confidence_base": (0.4, 0.8, float),
    },
    "Breakout": {
        "lookback": (10, 50, int),
        "volume_mult": (1.2, 3.0, float),
        "atr_mult": (0.5, 1.5, float),  # bar range ≥ atr_mult×ATR; 0.5 lenient, 1.5 strict
        "confidence_base": (0.4, 0.8, float),
    },
    "Grid": {
        "grid_size_pct": (0.5, 3.0, float),
        "num_levels": (3, 10, int),
        "confidence_base": (0.3, 0.7, float),
    },
    "Scalping": {
        "spread_threshold": (0.001, 0.005, float),
        "volume_spike": (1.5, 4.0, float),
        "rsi_range_low": (35, 50, int),
        "rsi_range_high": (50, 65, int),
        "confidence_base": (0.3, 0.7, float),
    },
    "Sentiment": {
        "fear_threshold": (15, 35, int),
        "greed_threshold": (65, 85, int),
        "composite_threshold": (0.1, 0.5, float),
        "confidence_base": (0.3, 0.7, float),
    },
    # v9.1: Ichimoku Cloud
    "Ichimoku": {
        "adx_threshold": (15, 35, int),
        "confidence_base": (0.4, 0.8, float),
    },
    # v10: RSI Divergence + Stochastic strategy
    "RSIDivergence": {
        "lookback": (15, 30, int),
        "stoch_threshold": (25.0, 40.0, float),
    },
}

# Default parameters (current v4 values)
DEFAULT_PARAMS = {
    "Momentum": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "macd_threshold": 0.0,
        "adx_threshold": 25,
        "confidence_base": 0.6,
    },
    "MeanReversion": {
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_low": 25,
        "rsi_high": 75,
        "confidence_base": 0.6,
    },
    "Breakout": {
        "lookback": 20,
        "volume_mult": 1.5,
        "atr_mult": 0.8,  # bar range must be ≥ 80% of ATR to qualify as a real breakout candle
        "confidence_base": 0.6,
    },
    "Grid": {
        "grid_size_pct": 1.0,
        "num_levels": 5,
        "confidence_base": 0.5,
    },
    "Scalping": {
        "spread_threshold": 0.002,
        "volume_spike": 2.0,
        "rsi_range_low": 40,
        "rsi_range_high": 60,
        "confidence_base": 0.5,
    },
    "Sentiment": {
        "fear_threshold": 25,
        "greed_threshold": 75,
        "composite_threshold": 0.3,
        "confidence_base": 0.5,
    },
    # v9.1: Ichimoku Cloud
    "Ichimoku": {
        "adx_threshold": 20,
        "confidence_base": 0.6,
    },
    # v10: RSI Divergence + Stochastic strategy
    "RSIDivergence": {
        "lookback": 20,
        "stoch_threshold": 35.0,
    },
}


class StrategyEvolver:
    """
    Genetic algorithm that evolves strategy parameters.
    Evaluates fitness via backtest metrics and breeds top performers.
    """

    def __init__(self, population_size: int = 20, mutation_rate: float = 0.1, elite_fraction: float = 0.3) -> None:
        """Initialise the genetic algorithm evolver.

        Args:
            population_size: Number of genomes per strategy population.
            mutation_rate: Probability (0–1) that any single parameter is
                mutated when creating an offspring genome.
            elite_fraction: Fraction of each population that is carried
                unchanged into the next generation as elites.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction
        self.populations: dict[str, list[Genome]] = {}
        self.generation: int = 0
        self.best_genomes: dict[str, Genome] = {}
        self.evolution_history: deque[dict[str, Any]] = deque(maxlen=50)
        self._initialized = False

    def initialize_population(self, strategy_names: list[str] | None = None) -> None:
        """Create the initial genome population for each strategy.

        The first genome in each population uses the default parameters from
        ``DEFAULT_PARAMS`` exactly.  The remaining genomes are Gaussian
        perturbations around those defaults, clipped to ``PARAM_BOUNDS``.

        Args:
            strategy_names: List of strategy names to initialise. Defaults to
                all keys in ``DEFAULT_PARAMS`` when ``None``.
        """
        if strategy_names is None:
            strategy_names = list(DEFAULT_PARAMS.keys())

        for name in strategy_names:
            if name not in PARAM_BOUNDS:
                continue

            pop = []
            defaults = DEFAULT_PARAMS.get(name, {})
            bounds = PARAM_BOUNDS[name]

            # First genome: exact defaults
            pop.append(
                Genome(
                    strategy_name=name,
                    parameters=copy.deepcopy(defaults),
                    generation=0,
                )
            )

            # Rest: random variations around defaults
            for _ in range(self.population_size - 1):
                params = {}
                for key, (lo, hi, dtype) in bounds.items():
                    default_val = defaults.get(key, (lo + hi) / 2)
                    # Random perturbation within bounds
                    spread = (hi - lo) * 0.3
                    val = default_val + random.gauss(0, spread)
                    val = max(lo, min(hi, val))
                    params[key] = dtype(val) if dtype is int else round(val, 4)
                pop.append(
                    Genome(
                        strategy_name=name,
                        parameters=params,
                        generation=0,
                    )
                )

            self.populations[name] = pop
            self.best_genomes[name] = pop[0]  # Default as initial best

        self._initialized = True
        print(
            f"  [Evolver] Initialized {len(self.populations)} strategy populations "
            f"({self.population_size} genomes each)"
        )

    def evaluate_fitness(self, genome: Genome, backtest_metrics: dict[str, Any]) -> None:
        """Update a genome's fitness attributes from backtest results.

        Fitness formula: ``Sharpe × sqrt(trades)`` with a bonus for win rate
        above 50 %, a drawdown penalty above 5 %, and a penalty factor of 0.5
        for fewer than 5 trades to discourage overfitted low-activity genomes.
        The result is stored directly on ``genome.fitness_score``.

        Args:
            genome: The genome to score (mutated in place).
            backtest_metrics: Metrics dict as returned by
                ``Backtester.run()``, expected keys: ``sharpe_ratio``,
                ``total_trades``, ``max_drawdown_pct``, ``win_rate``.
        """
        sharpe = backtest_metrics.get("sharpe_ratio", 0)
        trades = backtest_metrics.get("total_trades", 0)
        drawdown = abs(backtest_metrics.get("max_drawdown_pct", 0))
        win_rate = backtest_metrics.get("win_rate", 0)

        # Core fitness: Sharpe scaled by trade activity
        trade_factor = np.sqrt(max(trades, 1))
        fitness = sharpe * trade_factor

        # Bonus for win rate above 50%
        if win_rate > 0.5:
            fitness *= 1 + (win_rate - 0.5)

        # Penalty for excessive drawdown
        if drawdown > 5:
            fitness *= max(0.1, 1 - (drawdown - 5) / 20)

        # Penalty for too few trades (overfitting risk)
        if trades < 5:
            fitness *= 0.5

        genome.fitness_score = round(fitness, 4)
        genome.trade_count = trades
        genome.sharpe = round(sharpe, 4)
        genome.max_drawdown = round(drawdown, 4)

    def evolve(self, strategy_name: str) -> list[Genome]:
        """Evolve the population for a single strategy by one generation.

        Applies the standard genetic algorithm pipeline:
        Select (elites) → Crossover (uniform) → Mutate (Gaussian) → Replace.
        The elite fraction is carried over unchanged; the remaining slots are
        filled with children produced from two randomly selected elites.

        Args:
            strategy_name: Name of the strategy population to evolve (must
                be a key in ``self.populations``).

        Returns:
            The updated population list after evolution.  Returns the
            unmodified population if it has fewer than 4 genomes or no
            ``PARAM_BOUNDS`` entry exists for the strategy.
        """
        pop = self.populations.get(strategy_name, [])
        if len(pop) < 4:
            return pop

        bounds = PARAM_BOUNDS.get(strategy_name, {})
        if not bounds:
            return pop

        # Sort by fitness
        pop.sort(key=lambda g: g.fitness_score, reverse=True)

        # Elite selection (top 30%)
        n_elite = max(2, int(len(pop) * self.elite_fraction))
        elites = pop[:n_elite]

        # Update best genome
        if elites[0].fitness_score > self.best_genomes.get(strategy_name, Genome("", {})).fitness_score:
            self.best_genomes[strategy_name] = copy.deepcopy(elites[0])

        # New population
        new_pop = [copy.deepcopy(g) for g in elites]  # Keep elites

        # Fill rest with crossover + mutation
        while len(new_pop) < self.population_size:
            parent_a = random.choice(elites)
            parent_b = random.choice(elites)
            child = self._crossover(parent_a, parent_b, bounds, strategy_name)
            child = self._mutate(child, bounds)
            child.generation = self.generation + 1
            child.fitness_score = 0  # Reset fitness for new generation
            new_pop.append(child)

        self.populations[strategy_name] = new_pop
        self.generation += 1

        # Log evolution
        best = elites[0]
        self.evolution_history.append(
            {
                "generation": self.generation,
                "strategy": strategy_name,
                "best_fitness": best.fitness_score,
                "best_sharpe": best.sharpe,
                "avg_fitness": round(np.mean([g.fitness_score for g in elites]), 4),
            }
        )

        print(
            f"  [Evolver] {strategy_name} gen {self.generation}: best={best.fitness_score:.3f} sharpe={best.sharpe:.3f}"
        )

        return new_pop

    def _crossover(self, parent_a: Genome, parent_b: Genome, bounds: dict, strategy_name: str) -> Genome:
        """Uniform crossover: randomly pick each param from either parent."""
        child_params = {}
        for key in bounds:
            if random.random() < 0.5:
                child_params[key] = parent_a.parameters.get(key, parent_b.parameters.get(key))
            else:
                child_params[key] = parent_b.parameters.get(key, parent_a.parameters.get(key))
        return Genome(strategy_name=strategy_name, parameters=child_params)

    def _mutate(self, genome: Genome, bounds: dict) -> Genome:
        """Mutate random parameters within bounds."""
        for key, (lo, hi, dtype) in bounds.items():
            if random.random() < self.mutation_rate:
                current = genome.parameters.get(key, (lo + hi) / 2)
                spread = (hi - lo) * 0.15  # Small mutation
                new_val = current + random.gauss(0, spread)
                new_val = max(lo, min(hi, new_val))
                genome.parameters[key] = dtype(new_val) if dtype is int else round(new_val, 4)
        return genome

    def get_best_params(self, strategy_name: str) -> dict[str, Any] | None:
        """Return the best evolved parameters for a strategy.

        Args:
            strategy_name: Strategy name to look up in ``best_genomes``.

        Returns:
            Parameter dict for the best genome, or ``None`` if no genome
            with positive fitness has been recorded for that strategy.
        """
        best = self.best_genomes.get(strategy_name)
        if best and best.fitness_score > 0:
            return best.parameters
        return None

    def get_all_best(self) -> dict[str, Any]:
        """Return the best genome serialized dict for every strategy.

        Returns:
            Dict mapping strategy name to the result of
            ``Genome.to_dict()`` for each strategy that has at least one
            genome with positive fitness.
        """
        return {name: genome.to_dict() for name, genome in self.best_genomes.items() if genome.fitness_score > 0}

    def get_status(self) -> dict[str, Any]:
        """Return a summary of the current evolution state.

        Returns:
            Dict with keys ``generation``, ``initialized``,
            ``strategies`` (per-strategy population size, best fitness,
            and best Sharpe), and ``recent_evolution`` (last 5 evolution
            log entries).
        """
        return {
            "generation": self.generation,
            "initialized": self._initialized,
            "strategies": {
                name: {
                    "population_size": len(pop),
                    "best_fitness": self.best_genomes.get(name, Genome("", {})).fitness_score,
                    "best_sharpe": self.best_genomes.get(name, Genome("", {})).sharpe,
                }
                for name, pop in self.populations.items()
            },
            "recent_evolution": list(self.evolution_history)[-5:] if self.evolution_history else [],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evolver state for persistence.

        Returns:
            Dict with keys ``generation``, ``best_genomes`` (mapping of
            strategy name to serialized genome dict), and
            ``evolution_history`` (list of evolution log entries).
        """
        return {
            "generation": self.generation,
            "best_genomes": {name: g.to_dict() for name, g in self.best_genomes.items()},
            "evolution_history": list(self.evolution_history),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore evolver state from a previously serialized dict.

        Args:
            data: Dict as produced by ``to_dict()``.
        """
        self.generation = data.get("generation", 0)
        for name, gdata in data.get("best_genomes", {}).items():
            self.best_genomes[name] = Genome.from_dict(gdata)
        self.evolution_history = deque(data.get("evolution_history", []), maxlen=50)
        if self.best_genomes:
            self._initialized = True
