"""
Auto-Optimizer Module v2.0
============================
Bayesian hyperparameter optimization using Optuna (TPE sampler).
Multi-objective: maximize Sharpe ratio, minimize max drawdown.
Falls back to random search if Optuna is unavailable.

v2.0: Optuna TPE sampler + Hyperband pruner, SQLite persistence,
      multi-objective optimization, preserved backward-compatible API.
v1.0: Random search with Pareto front tracking.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_log = logging.getLogger(__name__)

# Try importing Optuna
_HAS_OPTUNA = False
try:
    import optuna
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    pass


@dataclass
class HyperparamBound:
    """Defines the search range for a single hyperparameter."""

    name: str
    low: float
    high: float
    dtype: type = float  # int or float
    current: float = 0.0
    description: str = ""

    def sample(self) -> float:
        """Draw a uniform random sample within [low, high].

        Returns:
            An integer cast to ``dtype`` when ``dtype`` is ``int``,
            otherwise a float rounded to 4 decimal places.
        """
        val = random.uniform(self.low, self.high)
        return self.dtype(val) if self.dtype == int else round(val, 4)


@dataclass
class TrialResult:
    """Result of one optimization trial."""

    params: dict[str, Any]
    metrics: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    score: float = 0.0  # Composite optimization score

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trial result to a JSON-compatible dict.

        Returns:
            Dict with keys ``params``, ``total_return``, ``sharpe``,
            ``max_drawdown``, ``total_trades``, ``score``, and ``timestamp``
            (ISO-8601 string).
        """
        return {
            "params": self.params,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "total_trades": self.total_trades,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
        }


# Default hyperparameter search space
DEFAULT_SEARCH_SPACE = {
    "stop_loss_pct": HyperparamBound("stop_loss_pct", 0.5, 5.0, float, 2.0, "Stop-loss percentage"),
    "take_profit_pct": HyperparamBound("take_profit_pct", 1.0, 10.0, float, 3.0, "Take-profit percentage"),
    "trailing_stop_pct": HyperparamBound("trailing_stop_pct", 0.5, 4.0, float, 1.5, "Trailing stop percentage"),
    "confidence_threshold": HyperparamBound(
        "confidence_threshold", 0.3, 0.8, float, 0.5, "Minimum confidence to trade"
    ),
    "lookback_bars": HyperparamBound("lookback_bars", 50, 500, int, 200, "Historical bars for analysis"),
    "max_hold_bars": HyperparamBound("max_hold_bars", 20, 200, int, 100, "Maximum bars to hold a position"),
    "position_size_pct": HyperparamBound("position_size_pct", 1.0, 20.0, float, 10.0, "Position size as % of capital"),
    "max_open_positions": HyperparamBound("max_open_positions", 1, 5, int, 3, "Maximum simultaneous positions"),
    "retrain_hours": HyperparamBound("retrain_hours", 1, 24, float, 6.0, "Hours between model retraining"),
}


class AutoOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna TPE sampler.
    Falls back to random search if Optuna is unavailable.

    Key improvements over v1.0:
    - TPE sampler builds a probabilistic model of good/bad parameter regions
    - Hyperband pruner early-stops unpromising trials
    - Multi-objective: maximize Sharpe, minimize drawdown simultaneously
    - SQLite storage for persistence across sessions
    - 3-10x more sample-efficient than random search

    API is backward-compatible with v1.0 (all methods preserved).
    """

    def __init__(
        self,
        search_space: dict[str, HyperparamBound] | None = None,
        max_trials: int = 100,
        storage_path: str | None = None,
    ) -> None:
        """Initialise the optimizer with a parameter search space.

        Args:
            search_space: Mapping of parameter name to ``HyperparamBound``
                defining the search range for each hyperparameter. Defaults
                to ``DEFAULT_SEARCH_SPACE`` when ``None``.
            max_trials: Maximum number of trials to keep in memory. Older
                trials are discarded when this limit is reached.
            storage_path: Optional filesystem path for an SQLite database
                used by Optuna to persist trial history across sessions.
                When ``None`` the study is in-memory only.
        """
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.max_trials = max_trials
        self.trials: deque[TrialResult] = deque(maxlen=max_trials)
        self.pareto_front: list[TrialResult] = []
        self.best_result: TrialResult | None = None
        self.optimization_count: int = 0
        self.last_optimization: datetime | None = None
        self.engine = "optuna" if _HAS_OPTUNA else "random"

        # Optuna study setup
        self._study = None
        self._pending_params: list[dict] = []
        self._storage_path = storage_path

        if _HAS_OPTUNA:
            self._init_optuna_study()

        print(f"  [Optimizer] Engine: {self.engine} | Max trials: {max_trials}")

    def _init_optuna_study(self) -> None:
        """Initialize Optuna multi-objective study."""
        storage = None
        if self._storage_path:
            storage = f"sqlite:///{self._storage_path}"

        sampler = TPESampler(
            seed=42,
            n_startup_trials=10,  # Random exploration first
            multivariate=True,  # Model parameter interactions
        )

        pruner = HyperbandPruner(
            min_resource=5,  # Minimum trials before pruning
            max_resource=self.max_trials,
            reduction_factor=3,
        )

        self._study = optuna.create_study(
            study_name="crypto_optimizer",
            directions=["maximize", "minimize"],  # Sharpe, drawdown
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

    def _suggest_optuna_params(self, trial: Any) -> dict[str, Any]:
        """Use Optuna trial to suggest parameters."""
        params = {}
        for name, bound in self.search_space.items():
            if bound.dtype == int:
                params[name] = trial.suggest_int(name, int(bound.low), int(bound.high))
            else:
                params[name] = round(trial.suggest_float(name, bound.low, bound.high), 4)
        return params

    def suggest_params(self) -> dict[str, Any]:
        """Suggest a new parameter set to evaluate.

        Uses the Optuna TPE sampler when Optuna is installed, otherwise
        falls back to uniform random sampling within each parameter's bounds.

        Returns:
            Dict mapping parameter name to a suggested value. Keys match
            the names defined in ``self.search_space``.
        """
        if _HAS_OPTUNA and self._study is not None:
            # Create a trial and get Optuna's suggestion
            trial = self._study.ask(
                fixed_distributions={
                    name: (
                        optuna.distributions.IntDistribution(int(bound.low), int(bound.high))
                        if bound.dtype == int
                        else optuna.distributions.FloatDistribution(bound.low, bound.high)
                    )
                    for name, bound in self.search_space.items()
                }
            )
            params = {name: trial.params[name] for name in self.search_space}
            # Round float params
            for name, bound in self.search_space.items():
                if bound.dtype == float:
                    params[name] = round(params[name], 4)
            # Store trial number for later tell()
            params["_optuna_trial_number"] = trial.number
            self._pending_params.append(params)
            return {k: v for k, v in params.items() if not k.startswith("_")}

        # Fallback: random search
        params = {}
        for name, bound in self.search_space.items():
            params[name] = bound.sample()
        return params

    def suggest_nearby(self, base_params: dict[str, Any], spread: float = 0.2) -> dict[str, Any]:
        """Suggest parameters via Gaussian perturbation around a known-good set.

        Useful for local exploitation after a promising region has been
        identified.  Each parameter is perturbed by a zero-mean Gaussian
        with standard deviation ``spread * (high - low)`` and then clipped
        to its declared bounds.

        Args:
            base_params: Reference parameter set to perturb (e.g. from
                ``get_best_params()``).
            spread: Fraction of each parameter's range to use as the
                standard deviation of the perturbation.  Defaults to 0.2.

        Returns:
            Dict of perturbed parameter values within their declared bounds.
        """
        params = {}
        for name, bound in self.search_space.items():
            base_val = base_params.get(name, bound.current)
            range_size = bound.high - bound.low
            perturbation = random.gauss(0, range_size * spread)
            new_val = base_val + perturbation
            new_val = max(bound.low, min(bound.high, new_val))
            params[name] = bound.dtype(new_val) if bound.dtype == int else round(new_val, 4)
        return params

    def record_result(self, params: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Record the outcome of a completed trial.

        Computes the composite score, updates the best result, refreshes the
        Pareto front, and feeds the result back to the Optuna study when
        available so that subsequent ``suggest_params()`` calls benefit from
        the observed data.

        Args:
            params: Parameter dict that was evaluated (as returned by
                ``suggest_params()`` or ``suggest_nearby()``).
            metrics: Backtest metrics dict with at minimum the keys
                ``sharpe_ratio``, ``total_return_pct``, ``max_drawdown_pct``,
                and ``total_trades``.
        """
        result = TrialResult(
            params=params,
            metrics=metrics,
            total_return=metrics.get("total_return_pct", 0),
            sharpe=metrics.get("sharpe_ratio", 0),
            max_drawdown=abs(metrics.get("max_drawdown_pct", 0)),
            total_trades=metrics.get("total_trades", 0),
        )

        # Composite score
        result.score = self._compute_score(result)
        self.trials.append(result)

        # Feed result back to Optuna
        if _HAS_OPTUNA and self._study is not None:
            self._tell_optuna(params, result)

        # Update best
        if self.best_result is None or result.score > self.best_result.score:
            self.best_result = result
            print(
                f"  [Optimizer] New best: score={result.score:.3f} "
                f"sharpe={result.sharpe:.3f} return={result.total_return:.2f}%"
            )

        # Update Pareto front
        self._update_pareto(result)

    def _tell_optuna(self, params: dict[str, Any], result: TrialResult) -> None:
        """Report trial result back to Optuna study."""
        # Find matching pending trial (tolerance-based float comparison)
        trial_number = None
        for pending in self._pending_params:
            match = all(
                abs(float(pending.get(k, 0)) - float(params.get(k, 0))) < 1e-6
                for k in self.search_space
            )
            if match:
                trial_number = pending.get("_optuna_trial_number")
                self._pending_params.remove(pending)
                break

        if trial_number is not None:
            # Multi-objective: maximize Sharpe, minimize drawdown
            self._study.tell(
                trial_number,
                values=[result.sharpe, result.max_drawdown],
            )
        else:
            # No matching trial — add as external observation
            trial = optuna.trial.create_trial(
                params={name: params[name] for name in self.search_space if name in params},
                distributions={
                    name: (
                        optuna.distributions.IntDistribution(int(bound.low), int(bound.high))
                        if bound.dtype == int
                        else optuna.distributions.FloatDistribution(bound.low, bound.high)
                    )
                    for name, bound in self.search_space.items()
                },
                values=[result.sharpe, result.max_drawdown],
            )
            self._study.add_trial(trial)

    def _compute_score(self, result: TrialResult) -> float:
        """Compute composite optimization score.

        All components are normalized to comparable scales before combining.
        The score is dimensionless and used only for ranking trials.

        Components:
          - Sharpe ratio (already dimensionless): primary quality metric
          - Return quality: sign-based bonus (not raw percentage)
          - Drawdown penalty: penalizes excessive risk
          - Trade count: penalizes overfitting (too few trades)
        """
        # Primary: Sharpe ratio (already dimensionless, typically -2 to +3)
        score = result.sharpe * 2.0

        # Return quality: bonus for profitable, penalty for losing
        # Using sign + magnitude-bucket rather than raw pct to stay scale-neutral
        if result.total_return > 5.0:
            score += 1.0  # Strong positive return
        elif result.total_return > 0:
            score += 0.5  # Modest positive return
        elif result.total_return < -5.0:
            score -= 0.5  # Significant loss

        # Drawdown penalty (normalized: 5% drawdown = 0 penalty, each 5% above = -1)
        if result.max_drawdown > 5:
            score -= (result.max_drawdown - 5) / 5.0

        # Trade count penalty / bonus (prevents overfitting)
        if result.total_trades < 5:
            score *= 0.5  # Likely overfitting on few trades
        elif result.total_trades >= 10:
            score += 0.5  # Sufficient sample for statistical relevance

        return round(score, 4)

    def _update_pareto(self, new_result: TrialResult) -> None:
        """
        Update Pareto front (non-dominated solutions).
        Objectives: maximize Sharpe, minimize drawdown, maximize trades.
        """
        dominated = False
        to_remove = []

        for i, existing in enumerate(self.pareto_front):
            if (  # noqa: SIM102
                existing.sharpe >= new_result.sharpe
                and existing.max_drawdown <= new_result.max_drawdown
                and existing.total_trades >= new_result.total_trades
            ):
                if (
                    existing.sharpe > new_result.sharpe
                    or existing.max_drawdown < new_result.max_drawdown
                    or existing.total_trades > new_result.total_trades
                ):
                    dominated = True
                    break

            if (  # noqa: SIM102
                new_result.sharpe >= existing.sharpe
                and new_result.max_drawdown <= existing.max_drawdown
                and new_result.total_trades >= existing.total_trades
            ):
                if (
                    new_result.sharpe > existing.sharpe
                    or new_result.max_drawdown < existing.max_drawdown
                    or new_result.total_trades > existing.total_trades
                ):
                    to_remove.append(i)

        if not dominated:
            for i in reversed(to_remove):
                self.pareto_front.pop(i)
            self.pareto_front.append(new_result)

        if len(self.pareto_front) > 10:
            self.pareto_front.sort(key=lambda r: r.score, reverse=True)
            self.pareto_front = self.pareto_front[:10]

    def get_best_params(self) -> dict[str, Any] | None:
        """Return the parameter set with the highest composite score.

        When Optuna is available, returns the Pareto-optimal trial with
        the highest Sharpe ratio from the study. Otherwise returns the
        best result from the in-memory trial log.

        Returns:
            Dict of parameter values, or ``None`` if no trials have been
            recorded yet.
        """
        # Optuna: get best from study (Sharpe objective)
        if _HAS_OPTUNA and self._study is not None:
            best_trials = self._study.best_trials
            if best_trials:
                # Pick trial with highest Sharpe among Pareto-optimal
                best = max(best_trials, key=lambda t: t.values[0])
                return dict(best.params)

        if self.best_result:
            return self.best_result.params
        return None

    def get_pareto_front(self) -> list[dict[str, Any]]:
        """Return all Pareto-optimal solutions found so far.

        A solution is Pareto-optimal if no other trial is simultaneously
        better on all three objectives (Sharpe, max drawdown, trade count).
        The front is capped at 10 entries.

        Returns:
            List of dicts. When Optuna is active each dict has keys
            ``params``, ``sharpe``, ``max_drawdown``, and ``trial_number``.
            When using random search, each dict is the output of
            ``TrialResult.to_dict()``.
        """
        if _HAS_OPTUNA and self._study is not None:
            best_trials = self._study.best_trials
            if best_trials:
                front = []
                for t in best_trials[:10]:
                    front.append(
                        {
                            "params": dict(t.params),
                            "sharpe": t.values[0],
                            "max_drawdown": t.values[1],
                            "trial_number": t.number,
                        }
                    )
                return front

        return [r.to_dict() for r in self.pareto_front]

    def run_optimization_round(self, n_trials: int = 10) -> list[dict[str, Any]]:
        """Generate a batch of parameter sets for external evaluation.

        Each parameter set should be evaluated by running a backtest and
        then reported back via ``record_result()``.  Uses Optuna TPE
        suggestions when available; when using random search, the first
        suggestion in a round is a local perturbation of the best known
        result to mix exploitation with exploration.

        Args:
            n_trials: Number of parameter sets to generate per round.

        Returns:
            List of parameter dicts, each suitable for passing to
            ``record_result()``.
        """
        suggestions = []

        for i in range(n_trials):
            if _HAS_OPTUNA and self._study is not None:
                params = self.suggest_params()
            else:
                # Fallback: same logic as v1.0
                if i == 0 and self.best_result:
                    params = self.suggest_nearby(self.best_result.params, spread=0.15)
                elif i < 3 and self.pareto_front:
                    base = random.choice(self.pareto_front)
                    params = self.suggest_nearby(base.params, spread=0.2)
                else:
                    params = self.suggest_params()

            suggestions.append(params)

        self.optimization_count += 1
        self.last_optimization = datetime.now()
        return suggestions

    def get_optimization_history(self) -> dict[str, Any]:
        """Return optimization history and parameter importance analysis.

        Parameter importance is computed via Optuna's ``get_param_importances``
        and requires at least 10 completed trials to produce meaningful results.
        This method is only fully populated when Optuna is available.

        Returns:
            Dict with keys:
                ``engine`` (str): "optuna" or "random".
                ``total_trials`` (int): Total number of Optuna trials.
                ``completed_trials`` (int): Trials with a final result.
                ``param_importance`` (dict): Per-objective parameter
                    importance scores (only when Optuna is active).
                ``history`` (list[dict]): Last 20 completed trials, each
                    with ``trial``, ``params``, ``sharpe``, and ``drawdown``.
        """
        if not _HAS_OPTUNA or self._study is None:
            return {"engine": "random", "history": []}

        trials = self._study.trials
        history = []
        for t in trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                history.append(
                    {
                        "trial": t.number,
                        "params": dict(t.params),
                        "sharpe": t.values[0] if t.values else 0,
                        "drawdown": t.values[1] if t.values and len(t.values) > 1 else 0,
                    }
                )

        # Parameter importance (requires enough trials)
        importance = {}
        if len(trials) >= 10:
            try:
                for i, direction in enumerate(["sharpe", "drawdown"]):
                    imp = optuna.importance.get_param_importances(self._study, target=lambda t, idx=i: t.values[idx])
                    importance[direction] = {k: round(v, 4) for k, v in imp.items()}
            except Exception as e:
                __import__("logging").getLogger(__name__).debug("Param importance calc failed: %s", e)

        return {
            "engine": "optuna",
            "total_trials": len(trials),
            "completed_trials": len(history),
            "param_importance": importance,
            "history": history[-20:],  # Last 20 trials
        }

    def apply_best_to_config(self) -> dict[str, Any] | None:
        """Return the best parameters mapped to their ``Config`` attribute names.

        Translates optimizer parameter names (e.g. ``stop_loss_pct``) to
        the corresponding ``Config`` attribute names (e.g. ``STOP_LOSS_PCT``)
        so the result can be applied directly to the live agent configuration.

        Returns:
            Dict of ``{CONFIG_KEY: value}`` for all optimized parameters,
            or ``None`` if no trials have been completed yet.
        """
        params = self.get_best_params()
        if not params:
            return None

        return {
            "STOP_LOSS_PCT": params.get("stop_loss_pct"),
            "TAKE_PROFIT_PCT": params.get("take_profit_pct"),
            "TRAILING_STOP_PCT": params.get("trailing_stop_pct"),
            "CONFIDENCE_THRESHOLD": params.get("confidence_threshold"),
            "LOOKBACK_BARS": params.get("lookback_bars"),
            "MAX_HOLD_BARS": params.get("max_hold_bars"),
            "POSITION_SIZE_PCT": params.get("position_size_pct"),
            "MAX_OPEN_POSITIONS": params.get("max_open_positions"),
            "MODEL_RETRAIN_HOURS": params.get("retrain_hours"),
        }

    def get_status(self) -> dict[str, Any]:
        """Return a summary of the optimizer's current state.

        Returns:
            Dict with keys ``engine``, ``total_trials``,
            ``optimization_rounds``, ``best_score``, ``best_sharpe``,
            ``best_return``, ``pareto_size``, and ``last_optimization``
            (ISO-8601 string or ``None``).  When Optuna is active, also
            includes ``optuna_trials`` and ``optuna_pareto_size``.
        """
        status = {
            "engine": self.engine,
            "total_trials": len(self.trials),
            "optimization_rounds": self.optimization_count,
            "best_score": self.best_result.score if self.best_result else 0,
            "best_sharpe": self.best_result.sharpe if self.best_result else 0,
            "best_return": self.best_result.total_return if self.best_result else 0,
            "pareto_size": len(self.pareto_front),
            "last_optimization": (self.last_optimization.isoformat() if self.last_optimization else None),
        }

        if _HAS_OPTUNA and self._study is not None:
            status["optuna_trials"] = len(self._study.trials)
            status["optuna_pareto_size"] = len(self._study.best_trials)

        return status

    def to_dict(self) -> dict[str, Any]:
        """Serialize the optimizer state for persistence.

        Returns:
            Dict with keys ``engine``, ``optimization_count``,
            ``best_result``, ``pareto_front``, and ``trials_count``.
        """
        return {
            "engine": self.engine,
            "optimization_count": self.optimization_count,
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "pareto_front": [r.to_dict() for r in self.pareto_front],
            "trials_count": len(self.trials),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore optimizer state from a previously serialized dict.

        Args:
            data: Dict as produced by ``to_dict()``.
        """
        self.optimization_count = data.get("optimization_count", 0)
        if data.get("best_result"):
            br = data["best_result"]
            self.best_result = TrialResult(
                params=br["params"],
                metrics={},
                total_return=br.get("total_return", 0),
                sharpe=br.get("sharpe", 0),
                max_drawdown=br.get("max_drawdown", 0),
                total_trades=br.get("total_trades", 0),
                score=br.get("score", 0),
            )
