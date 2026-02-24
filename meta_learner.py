"""
Meta-Learning Module v2.0 — A/B Testing Framework
===================================================
Learns optimal configuration from trading outcomes WITH statistical
validation before adopting new weights.

v2.0 enhancements:
  - Split-capital A/B testing: control (old weights) vs treatment (new weights)
  - Welch's t-test for statistical significance (p < 0.05)
  - Automatic rollback for failed experiments
  - Experiment history with win/loss tracking
  - Gradual weight adoption: treatment starts at 30% capital allocation
  - All original learning algorithms preserved
  - Dashboard-friendly status reporting

Architecture:
  1. MetaLearner.learn() proposes new config → creates Experiment
  2. Each trade is split: control uses old config, treatment uses new config
  3. After MIN_TRADES trades, Welch's t-test determines significance
  4. If treatment wins (p < 0.05, better Sharpe): adopt new config
  5. If treatment loses or insufficient evidence: rollback to control
  6. Failed experiments count toward adaptive learning rate

Research basis:
  - Frequentist A/B testing with Welch's t-test (unequal variances)
  - Sequential testing with early stopping for clear losers
  - Bayesian treatment allocation (shift capital toward winner)
"""

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────


@dataclass
class MetaConfig:
    """Learned configuration that replaces hardcoded values."""

    strategy_weight: float = 0.6
    ml_weight: float = 0.4
    agreement_bonus: float = 0.1
    disagreement_penalty: float = 0.75
    position_size_method: str = "fixed"
    kelly_fraction: float = 0.25
    retrain_hours: float = 6.0
    min_confidence: float = 0.3
    regime_adjustments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return {
            "strategy_weight": round(self.strategy_weight, 4),
            "ml_weight": round(self.ml_weight, 4),
            "agreement_bonus": round(self.agreement_bonus, 4),
            "disagreement_penalty": round(self.disagreement_penalty, 4),
            "position_size_method": self.position_size_method,
            "kelly_fraction": round(self.kelly_fraction, 4),
            "retrain_hours": round(self.retrain_hours, 2),
            "min_confidence": round(self.min_confidence, 4),
            "regime_adjustments": self.regime_adjustments,
        }

    def copy(self) -> MetaConfig:
        """Deep copy for experiment forking."""
        new = MetaConfig(
            strategy_weight=self.strategy_weight,
            ml_weight=self.ml_weight,
            agreement_bonus=self.agreement_bonus,
            disagreement_penalty=self.disagreement_penalty,
            position_size_method=self.position_size_method,
            kelly_fraction=self.kelly_fraction,
            retrain_hours=self.retrain_hours,
            min_confidence=self.min_confidence,
            regime_adjustments=copy.deepcopy(self.regime_adjustments),
        )
        return new


@dataclass
class TradeObservation:
    """A recorded trade with context for learning."""

    pnl: float
    signal_source: str  # "strategy", "ml", "both"
    strategy_signal: str  # BUY/SELL/HOLD
    ml_signal: str
    final_signal: str
    strategy_confidence: float
    ml_confidence: float
    regime: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize trade observation to dict."""
        return {
            "pnl": self.pnl,
            "signal_source": self.signal_source,
            "strategy_signal": self.strategy_signal,
            "ml_signal": self.ml_signal,
            "final_signal": self.final_signal,
            "strategy_confidence": self.strategy_confidence,
            "ml_confidence": self.ml_confidence,
            "regime": self.regime,
            "timestamp": self.timestamp.isoformat(),
        }


class ExperimentStatus(Enum):
    """State of an A/B experiment."""

    RUNNING = "running"
    TREATMENT_WINS = "treatment_wins"
    CONTROL_WINS = "control_wins"
    INCONCLUSIVE = "inconclusive"
    EARLY_STOPPED = "early_stopped"


@dataclass
class ABExperiment:
    """
    A single A/B test comparing control vs treatment config.

    Capital is split: control_pct goes to old config, rest to new.
    Each trade records PnL under both configs (simulated for treatment).
    """

    experiment_id: int
    control_config: MetaConfig
    treatment_config: MetaConfig
    changes_description: str
    created_at: datetime = field(default_factory=datetime.now)
    status: ExperimentStatus = ExperimentStatus.RUNNING

    # Trade PnL tracking
    control_pnls: list[float] = field(default_factory=list)
    treatment_pnls: list[float] = field(default_factory=list)

    # Capital allocation (shifts toward winner)
    control_allocation: float = 0.70  # 70% to proven config
    treatment_allocation: float = 0.30  # 30% to experiment

    # Test results
    p_value: float = 1.0
    control_sharpe: float = 0.0
    treatment_sharpe: float = 0.0
    conclusion: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize experiment state for dashboard display."""
        return {
            "id": self.experiment_id,
            "status": self.status.value,
            "control_trades": len(self.control_pnls),
            "treatment_trades": len(self.treatment_pnls),
            "control_avg_pnl": round(np.mean(self.control_pnls), 6) if self.control_pnls else 0.0,
            "treatment_avg_pnl": round(np.mean(self.treatment_pnls), 6) if self.treatment_pnls else 0.0,
            "control_allocation": round(self.control_allocation, 2),
            "treatment_allocation": round(self.treatment_allocation, 2),
            "p_value": round(self.p_value, 4),
            "control_sharpe": round(self.control_sharpe, 3),
            "treatment_sharpe": round(self.treatment_sharpe, 3),
            "changes": self.changes_description,
            "conclusion": self.conclusion,
        }


# ── A/B Testing Engine ─────────────────────────────────────────────


class ABTestEngine:
    """
    Statistical A/B testing engine for meta-learner experiments.

    Uses Welch's t-test (unequal variances) to determine if treatment
    config significantly outperforms control config.
    """

    # Experiment parameters
    MIN_TRADES_PER_ARM = 25  # Minimum trades before testing
    MAX_TRADES_PER_ARM = 150  # Force conclusion after this
    SIGNIFICANCE_LEVEL = 0.05  # p < 0.05 to adopt
    EARLY_STOP_THRESHOLD = 3.0  # Stop early if treatment avg < -3x control
    MAX_CONCURRENT_EXPERIMENTS = 1  # One experiment at a time
    ALLOCATION_SHIFT_RATE = 0.05  # Shift allocation toward interim winner

    def __init__(self) -> None:
        """
        Initialize the A/B testing engine.

        Starts with no active experiments and zeroed counters for
        adopted, rejected, and inconclusive outcomes.
        """
        self._experiments: list[ABExperiment] = []
        self._active_experiment: ABExperiment | None = None
        self._experiment_counter: int = 0
        self._total_adopted: int = 0
        self._total_rejected: int = 0
        self._total_inconclusive: int = 0

    def has_active_experiment(self) -> bool:
        """Check if there's a running experiment."""
        return self._active_experiment is not None

    def start_experiment(
        self, control_config: MetaConfig, treatment_config: MetaConfig, description: str
    ) -> ABExperiment:
        """
        Start a new A/B experiment.

        Args:
            control_config: Current proven config
            treatment_config: Proposed new config from learning
            description: Human-readable description of changes

        Returns:
            The newly created ``ABExperiment`` instance with status
            ``RUNNING``, registered as the active experiment.
        """
        if self._active_experiment is not None:
            _log.warning("Experiment already running, concluding previous")
            self._force_conclude()

        self._experiment_counter += 1
        exp = ABExperiment(
            experiment_id=self._experiment_counter,
            control_config=control_config.copy(),
            treatment_config=treatment_config.copy(),
            changes_description=description,
        )
        self._active_experiment = exp
        self._experiments.append(exp)

        _log.info("A/B Experiment #%d started: %s", exp.experiment_id, description)
        return exp

    def record_trade(self, pnl: float, obs: TradeObservation) -> dict[str, Any]:
        """
        Record a trade outcome for the active experiment.

        Simulates what the trade would have earned under both configs.
        The actual PnL is from the blended allocation.

        Returns dict with allocation-weighted PnL breakdown.
        """
        if self._active_experiment is None:
            return {"actual_pnl": pnl, "experiment": None}

        exp = self._active_experiment

        # Simulate PnL under both configs
        # Control uses the actual PnL (it's the current live config)
        control_pnl = pnl

        # Treatment PnL: adjusted by the difference in signal weights
        # This is an approximation — in practice, treatment signals may
        # differ, but we use PnL scaling as a proxy
        treatment_pnl = self._simulate_treatment_pnl(pnl, obs, exp.control_config, exp.treatment_config)

        exp.control_pnls.append(control_pnl)
        exp.treatment_pnls.append(treatment_pnl)

        # Allocation-weighted actual PnL
        actual_pnl = control_pnl * exp.control_allocation + treatment_pnl * exp.treatment_allocation

        # Adaptive allocation: shift toward interim winner
        self._update_allocation(exp)

        # Check if we should conclude
        result = {"actual_pnl": actual_pnl, "experiment": exp.experiment_id}

        n = min(len(exp.control_pnls), len(exp.treatment_pnls))
        if n >= self.MIN_TRADES_PER_ARM:
            conclusion = self._evaluate_experiment(exp)
            if conclusion is not None:
                result["conclusion"] = conclusion
        elif n >= 10:
            # Early stop check for clearly bad treatments
            if self._should_early_stop(exp):
                exp.status = ExperimentStatus.EARLY_STOPPED
                exp.conclusion = "Treatment clearly underperforming, stopped early"
                self._total_rejected += 1
                self._active_experiment = None
                result["conclusion"] = "early_stopped"
                _log.info("Experiment #%d early-stopped", exp.experiment_id)

        return result

    def get_active_config(self) -> tuple[MetaConfig | None, float]:
        """
        Get the config to use and its allocation weight.

        During experiments, returns control config (safe default).
        The caller applies treatment allocation separately.

        Returns:
            Tuple of (config, allocation) where config is the control
            ``MetaConfig`` during an active experiment (or ``None`` when
            no experiment is running) and allocation is the control's
            capital fraction (e.g. 0.70) or 1.0 when idle.
        """
        if self._active_experiment:
            return (self._active_experiment.control_config, self._active_experiment.control_allocation)
        return None, 1.0

    def get_treatment_config(self) -> tuple[MetaConfig | None, float]:
        """Get treatment config and its allocation, if experiment active.

        Returns:
            Tuple of (config, allocation) where config is the treatment
            ``MetaConfig`` during an active experiment (or ``None`` when
            no experiment is running) and allocation is the treatment's
            capital fraction (e.g. 0.30) or 0.0 when idle.
        """
        if self._active_experiment:
            return (self._active_experiment.treatment_config, self._active_experiment.treatment_allocation)
        return None, 0.0

    def get_status(self) -> dict[str, Any]:
        """Dashboard status.

        Returns:
            Dictionary with keys ``"active_experiment"`` (dict or
            ``None``), ``"total_experiments"`` (int), ``"adopted"``
            (int), ``"rejected"`` (int), ``"inconclusive"`` (int), and
            ``"history"`` (list of up to 5 recent experiment dicts).
        """
        return {
            "active_experiment": self._active_experiment.to_dict() if self._active_experiment else None,
            "total_experiments": self._experiment_counter,
            "adopted": self._total_adopted,
            "rejected": self._total_rejected,
            "inconclusive": self._total_inconclusive,
            "history": [e.to_dict() for e in self._experiments[-5:]],
        }

    # ── Internal: Statistical Testing ────────────────────────────────

    def _evaluate_experiment(self, exp: ABExperiment) -> str | None:
        """
        Run Welch's t-test to evaluate experiment outcome.

        Returns conclusion string if experiment should end, None to continue.
        """
        c_pnls = np.array(exp.control_pnls)
        t_pnls = np.array(exp.treatment_pnls)
        n_c, n_t = len(c_pnls), len(t_pnls)

        if n_c < self.MIN_TRADES_PER_ARM or n_t < self.MIN_TRADES_PER_ARM:
            return None

        # Compute Sharpe ratios (annualized assuming ~100 trades/day)
        c_mean, c_std = np.mean(c_pnls), np.std(c_pnls, ddof=1)
        t_mean, t_std = np.mean(t_pnls), np.std(t_pnls, ddof=1)

        exp.control_sharpe = (c_mean / c_std * np.sqrt(252)) if c_std > 0 else 0.0
        exp.treatment_sharpe = (t_mean / t_std * np.sqrt(252)) if t_std > 0 else 0.0

        # Welch's t-test (treatment - control)
        p_value = self._welch_ttest(t_pnls, c_pnls)
        exp.p_value = p_value

        # Force conclusion if max trades reached
        force = n_c >= self.MAX_TRADES_PER_ARM or n_t >= self.MAX_TRADES_PER_ARM

        # Decision
        if p_value < self.SIGNIFICANCE_LEVEL and t_mean > c_mean:
            exp.status = ExperimentStatus.TREATMENT_WINS
            exp.conclusion = (
                f"Treatment wins! p={p_value:.4f}, Sharpe {exp.treatment_sharpe:.2f} vs {exp.control_sharpe:.2f}"
            )
            self._total_adopted += 1
            self._active_experiment = None
            _log.info("Experiment #%d: TREATMENT ADOPTED — %s", exp.experiment_id, exp.conclusion)
            return "treatment_wins"

        elif p_value < self.SIGNIFICANCE_LEVEL and c_mean > t_mean:
            exp.status = ExperimentStatus.CONTROL_WINS
            exp.conclusion = (
                f"Control wins. p={p_value:.4f}, Sharpe {exp.control_sharpe:.2f} vs {exp.treatment_sharpe:.2f}"
            )
            self._total_rejected += 1
            self._active_experiment = None
            _log.info("Experiment #%d: TREATMENT REJECTED — %s", exp.experiment_id, exp.conclusion)
            return "control_wins"

        elif force:
            exp.status = ExperimentStatus.INCONCLUSIVE
            exp.conclusion = f"Inconclusive after {n_c}/{n_t} trades, p={p_value:.4f}"
            self._total_inconclusive += 1
            self._active_experiment = None
            _log.info("Experiment #%d: INCONCLUSIVE — %s", exp.experiment_id, exp.conclusion)
            return "inconclusive"

        return None  # Continue collecting data

    def _welch_ttest(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Welch's t-test for two samples with potentially unequal variances.

        Returns two-sided p-value.
        Uses Student's t-distribution approximation.
        """
        n_a, n_b = len(a), len(b)
        if n_a < 2 or n_b < 2:
            return 1.0

        mean_a, mean_b = np.mean(a), np.mean(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

        se_a = var_a / n_a
        se_b = var_b / n_b
        se_diff = np.sqrt(se_a + se_b)

        if se_diff < 1e-12:
            return 1.0 if abs(mean_a - mean_b) < 1e-12 else 0.0

        t_stat = (mean_a - mean_b) / se_diff

        # Welch-Satterthwaite degrees of freedom
        numerator = (se_a + se_b) ** 2
        denominator = se_a**2 / (n_a - 1) + se_b**2 / (n_b - 1)
        if denominator < 1e-12:
            df = n_a + n_b - 2
        else:
            df = numerator / denominator

        df = max(df, 1.0)

        # Approximate p-value using the t-distribution
        # Using the incomplete beta function approximation
        p_value = self._t_distribution_pvalue(abs(t_stat), df)
        return p_value

    @staticmethod
    def _t_distribution_pvalue(t: float, df: float) -> float:
        """
        Approximate two-sided p-value from Student's t-distribution.

        Uses the approximation: p ≈ 2 * (1 - Φ(t * sqrt(df/(df-2+t²))))
        where Φ is the standard normal CDF. Accurate for df > 5.
        """
        if df <= 0:
            return 1.0

        # For large df, t ≈ z (normal)
        if df > 100:
            # Normal approximation
            z = t
        else:
            # Adjusted t to z mapping
            z = t * np.sqrt(df / (df - 2 + t**2)) if df > 2 else t * 0.7

        # Standard normal CDF approximation (Abramowitz & Stegun)
        p = 0.5 * (1 + math.erf(-z / np.sqrt(2)))

        # Two-sided
        return min(2 * p, 1.0)

    def _should_early_stop(self, exp: ABExperiment) -> bool:
        """
        Early stop if treatment is clearly worse.

        Stops if treatment avg PnL < -THRESHOLD × |control avg PnL|
        """
        if len(exp.treatment_pnls) < 10:
            return False

        c_mean = np.mean(exp.control_pnls) if exp.control_pnls else 0
        t_mean = np.mean(exp.treatment_pnls)

        # If control is profitable and treatment is losing badly
        if c_mean > 0 and t_mean < -self.EARLY_STOP_THRESHOLD * c_mean:
            return True

        # If both are losing but treatment is much worse
        return t_mean < 0 and c_mean >= 0 and abs(t_mean) > abs(c_mean) * 2

    def _update_allocation(self, exp: ABExperiment) -> None:
        """
        Adaptively shift capital allocation toward interim winner.

        This is a form of Thompson Sampling / adaptive allocation.
        """
        if len(exp.control_pnls) < 10 or len(exp.treatment_pnls) < 10:
            return

        c_mean = np.mean(exp.control_pnls[-20:])  # Recent window
        t_mean = np.mean(exp.treatment_pnls[-20:])

        if t_mean > c_mean:
            # Treatment doing well — increase its allocation
            exp.treatment_allocation = min(
                0.50,  # Never more than 50% to unproven
                exp.treatment_allocation + self.ALLOCATION_SHIFT_RATE,
            )
        else:
            # Control doing better — decrease treatment
            exp.treatment_allocation = max(
                0.15,  # Keep at least 15% for statistical power
                exp.treatment_allocation - self.ALLOCATION_SHIFT_RATE,
            )

        exp.control_allocation = 1.0 - exp.treatment_allocation

    def _simulate_treatment_pnl(
        self, actual_pnl: float, obs: TradeObservation, control: MetaConfig, treatment: MetaConfig
    ) -> float:
        """
        Estimate what PnL would have been under treatment config.

        Re-computes the signal combination under treatment weights to determine:
        1. Would treatment have taken this trade at all? (confidence gate)
        2. Would treatment have reached a different combined signal strength?
        3. How would position sizing differ under treatment config?

        The PnL is then scaled by the ratio of treatment vs control signal
        strength, reflecting that stronger conviction → larger position → more PnL.

        Note: This is still an approximation since we can't replay the exact
        market impact. True parallel execution would require splitting capital
        at the order level, which is a future enhancement.
        """
        # Step 1: Re-compute combined confidence under treatment weights
        control_combined_conf = (
            obs.strategy_confidence * control.strategy_weight
            + obs.ml_confidence * control.ml_weight
        )
        treatment_combined_conf = (
            obs.strategy_confidence * treatment.strategy_weight
            + obs.ml_confidence * treatment.ml_weight
        )

        # Agreement bonus: if both signals agree, add bonus
        if obs.strategy_signal == obs.ml_signal and obs.strategy_signal != "HOLD":
            control_combined_conf += control.agreement_bonus
            treatment_combined_conf += treatment.agreement_bonus

        # Disagreement penalty: if signals disagree, dampen
        if obs.strategy_signal != obs.ml_signal:
            control_combined_conf *= control.disagreement_penalty
            treatment_combined_conf *= treatment.disagreement_penalty

        # Step 2: Would treatment have traded?
        if treatment_combined_conf < treatment.min_confidence:
            return 0.0  # Treatment would have skipped this trade

        # Step 3: Signal strength ratio determines PnL scaling
        # Stronger combined confidence → larger effective position
        if control_combined_conf > 1e-8:
            signal_ratio = treatment_combined_conf / control_combined_conf
        else:
            signal_ratio = 1.0

        # Step 4: Position sizing adjustment
        sizing_ratio = 1.0
        if treatment.position_size_method == "kelly" and control.position_size_method == "kelly":
            sizing_ratio = treatment.kelly_fraction / max(control.kelly_fraction, 0.01)
        elif treatment.position_size_method == "kelly" and control.position_size_method == "fixed":
            # Kelly vs fixed: scale by kelly fraction relative to baseline
            sizing_ratio = treatment.kelly_fraction / 0.25  # 0.25 = default fixed sizing baseline
        elif treatment.position_size_method == "fixed" and control.position_size_method == "kelly":
            sizing_ratio = 0.25 / max(control.kelly_fraction, 0.01)

        # Combined scale factor (bounded to prevent wild swings)
        scale = signal_ratio * sizing_ratio
        scale = max(0.2, min(3.0, scale))

        return actual_pnl * scale

    def _force_conclude(self) -> None:
        """
        Force-conclude the active experiment as inconclusive.

        Called when a new experiment is started while one is already running.
        Marks the current experiment ``INCONCLUSIVE`` and clears the active
        reference so the new experiment can be registered.
        """
        if self._active_experiment:
            self._active_experiment.status = ExperimentStatus.INCONCLUSIVE
            self._active_experiment.conclusion = "Force-concluded for new experiment"
            self._total_inconclusive += 1
            self._active_experiment = None


# ── Meta-Learner with A/B Testing ─────────────────────────────────


class MetaLearner:
    """
    v2.0: Learns optimal trading configuration WITH A/B testing validation.

    Instead of immediately applying learned weights, proposes changes as
    experiments. New config only adopted if it statistically outperforms
    the current config (p < 0.05).

    Public interface is backward-compatible with v1.0.
    """

    def __init__(self, window_size: int = 200) -> None:
        """
        Initialize the MetaLearner with A/B testing support.

        Args:
            window_size: Maximum number of trade observations to retain in
                the rolling learning window. Older observations are evicted
                when the window is full.
        """
        self.window_size = window_size
        self.observations: deque[TradeObservation] = deque(maxlen=window_size)
        self.config = MetaConfig()
        self._regime_observations: dict[str, deque[TradeObservation]] = defaultdict(lambda: deque(maxlen=100))
        self.learning_count: int = 0
        self.last_learn_time: datetime | None = None
        self._drift_events: deque[dict[str, Any]] = deque(maxlen=50)
        self._initial_retrain_hours: float = 6.0

        # v2.0: A/B Testing Engine
        self._ab_engine = ABTestEngine()
        self._learning_rate: float = 0.3
        self._consecutive_rejections: int = 0
        self._adoption_history: deque = deque(maxlen=20)

    # ── Public Interface (backward-compatible) ───────────────────────

    def observe_trade(
        self,
        pnl: float,
        strategy_signal: str,
        ml_signal: str,
        final_signal: str,
        strategy_confidence: float,
        ml_confidence: float,
        regime: str,
    ) -> None:
        """
        Record a trade outcome with full context.

        Appends the observation to the rolling window and, if an A/B
        experiment is active, feeds the PnL to the experiment engine.
        Automatically handles experiment conclusions (adoption / rejection /
        inconclusive) when the engine signals a result.

        Args:
            pnl: Realised profit-and-loss for this trade.
            strategy_signal: Signal produced by the strategy ensemble
                (``"BUY"``, ``"SELL"``, or ``"HOLD"``).
            ml_signal: Signal produced by the ML model.
            final_signal: The signal that was actually executed.
            strategy_confidence: Confidence score from the strategy
                ensemble in [0, 1].
            ml_confidence: Confidence score from the ML model in [0, 1].
            regime: Market regime label at trade time.
        """
        if strategy_signal == ml_signal:
            source = "both"
        elif final_signal == strategy_signal:
            source = "strategy"
        else:
            source = "ml"

        obs = TradeObservation(
            pnl=pnl,
            signal_source=source,
            strategy_signal=strategy_signal,
            ml_signal=ml_signal,
            final_signal=final_signal,
            strategy_confidence=strategy_confidence,
            ml_confidence=ml_confidence,
            regime=regime,
        )
        self.observations.append(obs)
        self._regime_observations[regime].append(obs)

        # v2.0: Feed trade to A/B engine if experiment is running
        if self._ab_engine.has_active_experiment():
            result = self._ab_engine.record_trade(pnl, obs)

            # Check if experiment concluded
            conclusion = result.get("conclusion")
            if conclusion == "treatment_wins":
                self._adopt_treatment()
            elif conclusion in ("control_wins", "early_stopped"):
                self._handle_rejection()
            elif conclusion == "inconclusive":
                self._handle_inconclusive()

    def learn(self) -> dict[str, Any]:
        """
        v2.0: Run learning algorithms and propose changes as A/B experiment.

        Instead of immediately modifying config, creates an experiment
        comparing old config vs proposed config.

        Returns:
            Dictionary with key ``"status"`` (one of
            ``"insufficient_data"``, ``"experiment_in_progress"``,
            ``"no_changes_proposed"``, or ``"experiment_started"``)
            and additional keys depending on status: ``"observations"``
            (int), ``"experiment"`` (dict), ``"round"`` (int),
            ``"changes"`` (dict), or ``"experiment_id"`` (int).
        """
        if len(self.observations) < 10:
            return {"status": "insufficient_data", "observations": len(self.observations)}

        # Don't start a new experiment while one is running
        if self._ab_engine.has_active_experiment():
            return {"status": "experiment_in_progress", "experiment": self._ab_engine.get_status()["active_experiment"]}

        # Propose new config through learning algorithms
        proposed = self.config.copy()
        changes = {}

        w_change = self._learn_signal_weights_proposed(proposed)
        if w_change:
            changes["signal_weights"] = w_change

        ps_change = self._learn_position_sizing_proposed(proposed)
        if ps_change:
            changes["position_sizing"] = ps_change

        rt_change = self._learn_retraining_frequency()
        if rt_change:
            changes["retrain_frequency"] = rt_change

        regime_change = self._learn_regime_adjustments_proposed(proposed)
        if regime_change:
            changes["regime_adjustments"] = regime_change

        conf_change = self._learn_confidence_threshold_proposed(proposed)
        if conf_change:
            changes["min_confidence"] = conf_change

        self.learning_count += 1
        self.last_learn_time = datetime.now()

        if not changes:
            return {"status": "no_changes_proposed", "round": self.learning_count}

        # v2.0: Start A/B experiment instead of immediate adoption
        description = ", ".join(changes.keys())
        exp = self._ab_engine.start_experiment(
            control_config=self.config,
            treatment_config=proposed,
            description=description,
        )

        _log.info(
            "[MetaLearner] Proposed changes (round %d): %s → A/B test #%d",
            self.learning_count,
            description,
            exp.experiment_id,
        )

        return {
            "status": "experiment_started",
            "changes": changes,
            "round": self.learning_count,
            "experiment_id": exp.experiment_id,
        }

    def learn_immediate(self) -> dict[str, Any]:
        """
        v1.0 compatibility: learn and apply configuration changes immediately.

        Runs all learning algorithms and applies proposed changes directly to
        ``self.config`` without creating an A/B experiment. Use this for
        initial warm-up or when statistical validation is not required.

        Returns:
            Dict with keys ``status``, ``changes``, and ``round``.
        """
        if len(self.observations) < 10:
            return {"status": "insufficient_data", "observations": len(self.observations)}

        changes = {}

        w_change = self._learn_signal_weights_proposed(self.config)
        if w_change:
            changes["signal_weights"] = w_change

        ps_change = self._learn_position_sizing_proposed(self.config)
        if ps_change:
            changes["position_sizing"] = ps_change

        rt_change = self._learn_retraining_frequency()
        if rt_change:
            changes["retrain_frequency"] = rt_change

        regime_change = self._learn_regime_adjustments_proposed(self.config)
        if regime_change:
            changes["regime_adjustments"] = regime_change

        conf_change = self._learn_confidence_threshold_proposed(self.config)
        if conf_change:
            changes["min_confidence"] = conf_change

        self.learning_count += 1
        self.last_learn_time = datetime.now()

        if changes:
            _log.info("[MetaLearner] Immediate update (round %d): %s", self.learning_count, list(changes.keys()))

        return {"status": "learned", "changes": changes, "round": self.learning_count}

    # ── A/B Testing Internals ─────────────────────────────────────────

    def _adopt_treatment(self) -> None:
        """
        Adopt the winning treatment config as the new live configuration.

        Called when the A/B engine reports ``"treatment_wins"``.  Resets the
        consecutive-rejection counter and records the adoption in
        ``_adoption_history``.
        """
        exp = self._ab_engine._experiments[-1]
        self.config = exp.treatment_config.copy()
        self._consecutive_rejections = 0
        self._adoption_history.append(
            {
                "round": self.learning_count,
                "experiment_id": exp.experiment_id,
                "result": "adopted",
                "sharpe_improvement": round(exp.treatment_sharpe - exp.control_sharpe, 3),
            }
        )
        _log.info("[MetaLearner] New config adopted! Sharpe: %.2f → %.2f", exp.control_sharpe, exp.treatment_sharpe)

    def _handle_rejection(self) -> None:
        """
        Handle a rejected treatment: keep current config, adjust learning rate.

        Increments the consecutive-rejection counter and reduces the
        adaptive learning rate after three or more consecutive rejections,
        making future proposals more conservative.
        """
        self._consecutive_rejections += 1
        self._adoption_history.append(
            {
                "round": self.learning_count,
                "result": "rejected",
                "consecutive": self._consecutive_rejections,
            }
        )

        # Reduce learning rate after consecutive rejections
        # This makes future proposals more conservative
        if self._consecutive_rejections >= 3:
            self._learning_rate = max(0.1, self._learning_rate * 0.8)
            _log.info(
                "[MetaLearner] %d consecutive rejections, learning rate → %.2f",
                self._consecutive_rejections,
                self._learning_rate,
            )

    def _handle_inconclusive(self) -> None:
        """
        Handle an inconclusive experiment outcome.

        No configuration change is made; the result is recorded in
        ``_adoption_history`` for dashboard visibility.
        """
        self._adoption_history.append(
            {
                "round": self.learning_count,
                "result": "inconclusive",
            }
        )

    # ── Learning Algorithms (propose to config, don't modify self.config) ──

    def _learn_signal_weights_proposed(self, target: MetaConfig) -> dict[str, Any] | None:
        """Learn optimal strategy vs ML weights, write to target config."""
        strat_profits, ml_profits, both_profits = [], [], []

        for obs in self.observations:
            if obs.signal_source == "strategy":
                strat_profits.append(obs.pnl)
            elif obs.signal_source == "ml":
                ml_profits.append(obs.pnl)
            elif obs.signal_source == "both":
                both_profits.append(obs.pnl)

        if len(strat_profits) < 3 and len(ml_profits) < 3:
            return None

        strat_avg = np.mean(strat_profits) if strat_profits else 0
        ml_avg = np.mean(ml_profits) if ml_profits else 0

        strat_wr = np.mean([1 if p > 0 else 0 for p in strat_profits]) if strat_profits else 0.5
        ml_wr = np.mean([1 if p > 0 else 0 for p in ml_profits]) if ml_profits else 0.5

        strat_score = strat_avg * strat_wr if strat_profits else 0
        ml_score = ml_avg * ml_wr if ml_profits else 0

        total = abs(strat_score) + abs(ml_score)
        if total < 1e-8:
            return None

        new_strat_w = max(0.2, min(0.8, (strat_score / total + 1) / 2))

        # Use adaptive learning rate
        alpha = self._learning_rate
        old_sw = target.strategy_weight
        target.strategy_weight = round(old_sw * (1 - alpha) + new_strat_w * alpha, 4)
        target.ml_weight = round(1 - target.strategy_weight, 4)

        if both_profits and np.mean(both_profits) > max(strat_avg, ml_avg):
            target.agreement_bonus = min(0.2, target.agreement_bonus + 0.01)

        return {
            "strategy_weight": target.strategy_weight,
            "ml_weight": target.ml_weight,
            "strat_score": round(strat_score, 4),
            "ml_score": round(ml_score, 4),
        }

    def _learn_position_sizing_proposed(self, target: MetaConfig) -> dict[str, Any] | None:
        """Learn best position sizing method, write to target config."""
        if len(self.observations) < 20:
            return None

        profits = [obs.pnl for obs in self.observations]
        win_rate = np.mean([1 if p > 0 else 0 for p in profits])
        avg_win = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
        avg_loss = abs(np.mean([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 1

        if avg_loss > 0 and avg_win > 0:
            b = avg_win / avg_loss
            kelly = (win_rate * b - (1 - win_rate)) / b
            kelly = max(0, min(0.5, kelly))

            if kelly > 0.05:
                target.position_size_method = "kelly"
                target.kelly_fraction = round(kelly * 0.5, 4)
                return {"method": "kelly", "kelly_f": target.kelly_fraction}

        target.position_size_method = "fixed"
        return {"method": "fixed"}

    def _learn_retraining_frequency(self) -> dict[str, Any] | None:
        """Adjust model retrain frequency (modifies self.config directly — safe)."""
        if len(self._drift_events) < 2:
            return None

        times = [d["timestamp"] for d in self._drift_events]
        if len(times) >= 2:
            intervals = []
            for i in range(1, len(times)):
                dt = (times[i] - times[i - 1]).total_seconds() / 3600
                intervals.append(dt)

            avg_interval = np.mean(intervals)
            new_hours = max(1, min(24, avg_interval * 0.5))
            old_hours = self.config.retrain_hours
            self.config.retrain_hours = round(old_hours * 0.7 + new_hours * 0.3, 2)
            return {"retrain_hours": self.config.retrain_hours}

        return None

    def _learn_regime_adjustments_proposed(self, target: MetaConfig) -> dict[str, Any] | None:
        """Learn regime-specific adjustments, write to target config."""
        adjustments = {}

        for regime, obs_deque in self._regime_observations.items():
            if len(obs_deque) < 5:
                continue

            obs_list = list(obs_deque)
            strat_pnls = [o.pnl for o in obs_list if o.signal_source in ("strategy", "both")]
            ml_pnls = [o.pnl for o in obs_list if o.signal_source in ("ml", "both")]

            strat_perf = np.mean(strat_pnls) if strat_pnls else 0
            ml_perf = np.mean(ml_pnls) if ml_pnls else 0

            if abs(strat_perf - ml_perf) > 0.001:
                bias = "strategy" if strat_perf > ml_perf else "ml"
                adjustments[regime] = {
                    "bias": bias,
                    "strat_perf": round(strat_perf, 4),
                    "ml_perf": round(ml_perf, 4),
                }

        if adjustments:
            target.regime_adjustments = adjustments
            return adjustments
        return None

    def _learn_confidence_threshold_proposed(self, target: MetaConfig) -> dict[str, Any] | None:
        """Learn minimum confidence threshold, write to target config."""
        if len(self.observations) < 20:
            return None

        buckets = defaultdict(list)
        for obs in self.observations:
            conf = max(obs.strategy_confidence, obs.ml_confidence)
            bucket = round(conf, 1)
            buckets[bucket].append(obs.pnl)

        best_threshold = 0.5
        for conf_level in sorted(buckets.keys()):
            if len(buckets[conf_level]) >= 3:
                avg_pnl = np.mean(buckets[conf_level])
                if avg_pnl > 0:
                    best_threshold = conf_level
                    break

        old_threshold = target.min_confidence
        new_threshold = round(old_threshold * 0.7 + best_threshold * 0.3, 4)
        new_threshold = max(0.3, min(0.8, new_threshold))

        if abs(new_threshold - old_threshold) > 0.01:
            target.min_confidence = new_threshold
            return {"min_confidence": new_threshold}
        return None

    # ── Original Public Methods (preserved) ───────────────────────────

    def record_drift_event(self) -> None:
        """Record a model drift event for retraining frequency learning."""
        self._drift_events.append({"timestamp": datetime.now()})

    def get_config(self) -> MetaConfig:
        """Get current learned configuration.

        Returns:
            The current live ``MetaConfig`` instance used by the agent.
        """
        return self.config

    def get_signal_weights(self, regime: str | None = None) -> tuple[float, float]:
        """Get strategy/ML weights, optionally regime-adjusted.

        Returns:
            Tuple of (strategy_weight, ml_weight) as floats in [0.0, 1.0]
            that sum to 1.0, with a +0.05 bias applied to whichever
            source performed better in ``regime`` (when provided and
            present in regime adjustments).
        """
        sw = self.config.strategy_weight
        mw = self.config.ml_weight

        if regime and regime in self.config.regime_adjustments:
            adj = self.config.regime_adjustments[regime]
            if adj.get("bias") == "strategy":
                sw = min(0.8, sw + 0.05)
                mw = 1 - sw
            elif adj.get("bias") == "ml":
                mw = min(0.8, mw + 0.05)
                sw = 1 - mw

        return sw, mw

    # ── v2.0: Enhanced Status ─────────────────────────────────────────

    def get_ab_status(self) -> dict[str, Any]:
        """A/B testing specific status.

        Returns:
            Dictionary from :meth:`ABTestEngine.get_status` with keys
            ``"active_experiment"``, ``"total_experiments"``,
            ``"adopted"``, ``"rejected"``, ``"inconclusive"``, and
            ``"history"``.
        """
        return self._ab_engine.get_status()

    def get_status(self) -> dict[str, Any]:
        """Full meta-learner status with A/B testing info.

        Returns:
            Dictionary with keys ``"config"`` (dict), ``"observations"``
            (int), ``"learning_rounds"`` (int), ``"last_learn"`` (ISO
            str or ``None``), ``"drift_events"`` (int),
            ``"regime_data"`` (dict[str, int]), ``"ab_testing"`` (dict),
            ``"learning_rate"`` (float), ``"consecutive_rejections"``
            (int), and ``"adoption_history"`` (list of up to 5 dicts).
        """
        return {
            "config": self.config.to_dict(),
            "observations": len(self.observations),
            "learning_rounds": self.learning_count,
            "last_learn": self.last_learn_time.isoformat() if self.last_learn_time else None,
            "drift_events": len(self._drift_events),
            "regime_data": {regime: len(obs) for regime, obs in self._regime_observations.items()},
            # v2.0 additions
            "ab_testing": self._ab_engine.get_status(),
            "learning_rate": round(self._learning_rate, 3),
            "consecutive_rejections": self._consecutive_rejections,
            "adoption_history": list(self._adoption_history)[-5:],
        }

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize for state persistence.

        Returns:
            Dictionary with keys ``"config"`` (dict), ``"learning_count"``
            (int), ``"observations"`` (list of up to 50 serialized
            ``TradeObservation`` dicts), ``"learning_rate"`` (float),
            ``"consecutive_rejections"`` (int), and
            ``"adoption_history"`` (list of dicts).
        """
        return {
            "config": self.config.to_dict(),
            "learning_count": self.learning_count,
            "observations": [o.to_dict() for o in list(self.observations)[-50:]],
            # v2.0
            "learning_rate": self._learning_rate,
            "consecutive_rejections": self._consecutive_rejections,
            "adoption_history": list(self._adoption_history),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """
        Restore learner state from a serialized dict.

        Args:
            data: Dict previously returned by :meth:`to_dict`.
        """
        self.learning_count = data.get("learning_count", 0)
        cfg = data.get("config", {})
        if cfg:
            self.config.strategy_weight = cfg.get("strategy_weight", 0.6)
            self.config.ml_weight = cfg.get("ml_weight", 0.4)
            self.config.agreement_bonus = cfg.get("agreement_bonus", 0.1)
            self.config.disagreement_penalty = cfg.get("disagreement_penalty", 0.6)
            self.config.position_size_method = cfg.get("position_size_method", "fixed")
            self.config.kelly_fraction = cfg.get("kelly_fraction", 0.25)
            self.config.retrain_hours = cfg.get("retrain_hours", 6.0)
            self.config.min_confidence = cfg.get("min_confidence", 0.5)
            self.config.regime_adjustments = cfg.get("regime_adjustments", {})

        # v2.0
        self._learning_rate = data.get("learning_rate", 0.3)
        self._consecutive_rejections = data.get("consecutive_rejections", 0)
        for item in data.get("adoption_history", []):
            self._adoption_history.append(item)
