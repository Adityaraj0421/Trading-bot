"""
Decision Engine — Central Orchestration Brain
===============================================
Coordinates all autonomous subsystems: self-healing, strategy evolution,
meta-learning, and hyperparameter optimization. Enforces safety limits
and manages the system's operational state.
"""

from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

from self_healer import SelfHealer, ErrorSeverity, CircuitState
from strategy_evolver import StrategyEvolver
from meta_learner import MetaLearner
from auto_optimizer import AutoOptimizer


class DecisionState(Enum):
    NORMAL = "normal"         # Full trading
    CAUTIOUS = "cautious"     # Reduced position sizes after losses
    DEFENSIVE = "defensive"   # Minimal trading, near safety limits
    HALTED = "halted"         # No trading, safety triggered


@dataclass
class AutonomousConfig:
    """Safety limits for autonomous operation."""
    max_daily_loss_pct: float = 5.0        # Max % loss in a day
    max_consecutive_losses: int = 5         # Max losing streak before CAUTIOUS
    min_capital_pct: float = 50.0           # Min % of initial capital before HALTED
    cautious_position_mult: float = 0.5     # Position size multiplier in CAUTIOUS
    defensive_position_mult: float = 0.25   # Position size multiplier in DEFENSIVE
    evolution_interval: int = 500           # Cycles between evolution rounds
    learning_interval: int = 100            # Cycles between meta-learning
    optimization_interval: int = 1000       # Cycles between optimization
    health_check_interval: int = 10         # Cycles between health checks


@dataclass
class AutonomousEvent:
    """Logged autonomous decision or action."""
    event_type: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.event_type,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


class DecisionEngine:
    """
    Central brain that coordinates all autonomous subsystems.
    Manages system state, enforces safety, and schedules autonomous tasks.
    """

    def __init__(self, initial_capital: float = 10000):
        self.healer = SelfHealer()
        self.evolver = StrategyEvolver()
        self.meta = MetaLearner()
        self.optimizer = AutoOptimizer()
        self.config = AutonomousConfig()

        self.state = DecisionState.NORMAL
        self.initial_capital = initial_capital
        self.event_log: deque = deque(maxlen=1000)

        # Tracking
        self._daily_pnl: float = 0.0
        self._daily_reset_date: Optional[datetime] = None
        self._consecutive_losses: int = 0
        self._last_evolution_cycle: int = 0
        self._last_learning_cycle: int = 0
        self._last_optimization_cycle: int = 0
        self._state_change_time: Optional[datetime] = None
        self._total_autonomous_decisions: int = 0
        self._latest_evolved_params: dict = {}  # Best params from strategy evolution

        # Production safeguards
        self._manual_halt: bool = False
        self._halt_reason: str = ""
        self._force_close_all: bool = False
        self._alerts: list[dict] = []

    # --- Production safeguards (kill switch, manual override) ---

    def emergency_halt(self, reason: str = "Manual kill switch"):
        """Immediately halt all trading. Called via API kill switch."""
        self._manual_halt = True
        self._halt_reason = reason
        self.state = DecisionState.HALTED
        self._log_event("emergency_halt", reason)
        self._add_alert("critical", f"EMERGENCY HALT: {reason}")
        print(f"\n  [KILL SWITCH] Trading HALTED: {reason}")

    def emergency_resume(self):
        """Resume trading after manual halt."""
        self._manual_halt = False
        self._halt_reason = ""
        self.state = DecisionState.NORMAL
        self._log_event("emergency_resume", "Trading resumed manually")
        print(f"\n  [RESUME] Trading resumed")

    def force_close_all_positions(self):
        """Signal agent to close all open positions immediately."""
        self._force_close_all = True
        self._log_event("force_close", "Force close all positions triggered")
        self._add_alert("warning", "Force closing all positions")

    def _add_alert(self, severity: str, message: str):
        """Add an alert to the alert queue."""
        alert = {
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False,
        }
        self._alerts.append(alert)
        # Keep last 100 alerts
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def get_alerts(self, unacknowledged_only: bool = False) -> list[dict]:
        """Get alerts, optionally filtering to unacknowledged only."""
        if unacknowledged_only:
            return [a for a in self._alerts if not a["acknowledged"]]
        return self._alerts.copy()

    def acknowledge_alerts(self):
        """Mark all alerts as acknowledged."""
        for a in self._alerts:
            a["acknowledged"] = True

    def orchestrate(self, cycle_count: int, current_capital: float,
                    current_pnl: float = 0) -> dict:
        """
        Main orchestration method — called at the start of each cycle.
        Returns dict with instructions for the agent.
        """
        instructions = {
            "should_trade": True,
            "position_multiplier": 1.0,
            "signal_weights": None,  # None = use defaults
            "skip_reason": None,
            "autonomous_actions": [],
            "force_close_all": False,
        }

        # 0. Manual halt check (kill switch)
        if self._manual_halt:
            instructions["should_trade"] = False
            instructions["skip_reason"] = f"MANUAL HALT: {self._halt_reason}"
            return instructions

        # 0b. Force close all positions
        if self._force_close_all:
            instructions["force_close_all"] = True
            self._force_close_all = False  # Reset after signaling

        # 1. Health check (periodic)
        if cycle_count % self.config.health_check_interval == 0:
            health = self.healer.check_health()
            if not health.overall_healthy:
                self._log_event("health_warning", "System health degraded",
                               health.to_dict())
                # Don't halt for health — just log. Self-healer handles recovery.

        # 2. Safety check
        state, reason = self._check_safety(current_capital, current_pnl)
        if state != self.state:
            old_state = self.state
            self.state = state
            self._state_change_time = datetime.now()
            self._log_event("state_change",
                           f"{old_state.value} -> {state.value}: {reason}",
                           {"old": old_state.value, "new": state.value, "reason": reason})
            print(f"  [DecisionEngine] State: {old_state.value} -> {state.value} ({reason})")

        # Apply state-based restrictions
        if self.state == DecisionState.HALTED:
            instructions["should_trade"] = False
            instructions["skip_reason"] = f"HALTED: {reason}"
            return instructions
        elif self.state == DecisionState.DEFENSIVE:
            instructions["position_multiplier"] = self.config.defensive_position_mult
        elif self.state == DecisionState.CAUTIOUS:
            instructions["position_multiplier"] = self.config.cautious_position_mult

        # 3. Get learned signal weights
        instructions["signal_weights"] = {
            "strategy": self.meta.config.strategy_weight,
            "ml": self.meta.config.ml_weight,
            "agreement_bonus": self.meta.config.agreement_bonus,
        }

        # 4. Scheduled autonomous tasks
        actions = self._run_scheduled_tasks(cycle_count)
        instructions["autonomous_actions"] = actions

        # 5. Include evolved strategy params for hot-reload
        if self._latest_evolved_params:
            instructions["evolved_params"] = self._latest_evolved_params

        # 6. Apply optimized hyperparams to Config if available
        optimized = self.optimizer.apply_best_to_config()
        if optimized:
            instructions["optimized_config"] = optimized

        return instructions

    def _check_safety(self, current_capital: float, recent_pnl: float) -> tuple:
        """Check safety limits and determine system state."""
        now = datetime.now()

        # Reset daily PnL tracking at midnight
        if self._daily_reset_date is None or self._daily_reset_date.date() != now.date():
            self._daily_pnl = 0.0
            self._daily_reset_date = now

        # Track consecutive losses
        # (This is updated by record_trade_result, not here)

        # Check capital floor
        capital_pct = (current_capital / self.initial_capital) * 100
        if capital_pct < self.config.min_capital_pct:
            return DecisionState.HALTED, f"capital at {capital_pct:.1f}% (min {self.config.min_capital_pct}%)"

        # Check daily loss
        daily_loss_pct = abs(min(0, self._daily_pnl)) / self.initial_capital * 100
        if daily_loss_pct > self.config.max_daily_loss_pct:
            return DecisionState.DEFENSIVE, f"daily loss {daily_loss_pct:.1f}% (max {self.config.max_daily_loss_pct}%)"

        # Check consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            return DecisionState.CAUTIOUS, f"{self._consecutive_losses} consecutive losses"

        # Check if we can recover from CAUTIOUS/DEFENSIVE
        if self.state in (DecisionState.CAUTIOUS, DecisionState.DEFENSIVE):
            if self._state_change_time:
                elapsed = (now - self._state_change_time).total_seconds()
                # Auto-recover after 30 minutes if conditions improve
                if elapsed > 1800 and self._consecutive_losses < 3:
                    return DecisionState.NORMAL, "conditions improved"

        return self.state if self.state != DecisionState.HALTED else DecisionState.NORMAL, "ok"

    def record_trade_result(self, pnl: float, strategy_signal: str, ml_signal: str,
                            final_signal: str, strategy_confidence: float,
                            ml_confidence: float, regime: str):
        """Record a trade result — updates all subsystems."""
        # Track daily PnL
        self._daily_pnl += pnl

        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Alert on large losses (> 2% of capital)
        loss_pct = abs(pnl) / self.initial_capital * 100
        if pnl < 0 and loss_pct > 2.0:
            self._add_alert("warning", f"Large loss: ${pnl:.2f} ({loss_pct:.1f}% of capital)")

        # Alert on losing streaks
        if self._consecutive_losses >= 3:
            self._add_alert("warning", f"Losing streak: {self._consecutive_losses} consecutive losses")

        # Feed to meta-learner
        self.meta.observe_trade(
            pnl=pnl,
            strategy_signal=strategy_signal,
            ml_signal=ml_signal,
            final_signal=final_signal,
            strategy_confidence=strategy_confidence,
            ml_confidence=ml_confidence,
            regime=regime,
        )

        self._total_autonomous_decisions += 1

    def record_drift_event(self):
        """Record a model drift event."""
        self.meta.record_drift_event()
        self._log_event("drift_detected", "Model drift detected, retraining triggered")

    def _run_scheduled_tasks(self, cycle_count: int) -> list[str]:
        """Run autonomous tasks on schedule."""
        actions = []

        # Meta-learning (every 100 cycles)
        if (cycle_count - self._last_learning_cycle >= self.config.learning_interval
                and len(self.meta.observations) >= 10):
            result = self.meta.learn()
            if result.get("changes"):
                actions.append(f"meta_learn: {list(result['changes'].keys())}")
                self._log_event("meta_learning", "Updated trading config",
                               result.get("changes", {}))
            self._last_learning_cycle = cycle_count

        # Strategy evolution (every 500 cycles) — backtest unscored genomes, then evolve
        if cycle_count - self._last_evolution_cycle >= self.config.evolution_interval:
            if self.evolver._initialized:
                # First: evaluate fitness of unscored genomes via backtesting
                scored = self._evaluate_population_fitness()
                if scored:
                    actions.append(f"fitness_eval: scored {scored} genomes via backtest")

                # Then: evolve populations using fitness scores
                for name in self.evolver.populations:
                    self.evolver.evolve(name)
                # Collect best evolved params for strategy hot-reload
                evolved = self.evolver.get_all_best()
                self._latest_evolved_params = {
                    name: data["parameters"]
                    for name, data in evolved.items()
                    if data.get("parameters")
                }
                actions.append(f"evolution: gen {self.evolver.generation}")
                self._log_event("evolution", f"Evolved to generation {self.evolver.generation}",
                               {"evolved_strategies": list(self._latest_evolved_params.keys()),
                                "genomes_scored": scored})
            self._last_evolution_cycle = cycle_count

        # Auto-optimization (every 1000 cycles) — run backtest trials
        if cycle_count - self._last_optimization_cycle >= self.config.optimization_interval:
            try:
                results = self._run_optimization_trials()
                if results:
                    actions.append(f"optimization: {len(results)} trials, "
                                   f"best_score={self.optimizer.best_result.score:.3f}")
                    self._log_event("optimization",
                                   f"Ran {len(results)} trials",
                                   {"best_score": self.optimizer.best_result.score if self.optimizer.best_result else 0})
            except Exception as e:
                self._log_event("optimization_error", str(e))
            self._last_optimization_cycle = cycle_count

        return actions

    def _evaluate_population_fitness(self, max_per_strategy: int = 5) -> int:
        """Backtest unscored genomes in all strategy populations.

        This bridges the strategy_evolver with the backtester so that
        evolution is driven by actual backtest performance rather than
        random fitness values.

        Returns:
            Number of genomes scored.
        """
        from backtester import Backtester
        from demo_data import generate_demo_ohlcv

        scored = 0
        df = generate_demo_ohlcv(periods=500)  # Longer series for reliable fitness

        for strat_name, population in self.evolver.populations.items():
            evaluated = 0
            for genome in population:
                if genome.fitness is not None:
                    continue  # Already scored
                if evaluated >= max_per_strategy:
                    break

                try:
                    import config as cfg
                    # Snapshot config values
                    orig = {
                        "sl": cfg.Config.STOP_LOSS_PCT,
                        "tp": cfg.Config.TAKE_PROFIT_PCT,
                        "trail": cfg.Config.TRAILING_STOP_PCT,
                        "conf": cfg.Config.MIN_CONFIDENCE,
                        "hold": cfg.Config.MAX_HOLD_BARS,
                        "positions": cfg.Config.MAX_OPEN_POSITIONS,
                    }

                    # Apply genome parameters to Config
                    params = genome.parameters
                    cfg.Config.STOP_LOSS_PCT = params.get("stop_loss_pct", orig["sl"] * 100) / 100.0
                    cfg.Config.TAKE_PROFIT_PCT = params.get("take_profit_pct", orig["tp"] * 100) / 100.0
                    cfg.Config.TRAILING_STOP_PCT = params.get("trailing_stop_pct", orig["trail"] * 100) / 100.0
                    cfg.Config.MIN_CONFIDENCE = params.get("confidence_threshold", orig["conf"])
                    cfg.Config.MAX_HOLD_BARS = int(params.get("max_hold_bars", orig["hold"]))
                    cfg.Config.MAX_OPEN_POSITIONS = int(params.get("max_open_positions", orig["positions"]))

                    bt = Backtester(fee_pct=cfg.Config.FEE_PCT, slippage_pct=0.0005)
                    metrics = bt.run(df, verbose=False)

                    # Use evolver's fitness evaluation
                    self.evolver.evaluate_fitness(genome, metrics)
                    scored += 1
                    evaluated += 1

                except Exception as e:
                    _log = __import__("logging").getLogger(__name__)
                    _log.debug("Fitness eval failed for %s genome: %s", strat_name, e)
                finally:
                    # Always restore Config
                    cfg.Config.STOP_LOSS_PCT = orig["sl"]
                    cfg.Config.TAKE_PROFIT_PCT = orig["tp"]
                    cfg.Config.TRAILING_STOP_PCT = orig["trail"]
                    cfg.Config.MIN_CONFIDENCE = orig["conf"]
                    cfg.Config.MAX_HOLD_BARS = orig["hold"]
                    cfg.Config.MAX_OPEN_POSITIONS = orig["positions"]

        return scored

    def _run_optimization_trials(self, n_trials: int = 5) -> list[dict]:
        """Run mini-backtest trials for hyperparameter optimization."""
        from backtester import Backtester
        from demo_data import generate_demo_ohlcv
        import config as cfg

        suggestions = self.optimizer.run_optimization_round(n_trials=n_trials)
        results = []

        df = generate_demo_ohlcv(periods=500)

        for params in suggestions:
            # Snapshot original config
            orig = {
                "sl": cfg.Config.STOP_LOSS_PCT,
                "tp": cfg.Config.TAKE_PROFIT_PCT,
                "trail": cfg.Config.TRAILING_STOP_PCT,
                "conf": cfg.Config.MIN_CONFIDENCE,
                "hold": cfg.Config.MAX_HOLD_BARS,
                "positions": cfg.Config.MAX_OPEN_POSITIONS,
            }
            try:
                cfg.Config.STOP_LOSS_PCT = params.get("stop_loss_pct", 2.0) / 100.0
                cfg.Config.TAKE_PROFIT_PCT = params.get("take_profit_pct", 3.0) / 100.0
                cfg.Config.TRAILING_STOP_PCT = params.get("trailing_stop_pct", 1.5) / 100.0
                cfg.Config.MIN_CONFIDENCE = params.get("confidence_threshold", 0.5)
                cfg.Config.MAX_HOLD_BARS = int(params.get("max_hold_bars", 100))
                cfg.Config.MAX_OPEN_POSITIONS = int(params.get("max_open_positions", orig["positions"]))

                bt = Backtester(fee_pct=cfg.Config.FEE_PCT, slippage_pct=0.0005)
                metrics = bt.run(df, verbose=False)

                self.optimizer.record_result(params, metrics)
                results.append({"params": params, "metrics": metrics})
            except Exception as e:
                __import__("logging").getLogger(__name__).debug("Optimization trial failed: %s", e)
                continue
            finally:
                # Always restore Config
                cfg.Config.STOP_LOSS_PCT = orig["sl"]
                cfg.Config.TAKE_PROFIT_PCT = orig["tp"]
                cfg.Config.TRAILING_STOP_PCT = orig["trail"]
                cfg.Config.MIN_CONFIDENCE = orig["conf"]
                cfg.Config.MAX_HOLD_BARS = orig["hold"]
                cfg.Config.MAX_OPEN_POSITIONS = orig["positions"]

        return results

    def should_override_signal(self, signal: str, confidence: float) -> tuple[str, float]:
        """
        Override or dampen signals based on system state.

        v7.0: Relaxed thresholds so bot trades in NORMAL state.
        Safety states (DEFENSIVE/CAUTIOUS) still reduce sizing but don't
        kill signals as aggressively.
        """
        if self.state == DecisionState.HALTED:
            return "HOLD", 0.0

        if self.state == DecisionState.DEFENSIVE:
            # Only allow moderate+ confidence trades (was 0.75 → 0.5)
            if confidence < 0.5:
                return "HOLD", confidence
            confidence *= 0.8  # Slight dampen (was 0.7)

        if self.state == DecisionState.CAUTIOUS:
            # Slight dampening (was 0.8 → 0.9)
            confidence *= 0.9

        # Check minimum confidence from meta-learner
        if confidence < self.meta.config.min_confidence:
            return "HOLD", confidence

        return signal, confidence

    def _log_event(self, event_type: str, description: str, data: dict = None):
        """Log an autonomous event."""
        event = AutonomousEvent(
            event_type=event_type,
            description=description,
            data=data or {},
        )
        self.event_log.append(event)

    def get_autonomous_status(self) -> dict:
        """Comprehensive status report of all autonomous subsystems."""
        return {
            "state": self.state.value,
            "daily_pnl": round(self._daily_pnl, 2),
            "consecutive_losses": self._consecutive_losses,
            "total_autonomous_decisions": self._total_autonomous_decisions,
            "healer": self.healer.get_status(),
            "evolver": self.evolver.get_status(),
            "meta_learner": self.meta.get_status(),
            "optimizer": self.optimizer.get_status(),
            "recent_events": [
                e.to_dict() for e in list(self.event_log)[-10:]
            ],
        }

    def print_autonomous_summary(self):
        """Print concise autonomous system status."""
        state_icons = {
            DecisionState.NORMAL: "[OK]",
            DecisionState.CAUTIOUS: "[!!]",
            DecisionState.DEFENSIVE: "[XX]",
            DecisionState.HALTED: "[STOP]",
        }
        icon = state_icons.get(self.state, "?")
        print(f"\n  AUTONOMOUS {icon} State={self.state.value} | "
              f"DailyPnL=${self._daily_pnl:,.2f} | "
              f"LossStreak={self._consecutive_losses} | "
              f"Decisions={self._total_autonomous_decisions}")

        # Signal weights
        sw = self.meta.config.strategy_weight
        mw = self.meta.config.ml_weight
        print(f"  Weights: Strategy={sw:.0%} ML={mw:.0%} | "
              f"Sizing={self.meta.config.position_size_method} | "
              f"Retrain={self.meta.config.retrain_hours:.1f}h")

        # Evolution
        if self.evolver._initialized:
            print(f"  Evolution: Gen {self.evolver.generation} | "
                  f"Best strategies: {list(self.evolver.best_genomes.keys())[:3]}")

        # Health
        health = self.healer._cached_health
        if health:
            h_count = health.components_healthy
            h_total = health.components_total
            print(f"  Health: {h_count}/{h_total} components | "
                  f"Errors/min={health.error_rate:.1f}")

        # Recent events
        recent = list(self.event_log)[-3:]
        if recent:
            print(f"  Recent: {', '.join(e.event_type for e in recent)}")

    def to_dict(self) -> dict:
        """Serialize for state persistence."""
        return {
            "state": self.state.value,
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "total_decisions": self._total_autonomous_decisions,
            "last_evolution_cycle": self._last_evolution_cycle,
            "last_learning_cycle": self._last_learning_cycle,
            "last_optimization_cycle": self._last_optimization_cycle,
            "config": {
                "max_daily_loss_pct": self.config.max_daily_loss_pct,
                "max_consecutive_losses": self.config.max_consecutive_losses,
                "min_capital_pct": self.config.min_capital_pct,
            },
            "healer": self.healer.to_dict(),
            "evolver": self.evolver.to_dict(),
            "meta": self.meta.to_dict(),
            "optimizer": self.optimizer.to_dict(),
        }

    def from_dict(self, data: dict):
        """Restore from state."""
        state_str = data.get("state", "normal")
        self.state = DecisionState(state_str)
        self._daily_pnl = data.get("daily_pnl", 0)
        self._consecutive_losses = data.get("consecutive_losses", 0)
        self._total_autonomous_decisions = data.get("total_decisions", 0)
        self._last_evolution_cycle = data.get("last_evolution_cycle", 0)
        self._last_learning_cycle = data.get("last_learning_cycle", 0)
        self._last_optimization_cycle = data.get("last_optimization_cycle", 0)

        # Restore subsystems
        if "healer" in data:
            self.healer.from_dict(data["healer"])
        if "evolver" in data:
            self.evolver.from_dict(data["evolver"])
        if "meta" in data:
            self.meta.from_dict(data["meta"])
        if "optimizer" in data:
            self.optimizer.from_dict(data["optimizer"])
