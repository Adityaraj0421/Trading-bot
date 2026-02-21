"""
Monte Carlo Simulator
======================
Simulates thousands of possible future equity paths based on historical
trade return distribution. Computes VaR, CVaR, and probability of ruin.
"""

import numpy as np
from dataclasses import dataclass, field
from config import Config


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation run."""
    n_simulations: int
    n_days: int
    initial_equity: float
    var_95: float              # 95% Value at Risk (loss)
    var_99: float              # 99% Value at Risk (loss)
    cvar_95: float             # Expected loss beyond 95% VaR
    cvar_99: float             # Expected loss beyond 99% VaR
    max_drawdown_median: float
    max_drawdown_95th: float
    probability_of_ruin: float  # P(equity <= 0)
    median_final_equity: float
    percentile_5: float        # Worst 5% outcome
    percentile_25: float
    percentile_75: float
    percentile_95: float       # Best 5% outcome
    mean_final_equity: float
    paths_summary: list = field(default_factory=list)  # Percentile bands for chart

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "n_days": self.n_days,
            "initial_equity": round(self.initial_equity, 2),
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "cvar_95": round(self.cvar_95, 2),
            "cvar_99": round(self.cvar_99, 2),
            "max_drawdown_median": round(self.max_drawdown_median, 4),
            "max_drawdown_95th": round(self.max_drawdown_95th, 4),
            "probability_of_ruin": round(self.probability_of_ruin, 4),
            "median_final_equity": round(self.median_final_equity, 2),
            "percentile_5": round(self.percentile_5, 2),
            "percentile_25": round(self.percentile_25, 2),
            "percentile_75": round(self.percentile_75, 2),
            "percentile_95": round(self.percentile_95, 2),
            "mean_final_equity": round(self.mean_final_equity, 2),
            "paths_summary": self.paths_summary,
            "interpretation": self._interpret(),
        }

    def _interpret(self) -> dict:
        """Beginner-friendly interpretation of results."""
        risk_level = "low"
        if self.probability_of_ruin > 0.05:
            risk_level = "high"
        elif self.probability_of_ruin > 0.01:
            risk_level = "moderate"

        return {
            "risk_level": risk_level,
            "var_95_explanation": f"There's a 95% chance your daily loss won't exceed ${abs(self.var_95):.2f}",
            "cvar_95_explanation": f"If losses do exceed VaR, the average loss would be ${abs(self.cvar_95):.2f}",
            "ruin_explanation": f"{self.probability_of_ruin:.2%} chance of losing all capital over {self.n_days} days",
            "median_explanation": f"The most likely outcome is ending with ${self.median_final_equity:.2f}",
            "best_case": f"In the best 5% of scenarios, you'd have ${self.percentile_95:.2f}",
            "worst_case": f"In the worst 5% of scenarios, you'd have ${self.percentile_5:.2f}",
        }


class MonteCarloSimulator:
    """Runs Monte Carlo simulations on trading strategy returns."""

    def __init__(self, n_simulations: int = None, n_days: int = None):
        self.n_simulations = n_simulations or Config.MC_SIMULATIONS
        self.n_days = n_days or Config.MC_HORIZON_DAYS

    def run(self, trade_returns: list[float], initial_equity: float = None) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Args:
            trade_returns: List of percentage returns per trade (e.g., [0.02, -0.01, 0.03])
            initial_equity: Starting equity value
        """
        initial_equity = initial_equity or Config.INITIAL_CAPITAL

        if not trade_returns or len(trade_returns) < 5:
            # Not enough data — return conservative estimate
            return self._default_result(initial_equity)

        returns = np.array(trade_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Generate random paths
        # Each path: initial_equity * cumulative product of (1 + random_return)
        random_returns = np.random.normal(
            mean_return, std_return,
            size=(self.n_simulations, self.n_days)
        )

        # Build equity paths
        equity_paths = initial_equity * np.cumprod(1 + random_returns, axis=1)

        # Compute final equity values
        final_equities = equity_paths[:, -1]

        # Value at Risk (VaR) — daily
        daily_returns_all = random_returns.flatten()
        var_95 = np.percentile(daily_returns_all, 5) * initial_equity
        var_99 = np.percentile(daily_returns_all, 1) * initial_equity

        # Conditional VaR (CVaR) — expected loss beyond VaR
        losses_beyond_95 = daily_returns_all[daily_returns_all <= np.percentile(daily_returns_all, 5)]
        cvar_95 = np.mean(losses_beyond_95) * initial_equity if len(losses_beyond_95) > 0 else var_95

        losses_beyond_99 = daily_returns_all[daily_returns_all <= np.percentile(daily_returns_all, 1)]
        cvar_99 = np.mean(losses_beyond_99) * initial_equity if len(losses_beyond_99) > 0 else var_99

        # Max drawdown per simulation
        max_drawdowns = []
        for path in equity_paths:
            running_max = np.maximum.accumulate(path)
            drawdowns = (running_max - path) / running_max
            max_drawdowns.append(np.max(drawdowns))

        max_drawdowns = np.array(max_drawdowns)

        # Probability of ruin
        probability_of_ruin = np.mean(np.any(equity_paths <= 0, axis=1))

        # Percentile bands for chart (every 10th day)
        step = max(1, self.n_days // 50)
        paths_summary = []
        for day_idx in range(0, self.n_days, step):
            day_values = equity_paths[:, day_idx]
            paths_summary.append({
                "day": day_idx + 1,
                "p5": round(float(np.percentile(day_values, 5)), 2),
                "p25": round(float(np.percentile(day_values, 25)), 2),
                "p50": round(float(np.percentile(day_values, 50)), 2),
                "p75": round(float(np.percentile(day_values, 75)), 2),
                "p95": round(float(np.percentile(day_values, 95)), 2),
            })

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            n_days=self.n_days,
            initial_equity=initial_equity,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown_median=float(np.median(max_drawdowns)),
            max_drawdown_95th=float(np.percentile(max_drawdowns, 95)),
            probability_of_ruin=float(probability_of_ruin),
            median_final_equity=float(np.median(final_equities)),
            percentile_5=float(np.percentile(final_equities, 5)),
            percentile_25=float(np.percentile(final_equities, 25)),
            percentile_75=float(np.percentile(final_equities, 75)),
            percentile_95=float(np.percentile(final_equities, 95)),
            mean_final_equity=float(np.mean(final_equities)),
            paths_summary=paths_summary,
        )

    def _default_result(self, initial_equity: float) -> MonteCarloResult:
        """Return a default result when there's insufficient data."""
        return MonteCarloResult(
            n_simulations=0,
            n_days=0,
            initial_equity=initial_equity,
            var_95=0,
            var_99=0,
            cvar_95=0,
            cvar_99=0,
            max_drawdown_median=0,
            max_drawdown_95th=0,
            probability_of_ruin=0,
            median_final_equity=initial_equity,
            percentile_5=initial_equity,
            percentile_25=initial_equity,
            percentile_75=initial_equity,
            percentile_95=initial_equity,
            mean_final_equity=initial_equity,
            paths_summary=[],
        )
