"""
Historical Black Swan Event Replayer
======================================
Applies real historical crash patterns to the current portfolio.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class StressScenario:
    name: str
    description: str
    daily_returns: list[float]  # Sequence of daily returns during the event
    duration_days: int
    total_drawdown: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "duration_days": self.duration_days,
            "total_drawdown": round(self.total_drawdown * 100, 2),
        }


# Historical black swan events
HISTORICAL_SCENARIOS = [
    StressScenario(
        name="COVID Crash (March 2020)",
        description="BTC dropped ~50% in 2 days as global markets panicked",
        daily_returns=[-0.15, -0.30, -0.10, 0.05, -0.08, 0.12, 0.05],
        duration_days=7,
        total_drawdown=-0.50,
    ),
    StressScenario(
        name="China Mining Ban (May 2021)",
        description="BTC dropped ~55% over 2 weeks after China banned crypto mining",
        daily_returns=[-0.05, -0.08, -0.03, -0.10, 0.02, -0.06, -0.04,
                       -0.07, 0.03, -0.05, -0.03, -0.02, 0.01, -0.04],
        duration_days=14,
        total_drawdown=-0.55,
    ),
    StressScenario(
        name="FTX Collapse (Nov 2022)",
        description="BTC dropped ~25% in a week after FTX exchange collapsed",
        daily_returns=[-0.05, -0.12, -0.08, -0.03, 0.02, -0.04, 0.01],
        duration_days=7,
        total_drawdown=-0.25,
    ),
    StressScenario(
        name="Terra/Luna Crash (May 2022)",
        description="Market-wide contagion as UST/LUNA collapsed",
        daily_returns=[-0.03, -0.05, -0.10, -0.15, -0.08, 0.05, -0.03,
                       -0.02, 0.01, -0.04],
        duration_days=10,
        total_drawdown=-0.40,
    ),
]


class StressTestRunner:
    """Applies historical crash patterns to a portfolio."""

    def run_stress_test(self, initial_equity: float,
                        scenarios: list[StressScenario] = None) -> list[dict]:
        """
        Run all stress test scenarios against the current portfolio.
        Returns projected losses for each scenario.
        """
        scenarios = scenarios or HISTORICAL_SCENARIOS
        results = []

        for scenario in scenarios:
            equity = initial_equity
            equity_path = [equity]

            for daily_return in scenario.daily_returns:
                equity *= (1 + daily_return)
                equity_path.append(round(equity, 2))

            final_equity = equity_path[-1]
            loss = initial_equity - final_equity
            loss_pct = loss / initial_equity

            results.append({
                **scenario.to_dict(),
                "initial_equity": round(initial_equity, 2),
                "final_equity": round(final_equity, 2),
                "projected_loss": round(loss, 2),
                "projected_loss_pct": round(loss_pct * 100, 2),
                "equity_path": equity_path,
                "survived": final_equity > 0,
            })

        return results

    def list_scenarios(self) -> list[dict]:
        return [s.to_dict() for s in HISTORICAL_SCENARIOS]
