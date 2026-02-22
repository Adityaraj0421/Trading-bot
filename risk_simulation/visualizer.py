"""
Risk Visualization Data Generator
====================================
Generates data structures suitable for dashboard charting.
"""


class RiskVisualizer:
    """Generates chart-ready data for the React dashboard."""

    @staticmethod
    def fan_chart_data(mc_result) -> list[dict]:
        """
        Generate fan chart data from Monte Carlo paths.
        Returns list of {day, p5, p25, p50, p75, p95} for Recharts.
        """
        return mc_result.paths_summary if mc_result.paths_summary else []

    @staticmethod
    def drawdown_histogram(max_drawdowns: list[float], bins: int = 20) -> list[dict]:
        """Generate histogram data for max drawdown distribution."""
        import numpy as np

        if not max_drawdowns:
            return []

        arr = np.array(max_drawdowns)
        counts, edges = np.histogram(arr, bins=bins)

        return [
            {
                "range_start": round(float(edges[i]) * 100, 2),
                "range_end": round(float(edges[i + 1]) * 100, 2),
                "count": int(counts[i]),
                "label": f"{edges[i] * 100:.1f}%-{edges[i + 1] * 100:.1f}%",
            }
            for i in range(len(counts))
        ]

    @staticmethod
    def equity_distribution(final_equities: list[float], bins: int = 30) -> list[dict]:
        """Generate histogram for final equity distribution."""
        import numpy as np

        if not final_equities:
            return []

        arr = np.array(final_equities)
        counts, edges = np.histogram(arr, bins=bins)

        return [
            {
                "range_start": round(float(edges[i]), 2),
                "range_end": round(float(edges[i + 1]), 2),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]

    @staticmethod
    def stress_test_comparison(stress_results: list[dict]) -> list[dict]:
        """Format stress test results for bar chart comparison."""
        return [
            {
                "scenario": r["name"],
                "loss_pct": r["projected_loss_pct"],
                "survived": r["survived"],
            }
            for r in stress_results
        ]
