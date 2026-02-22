"""
Value at Risk (VaR) Calculator
================================
Computes VaR and CVaR from historical trade data.
Supports parametric, historical, and Monte Carlo methods.
"""

import numpy as np


class VaRCalculator:
    """Computes Value at Risk metrics."""

    @staticmethod
    def historical_var(returns: list[float], confidence: float = 0.95, portfolio_value: float = 1000.0) -> dict:
        """
        Historical VaR — uses actual return distribution.

        Args:
            returns: List of historical returns (proportional, e.g. 0.02 = 2%)
            confidence: Confidence level (0.95 = 95%)
            portfolio_value: Current portfolio value
        """
        if not returns or len(returns) < 5:
            return {"var": 0, "cvar": 0, "method": "historical", "error": "insufficient_data"}

        arr = np.array(returns)
        percentile = (1 - confidence) * 100
        var = np.percentile(arr, percentile) * portfolio_value
        # CVaR = expected loss given loss exceeds VaR
        threshold = np.percentile(arr, percentile)
        tail_losses = arr[arr <= threshold]
        cvar = np.mean(tail_losses) * portfolio_value if len(tail_losses) > 0 else var

        return {
            "var": round(float(var), 2),
            "cvar": round(float(cvar), 2),
            "confidence": confidence,
            "method": "historical",
            "sample_size": len(returns),
        }

    @staticmethod
    def parametric_var(returns: list[float], confidence: float = 0.95, portfolio_value: float = 1000.0) -> dict:
        """
        Parametric (Gaussian) VaR — assumes normal distribution.

        Faster but less accurate for fat-tailed crypto returns.
        """
        if not returns or len(returns) < 5:
            return {"var": 0, "cvar": 0, "method": "parametric", "error": "insufficient_data"}

        from scipy import stats

        arr = np.array(returns)
        mean = np.mean(arr)
        std = np.std(arr)

        z_score = stats.norm.ppf(1 - confidence)
        var = (mean + z_score * std) * portfolio_value

        # CVaR for normal distribution
        cvar = (mean - std * stats.norm.pdf(z_score) / (1 - confidence)) * portfolio_value

        return {
            "var": round(float(var), 2),
            "cvar": round(float(cvar), 2),
            "confidence": confidence,
            "method": "parametric",
            "mean_return": round(float(mean), 6),
            "std_return": round(float(std), 6),
            "sample_size": len(returns),
        }
