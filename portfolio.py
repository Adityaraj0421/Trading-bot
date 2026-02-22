"""
Multi-Pair Portfolio Manager
==============================
Manages trading across multiple crypto pairs simultaneously.
Tracks correlation, allocates capital per-pair, prevents over-exposure.

Usage:
    portfolio = PortfolioManager(pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    portfolio.analyze_correlations(price_data)
    allocation = portfolio.get_allocation("ETH/USDT")
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from dataclasses import dataclass
from config import Config

_log = logging.getLogger(__name__)


@dataclass
class PairAllocation:
    """Capital allocation for a single trading pair after risk adjustments."""

    pair: str
    weight: float           # Portfolio weight (0-1)
    max_position_pct: float # Max capital to allocate
    correlation_penalty: float  # Reduce if highly correlated with existing positions
    is_tradeable: bool


class PortfolioManager:
    """
    Manages capital allocation across multiple trading pairs.
    Prevents over-concentration and correlation-based risk.
    """

    def __init__(self, pairs: list[str] | None = None) -> None:
        self.pairs = pairs or Config.TRADING_PAIRS
        self.correlations: pd.DataFrame = pd.DataFrame()
        self.pair_returns: dict[str, pd.Series] = {}
        self.pair_volatility: dict[str, float] = {}

        # Equal weight by default
        n = len(self.pairs)
        self.weights = {p: 1.0 / n for p in self.pairs}

        # Track which pairs have open positions
        self.active_pairs: set = set()

    def update_prices(self, pair: str, prices: pd.Series) -> None:
        """Update price series for a pair (for correlation calculation)."""
        returns = prices.pct_change().dropna()
        self.pair_returns[pair] = returns
        self.pair_volatility[pair] = float(returns.std()) if len(returns) > 10 else 0.02

    def compute_correlations(self) -> pd.DataFrame:
        """Compute rolling correlation matrix between all pairs."""
        if len(self.pair_returns) < 2:
            return pd.DataFrame()

        # Align all return series
        aligned = pd.DataFrame(self.pair_returns)
        aligned = aligned.dropna()

        if len(aligned) < 20:
            return pd.DataFrame()

        self.correlations = aligned.corr()
        return self.correlations

    def get_allocation(self, pair: str, existing_positions: list[str] | None = None) -> PairAllocation:
        """
        Get capital allocation for a pair, accounting for:
        - Base portfolio weight
        - Correlation with existing positions
        - Volatility adjustment (inverse volatility weighting)
        """
        base_weight = self.weights.get(pair, 1.0 / len(self.pairs))

        # Volatility adjustment: lower vol pairs get slightly more allocation
        vol = self.pair_volatility.get(pair, 0.02)
        all_vols = list(self.pair_volatility.values()) or [0.02]
        avg_vol = np.mean(all_vols)
        vol_adj = min(avg_vol / vol, 1.5) if vol > 0 else 1.0  # Cap at 1.5x

        # Correlation penalty: reduce allocation if highly correlated
        # with existing open positions
        corr_penalty = 1.0
        if existing_positions and not self.correlations.empty:
            for pos_pair in existing_positions:
                if pos_pair in self.correlations.columns and pair in self.correlations.index:
                    corr = abs(self.correlations.loc[pair, pos_pair])
                    if corr > 0.7:
                        corr_penalty *= (1 - (corr - 0.7))  # Reduce by (corr-0.7)

        adjusted_weight = base_weight * vol_adj * corr_penalty
        max_pct = Config.MAX_POSITION_PCT * adjusted_weight / base_weight

        # Don't trade if too many correlated positions already open
        is_tradeable = corr_penalty > 0.3

        return PairAllocation(
            pair=pair,
            weight=round(adjusted_weight, 4),
            max_position_pct=round(max_pct, 4),
            correlation_penalty=round(1 - corr_penalty, 4),
            is_tradeable=is_tradeable,
        )

    def get_portfolio_risk(self, positions: list[Any]) -> dict[str, Any]:
        """
        Calculate portfolio-level risk metrics.
        """
        if not positions:
            return {"total_exposure": 0, "concentration": 0, "corr_risk": "low"}

        # Exposure by pair
        pair_exposure = {}
        for pos in positions:
            pair = pos.symbol if hasattr(pos, 'symbol') else pos.get('symbol', '')
            value = pos.notional_value if hasattr(pos, 'notional_value') else 0
            pair_exposure[pair] = pair_exposure.get(pair, 0) + value

        total = sum(pair_exposure.values())
        if total == 0:
            return {"total_exposure": 0, "concentration": 0, "corr_risk": "low"}

        # Herfindahl index (concentration)
        weights = [v / total for v in pair_exposure.values()]
        herfindahl = sum(w ** 2 for w in weights)

        # Correlation risk
        corr_risk = "low"
        if not self.correlations.empty and len(pair_exposure) > 1:
            pairs = list(pair_exposure.keys())
            avg_corr = 0
            count = 0
            for i, p1 in enumerate(pairs):
                for p2 in pairs[i+1:]:
                    if p1 in self.correlations.index and p2 in self.correlations.columns:
                        avg_corr += abs(self.correlations.loc[p1, p2])
                        count += 1
            if count > 0:
                avg_corr /= count
                if avg_corr > 0.8:
                    corr_risk = "high"
                elif avg_corr > 0.6:
                    corr_risk = "medium"

        return {
            "total_exposure": round(total, 2),
            "concentration": round(herfindahl, 4),
            "pair_exposure": {k: round(v, 2) for k, v in pair_exposure.items()},
            "corr_risk": corr_risk,
        }

    def rebalance_weights(self) -> None:
        """
        Rebalance portfolio weights using inverse-volatility weighting.
        Lower volatility pairs get higher allocation.
        """
        if not self.pair_volatility:
            return

        inv_vols = {}
        for pair in self.pairs:
            vol = self.pair_volatility.get(pair, 0.02)
            inv_vols[pair] = 1.0 / vol if vol > 0 else 50.0

        total_inv_vol = sum(inv_vols.values())
        self.weights = {
            pair: round(inv_vol / total_inv_vol, 4)
            for pair, inv_vol in inv_vols.items()
        }
