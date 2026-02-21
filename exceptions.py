"""
Custom exception hierarchy for the trading agent.
All domain-specific exceptions inherit from TradingError.
"""


class TradingError(Exception):
    """Base exception for all trading agent errors."""


class DataFetchError(TradingError):
    """Failed to fetch market data from exchange or API."""


class ExecutionError(TradingError):
    """Failed to execute, modify, or cancel an order."""


class ValidationError(TradingError):
    """Configuration or input validation failure."""


class ModelError(TradingError):
    """ML model training or prediction failure."""
