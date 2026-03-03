"""
Configuration loader for the crypto trading agent v7.0.
Reads settings from .env file and provides defaults.
Supports multi-pair, trailing stops, transaction costs, websockets,
notifications, trade database, and all intelligence modules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_log = logging.getLogger(__name__)

# Use explicit path so .env is found regardless of working directory
load_dotenv(Path(__file__).parent / ".env")


class Config:
    """Central configuration loaded from environment variables with sensible defaults."""

    # Exchange settings
    EXCHANGE_ID: str = os.getenv("EXCHANGE_ID", "binance")
    API_KEY: str = os.getenv("API_KEY", "")
    API_SECRET: str = os.getenv("API_SECRET", "")
    API_PRIVATE_KEY_PATH: str = os.getenv("API_PRIVATE_KEY_PATH", "")  # Ed25519 private key file

    # Trading settings
    TRADING_PAIR: str = os.getenv("TRADING_PAIR", "BTC/USDT")
    TRADING_PAIRS: list[str] = [p.strip() for p in os.getenv("TRADING_PAIRS", "BTC/USDT").split(",") if p.strip()]
    # v9.1: Dynamic pair selection — pool of candidates, top N selected each scoring cycle
    PAIR_POOL: list[str] = [p.strip() for p in os.getenv("PAIR_POOL", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,AVAX/USDT").split(",") if p.strip()]
    PAIR_SELECTOR_TOP_N: int = int(os.getenv("PAIR_SELECTOR_TOP_N", "3"))
    PAIR_SCORER_INTERVAL_CYCLES: int = int(os.getenv("PAIR_SCORER_INTERVAL_CYCLES", "200"))
    TIMEFRAME: str = os.getenv("TIMEFRAME", "1h")
    CONFIRMATION_TIMEFRAME: str = os.getenv("CONFIRMATION_TIMEFRAME", "4h")
    INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "1000"))

    # Risk management
    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.02"))
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.02"))
    TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
    TRAILING_STOP_PCT: float = float(os.getenv("TRAILING_STOP_PCT", "0.015"))
    ATR_TRAILING_MULT: float = float(os.getenv("ATR_TRAILING_MULT", "2.0"))  # v9.1: ATR × mult replaces fixed trailing %
    # Per-pair trailing stop overrides (Phase 6: SOL needs wider stop due to ~2-3× higher ATR)
    SYMBOL_TRAILING_STOP_PCT: dict[str, float] = {
        "BTC/USDT": 0.025,
        "ETH/USDT": 0.025,
        "SOL/USDT": 0.040,
    }
    BREAKEVEN_TRIGGER_PCT: float = float(os.getenv("BREAKEVEN_TRIGGER_PCT", "0.6"))  # v9.1: move SL to entry at 60% of TP dist
    MAX_DAILY_LOSS_PCT: float = 0.05
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    MAX_HOLD_BARS: int = int(os.getenv("MAX_HOLD_BARS", "100"))

    # Transaction costs
    FEE_PCT: float = float(os.getenv("FEE_PCT", "0.001"))
    SLIPPAGE_PCT: float = float(os.getenv("SLIPPAGE_PCT", "0.0005"))

    # Agent settings
    TRADING_MODE: str = os.getenv("TRADING_MODE", "paper")
    LOOKBACK_BARS: int = 200
    AGENT_INTERVAL_SECONDS: int = int(os.getenv("AGENT_INTERVAL_SECONDS", "300"))

    # Model settings
    MODEL_RETRAIN_HOURS: float = float(os.getenv("MODEL_RETRAIN_HOURS", "24"))
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.6"))
    DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
    ML_LABEL_THRESHOLD: float = float(os.getenv("ML_LABEL_THRESHOLD", "0.005"))
    ML_FEATURE_PRUNING: bool = os.getenv("ML_FEATURE_PRUNING", "false").lower() == "true"
    ML_TOP_FEATURES: int = int(os.getenv("ML_TOP_FEATURES", "14"))

    # State persistence
    STATE_FILE: str = os.getenv("STATE_FILE", "")

    # Logging
    LOG_FILE: str = os.getenv("LOG_FILE", "")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # API Server
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    ENABLE_API: bool = os.getenv("ENABLE_API", "true").lower() == "true"
    API_AUTH_KEY: str = os.getenv("API_AUTH_KEY", "")  # API key for authentication (empty = no auth)
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

    # Data directory (for Docker volume mounts)
    DATA_DIR: str = os.getenv("DATA_DIR", ".")

    # Multi-exchange (for arbitrage)
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
    COINBASE_API_KEY: str = os.getenv("COINBASE_API_KEY", "")
    COINBASE_API_SECRET: str = os.getenv("COINBASE_API_SECRET", "")
    KRAKEN_API_KEY: str = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET: str = os.getenv("KRAKEN_API_SECRET", "")

    # Intelligence toggles
    ENABLE_ONCHAIN: bool = os.getenv("ENABLE_ONCHAIN", "false").lower() == "true"
    ENABLE_WHALE_TRACKING: bool = os.getenv("ENABLE_WHALE_TRACKING", "false").lower() == "true"
    ENABLE_NEWS_NLP: bool = os.getenv("ENABLE_NEWS_NLP", "false").lower() == "true"
    ENABLE_CORRELATION: bool = os.getenv("ENABLE_CORRELATION", "false").lower() == "true"
    ENABLE_ORDERBOOK: bool = os.getenv("ENABLE_ORDERBOOK", "false").lower() == "true"
    ENABLE_FUNDING_OI: bool = os.getenv("ENABLE_FUNDING_OI", "true").lower() == "true"
    ENABLE_LIQUIDATION: bool = os.getenv("ENABLE_LIQUIDATION", "true").lower() == "true"
    INTELLIGENCE_INTERVAL_SECONDS: int = int(os.getenv("INTELLIGENCE_INTERVAL_SECONDS", "300"))
    CRYPTOPANIC_API_KEY: str = os.getenv("CRYPTOPANIC_API_KEY", "")

    # Arbitrage
    ARBITRAGE_ENABLED: bool = os.getenv("ARBITRAGE_ENABLED", "false").lower() == "true"
    ARBITRAGE_MIN_SPREAD_PCT: float = float(os.getenv("ARBITRAGE_MIN_SPREAD_PCT", "0.003"))
    ARBITRAGE_INTERVAL_SECONDS: int = int(os.getenv("ARBITRAGE_INTERVAL_SECONDS", "60"))
    ARBITRAGE_CAPITAL_PCT: float = float(os.getenv("ARBITRAGE_CAPITAL_PCT", "0.1"))

    # Monte Carlo
    MC_SIMULATIONS: int = int(os.getenv("MC_SIMULATIONS", "10000"))
    MC_HORIZON_DAYS: int = int(os.getenv("MC_HORIZON_DAYS", "252"))

    # WebSocket streaming
    ENABLE_WEBSOCKET: bool = os.getenv("ENABLE_WEBSOCKET", "false").lower() == "true"

    # Notifications
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_WEBHOOK_URL: str = os.getenv("TELEGRAM_WEBHOOK_URL", "")
    TELEGRAM_WEBHOOK_SECRET: str = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    TELEGRAM_TRADE_CONFIRMATION: bool = os.getenv("TELEGRAM_TRADE_CONFIRMATION", "false").lower() == "true"
    TELEGRAM_CONFIRMATION_TIMEOUT: int = int(os.getenv("TELEGRAM_CONFIRMATION_TIMEOUT", "60"))
    TELEGRAM_CONFIRMATION_DEFAULT: str = os.getenv("TELEGRAM_CONFIRMATION_DEFAULT", "reject")
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    EMAIL_ALERTS_ENABLED: bool = os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true"
    EMAIL_SMTP_HOST: str = os.getenv("EMAIL_SMTP_HOST", "")
    EMAIL_SMTP_PORT: int = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    EMAIL_USER: str = os.getenv("EMAIL_USER", "")
    EMAIL_PASS: str = os.getenv("EMAIL_PASS", "")
    EMAIL_TO: str = os.getenv("EMAIL_TO", "")

    # Trade Database
    ENABLE_TRADE_DB: bool = os.getenv("ENABLE_TRADE_DB", "true").lower() == "true"
    TRADE_DB_PATH: str = os.getenv("TRADE_DB_PATH", "")

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "1200"))
    MAX_ORDERS_PER_MINUTE: int = int(os.getenv("MAX_ORDERS_PER_MINUTE", "10"))

    # Autonomous subsystem intervals (cycles)
    EVOLUTION_INTERVAL = int(os.getenv("EVOLUTION_INTERVAL", "500"))
    META_LEARNING_INTERVAL = int(os.getenv("META_LEARNING_INTERVAL", "100"))
    OPTIMIZATION_INTERVAL = int(os.getenv("OPTIMIZATION_INTERVAL", "1000"))
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "10"))
    EQUITY_SNAPSHOT_INTERVAL = int(os.getenv("EQUITY_SNAPSHOT_INTERVAL", "5"))
    STATE_SAVE_INTERVAL = int(os.getenv("STATE_SAVE_INTERVAL", "10"))
    PORTFOLIO_REBALANCE_INTERVAL = int(os.getenv("PORTFOLIO_REBALANCE_INTERVAL", "20"))

    # Self-healer
    CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
    CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
    DATA_FRESHNESS_SECONDS = int(os.getenv("DATA_FRESHNESS_SECONDS", "600"))
    MODEL_STALENESS_HOURS = int(os.getenv("MODEL_STALENESS_HOURS", "24"))

    # Safety recovery
    SAFETY_RECOVERY_TIMEOUT_SECONDS = int(os.getenv("SAFETY_RECOVERY_TIMEOUT_SECONDS", "1800"))
    DRIFT_BASELINE_CONFIDENCE = float(os.getenv("DRIFT_BASELINE_CONFIDENCE", "0.7"))

    # A/B Testing
    AB_MIN_TRADES = int(os.getenv("AB_MIN_TRADES", "25"))
    AB_MAX_TRADES = int(os.getenv("AB_MAX_TRADES", "150"))
    AB_SIGNIFICANCE_LEVEL = float(os.getenv("AB_SIGNIFICANCE_LEVEL", "0.05"))

    # -------------------------------------------------------------------------
    # Phase 9 — Context + Trigger Architecture
    # -------------------------------------------------------------------------
    # Phase 9 is the primary pipeline — this flag defaults on.
    # Set USE_PHASE9_PIPELINE=false in .env only to run Phase 8 backtest scripts.
    USE_PHASE9_PIPELINE: bool = os.getenv("USE_PHASE9_PIPELINE", "true").lower() == "true"

    # RiskSupervisor thresholds
    PHASE9_DAILY_DRAWDOWN_LIMIT: float = float(os.getenv("PHASE9_DAILY_DRAWDOWN_LIMIT", "0.03"))
    PHASE9_CONSECUTIVE_LOSS_LIMIT: int = int(os.getenv("PHASE9_CONSECUTIVE_LOSS_LIMIT", "4"))
    PHASE9_ATR_SIGMA_THRESHOLD: float = float(os.getenv("PHASE9_ATR_SIGMA_THRESHOLD", "3.0"))
    PHASE9_API_ERROR_RATE_THRESHOLD: float = float(os.getenv("PHASE9_API_ERROR_RATE_THRESHOLD", "0.5"))
    PHASE9_RISK_COOLDOWN_MINUTES: int = int(os.getenv("PHASE9_RISK_COOLDOWN_MINUTES", "120"))

    # ContextEngine cadence
    PHASE9_CONTEXT_INTERVAL_MINUTES: int = int(os.getenv("PHASE9_CONTEXT_INTERVAL_MINUTES", "15"))

    # DecisionLogger output path (empty = log to agent.log only)
    PHASE9_DECISION_LOG_PATH: str = os.getenv("PHASE9_DECISION_LOG_PATH", "")

    @classmethod
    def any_intelligence_enabled(cls) -> bool:
        """Return True if at least one intelligence module is enabled.

        Returns:
            True when any of the ENABLE_ONCHAIN, ENABLE_WHALE_TRACKING,
            ENABLE_NEWS_NLP, ENABLE_CORRELATION, ENABLE_ORDERBOOK,
            ENABLE_FUNDING_OI, or ENABLE_LIQUIDATION flags is True.
        """
        return any(
            [
                cls.ENABLE_ONCHAIN,
                cls.ENABLE_WHALE_TRACKING,
                cls.ENABLE_NEWS_NLP,
                cls.ENABLE_CORRELATION,
                cls.ENABLE_ORDERBOOK,
                cls.ENABLE_FUNDING_OI,
                cls.ENABLE_LIQUIDATION,
            ]
        )

    @classmethod
    def any_notifications_enabled(cls) -> bool:
        """Return True if at least one notification channel is configured.

        Returns:
            True when Telegram (bot token + chat ID), Discord webhook URL,
            or email alerts are configured.
        """
        return any(
            [
                cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID,
                cls.DISCORD_WEBHOOK_URL,
                cls.EMAIL_ALERTS_ENABLED,
            ]
        )

    @classmethod
    def _resolve_paths(cls) -> None:
        """Resolve file paths relative to DATA_DIR when not explicitly set."""
        if not cls.STATE_FILE:
            cls.STATE_FILE = os.path.join(cls.DATA_DIR, "agent_state.json")
        if not cls.LOG_FILE:
            cls.LOG_FILE = os.path.join(cls.DATA_DIR, "agent.log")
        if not cls.TRADE_DB_PATH:
            cls.TRADE_DB_PATH = os.path.join(cls.DATA_DIR, "trades.db")

    @classmethod
    def is_paper_mode(cls) -> bool:
        """Return True if the agent is running in paper-trading mode.

        Returns:
            True when TRADING_MODE is set to ``"paper"`` (case-insensitive).
        """
        return cls.TRADING_MODE.lower() == "paper"

    @classmethod
    def get_trailing_stop_pct(cls, symbol: str) -> float:
        """Return the trailing stop percentage for a given symbol.

        Uses per-pair overrides from ``SYMBOL_TRAILING_STOP_PCT`` when
        available, falling back to the global ``TRAILING_STOP_PCT``.

        Args:
            symbol: Trading pair symbol (e.g. ``"SOL/USDT"``).

        Returns:
            Trailing stop as a decimal fraction (e.g. ``0.040`` for 4%).
        """
        return cls.SYMBOL_TRAILING_STOP_PCT.get(symbol, cls.TRAILING_STOP_PCT)

    @classmethod
    def validate(cls) -> None:
        """Validate all configuration values and print a summary.

        Raises:
            ValueError: If any hard constraint is violated (e.g. missing API
                keys in live mode, invalid risk parameters).

        Note:
            Prints a human-readable configuration summary to stdout after all
            checks pass.
        """
        cls._resolve_paths()

        # ── Hard failures (ValueError → agent won't start) ────────────

        if not cls.is_paper_mode() and (not cls.API_KEY or not cls.API_SECRET):
            raise ValueError(
                "API_KEY and API_SECRET are required for live trading. Set TRADING_MODE=paper for simulation."
            )

        if cls.INITIAL_CAPITAL <= 0:
            raise ValueError(f"INITIAL_CAPITAL must be > 0, got {cls.INITIAL_CAPITAL}")

        if not (0 < cls.STOP_LOSS_PCT < 1):
            raise ValueError(f"STOP_LOSS_PCT must be between 0 and 1 (exclusive), got {cls.STOP_LOSS_PCT}")

        if not (0 < cls.TAKE_PROFIT_PCT < 1):
            raise ValueError(f"TAKE_PROFIT_PCT must be between 0 and 1 (exclusive), got {cls.TAKE_PROFIT_PCT}")

        if cls.STOP_LOSS_PCT >= cls.TAKE_PROFIT_PCT:
            raise ValueError(
                f"STOP_LOSS_PCT ({cls.STOP_LOSS_PCT}) must be < TAKE_PROFIT_PCT ({cls.TAKE_PROFIT_PCT}). "
                "Stop loss should be tighter than take profit."
            )

        total_cost = cls.FEE_PCT + cls.SLIPPAGE_PCT
        if total_cost >= cls.STOP_LOSS_PCT:
            raise ValueError(
                f"FEE_PCT ({cls.FEE_PCT}) + SLIPPAGE_PCT ({cls.SLIPPAGE_PCT}) = {total_cost} "
                f"must be < STOP_LOSS_PCT ({cls.STOP_LOSS_PCT}). "
                "Transaction costs would consume the entire stop loss."
            )

        max_allocation = cls.MAX_POSITION_PCT * cls.MAX_OPEN_POSITIONS
        if max_allocation > 1.0:
            raise ValueError(
                f"MAX_POSITION_PCT ({cls.MAX_POSITION_PCT}) × MAX_OPEN_POSITIONS ({cls.MAX_OPEN_POSITIONS}) "
                f"= {max_allocation:.2f}, which exceeds 1.0. Capital would be overallocated."
            )

        if not (0 < cls.MIN_CONFIDENCE <= 1.0):
            raise ValueError(
                f"MIN_CONFIDENCE must be between 0 (exclusive) and 1.0 (inclusive), got {cls.MIN_CONFIDENCE}"
            )

        if cls.AGENT_INTERVAL_SECONDS < 10:
            raise ValueError(
                f"AGENT_INTERVAL_SECONDS must be >= 10, got {cls.AGENT_INTERVAL_SECONDS}. "
                "Lower values risk exchange rate limits and IP bans."
            )

        if not (1 <= cls.PAIR_SELECTOR_TOP_N <= len(cls.PAIR_POOL)):
            raise ValueError(
                f"PAIR_SELECTOR_TOP_N ({cls.PAIR_SELECTOR_TOP_N}) must be between 1 "
                f"and len(PAIR_POOL) ({len(cls.PAIR_POOL)})"
            )

        # ── Soft warnings (log but don't crash) ──────────────────────

        if cls.TELEGRAM_BOT_TOKEN and not cls.TELEGRAM_CHAT_ID:
            _log.warning(
                "TELEGRAM_BOT_TOKEN is set but TELEGRAM_CHAT_ID is missing. Telegram notifications will silently fail."
            )
        if cls.TELEGRAM_CHAT_ID and not cls.TELEGRAM_BOT_TOKEN:
            _log.warning(
                "TELEGRAM_CHAT_ID is set but TELEGRAM_BOT_TOKEN is missing. Telegram notifications will silently fail."
            )

        if cls.MIN_CONFIDENCE < 0.4:
            _log.warning(
                "MIN_CONFIDENCE is very low (%.2f). The agent may generate many low-quality signals.",
                cls.MIN_CONFIDENCE,
            )

        if total_cost > cls.STOP_LOSS_PCT * 0.5:
            _log.warning(
                "Transaction costs (%.4f) exceed 50%% of STOP_LOSS_PCT (%.4f). "
                "Most stopped-out trades will be net negative.",
                total_cost,
                cls.STOP_LOSS_PCT,
            )

        if max_allocation > 0.8:
            _log.warning(
                "MAX_POSITION_PCT × MAX_OPEN_POSITIONS = %.2f. "
                "Over 80%% capital allocation leaves little reserve for drawdowns.",
                max_allocation,
            )

        # ── Config summary ───────────────────────────────────────────

        print(f"  Mode:     {cls.TRADING_MODE.upper()}")
        print(f"  Exchange: {cls.EXCHANGE_ID}")
        print(f"  Pair:     {cls.TRADING_PAIR}")
        if len(cls.TRADING_PAIRS) > 1:
            print(f"  Pairs:    {', '.join(cls.TRADING_PAIRS)}")
        print(f"  Timeframe:{cls.TIMEFRAME} (confirm: {cls.CONFIRMATION_TIMEFRAME})")
        print(f"  Capital:  ${cls.INITIAL_CAPITAL:,.2f}")
        print(f"  Fees:     {cls.FEE_PCT:.2%} + {cls.SLIPPAGE_PCT:.2%} slippage")
        print(f"  Trailing: {cls.TRAILING_STOP_PCT:.2%}")
        print(f"  MinConf:  {cls.MIN_CONFIDENCE:.2%}")
        print(f"  ML Label: ±{cls.ML_LABEL_THRESHOLD:.2%}")
        if cls.ENABLE_WEBSOCKET:
            print("  WebSocket: ENABLED")
        intel = []
        if cls.ENABLE_ONCHAIN:
            intel.append("OnChain")
        if cls.ENABLE_WHALE_TRACKING:
            intel.append("Whales")
        if cls.ENABLE_NEWS_NLP:
            intel.append("News")
        if cls.ENABLE_CORRELATION:
            intel.append("Correlation")
        if cls.ENABLE_ORDERBOOK:
            intel.append("OrderBook")
        if intel:
            print(f"  Intel:    {', '.join(intel)}")
        notif = []
        if cls.TELEGRAM_BOT_TOKEN:
            notif.append("Telegram")
        if cls.DISCORD_WEBHOOK_URL:
            notif.append("Discord")
        if cls.EMAIL_ALERTS_ENABLED:
            notif.append("Email")
        if notif:
            print(f"  Alerts:   {', '.join(notif)}")
        if cls.ENABLE_TRADE_DB:
            print(f"  TradeDB:  {cls.TRADE_DB_PATH}")
