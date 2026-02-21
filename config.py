"""
Configuration loader for the crypto trading agent v7.0.
Reads settings from .env file and provides defaults.
Supports multi-pair, trailing stops, transaction costs, websockets,
notifications, trade database, and all intelligence modules.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Use explicit path so .env is found regardless of working directory
load_dotenv(Path(__file__).parent / ".env")


class Config:
    # Exchange settings
    EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")
    API_KEY = os.getenv("API_KEY", "")
    API_SECRET = os.getenv("API_SECRET", "")
    API_PRIVATE_KEY_PATH = os.getenv("API_PRIVATE_KEY_PATH", "")  # Ed25519 private key file

    # Trading settings
    TRADING_PAIR = os.getenv("TRADING_PAIR", "BTC/USDT")
    TRADING_PAIRS = [p.strip() for p in os.getenv("TRADING_PAIRS", "BTC/USDT").split(",") if p.strip()]
    TIMEFRAME = os.getenv("TIMEFRAME", "1h")
    CONFIRMATION_TIMEFRAME = os.getenv("CONFIRMATION_TIMEFRAME", "4h")
    INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000"))

    # Risk management
    MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.02"))
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
    TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
    TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.015"))
    MAX_DAILY_LOSS_PCT = 0.05
    MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "100"))

    # Transaction costs
    FEE_PCT = float(os.getenv("FEE_PCT", "0.001"))
    SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0005"))

    # Agent settings
    TRADING_MODE = os.getenv("TRADING_MODE", "paper")
    LOOKBACK_BARS = 200
    AGENT_INTERVAL_SECONDS = int(os.getenv("AGENT_INTERVAL_SECONDS", "300"))

    # Model settings
    MODEL_RETRAIN_HOURS = float(os.getenv("MODEL_RETRAIN_HOURS", "24"))
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.6"))
    DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
    ML_LABEL_THRESHOLD = float(os.getenv("ML_LABEL_THRESHOLD", "0.005"))

    # State persistence
    STATE_FILE = os.getenv("STATE_FILE", "")

    # Logging
    LOG_FILE = os.getenv("LOG_FILE", "")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # API Server
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    ENABLE_API = os.getenv("ENABLE_API", "true").lower() == "true"
    API_AUTH_KEY = os.getenv("API_AUTH_KEY", "")  # API key for authentication (empty = no auth)

    # Data directory (for Docker volume mounts)
    DATA_DIR = os.getenv("DATA_DIR", ".")

    # Multi-exchange (for arbitrage)
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY", "")
    COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET", "")
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

    # Intelligence toggles
    ENABLE_ONCHAIN = os.getenv("ENABLE_ONCHAIN", "false").lower() == "true"
    ENABLE_WHALE_TRACKING = os.getenv("ENABLE_WHALE_TRACKING", "false").lower() == "true"
    ENABLE_NEWS_NLP = os.getenv("ENABLE_NEWS_NLP", "false").lower() == "true"
    ENABLE_CORRELATION = os.getenv("ENABLE_CORRELATION", "false").lower() == "true"
    ENABLE_ORDERBOOK = os.getenv("ENABLE_ORDERBOOK", "false").lower() == "true"
    ENABLE_FUNDING_OI = os.getenv("ENABLE_FUNDING_OI", "true").lower() == "true"
    ENABLE_LIQUIDATION = os.getenv("ENABLE_LIQUIDATION", "true").lower() == "true"
    INTELLIGENCE_INTERVAL_SECONDS = int(os.getenv("INTELLIGENCE_INTERVAL_SECONDS", "300"))
    CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")

    # Arbitrage
    ARBITRAGE_ENABLED = os.getenv("ARBITRAGE_ENABLED", "false").lower() == "true"
    ARBITRAGE_MIN_SPREAD_PCT = float(os.getenv("ARBITRAGE_MIN_SPREAD_PCT", "0.003"))
    ARBITRAGE_INTERVAL_SECONDS = int(os.getenv("ARBITRAGE_INTERVAL_SECONDS", "60"))
    ARBITRAGE_CAPITAL_PCT = float(os.getenv("ARBITRAGE_CAPITAL_PCT", "0.1"))

    # Monte Carlo
    MC_SIMULATIONS = int(os.getenv("MC_SIMULATIONS", "10000"))
    MC_HORIZON_DAYS = int(os.getenv("MC_HORIZON_DAYS", "252"))

    # WebSocket streaming
    ENABLE_WEBSOCKET = os.getenv("ENABLE_WEBSOCKET", "false").lower() == "true"

    # Notifications
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "")
    TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    TELEGRAM_TRADE_CONFIRMATION = os.getenv("TELEGRAM_TRADE_CONFIRMATION", "false").lower() == "true"
    TELEGRAM_CONFIRMATION_TIMEOUT = int(os.getenv("TELEGRAM_CONFIRMATION_TIMEOUT", "60"))
    TELEGRAM_CONFIRMATION_DEFAULT = os.getenv("TELEGRAM_CONFIRMATION_DEFAULT", "reject")
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
    EMAIL_ALERTS_ENABLED = os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true"
    EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "")
    EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    EMAIL_USER = os.getenv("EMAIL_USER", "")
    EMAIL_PASS = os.getenv("EMAIL_PASS", "")
    EMAIL_TO = os.getenv("EMAIL_TO", "")

    # Trade Database
    ENABLE_TRADE_DB = os.getenv("ENABLE_TRADE_DB", "true").lower() == "true"
    TRADE_DB_PATH = os.getenv("TRADE_DB_PATH", "")

    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "1200"))
    MAX_ORDERS_PER_MINUTE = int(os.getenv("MAX_ORDERS_PER_MINUTE", "10"))

    @classmethod
    def any_intelligence_enabled(cls) -> bool:
        return any([
            cls.ENABLE_ONCHAIN,
            cls.ENABLE_WHALE_TRACKING,
            cls.ENABLE_NEWS_NLP,
            cls.ENABLE_CORRELATION,
            cls.ENABLE_ORDERBOOK,
            cls.ENABLE_FUNDING_OI,
            cls.ENABLE_LIQUIDATION,
        ])

    @classmethod
    def any_notifications_enabled(cls) -> bool:
        return any([
            cls.TELEGRAM_BOT_TOKEN and cls.TELEGRAM_CHAT_ID,
            cls.DISCORD_WEBHOOK_URL,
            cls.EMAIL_ALERTS_ENABLED,
        ])

    @classmethod
    def _resolve_paths(cls):
        if not cls.STATE_FILE:
            cls.STATE_FILE = os.path.join(cls.DATA_DIR, "agent_state.json")
        if not cls.LOG_FILE:
            cls.LOG_FILE = os.path.join(cls.DATA_DIR, "agent.log")
        if not cls.TRADE_DB_PATH:
            cls.TRADE_DB_PATH = os.path.join(cls.DATA_DIR, "trades.db")

    @classmethod
    def is_paper_mode(cls) -> bool:
        return cls.TRADING_MODE.lower() == "paper"

    @classmethod
    def validate(cls):
        cls._resolve_paths()
        if not cls.is_paper_mode() and (not cls.API_KEY or not cls.API_SECRET):
            raise ValueError(
                "API_KEY and API_SECRET are required for live trading. "
                "Set TRADING_MODE=paper for simulation."
            )
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
            print(f"  WebSocket: ENABLED")
        intel = []
        if cls.ENABLE_ONCHAIN: intel.append("OnChain")
        if cls.ENABLE_WHALE_TRACKING: intel.append("Whales")
        if cls.ENABLE_NEWS_NLP: intel.append("News")
        if cls.ENABLE_CORRELATION: intel.append("Correlation")
        if cls.ENABLE_ORDERBOOK: intel.append("OrderBook")
        if intel:
            print(f"  Intel:    {', '.join(intel)}")
        notif = []
        if cls.TELEGRAM_BOT_TOKEN: notif.append("Telegram")
        if cls.DISCORD_WEBHOOK_URL: notif.append("Discord")
        if cls.EMAIL_ALERTS_ENABLED: notif.append("Email")
        if notif:
            print(f"  Alerts:   {', '.join(notif)}")
        if cls.ENABLE_TRADE_DB:
            print(f"  TradeDB:  {cls.TRADE_DB_PATH}")
