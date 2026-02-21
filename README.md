# Adaptive Crypto Trading Agent v2.0

An AI-powered, regime-adaptive crypto trading bot that automatically switches between 6 strategies based on market conditions. Built with Python, CCXT, scikit-learn, and Hidden Markov Models.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your settings (paper mode works without API keys)

# 3. Run in paper trading mode
python agent.py

# Run for a specific number of cycles
python agent.py 10
```

## Architecture

```
agent.py             -> Main adaptive loop (orchestrates everything)
|-- config.py        -> Settings from .env
|-- data_fetcher.py  -> CCXT exchange data + demo fallback
|-- indicators.py    -> 15+ technical indicators
|-- regime_detector.py -> HMM + volatility + trend regime detection
|-- sentiment.py     -> Fear/Greed Index + volume sentiment
|-- strategies.py    -> 6 strategies + adaptive Strategy Engine
|-- model.py         -> Random Forest + Gradient Boosting ensemble ML
|-- risk_manager.py  -> Position sizing, stop-loss, daily limits
|-- executor.py      -> Paper simulator / live order placement
|-- demo_data.py     -> Synthetic data for offline testing
```

## How It Works

1. **Fetches** OHLCV data from your chosen exchange via CCXT
2. **Computes** 15+ technical features (RSI, MACD, Bollinger Bands, ATR, etc.)
3. **Detects market regime** using Hidden Markov Model + volatility + trend analysis
4. **Analyzes sentiment** via Fear & Greed Index + volume-price divergence
5. **Selects strategies** appropriate for the detected regime
6. **Runs Strategy Engine** (weighted ensemble of active strategies)
7. **Cross-validates** with ML model (Random Forest + Gradient Boosting)
8. **Checks risk rules** (position limits, daily loss cap, confidence threshold)
9. **Executes** trades via paper simulator or live exchange
10. **Repeats** every N seconds

## 6 Trading Strategies

| Strategy | Best Regime | Approach |
|----------|------------|----------|
| **Momentum** | Trending Up/Down | Ride trends using MA alignment + MACD crossovers |
| **Mean Reversion** | Ranging/Sideways | Buy oversold dips, sell overbought rallies (Bollinger + RSI) |
| **Breakout** | Ranging -> Trending | Catch volatility expansion after Bollinger squeeze |
| **Grid** | Low-vol Ranging | Profit from oscillation at grid levels around fair value |
| **Scalping** | Any (high volume) | Quick reversals on pin bars + RSI extremes |
| **Sentiment** | Any | Contrarian trades at Fear/Greed extremes |

## Regime Detection

The agent detects 4 market regimes and adapts:

| Regime | Primary Strategy | Secondary |
|--------|-----------------|-----------|
| Trending Up | Momentum (60%) | Sentiment (25%) + Breakout (15%) |
| Trending Down | Momentum (50%) | Sentiment (30%) + Scalping (20%) |
| Ranging | Mean Reversion (50%) | Grid (25%) + Breakout (25%) |
| High Volatility | Breakout (50%) | Scalping (25%) + Sentiment (25%) |

Detection uses 3 methods combined by weighted voting:
- **Hidden Markov Model** (Gaussian HMM on returns + volatility) — 40% weight
- **Trend Analysis** (MA alignment + ADX-like strength) — 40% weight
- **Volatility Clustering** (ATR percentile + BB width) — 20% weight

## Configuration

Edit `.env` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| EXCHANGE_ID | binance | Exchange to connect to |
| TRADING_PAIR | BTC/USDT | Crypto pair to trade |
| TIMEFRAME | 1h | Candlestick timeframe |
| INITIAL_CAPITAL | 1000 | Starting capital in USDT |
| MAX_POSITION_PCT | 0.02 | Max 2% of capital per trade |
| STOP_LOSS_PCT | 0.02 | 2% stop loss (strategies override with dynamic stops) |
| TAKE_PROFIT_PCT | 0.05 | 5% take profit |
| TRADING_MODE | paper | "paper" or "live" |

## Risk Management

- Max 2% capital per trade
- Strategy-specific dynamic stop-losses (ATR-based in trends, tighter in scalps)
- 5% daily loss limit (auto-stops trading)
- Max 3 concurrent positions
- 60% minimum confidence to trade
- ML cross-validation (strategy + ML must agree or confidence is reduced)

## Disclaimer

This is for educational purposes. Crypto trading involves substantial risk.
Always start with paper trading. Never invest more than you can afford to lose.
This is not financial advice.
