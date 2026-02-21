"""
Adaptive Crypto Trading Agent v7.0 — FULLY AUTONOMOUS
======================================================
New in v7.0:
  - Real-time WebSocket streaming (Binance, Kraken, Coinbase, Bybit)
  - Multi-channel notifications (Telegram, Discord, Email)
  - SQLite trade database with analytics
  - Graceful shutdown with signal handling
  - Exchange API rate limiting
  - Backtester-driven strategy evolution fitness
  - Real intelligence providers (on-chain, whale, news, orderbook, correlation)

Carried from v5.0/v6.0:
  - Self-healing with circuit breakers
  - Strategy evolution via genetic algorithm
  - Meta-learning, auto-optimization, decision engine
  - Multi-pair portfolio, arbitrage scanning
  - Trailing stop-loss, transaction cost modeling
  - Multi-timeframe confirmation, walk-forward retraining
  - State persistence, per-strategy attribution, structured logging
"""

import logging
import time
import json
import sys
from datetime import datetime
from config import Config
from data_fetcher import DataFetcher
from indicators import Indicators
from model import TradingModel, Signal
from risk_manager import RiskManager
from executor import PaperExecutor, LiveExecutor
from regime_detector import RegimeDetector, MarketRegime
from sentiment import SentimentAnalyzer
from strategies import StrategyEngine, StrategySignal
from multi_timeframe import MultiTimeframeConfirmer
from drift_detector import DriftDetector
from state_manager import StateManager
from logger import StructuredLogger
from portfolio import PortfolioManager

# v5.0 Autonomous modules
from arbitrage.opportunity_detector import ArbitrageDetector
from arbitrage.execution_engine import ArbitrageExecutor
from self_healer import SelfHealer, ErrorSeverity
from strategy_evolver import StrategyEvolver
from meta_learner import MetaLearner
from auto_optimizer import AutoOptimizer
from decision_engine import DecisionEngine, DecisionState
from intelligence.aggregator import IntelligenceAggregator
from rl_ensemble import RLEnsemble

_log = logging.getLogger(__name__)

# v7.0: Production modules
from websocket_streamer import WebSocketStreamer
from notifier import Notifier
from trade_db import TradeDB
from graceful_shutdown import GracefulShutdown, RateLimiter

# Optional API integration
_data_store = None

def set_data_store(store, agent=None):
    """Allow the API server to inject a DataStore instance."""
    global _data_store
    _data_store = store
    # Wire DecisionEngine reference for kill switch / alerts
    if agent is not None and hasattr(agent, 'decision'):
        store.set_decision_engine(agent.decision)


class TradingAgent:
    def __init__(self, restore_state: bool = True):
        self._print_banner()
        Config.validate()
        print()

        self.fetcher = DataFetcher()
        self.model = TradingModel()
        self.risk = RiskManager()
        self.regime_detector = RegimeDetector()
        self.sentiment = SentimentAnalyzer()
        self.strategy_engine = StrategyEngine()
        self.mtf = MultiTimeframeConfirmer()
        self.drift = DriftDetector()
        self.state_mgr = StateManager()
        self.log = StructuredLogger()

        # v5.0: Autonomous subsystems
        self.decision = DecisionEngine(initial_capital=Config.INITIAL_CAPITAL)
        self._register_recovery_actions()

        if Config.is_paper_mode():
            self.executor = PaperExecutor()
        else:
            self.executor = LiveExecutor(self.fetcher.exchange)

        self.cycle_count = 0
        self.last_train_time = None
        self._last_data_hash = None
        self._last_regime = None
        self._last_price = None

        # v6.0: Multi-pair portfolio manager
        self.portfolio = PortfolioManager()
        self.multi_pair = len(Config.TRADING_PAIRS) > 1

        # v6.0: Intelligence aggregator (external signal sources)
        if Config.any_intelligence_enabled():
            self.intelligence = IntelligenceAggregator(exchange=self.fetcher.exchange)
        else:
            self.intelligence = None
        self._last_intelligence = None
        self._last_intelligence_time = None

        # v6.0: Arbitrage scanner + executor
        if Config.ARBITRAGE_ENABLED:
            self.arb_detector = ArbitrageDetector()
            self.arb_executor = ArbitrageExecutor()
        else:
            self.arb_detector = None
            self.arb_executor = None
        self._last_arb_scan = None
        self._last_arb_time = None

        # v8.0: RL Ensemble (lightweight Q-learning agents)
        self.rl_ensemble = RLEnsemble()

        # Track signal sources for meta-learner
        self._last_strategy_signal = None
        self._last_ml_signal = None
        self._last_strategy_conf = 0.0
        self._last_ml_conf = 0.0

        # Restore state if available
        if restore_state and self.state_mgr.exists():
            self.state_mgr.load(self)

        # Initialize strategy evolution
        if not self.decision.evolver._initialized:
            self.decision.evolver.initialize_population()

        # v7.0: Production modules
        self.notifier = Notifier()
        self.telegram_bot = None  # Set by api/server.py for interactive Telegram commands
        self.trade_db = TradeDB() if Config.ENABLE_TRADE_DB else None
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=Config.MAX_REQUESTS_PER_MINUTE,
            max_orders_per_minute=Config.MAX_ORDERS_PER_MINUTE,
        )

        # v7.0: WebSocket streamer (optional real-time data)
        self.ws_streamer = None
        if Config.ENABLE_WEBSOCKET:
            try:
                self.ws_streamer = WebSocketStreamer(
                    exchange_id=Config.EXCHANGE_ID,
                    trading_pair=Config.TRADING_PAIR,
                    timeframe=Config.TIMEFRAME,
                )
            except Exception as e:
                logging.getLogger(__name__).warning("WebSocket init failed: %s", e)

        # Graceful shutdown (must be after all modules are initialized)
        self.shutdown_handler = GracefulShutdown()
        self._register_shutdown_callbacks()

    def _print_banner(self):
        print("=" * 60)
        print("  ADAPTIVE CRYPTO TRADING AGENT v7.0 (FULLY AUTONOMOUS)")
        print("  Self-Healing | Evolving | Meta-Learning | Auto-Optimizing")
        print("  WebSocket | Notifications | TradeDB | Graceful Shutdown")
        print("=" * 60)

    def _register_shutdown_callbacks(self):
        """Register cleanup callbacks for graceful shutdown."""
        # Stop WebSocket streamer
        if self.ws_streamer:
            self.shutdown_handler.register_callback(
                "websocket_streamer",
                lambda: self.ws_streamer.stop(),
            )

        # Save state
        self.shutdown_handler.register_callback(
            "state_manager",
            lambda: self.state_mgr.save(self),
        )

        # Close trade DB
        if self.trade_db:
            self.shutdown_handler.register_callback(
                "trade_db",
                lambda: self.trade_db.close(),
            )

        # Send shutdown notification
        self.shutdown_handler.register_callback(
            "notifier",
            lambda: self.notifier.notify_state_change(
                "normal", "halted",
                f"Agent shutting down gracefully after {self.cycle_count} cycles",
            ),
        )

    def _register_recovery_actions(self):
        """Register auto-recovery actions for each component."""
        healer = self.decision.healer

        # Data fetcher recovery: reset exchange connection
        def recover_data():
            self.fetcher.exchange = self.fetcher._init_exchange()
            self.fetcher.using_demo = False
        healer.register_recovery_action("data_fetcher", recover_data)

        # Model recovery: force retrain
        def recover_model():
            df = self.fetcher.fetch_ohlcv(limit=Config.LOOKBACK_BARS)
            if not df.empty:
                df_ind = Indicators.add_all(df)
                self.model.train(df=None, df_ind=df_ind)
        healer.register_recovery_action("model", recover_model)

        # Executor recovery: reinitialize
        def recover_executor():
            if Config.is_paper_mode():
                self.executor = PaperExecutor()
            else:
                self.executor = LiveExecutor(self.fetcher.exchange)
        healer.register_recovery_action("executor", recover_executor)

        # Intelligence recovery: reinitialize aggregator
        def recover_intelligence():
            if self.intelligence:
                self.intelligence = IntelligenceAggregator(exchange=self.fetcher.exchange)
        healer.register_recovery_action("intelligence", recover_intelligence)

        # Arbitrage recovery: reinitialize detector + executor
        def recover_arbitrage():
            if self.arb_detector:
                self.arb_detector = ArbitrageDetector()
                self.arb_executor = ArbitrageExecutor()
        healer.register_recovery_action("arbitrage", recover_arbitrage)

    def train_model(self, df_ind=None):
        """Train ML model with drift baseline."""
        print("\n[Agent] Training ML model...")
        if df_ind is None:
            df = self.fetcher.fetch_ohlcv(limit=Config.LOOKBACK_BARS)
            if df.empty:
                return False
            df_ind = Indicators.add_all(df)

        try:
            metrics = self.model.train(df=None, df_ind=df_ind)
            if "error" not in metrics:
                self.last_train_time = datetime.now()
                acc = metrics.get("cv_accuracy", 0.5)
                self.drift.set_baseline(acc, 0.7)
                self.drift.reset()
                self.log.log_model_train(acc, metrics.get("samples", 0))
                self.decision.healer.record_model_train()
                self.decision.healer.record_success("model")
                return True
            return False
        except Exception as e:
            self.decision.healer.record_error("model", e, ErrorSeverity.HIGH)
            return False

    def _should_retrain(self, data_hash) -> bool:
        if self.last_train_time is None:
            return True
        if data_hash == self._last_data_hash:
            return False

        # Use meta-learner's adaptive retrain frequency
        retrain_hours = self.decision.meta.config.retrain_hours
        hours = (datetime.now() - self.last_train_time).total_seconds() / 3600

        drift_result = self.drift.check_drift()
        if drift_result["drift_detected"]:
            self.log.log_model_train(
                drift_result["current_accuracy"], 0, drift_detected=True
            )
            self.decision.record_drift_event()
            return True

        return hours >= retrain_hours

    def _fetch_intelligence(self) -> dict | None:
        """Fetch intelligence signals if enabled, with rate limiting."""
        if self.intelligence is None:
            return None

        now = datetime.now()
        if (self._last_intelligence_time and
            (now - self._last_intelligence_time).total_seconds() < Config.INTELLIGENCE_INTERVAL_SECONDS):
            return self._last_intelligence

        try:
            result = self.intelligence.get_signals()
            self._last_intelligence = result
            self._last_intelligence_time = now
            self.decision.healer.record_success("intelligence")
            return result
        except Exception as e:
            self.decision.healer.record_error("intelligence", e, ErrorSeverity.LOW)
            return self._last_intelligence

    def _scan_arbitrage(self) -> dict | None:
        """Scan for arbitrage opportunities if enabled, with rate limiting."""
        if self.arb_detector is None:
            return None

        now = datetime.now()
        if (self._last_arb_time and
            (now - self._last_arb_time).total_seconds() < Config.ARBITRAGE_INTERVAL_SECONDS):
            return self._last_arb_scan

        try:
            opportunities = self.arb_detector.scan()

            # Allocate a fraction of current capital to arbitrage
            capital_for_arb = self.risk.current_capital * Config.ARBITRAGE_CAPITAL_PCT
            self.arb_executor.set_capital(capital_for_arb)

            # Execute any profitable opportunities found
            for opp in opportunities:
                self.arb_executor.execute(opp.to_dict())

            result = self.arb_detector.get_status()
            result["execution"] = self.arb_executor.get_summary()
            self._last_arb_scan = result
            self._last_arb_time = now
            self.decision.healer.record_success("arbitrage")
            return result
        except Exception as e:
            self.decision.healer.record_error("arbitrage", e, ErrorSeverity.LOW)
            return self._last_arb_scan

    def run_cycle(self):
        self.cycle_count += 1
        self.risk.set_bar(self.cycle_count)
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'_'*60}")
        print(f"  Cycle #{self.cycle_count} @ {now}")
        print(f"{'_'*60}")

        # v5.0: Orchestrate autonomous systems FIRST
        current_capital = self.risk.capital
        current_pnl = self.risk.total_pnl
        instructions = self.decision.orchestrate(
            self.cycle_count, current_capital, current_pnl
        )

        # v5.0: Apply evolved strategy params if available
        evolved = instructions.get("evolved_params")
        if evolved:
            self.strategy_engine.apply_evolved_params(evolved)
            print(f"  [EVOLUTION] Hot-reloaded params for: {list(evolved.keys())}")

        # v5.0: Apply optimized hyperparams to Config
        opt_cfg = instructions.get("optimized_config")
        if opt_cfg:
            for key, val in opt_cfg.items():
                if val is not None and hasattr(Config, key):
                    setattr(Config, key, val)
            print(f"  [OPTIMIZER] Applied best config: SL={Config.STOP_LOSS_PCT:.2%} "
                  f"TP={Config.TAKE_PROFIT_PCT:.2%} Conf={Config.MIN_CONFIDENCE:.2f}")

        # Production safeguard: force close all positions
        if instructions.get("force_close_all"):
            if self.risk.positions:
                print(f"  [FORCE CLOSE] Closing {len(self.risk.positions)} positions")
                current = self._last_price or 0
                for pos in list(self.risk.positions):
                    self.risk.close_position(pos, current, "force_close")
            else:
                print(f"  [FORCE CLOSE] No positions to close")

        if not instructions["should_trade"]:
            print(f"  [AUTONOMOUS] Trading suspended: {instructions['skip_reason']}")
            self.decision.print_autonomous_summary()
            return

        # Shared intelligence + arbitrage (fetched once per cycle, not per pair)
        intel_result = self._fetch_intelligence()
        intel_adjustment = 1.0
        intel_bias = "neutral"
        if intel_result:
            intel_adjustment = intel_result["adjustment_factor"]
            intel_bias = intel_result["bias"]

        arb_result = self._scan_arbitrage()

        # v5.0: Position multiplier from decision engine
        position_mult = instructions.get("position_multiplier", 1.0)

        # === Multi-pair loop ===
        pairs = Config.TRADING_PAIRS if self.multi_pair else [Config.TRADING_PAIR]
        for pair in pairs:
            self._run_pair_cycle(
                pair, position_mult,
                intel_result=intel_result,
                intel_adjustment=intel_adjustment,
                intel_bias=intel_bias,
                arb_result=arb_result,
            )

        # Rebalance portfolio weights every 20 cycles
        if self.multi_pair and self.cycle_count % 20 == 0:
            self.portfolio.compute_correlations()
            self.portfolio.rebalance_weights()
            print(f"  [PORTFOLIO] Rebalanced weights: "
                  + " | ".join(f"{p}={w:.1%}" for p, w in self.portfolio.weights.items()))

        # Portfolio-level risk summary
        if self.multi_pair:
            port_risk = self.portfolio.get_portfolio_risk(self.risk.positions)
            print(f"  [PORTFOLIO] Exposure: ${port_risk['total_exposure']:,.2f} | "
                  f"Concentration: {port_risk['concentration']:.2f} | "
                  f"Corr risk: {port_risk['corr_risk']}")

        # Autonomous summary
        self.decision.print_autonomous_summary()

        # v7.0: Record equity snapshot to trade DB
        if self.trade_db and self.cycle_count % 5 == 0:
            try:
                price = self._last_price or 0
                unrealized = sum(p.unrealized_pnl(price) for p in self.risk.positions)
                equity = self.risk.capital + unrealized
                self.trade_db.record_equity(
                    equity=equity, capital=self.risk.capital,
                    unrealized_pnl=unrealized,
                    open_positions=len(self.risk.positions),
                    cycle=self.cycle_count,
                )
            except Exception as e:
                _log.warning("Equity history update failed: %s", e, exc_info=True)

        # Auto-save state every 10 cycles
        if self.cycle_count % 10 == 0:
            self.state_mgr.save(self)

        # Push to API data store
        self._push_to_data_store()

        # v7.0: Check if shutdown was requested
        if self.shutdown_handler.shutdown_requested:
            raise KeyboardInterrupt("Graceful shutdown requested")

    def _run_pair_cycle(self, pair: str, position_mult: float,
                        intel_result=None, intel_adjustment: float = 1.0,
                        intel_bias: str = "neutral", arb_result=None):
        """Run the full analysis + execution pipeline for a single trading pair."""
        if self.multi_pair:
            print(f"\n  --- {pair} ---")

        # 1. Fetch data (with self-healing)
        try:
            df = self.fetcher.fetch_ohlcv(symbol=pair)
            if df.empty:
                self.decision.healer.record_data_fetch(False)
                self.log.log_error("data_fetcher", f"No data for {pair}")
                return
            self.decision.healer.record_data_fetch(True)
        except Exception as e:
            self.decision.healer.record_error("data_fetcher", e, ErrorSeverity.HIGH)
            return

        current_price = df["close"].iloc[-1]
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]
        self._last_price = current_price
        # v7.0: Track per-pair prices for heartbeat
        if not hasattr(self, "_last_prices"):
            self._last_prices = {}
        self._last_prices[pair] = current_price

        # Update portfolio price series for correlation tracking
        if self.multi_pair:
            self.portfolio.update_prices(pair, df["close"])

        self.log.log_cycle_start(self.cycle_count, current_price, pair)

        # 2. Compute indicators ONCE
        try:
            df_ind = Indicators.add_all(df)
        except Exception as e:
            self.decision.healer.record_error("indicators", e, ErrorSeverity.MEDIUM)
            return

        # 3. Retrain if needed (only on primary pair to avoid excess retraining)
        if pair == Config.TRADING_PAIR:
            data_hash = (len(df), float(current_price))
            if self._should_retrain(data_hash):
                self.train_model(df_ind=df_ind)
                self._last_data_hash = data_hash

        # 4. Check positions for THIS pair
        closed = self.risk.check_positions(current_price, current_high, current_low)
        pair_closed = [t for t in closed if t.symbol == pair]
        for trade in pair_closed:
            self.log.log_trade_close(
                trade.symbol, trade.side, trade.entry_price, trade.exit_price,
                trade.pnl_net, trade.pnl_gross, trade.fees_paid,
                trade.exit_reason, trade.hold_bars, trade.strategy_name,
            )
            ret = (trade.exit_price - trade.entry_price) / trade.entry_price
            if trade.side == "short":
                ret = -ret
            self.drift.record_outcome(ret)

            # v8.0: Feed reward to RL ensemble
            try:
                rl_reward = trade.pnl_net / max(1, Config.INITIAL_CAPITAL * 0.01)
                self.rl_ensemble.update_reward(
                    reward=rl_reward, next_df_ind=df_ind,
                    regime=self._last_regime.value if self._last_regime else "",
                )
            except Exception as e:
                _log.debug("RL reward update failed: %s", e)

            # v5.0: Feed trade result to meta-learner
            self.decision.record_trade_result(
                pnl=trade.pnl_net,
                strategy_signal=self._last_strategy_signal or "HOLD",
                ml_signal=self._last_ml_signal or "HOLD",
                final_signal="BUY" if trade.side == "long" else "SELL",
                strategy_confidence=self._last_strategy_conf,
                ml_confidence=self._last_ml_conf,
                regime=self._last_regime.value if self._last_regime else "unknown",
            )

            # v7.0: Record to trade DB
            if self.trade_db:
                try:
                    self.trade_db.record_trade_close(
                        symbol=trade.symbol, side=trade.side,
                        exit_price=trade.exit_price,
                        pnl_gross=trade.pnl_gross,
                        pnl_net=trade.pnl_net, fees=trade.fees_paid,
                        reason=trade.exit_reason, hold_bars=trade.hold_bars,
                    )
                except Exception as e:
                    _log.warning("Trade DB close record failed: %s", e, exc_info=True)

            self.notifier.notify_trade_close(
                symbol=trade.symbol, side=trade.side,
                entry=trade.entry_price, exit_price=trade.exit_price,
                pnl=trade.pnl_net, reason=trade.exit_reason,
                strategy=trade.strategy_name or "Unknown",
                hold_bars=trade.hold_bars,
            )

            # v7.0: Alert on large losses
            if trade.pnl_net < -(Config.INITIAL_CAPITAL * 0.01):
                capital_pct = abs(trade.pnl_net) / Config.INITIAL_CAPITAL * 100
                self.notifier.notify_large_loss(
                    pnl=trade.pnl_net, capital_pct=capital_pct,
                )

        # 5. Early exit if at max positions
        summary = self.risk.get_summary()
        if summary["open_positions"] >= Config.MAX_OPEN_POSITIONS and not pair_closed:
            print(f"  Max positions ({Config.MAX_OPEN_POSITIONS}) - monitoring only")
            self._print_portfolio(summary, current_price)
            return

        # 6. Full analysis pipeline
        try:
            regime_state = self.regime_detector.detect(df, df_ind=df_ind)
        except Exception as e:
            self.decision.healer.record_error("regime_detector", e, ErrorSeverity.MEDIUM)
            regime_state = type('Regime', (), {
                'regime': MarketRegime.RANGING, 'confidence': 0.5,
                'volatility': 0.02, 'regime_duration': 0
            })()

        if self._last_regime and self._last_regime != regime_state.regime:
            self.log.log_regime_change(
                self._last_regime.value, regime_state.regime.value,
                regime_state.confidence,
            )
        self._last_regime = regime_state.regime

        try:
            sentiment_state = self.sentiment.analyze(df, df_ind=df_ind)
        except Exception as e:
            self.decision.healer.record_error("sentiment", e, ErrorSeverity.LOW)
            sentiment_state = self.sentiment._default_state()

        try:
            strategy_signal = self.strategy_engine.run(df_ind, regime_state.regime, sentiment_state)
        except Exception as e:
            self.decision.healer.record_error("strategy_engine", e, ErrorSeverity.MEDIUM)
            strategy_signal = StrategySignal(
                signal="HOLD", confidence=0.0, strategy_name="Error",
                reason="strategy error", suggested_sl_pct=0.02, suggested_tp_pct=0.03,
            )

        try:
            ml_signal, ml_confidence = self.model.predict(df_ind=df_ind)
        except Exception as e:
            self.decision.healer.record_error("model", e, ErrorSeverity.MEDIUM)
            ml_signal, ml_confidence = "HOLD", 0.0

        # Track signal sources for meta-learner
        self._last_strategy_signal = strategy_signal.signal
        self._last_ml_signal = ml_signal
        self._last_strategy_conf = strategy_signal.confidence
        self._last_ml_conf = ml_confidence

        self.drift.record_prediction(ml_signal, ml_confidence)

        # v8.0: RL ensemble vote
        try:
            rl_signal, rl_confidence = self.rl_ensemble.predict(
                df_ind, regime=regime_state.regime.value
            )
        except Exception as e:
            _log.debug("RL ensemble error: %s", e)
            rl_signal, rl_confidence = "HOLD", 0.0

        # 7. Combine signals (v5.0: learned weights, v6.0: + intelligence, v8.0: + RL)
        final_signal, final_confidence = self._combine_signals(
            strategy_signal, ml_signal, ml_confidence,
            regime=regime_state.regime.value,
            intel_adjustment=intel_adjustment,
            intel_bias=intel_bias,
            rl_signal=rl_signal,
            rl_confidence=rl_confidence,
        )

        # 8. Multi-timeframe confirmation
        htf_bias = self.mtf.get_htf_bias(df, Config.CONFIRMATION_TIMEFRAME)
        pre_mtf_signal = final_signal
        pre_mtf_conf = final_confidence
        final_signal, final_confidence = self.mtf.confirm_signal(
            final_signal, final_confidence, htf_bias,
        )

        # v5.0: Decision engine override (safety governance)
        final_signal, final_confidence = self.decision.should_override_signal(
            final_signal, final_confidence
        )

        self.log.log_signal(final_signal, final_confidence, "final",
                           regime_state.regime.value)

        # Print status
        self._print_status(
            current_price, regime_state, sentiment_state,
            strategy_signal, ml_signal, ml_confidence,
            final_signal, final_confidence, htf_bias,
            pre_mtf_signal, pre_mtf_conf,
            intel_result=intel_result,
            arb_result=arb_result,
            pair=pair,
        )

        # 9. Execute if actionable (with portfolio-aware sizing)
        if final_signal in ("BUY", "SELL"):
            self._execute_trade(
                final_signal, final_confidence, current_price,
                df_ind, strategy_signal, position_mult=position_mult,
                intel_adjustment=intel_adjustment,
                pair=pair,
            )

        # 10. Portfolio summary
        self._print_portfolio(self.risk.get_summary(), current_price)

    def _combine_signals(self, strat_sig, ml_signal, ml_conf, regime: str = None,
                         intel_adjustment: float = 1.0, intel_bias: str = "neutral",
                         rl_signal: str = "HOLD", rl_confidence: float = 0.0):
        """Combine strategy + ML + RL signals using learned weights + intelligence.

        v8.0: Added RL ensemble vote with 15% weight allocation.
        v7.0: Rebalanced confidence math so the bot actually trades.
        """
        strat_signal = strat_sig.signal
        strat_conf = strat_sig.confidence

        # v5.0: Get adaptive weights from meta-learner
        sw, mw = self.decision.meta.get_signal_weights(regime)
        bonus = self.decision.meta.config.agreement_bonus

        if strat_signal == ml_signal:
            # Full agreement — strongest signal
            base_signal = strat_signal
            base_conf = min(strat_conf * sw + ml_conf * mw + bonus, 0.95)
        elif strat_signal == "HOLD" or ml_signal == "HOLD":
            # One source neutral — follow the active signal with moderate dampening
            active = strat_signal if strat_signal != "HOLD" else ml_signal
            active_conf = strat_conf if strat_signal != "HOLD" else ml_conf
            base_signal = active
            # v7.0: was 0.6, now 0.8 — HOLD from one source shouldn't kill the signal
            base_conf = active_conf * 0.8
        else:
            # Active disagreement — go with higher-weighted source, penalize
            if sw >= mw:
                base_signal = strat_signal
                base_conf = strat_conf * self.decision.meta.config.disagreement_penalty
            else:
                base_signal = ml_signal
                base_conf = ml_conf * self.decision.meta.config.disagreement_penalty

        # v6.0: Apply intelligence adjustment
        if intel_bias != "neutral" and base_signal != "HOLD":
            signal_is_bullish = (base_signal == "BUY")
            intel_is_bullish = (intel_bias == "bullish")

            if signal_is_bullish == intel_is_bullish:
                # Intelligence confirms signal direction — boost confidence
                conf_boost = abs(intel_adjustment - 1.0) * 0.3
                base_conf = min(base_conf + conf_boost, 0.95)
            else:
                # Intelligence contradicts signal direction — dampen confidence
                conf_penalty = abs(intel_adjustment - 1.0) * 0.4
                base_conf = max(base_conf - conf_penalty, 0.0)

        # v8.0: RL ensemble vote — 15% weight in final decision
        if rl_signal != "HOLD" and rl_confidence > 0.4 and base_signal != "HOLD":
            if rl_signal == base_signal:
                # RL confirms — boost by up to 0.08
                rl_boost = rl_confidence * 0.15 * 0.5
                base_conf = min(base_conf + rl_boost, 0.95)
            else:
                # RL disagrees — slight penalty
                rl_penalty = rl_confidence * 0.15 * 0.3
                base_conf = max(base_conf - rl_penalty, 0.0)

        return base_signal, base_conf

    def _execute_trade(self, signal, confidence, price, df_ind, strat_sig,
                       position_mult: float = 1.0, intel_adjustment: float = 1.0,
                       pair: str = None):
        pair = pair or Config.TRADING_PAIR
        allowed, reason = self.risk.can_open_position(signal, confidence, symbol=pair)
        if not allowed:
            print(f"  [!] Blocked: {reason}")
            return

        # v6.0: Portfolio-aware position sizing
        if self.multi_pair:
            existing_pairs = [p.symbol for p in self.risk.positions]
            allocation = self.portfolio.get_allocation(pair, existing_pairs)
            if not allocation.is_tradeable:
                print(f"  [!] Portfolio blocked: too correlated with open positions "
                      f"(penalty: {allocation.correlation_penalty:.2f})")
                return
            portfolio_mult = allocation.weight / (1.0 / len(Config.TRADING_PAIRS))
        else:
            portfolio_mult = 1.0

        side = "long" if signal == "BUY" else "short"
        # v8.0: Kelly-based dynamic position sizing
        quantity = self.risk.calculate_position_size(
            price, confidence=confidence,
            strategy_name=strat_sig.strategy_name,
            regime=self._last_regime.value if self._last_regime else "",
        )

        # v5.0: Apply autonomous position multiplier
        quantity = round(quantity * position_mult, 6)

        # v6.0: Apply portfolio allocation scaling
        quantity = round(quantity * portfolio_mult, 6)

        # v6.0: Apply intelligence-based position scaling
        intel_size_factor = max(0.5, min(1.5, intel_adjustment))
        if signal == "SELL":
            intel_size_factor = 2.0 - intel_size_factor
        quantity = round(quantity * intel_size_factor, 6)

        # v8.0: Regime-adaptive stop-loss and take-profit
        regime_val = self._last_regime.value if self._last_regime else ""
        sl_pct = strat_sig.suggested_sl_pct
        tp_pct = strat_sig.suggested_tp_pct
        atr_val = float(df_ind["atr"].iloc[-1]) if "atr" in df_ind.columns else None

        stop_loss, take_profit = self.risk.calculate_stop_take(
            price, side,
            sl_pct=sl_pct, tp_pct=tp_pct,
            atr=atr_val, regime=regime_val,
        )

        # v7.0: Rate limit check before placing order
        self.rate_limiter.wait_if_needed(is_order=True)

        # v7.1: Telegram trade confirmation (blocks until user approves/rejects/timeout)
        if (self.telegram_bot and Config.TELEGRAM_TRADE_CONFIRMATION
                and self.telegram_bot.enabled):
            conf = self.telegram_bot.request_confirmation(
                signal=signal, pair=pair, side=side,
                price=price, quantity=quantity,
                strategy=strat_sig.strategy_name, confidence=confidence,
            )
            timeout = Config.TELEGRAM_CONFIRMATION_TIMEOUT
            conf.event.wait(timeout=timeout)
            if conf.decision is None:
                # Timeout — apply default
                conf.decision = Config.TELEGRAM_CONFIRMATION_DEFAULT
                print(f"  [TG] Confirmation timeout -> {conf.decision}")
            if conf.decision == "rejected":
                print(f"  [TG] Trade REJECTED by user")
                return
            print(f"  [TG] Trade APPROVED")

        order = self.executor.place_order(pair, side, quantity, price)
        if "error" not in order:
            self.rate_limiter.record_order()
            pos = self.risk.open_position(
                pair, side, price, quantity, stop_loss, take_profit,
                strategy_name=strat_sig.strategy_name,
            )
            self.log.log_trade_open(
                pair, side, price, quantity,
                stop_loss, take_profit, pos.trailing_stop,
                strat_sig.strategy_name,
            )
            self.portfolio.active_pairs.add(pair)

            # v7.0: Record to trade DB
            if self.trade_db:
                try:
                    self.trade_db.record_trade_open(
                        symbol=pair, side=side, entry_price=price,
                        quantity=quantity, strategy=strat_sig.strategy_name,
                        regime=self._last_regime.value if self._last_regime else "unknown",
                        confidence=confidence, sl=stop_loss,
                        tp=take_profit, trailing=pos.trailing_stop,
                    )
                except Exception as e:
                    _log.warning("Trade DB open record failed: %s", e, exc_info=True)

            self.notifier.notify_trade_open(
                symbol=pair, side=side, price=price,
                quantity=quantity, strategy=strat_sig.strategy_name,
            )

    def _print_status(self, price, regime_state, sentiment_state,
                      strat_sig, ml_signal, ml_conf, final_signal, final_conf,
                      htf_bias, pre_mtf_signal, pre_mtf_conf,
                      intel_result=None, arb_result=None, pair: str = None):
        r = regime_state
        s = sentiment_state
        regime_icons = {
            MarketRegime.TRENDING_UP: "[UP]", MarketRegime.TRENDING_DOWN: "[DN]",
            MarketRegime.RANGING: "[==]", MarketRegime.HIGH_VOLATILITY: "[!!]",
        }
        pair_label = pair or Config.TRADING_PAIR
        print(f"\n  {pair_label} = ${price:,.2f}")
        print(f"  {regime_icons.get(r.regime, '?')} Regime: {r.regime.value} (conf:{r.confidence:.0%} vol:{r.volatility:.2%} dur:{r.regime_duration})")
        print(f"  Sentiment: F&G={s.fear_greed_index} ({s.fear_greed_label.value}) composite={s.composite_score:+.2f}")
        print(f"  Strategy: {strat_sig.strategy_name} -> {strat_sig.signal} ({strat_sig.confidence:.0%}) | {strat_sig.reason}")
        print(f"  ML Model: {ml_signal} ({ml_conf:.0%})")

        # v5.0: Show learned weights
        sw, mw = self.decision.meta.get_signal_weights()
        print(f"  Weights: Strategy={sw:.0%} ML={mw:.0%} (learned)")

        # v6.0: Intelligence signals
        if intel_result:
            bias_icon = {"bullish": "[+]", "bearish": "[-]", "neutral": "[=]"}.get(intel_result["bias"], "?")
            enabled_count = sum(1 for s in intel_result["signals"] if s["strength"] > 0)
            print(f"  Intelligence: {bias_icon} {intel_result['bias']} "
                  f"(adj:{intel_result['adjustment_factor']:.3f} "
                  f"net:{intel_result['net_score']:+.3f} "
                  f"sources:{enabled_count}/5)")

        if arb_result:
            n_opps = arb_result.get("active_opportunities", 0)
            exec_summary = arb_result.get("execution", {})
            pnl = exec_summary.get("total_pnl", 0)
            print(f"  Arbitrage: {n_opps} opportunities | PnL: ${pnl:.4f}")

        # v8.0: RL ensemble vote
        if hasattr(self, 'rl_ensemble'):
            rl_stats = self.rl_ensemble.get_stats()
            # Filter out non-dict entries (e.g. "engine": "deep_rl")
            agent_stats = [s for s in rl_stats.values() if isinstance(s, dict)]
            total_rl_trades = sum(s.get("total_trades", 0) for s in agent_stats)
            if total_rl_trades > 0 and agent_stats:
                avg_sharpe = sum(s.get("sharpe", 0) for s in agent_stats) / len(agent_stats)
                print(f"  RL Ensemble: trades={total_rl_trades} avg_sharpe={avg_sharpe:.3f}")

        htf_label = f"{htf_bias['bias']} ({htf_bias['strength']:.0%})"
        mtf_changed = pre_mtf_signal != final_signal or abs(pre_mtf_conf - final_conf) > 0.05
        mtf_mark = " [MTF adjusted]" if mtf_changed else ""
        print(f"  HTF ({Config.CONFIRMATION_TIMEFRAME}): {htf_label}{mtf_mark}")

        sig_icon = {"BUY": "[BUY]", "SELL": "[SELL]", "HOLD": "[---]"}.get(final_signal, "?")
        print(f"  FINAL: {sig_icon} {final_signal} ({final_conf:.0%})")

        drift = self.drift.check_drift()
        if drift["drift_detected"]:
            print(f"  [!] DRIFT: {drift['reason']}")

    def _print_portfolio(self, summary, current_price):
        fees_str = f" | Fees: ${summary.get('total_fees', 0):,.2f}" if summary.get('total_fees', 0) > 0 else ""
        print(
            f"\n  Capital: ${summary['capital']:,.2f} | "
            f"Open: {summary['open_positions']} | "
            f"Trades: {summary['total_trades']} | "
            f"PnL: ${summary['total_pnl']:,.2f} | "
            f"Win: {summary['win_rate']:.0%}{fees_str}"
        )

    def _push_to_data_store(self):
        """Push cycle snapshot to DataStore if available."""
        if _data_store is None:
            return
        try:
            summary = self.risk.get_summary()
            price = self._last_price or 0.0
            regime = self._last_regime.value if self._last_regime else "unknown"

            _data_store.update_snapshot({
                "cycle": self.cycle_count,
                "price": price,
                "pair": Config.EXCHANGE_ID,
                "trading_pair": Config.TRADING_PAIR,
                "trading_pairs": Config.TRADING_PAIRS,
                "multi_pair": self.multi_pair,
                "capital": summary["capital"],
                "total_pnl": summary["total_pnl"],
                "daily_pnl": summary["daily_pnl"],
                "total_fees": summary["total_fees"],
                "win_rate": summary["win_rate"],
                "total_trades": summary["total_trades"],
                "open_positions": summary["open_positions"],
                "regime": regime,
                "portfolio": self.portfolio.get_portfolio_risk(self.risk.positions) if self.multi_pair else None,
                "portfolio_weights": self.portfolio.weights if self.multi_pair else None,
                "positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "entry_price": p.entry_price,
                        "quantity": p.quantity,
                        "unrealized_pnl": p.unrealized_pnl(price),
                        "stop_loss": p.stop_loss,
                        "take_profit": p.take_profit,
                        "trailing_stop": p.trailing_stop,
                        "strategy": p.strategy_name,
                    }
                    for p in self.risk.positions
                ],
                "autonomous": self.decision.get_autonomous_status(),
                "intelligence": {
                    "bias": self._last_intelligence["bias"],
                    "adjustment_factor": self._last_intelligence["adjustment_factor"],
                    "net_score": self._last_intelligence["net_score"],
                } if self._last_intelligence else None,
                "arbitrage": {
                    "active_opportunities": self._last_arb_scan.get("active_opportunities", 0),
                    "scan_count": self._last_arb_scan.get("scan_count", 0),
                    "execution": self._last_arb_scan.get("execution", {}),
                } if self._last_arb_scan else None,
            })

            # Equity point
            equity_value = summary["capital"] + sum(
                p.unrealized_pnl(price) for p in self.risk.positions
            )
            _data_store.append_equity(
                round(equity_value, 2),
                datetime.now().isoformat(),
            )

            # Push new closed trades
            trade_count = len(self.risk.trade_history)
            existing_count = len(_data_store.get_trade_log())
            if trade_count > existing_count:
                for t in self.risk.trade_history[existing_count:]:
                    _data_store.append_trade({
                        "symbol": t.symbol,
                        "side": t.side,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "quantity": t.quantity,
                        "pnl_gross": round(t.pnl_gross, 4),
                        "pnl_net": round(t.pnl_net, 4),
                        "fees_paid": round(t.fees_paid, 4),
                        "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat(),
                        "exit_reason": t.exit_reason,
                        "strategy": t.strategy_name,
                        "hold_bars": t.hold_bars,
                    })

            # Push recent autonomous events
            for event in list(self.decision.event_log)[-5:]:
                _data_store.append_event(event.to_dict())

            # Push intelligence signals
            if self._last_intelligence:
                _data_store.update_intelligence(self._last_intelligence)

            # Push arbitrage data
            if self._last_arb_scan:
                _data_store.update_arbitrage(self._last_arb_scan)

        except Exception as e:
            logging.getLogger(__name__).debug("DataStore push failed: %s", e)

    def preflight_check(self) -> bool:
        """v7.0: Quick self-test before committing to an overnight run.
        Fetches one candle from each trading pair to verify exchange connectivity."""
        pairs = Config.TRADING_PAIRS if self.multi_pair else [Config.TRADING_PAIR]
        print("\n" + "=" * 50)
        print("  PREFLIGHT CHECK")
        print("=" * 50)
        all_ok = True
        for pair in pairs:
            try:
                df = self.fetcher.fetch_ohlcv(symbol=pair, limit=5)
                if df is not None and not df.empty:
                    price = df["close"].iloc[-1]
                    print(f"  [OK] {pair:12s} = ${price:>12,.2f}  ({len(df)} candles)")
                else:
                    print(f"  [!!] {pair:12s} = NO DATA (will use demo fallback)")
                    all_ok = False
            except Exception as e:
                print(f"  [!!] {pair:12s} = ERROR: {e}")
                all_ok = False

        # Quick intelligence check
        if Config.any_intelligence_enabled():
            try:
                intel = self.intelligence.get_signals()
                active = sum(1 for s in intel.get("signals", []) if s.get("strength", 0) > 0)
                total = len(intel.get("signals", []))
                print(f"  [{'OK' if active > 0 else '!!'}] Intelligence    = {active}/{total} providers active")
            except Exception as e:
                print(f"  [!!] Intelligence    = ERROR: {e}")

        # Notification check
        if Config.any_notifications_enabled():
            print(f"  [OK] Notifications  = {', '.join(c for c in ['Telegram', 'Discord', 'Email'] if getattr(Config, {'Telegram': 'TELEGRAM_BOT_TOKEN', 'Discord': 'DISCORD_WEBHOOK_URL', 'Email': 'EMAIL_ALERTS_ENABLED'}.get(c, ''), ''))}")

        print("=" * 50)
        if all_ok:
            print("  All systems GO — ready for overnight run")
        else:
            print("  Some checks failed — agent will continue with fallbacks")
        print("=" * 50 + "\n")
        return all_ok

    def run(self, cycles=None):
        # v7.0: Run preflight check before main loop
        self.preflight_check()

        label = "infinite" if cycles is None else cycles
        print(f"\n[Agent] Starting {label} cycle loop (FULLY AUTONOMOUS MODE)")
        print(f"[Agent] Interval: {Config.AGENT_INTERVAL_SECONDS}s")
        print(f"[Agent] Decision State: {self.decision.state.value}")
        print(f"[Agent] WebSocket: {'ON' if self.ws_streamer else 'OFF'}")
        print(f"[Agent] Notifications: {'ON' if self.notifier.has_channels() else 'OFF'}")
        print(f"[Agent] Trade DB: {'ON' if self.trade_db else 'OFF'}")
        print()

        # v7.0: Start WebSocket streamer
        if self.ws_streamer:
            try:
                self.ws_streamer.start()
                print("[Agent] WebSocket streamer started")
            except Exception as e:
                logging.getLogger(__name__).warning("WebSocket start failed: %s", e)

        # Send startup notification
        mode = "paper" if Config.is_paper_mode() else "live"
        pairs = Config.TRADING_PAIRS if self.multi_pair else [Config.TRADING_PAIR]
        self.notifier.notify_state_change(
            "offline", "normal",
            f"Agent started in {mode} mode with {', '.join(pairs)}",
        )

        count = 0
        try:
            while True:
                try:
                    self.run_cycle()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    # v5.0: Self-healing catches ALL cycle errors
                    self.decision.healer.record_error(
                        "agent", e, ErrorSeverity.HIGH
                    )
                    print(f"\n  [SelfHealer] Cycle error caught: {type(e).__name__}: {e}")
                    print(f"  [SelfHealer] Continuing to next cycle...")
                    self.notifier.notify_error(
                        component="agent",
                        error=f"Cycle {self.cycle_count}: {str(e)[:150]}",
                    )

                count += 1
                if cycles and count >= cycles:
                    break

                # v7.0: Send daily summary at end of each day (approx every 24h worth of cycles)
                cycles_per_day = 86400 // max(Config.AGENT_INTERVAL_SECONDS, 1)
                if count > 0 and count % cycles_per_day == 0:
                    summary = self.risk.get_summary()
                    self.notifier.notify_daily_summary(
                        capital=summary["capital"],
                        total_pnl=summary["total_pnl"],
                        win_rate=summary["win_rate"] * 100,
                        open_positions=summary["open_positions"],
                    )
                    # Also generate daily summary in trade DB
                    if self.trade_db:
                        try:
                            self.trade_db.generate_daily_summary()
                        except Exception as e:
                            _log.warning("Daily summary generation failed: %s", e)

                # v7.0: Hourly heartbeat (every ~12 cycles at 300s interval)
                heartbeat_interval = max(1, 3600 // max(Config.AGENT_INTERVAL_SECONDS, 1))
                if count > 0 and count % heartbeat_interval == 0:
                    try:
                        summary = self.risk.get_summary()
                        self.notifier.notify_heartbeat(
                            cycle=self.cycle_count,
                            capital=summary["capital"],
                            total_pnl=summary["total_pnl"],
                            open_positions=summary["open_positions"],
                            pairs=Config.TRADING_PAIRS if self.multi_pair else [Config.TRADING_PAIR],
                            prices=getattr(self, "_last_prices", {}),
                        )
                    except Exception as e:
                        _log.debug("Heartbeat notification failed: %s", e)

                print(f"\n  Next cycle in {Config.AGENT_INTERVAL_SECONDS}s...")
                time.sleep(Config.AGENT_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n\n[Agent] Shutting down...")
        finally:
            # Graceful shutdown handles: websocket stop, state save, db close, notification
            if not self.shutdown_handler.shutdown_requested:
                self.shutdown_handler.initiate_shutdown(reason="Agent run completed")
            self._print_final_report()

    def _print_final_report(self):
        summary = self.risk.get_summary()
        print("\n" + "=" * 60)
        print("  FINAL PERFORMANCE REPORT v7.0 (FULLY AUTONOMOUS)")
        print("=" * 60)
        print(f"  Cycles:      {self.cycle_count}")
        print(f"  Trades:      {summary['total_trades']}")
        print(f"  Win rate:    {summary['win_rate']:.0%}")
        print(f"  Total PnL:   ${summary['total_pnl']:,.2f}")
        print(f"  Total Fees:  ${summary.get('total_fees', 0):,.2f}")
        print(f"  Capital:     ${summary['capital']:,.2f}")
        ret = (summary['total_pnl'] / Config.INITIAL_CAPITAL) * 100
        print(f"  Return:      {ret:+.2f}%")

        drift = self.drift.check_drift()
        print(f"\n  Drift events: {drift['drift_count']}")

        # v6.0: Multi-pair portfolio report
        if self.multi_pair:
            print(f"\n  --- MULTI-PAIR PORTFOLIO ---")
            print(f"  Pairs: {', '.join(Config.TRADING_PAIRS)}")
            print(f"  Weights: " + " | ".join(
                f"{p}={w:.1%}" for p, w in self.portfolio.weights.items()
            ))
            port_risk = self.portfolio.get_portfolio_risk(self.risk.positions)
            print(f"  Correlation risk: {port_risk['corr_risk']}")
            if port_risk.get("pair_exposure"):
                for p, exp in port_risk["pair_exposure"].items():
                    print(f"    {p}: ${exp:,.2f}")

        # v5.0: Autonomous system report
        print(f"\n  --- AUTONOMOUS SYSTEM ---")
        print(f"  Decision state: {self.decision.state.value}")
        print(f"  Total autonomous decisions: {self.decision._total_autonomous_decisions}")

        meta = self.decision.meta.config
        print(f"  Learned weights: Strategy={meta.strategy_weight:.0%} ML={meta.ml_weight:.0%}")
        print(f"  Position sizing: {meta.position_size_method}")
        print(f"  Retrain frequency: {meta.retrain_hours:.1f}h")
        print(f"  Meta-learning rounds: {self.decision.meta.learning_count}")

        if self.decision.evolver._initialized:
            print(f"  Evolution generation: {self.decision.evolver.generation}")
            for name, genome in self.decision.evolver.best_genomes.items():
                if genome.fitness_score > 0:
                    print(f"    {name}: fitness={genome.fitness_score:.3f} sharpe={genome.sharpe:.3f}")

        health = self.decision.healer.check_health()
        print(f"  System health: {'OK' if health.overall_healthy else 'DEGRADED'}")
        print(f"  Components: {health.components_healthy}/{health.components_total}")
        print(f"  Error rate: {health.error_rate:.1f}/min")

        # v6.0: Intelligence summary
        if self._last_intelligence:
            intel = self._last_intelligence
            print(f"\n  --- INTELLIGENCE ---")
            print(f"  Last bias: {intel['bias']} (adjustment: {intel['adjustment_factor']:.3f})")
            for sig in intel["signals"]:
                status = f"{sig['signal']} ({sig['strength']:.2f})" if sig['strength'] > 0 else "disabled"
                print(f"    {sig['source']:20s} {status}")

        if self._last_arb_scan:
            exec_summary = self._last_arb_scan.get("execution", {})
            print(f"\n  --- ARBITRAGE ---")
            print(f"  Scans: {self._last_arb_scan.get('scan_count', 0)}")
            print(f"  Trades: {exec_summary.get('total_trades', 0)} "
                  f"(win: {exec_summary.get('win_rate', 0):.0%})")
            print(f"  PnL: ${exec_summary.get('total_pnl', 0):.4f}")

        # v7.0: Production modules status
        print(f"\n  --- PRODUCTION MODULES (v7.0) ---")
        print(f"  WebSocket: {'Active' if self.ws_streamer else 'Disabled'}")
        print(f"  Notifications: {self.notifier.has_channels()}")
        print(f"  Trade DB: {'Active' if self.trade_db else 'Disabled'}")
        rl_status = self.rate_limiter.get_status()
        print(f"  Rate limiter: {rl_status['requests_per_minute']}/{rl_status['max_requests_per_minute']} req/min | "
              f"{rl_status['orders_per_minute']}/{rl_status['max_orders_per_minute']} ord/min")

        if self.trade_db:
            try:
                db_stats = self.trade_db.get_total_stats()
                print(f"  DB trades: {db_stats.get('total_trades', 0)} | "
                      f"DB PnL: ${db_stats.get('total_pnl', 0):,.2f}")
            except Exception as e:
                _log.debug("Trade DB stats retrieval failed: %s", e)

        # Events log
        events = list(self.decision.event_log)
        if events:
            print(f"\n  Autonomous events ({len(events)}):")
            for e in events[-5:]:
                print(f"    [{e.event_type}] {e.description}")

        if self.regime_detector.regime_history:
            print(f"\n  Regime history:")
            from collections import Counter
            regimes = Counter(r.regime.value for r in self.regime_detector.regime_history)
            for regime, cnt in regimes.most_common():
                print(f"    {regime:20s} {cnt} cycles")

        if self.model.is_trained:
            print(f"\n  Top ML features:")
            for feat, imp in list(self.model.get_feature_importance().items())[:5]:
                bar = "|" * int(imp * 50)
                print(f"    {feat:20s} {imp:.3f} {bar}")

        if self.risk.trade_history:
            from collections import defaultdict, Counter
            strat_pnl = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
            for t in self.risk.trade_history:
                s = t.strategy_name or "Unknown"
                strat_pnl[s]["trades"] += 1
                strat_pnl[s]["pnl"] += t.pnl_net
                if t.pnl_net > 0:
                    strat_pnl[s]["wins"] += 1

            print(f"\n  Strategy Attribution:")
            for name, data in sorted(strat_pnl.items(), key=lambda x: -x[1]["pnl"]):
                wr = data["wins"] / data["trades"] * 100 if data["trades"] > 0 else 0
                print(f"    {name:25s} {data['trades']:>3d} trades | "
                      f"WR: {wr:>5.1f}% | PnL: ${data['pnl']:>8,.2f}")

            reasons = Counter(t.exit_reason for t in self.risk.trade_history)
            print(f"\n  Exit reasons:")
            for reason, cnt in reasons.most_common():
                print(f"    {reason:20s} {cnt}")

        print("=" * 60)

        if self.risk.trade_history:
            log = [{
                "symbol": t.symbol, "side": t.side,
                "entry": t.entry_price, "exit": t.exit_price,
                "qty": t.quantity,
                "pnl_gross": round(t.pnl_gross, 4),
                "pnl_net": round(t.pnl_net, 4),
                "fees": round(t.fees_paid, 4),
                "reason": t.exit_reason,
                "strategy": t.strategy_name,
                "hold_bars": t.hold_bars,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
            } for t in self.risk.trade_history]
            with open("trade_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print(f"  Trade log: trade_log.json | State: {Config.STATE_FILE}")


def main():
    agent = TradingAgent()
    cycles = None
    if len(sys.argv) > 1:
        try:
            cycles = int(sys.argv[1])
        except ValueError:
            pass
    agent.run(cycles=cycles)


if __name__ == "__main__":
    main()
