"""
Unit tests for risk_manager.py — Position dataclass and RiskManager.

Covers:
  - Position: trailing stops, exit conditions, unrealized PnL
  - RiskManager: gating, position sizing, open/close, fee tracking,
    daily loss limits, max hold duration, serialization
"""

from datetime import date, datetime

import pytest

from config import Config
from risk_manager import Position, RiskManager, TradeRecord

# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------


class TestPosition:
    """Tests for the Position dataclass and its methods."""

    def _make_pos(
        self, side="long", entry_price=50000.0, quantity=0.01, stop_loss=49000.0, take_profit=52500.0, entry_bar=0
    ):
        return Position(
            symbol="BTC/USDT",
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_bar=entry_bar,
        )

    def test_notional_value(self):
        pos = self._make_pos(entry_price=50000.0, quantity=0.02)
        assert pos.notional_value == 1000.0

    def test_trailing_stop_initialized_long(self):
        pos = self._make_pos(side="long", entry_price=50000.0)
        expected = 50000.0 * (1 - Config.TRAILING_STOP_PCT)
        assert pos.trailing_stop == expected

    def test_trailing_stop_initialized_short(self):
        pos = self._make_pos(side="short", entry_price=50000.0, stop_loss=51000.0, take_profit=47500.0)
        expected = 50000.0 * (1 + Config.TRAILING_STOP_PCT)
        assert pos.trailing_stop == expected

    # --- Trailing stop update ---

    def test_trailing_stop_ratchets_up_for_long(self):
        pos = self._make_pos(side="long", entry_price=50000.0)
        initial_trail = pos.trailing_stop

        # Price rises — trail should move up
        pos.update_trailing_stop(current_high=52000.0, current_low=51000.0)
        assert pos.trailing_stop > initial_trail
        assert pos.highest_price == 52000.0

    def test_trailing_stop_never_drops_for_long(self):
        pos = self._make_pos(side="long", entry_price=50000.0)
        pos.update_trailing_stop(current_high=55000.0, current_low=54000.0)
        high_trail = pos.trailing_stop

        # Price drops — trail must NOT decrease
        pos.update_trailing_stop(current_high=53000.0, current_low=52000.0)
        assert pos.trailing_stop == high_trail

    def test_trailing_stop_ratchets_down_for_short(self):
        pos = self._make_pos(side="short", entry_price=50000.0, stop_loss=51000.0, take_profit=47500.0)
        initial_trail = pos.trailing_stop

        # Price drops — trail should move down
        pos.update_trailing_stop(current_high=49000.0, current_low=48000.0)
        assert pos.trailing_stop < initial_trail
        assert pos.lowest_price == 48000.0

    def test_trailing_stop_never_rises_for_short(self):
        pos = self._make_pos(side="short", entry_price=50000.0, stop_loss=51000.0, take_profit=47500.0)
        pos.update_trailing_stop(current_high=49000.0, current_low=46000.0)
        low_trail = pos.trailing_stop

        # Price rises — trail must NOT increase
        pos.update_trailing_stop(current_high=49500.0, current_low=47000.0)
        assert pos.trailing_stop == low_trail

    # --- Exit conditions ---

    def test_stop_loss_exit_long(self):
        pos = self._make_pos(side="long", stop_loss=49000.0)
        assert pos.check_exit(48500.0) == "stop_loss"

    def test_take_profit_exit_long(self):
        pos = self._make_pos(side="long", take_profit=52500.0)
        assert pos.check_exit(53000.0) == "take_profit"

    def test_no_exit_in_range_long(self):
        pos = self._make_pos(side="long", stop_loss=49000.0, take_profit=52500.0)
        assert pos.check_exit(50500.0) is None

    def test_stop_loss_exit_short(self):
        pos = self._make_pos(side="short", entry_price=50000.0, stop_loss=51000.0, take_profit=47500.0)
        assert pos.check_exit(51500.0) == "stop_loss"

    def test_take_profit_exit_short(self):
        pos = self._make_pos(side="short", entry_price=50000.0, stop_loss=51000.0, take_profit=47500.0)
        assert pos.check_exit(47000.0) == "take_profit"

    def test_max_duration_exit(self):
        pos = self._make_pos(entry_bar=10)
        # current_bar exceeds entry_bar + MAX_HOLD_BARS
        far_bar = 10 + Config.MAX_HOLD_BARS
        assert pos.check_exit(50500.0, current_bar=far_bar) == "max_duration"

    def test_no_max_duration_exit_when_bars_zero(self):
        pos = self._make_pos(entry_bar=0)
        # When entry_bar is 0, max duration check is skipped
        assert pos.check_exit(50500.0, current_bar=0) is None

    def test_trailing_stop_exit_long(self):
        """Trailing stop triggers only when it's above fixed stop-loss."""
        pos = self._make_pos(side="long", entry_price=50000.0, stop_loss=48000.0, take_profit=60000.0)
        # Push price up so trailing stop ratchets above fixed SL
        pos.update_trailing_stop(current_high=56000.0, current_low=55000.0)
        trail = pos.trailing_stop
        assert trail > pos.stop_loss  # Precondition: trail above SL

        # Price drops to trailing stop level
        assert pos.check_exit(trail - 1) == "trailing_stop"

    # --- ATR-adaptive trailing stop ---

    def test_atr_trailing_uses_atr_mult(self):
        """When atr_pct is provided, trail_pct = ATR_TRAILING_MULT × atr_pct."""
        pos = self._make_pos(side="long", entry_price=50000.0, stop_loss=48000.0, take_profit=60000.0)
        atr_pct = 0.03  # 3% ATR → trail_pct = 2.0 × 0.03 = 0.06 (> fixed 0.015)
        pos.update_trailing_stop(current_high=55000.0, current_low=54000.0, atr_pct=atr_pct)
        expected_trail = 55000.0 * (1 - Config.ATR_TRAILING_MULT * atr_pct)
        assert pos.trailing_stop == pytest.approx(expected_trail)

    def test_atr_trailing_fallback_to_fixed_when_none(self):
        """When atr_pct=None, trailing distance falls back to TRAILING_STOP_PCT."""
        pos = self._make_pos(side="long", entry_price=50000.0, stop_loss=48000.0, take_profit=60000.0)
        pos.update_trailing_stop(current_high=55000.0, current_low=54000.0, atr_pct=None)
        expected_trail = 55000.0 * (1 - Config.TRAILING_STOP_PCT)
        assert pos.trailing_stop == pytest.approx(expected_trail)

    def test_atr_trailing_ratchet_still_applies(self):
        """ATR-based trailing still ratchets up for longs — only moves higher."""
        pos = self._make_pos(side="long", entry_price=50000.0, stop_loss=48000.0, take_profit=60000.0)
        pos.update_trailing_stop(current_high=55000.0, current_low=54000.0, atr_pct=0.02)
        trail_after_high = pos.trailing_stop

        # Price drops — trail must not decrease
        pos.update_trailing_stop(current_high=54000.0, current_low=53000.0, atr_pct=0.02)
        assert pos.trailing_stop == trail_after_high

    # --- Breakeven stop ---

    def _make_pos_qty1(
        self,
        side="long",
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=52500.0,
    ) -> "Position":
        """Position with quantity=1.0 so unrealized_pnl is in same scale as price distance."""
        return Position(
            symbol="BTC/USDT",
            side=side,
            entry_price=entry_price,
            quantity=1.0,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_bar=0,
        )

    def test_breakeven_triggers_at_threshold(self):
        """SL moves to entry + fee buffer when unrealized PnL >= BREAKEVEN_TRIGGER_PCT × tp_distance."""
        # tp_distance = |52500 - 50000| = 2500; trigger = 0.6 × 2500 = 1500
        # unrealized at 51600 = (51600 - 50000) × 1.0 = 1600 ≥ 1500 → triggers
        pos = self._make_pos_qty1()
        triggered = pos.check_breakeven(current_price=51600.0, fee_pct=Config.FEE_PCT)
        assert triggered is True
        assert pos.breakeven_triggered is True
        expected_sl = pos.entry_price + pos.entry_price * Config.FEE_PCT
        assert pos.stop_loss == pytest.approx(expected_sl)

    def test_breakeven_not_triggered_below_threshold(self):
        """SL stays unchanged when unrealized PnL is below the trigger threshold."""
        # trigger = 1500; unrealized at 51400 = 1400 < 1500 → no trigger
        pos = self._make_pos_qty1()
        original_sl = pos.stop_loss
        triggered = pos.check_breakeven(current_price=51400.0, fee_pct=Config.FEE_PCT)
        assert triggered is False
        assert pos.breakeven_triggered is False
        assert pos.stop_loss == original_sl

    def test_breakeven_one_shot(self):
        """After first trigger, subsequent calls are no-ops even at higher prices."""
        pos = self._make_pos_qty1()
        pos.check_breakeven(current_price=51600.0, fee_pct=Config.FEE_PCT)
        sl_after_first = pos.stop_loss

        triggered_again = pos.check_breakeven(current_price=52000.0, fee_pct=Config.FEE_PCT)
        assert triggered_again is False
        assert pos.stop_loss == sl_after_first  # Unchanged

    def test_breakeven_only_moves_sl_up_for_long(self):
        """Breakeven never lowers SL for a long position."""
        # Set stop_loss already very close to entry — new_sl must be > existing SL to apply
        pos = self._make_pos_qty1(stop_loss=49999.0)  # SL already near entry
        # Move SL manually above the computed breakeven level
        pos.stop_loss = 50060.0  # higher than entry + fee_buffer (~50050)
        triggered = pos.check_breakeven(current_price=51600.0, fee_pct=Config.FEE_PCT)
        assert triggered is False  # new_sl (50050) < existing SL (50060) → no change

    def test_breakeven_serialization_round_trip(self):
        """breakeven_triggered=True survives to_dict → from_dict round-trip."""
        pos = self._make_pos_qty1()
        pos.check_breakeven(current_price=51600.0, fee_pct=Config.FEE_PCT)
        assert pos.breakeven_triggered is True

        rm = RiskManager()
        rm.positions.append(pos)
        data = rm.to_dict()

        rm2 = RiskManager()
        rm2.from_dict(data)
        assert rm2.positions[0].breakeven_triggered is True

    def test_breakeven_default_false_on_old_state(self):
        """Old state dict without breakeven_triggered key loads with default False."""
        from datetime import date

        rm = RiskManager()
        rm.from_dict(
            {
                "capital": 1000.0,
                "total_pnl": 0.0,
                "total_fees": 0.0,
                "daily_pnl": 0.0,
                "daily_pnl_date": str(date.today()),
                "current_bar": 0,
                "positions": [
                    {
                        "symbol": "BTC/USDT",
                        "side": "long",
                        "entry_price": 50000.0,
                        "quantity": 0.01,
                        "entry_time": datetime.now().isoformat(),
                        "stop_loss": 49000.0,
                        "take_profit": 52500.0,
                        # Note: no "breakeven_triggered" key
                    }
                ],
            }
        )
        assert rm.positions[0].breakeven_triggered is False

    # --- Unrealized PnL ---

    def test_unrealized_pnl_long_profit(self):
        pos = self._make_pos(side="long", entry_price=50000.0, quantity=0.01)
        assert pos.unrealized_pnl(51000.0) == pytest.approx(10.0)

    def test_unrealized_pnl_long_loss(self):
        pos = self._make_pos(side="long", entry_price=50000.0, quantity=0.01)
        assert pos.unrealized_pnl(49000.0) == pytest.approx(-10.0)

    def test_unrealized_pnl_short_profit(self):
        pos = self._make_pos(side="short", entry_price=50000.0, quantity=0.01, stop_loss=51000.0, take_profit=47500.0)
        assert pos.unrealized_pnl(49000.0) == pytest.approx(10.0)

    def test_unrealized_pnl_short_loss(self):
        pos = self._make_pos(side="short", entry_price=50000.0, quantity=0.01, stop_loss=51000.0, take_profit=47500.0)
        assert pos.unrealized_pnl(51000.0) == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------


class TestRiskManager:
    """Tests for the RiskManager class."""

    @pytest.fixture()
    def rm(self):
        return RiskManager()

    # --- Position gating ---

    def test_can_open_with_high_confidence(self, rm):
        ok, msg = rm.can_open_position("BUY", 0.8)
        assert ok is True
        assert msg == "OK"

    def test_rejects_low_confidence(self, rm):
        ok, msg = rm.can_open_position("BUY", 0.1)
        assert ok is False
        assert "Confidence" in msg

    def test_rejects_at_max_positions(self, rm):
        # Fill up to max positions
        for i in range(Config.MAX_OPEN_POSITIONS):
            rm.positions.append(
                Position(
                    symbol=f"PAIR{i}/USDT",
                    side="long",
                    entry_price=50000.0,
                    quantity=0.01,
                    entry_time=datetime.now(),
                    stop_loss=49000.0,
                    take_profit=52000.0,
                )
            )
        ok, msg = rm.can_open_position("BUY", 0.9)
        assert ok is False
        assert "Max positions" in msg

    def test_rejects_conflicting_position(self, rm):
        rm.positions.append(
            Position(
                symbol="BTC/USDT",
                side="short",
                entry_price=50000.0,
                quantity=0.01,
                entry_time=datetime.now(),
                stop_loss=51000.0,
                take_profit=47500.0,
            )
        )
        ok, msg = rm.can_open_position("BUY", 0.9, symbol="BTC/USDT")
        assert ok is False
        assert "Conflicting" in msg

    def test_rejects_duplicate_direction(self, rm):
        rm.positions.append(
            Position(
                symbol="BTC/USDT",
                side="long",
                entry_price=50000.0,
                quantity=0.01,
                entry_time=datetime.now(),
                stop_loss=49000.0,
                take_profit=52000.0,
            )
        )
        ok, msg = rm.can_open_position("BUY", 0.9, symbol="BTC/USDT")
        assert ok is False
        assert "Already have a position" in msg

    def test_rejects_after_daily_loss_limit(self, rm):
        # Simulate daily loss exceeding limit
        rm.daily_pnl = -(Config.INITIAL_CAPITAL * Config.MAX_DAILY_LOSS_PCT + 1)
        rm.daily_pnl_date = date.today()
        ok, msg = rm.can_open_position("BUY", 0.9)
        assert ok is False
        assert "Daily loss" in msg

    # --- Position sizing ---

    def test_position_size_accounts_for_fees(self, rm):
        qty = rm.calculate_position_size(50000.0)
        # v9.0: Multi-layer sizing (Kelly + vol targeting + drawdown tiers)
        # Just verify it returns a reasonable positive quantity within bounds
        assert qty > 0
        max_qty = (Config.INITIAL_CAPITAL * Config.MAX_POSITION_PCT) / 50000.0
        assert qty <= max_qty * 1.1  # Allow small buffer for rounding

    def test_position_size_custom_fee(self, rm):
        qty_low = rm.calculate_position_size(50000.0, fee_pct=0.0001)
        qty_high = rm.calculate_position_size(50000.0, fee_pct=0.01)
        assert qty_low > qty_high  # Lower fees → larger position

    # --- Stop/Take calculation ---

    def test_stop_take_defaults_long(self, rm):
        sl, tp = rm.calculate_stop_take(50000.0, "long")
        assert sl < 50000.0
        assert tp > 50000.0
        assert sl == round(50000.0 - 50000.0 * Config.STOP_LOSS_PCT, 2)
        assert tp == round(50000.0 + 50000.0 * Config.TAKE_PROFIT_PCT, 2)

    def test_stop_take_defaults_short(self, rm):
        sl, tp = rm.calculate_stop_take(50000.0, "short")
        assert sl > 50000.0
        assert tp < 50000.0

    def test_stop_take_custom_pct(self, rm):
        sl, tp = rm.calculate_stop_take(50000.0, "long", sl_pct=0.01, tp_pct=0.03)
        assert sl == round(50000.0 - 50000.0 * 0.01, 2)
        assert tp == round(50000.0 + 50000.0 * 0.03, 2)

    def test_stop_take_with_atr(self, rm):
        atr = 500.0
        sl, tp = rm.calculate_stop_take(50000.0, "long", atr=atr)
        assert sl == round(50000.0 - 2 * atr, 2)  # 2x ATR
        assert tp == round(50000.0 + 3 * atr, 2)  # 3x ATR

    # --- Open / close positions ---

    def test_open_position_deducts_capital(self, rm):
        initial = rm.capital
        pos = rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        notional = 50000.0 * 0.01
        fee = notional * Config.FEE_PCT
        assert rm.capital == pytest.approx(initial - notional - fee)
        assert len(rm.positions) == 1
        assert pos.symbol == "BTC/USDT"

    def test_close_position_returns_capital(self, rm):
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        pos = rm.positions[0]
        capital_before_close = rm.capital

        record = rm.close_position(pos, 51000.0, "take_profit")
        assert isinstance(record, TradeRecord)
        assert record.exit_reason == "take_profit"
        assert record.pnl_gross > 0  # Price went up on a long

        # Capital should increase by notional - fee
        exit_notional = 51000.0 * 0.01
        exit_fee = exit_notional * Config.FEE_PCT
        assert rm.capital == pytest.approx(capital_before_close + exit_notional - exit_fee)
        assert len(rm.positions) == 0

    def test_close_losing_position_tracks_pnl(self, rm):
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        pos = rm.positions[0]
        record = rm.close_position(pos, 48000.0, "stop_loss")
        assert record.pnl_gross < 0
        assert rm.total_pnl < 0
        assert rm.daily_pnl < 0

    def test_fee_tracking_across_trades(self, rm):
        assert rm.total_fees == 0.0
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        fees_after_open = rm.total_fees
        assert fees_after_open > 0

        pos = rm.positions[0]
        rm.close_position(pos, 51000.0, "take_profit")
        assert rm.total_fees > fees_after_open

    def test_hold_bars_tracked_on_close(self, rm):
        rm.set_bar(10)
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        pos = rm.positions[0]
        assert pos.entry_bar == 10

        rm.set_bar(25)
        record = rm.close_position(pos, 51000.0, "manual")
        assert record.hold_bars == 15

    # --- check_positions (auto exit scan) ---

    def test_check_positions_triggers_stop_loss(self, rm):
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        closed = rm.check_positions(current_price=48500.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_loss"
        assert len(rm.positions) == 0

    def test_check_positions_triggers_take_profit(self, rm):
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        closed = rm.check_positions(current_price=52500.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "take_profit"

    def test_check_positions_no_exit_in_range(self, rm):
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        closed = rm.check_positions(current_price=50500.0)
        assert len(closed) == 0
        assert len(rm.positions) == 1

    def test_check_positions_updates_trailing(self, rm):
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 48000.0, 60000.0)
        pos = rm.positions[0]
        initial_trail = pos.trailing_stop

        # Push price up — trailing should ratchet
        rm.check_positions(current_price=55000.0, current_high=55000.0)
        assert pos.trailing_stop > initial_trail

    # --- Win rate & summary ---

    def test_win_rate_no_trades(self, rm):
        assert rm._win_rate() == 0.0

    def test_win_rate_calculation(self, rm):
        # Open and close two winning trades, one losing
        for exit_price in [51000.0, 52000.0, 48000.0]:
            rm.open_position("BTC/USDT", "long", 50000.0, 0.001, 47000.0, 55000.0)
            pos = rm.positions[-1]
            rm.close_position(pos, exit_price, "manual")
        # 2 wins, 1 loss
        assert rm._win_rate() == pytest.approx(2 / 3, abs=0.01)

    def test_get_summary_shape(self, rm):
        summary = rm.get_summary()
        assert "capital" in summary
        assert "open_positions" in summary
        assert "total_trades" in summary
        assert "total_pnl" in summary
        assert "daily_pnl" in summary
        assert "total_fees" in summary
        assert "win_rate" in summary

    # --- Serialization round-trip ---

    def test_to_dict_from_dict_round_trip(self, rm):
        rm.set_bar(42)
        rm.open_position("BTC/USDT", "long", 50000.0, 0.01, 49000.0, 52000.0)
        rm.open_position("ETH/USDT", "short", 3000.0, 0.5, 3100.0, 2800.0)

        data = rm.to_dict()
        assert data["capital"] == rm.capital
        assert data["current_bar"] == 42
        assert len(data["positions"]) == 2

        # Restore into fresh RiskManager
        rm2 = RiskManager()
        rm2.from_dict(data)
        assert rm2.capital == rm.capital
        assert rm2.current_bar == 42
        assert len(rm2.positions) == 2
        assert rm2.positions[0].symbol == "BTC/USDT"
        assert rm2.positions[1].side == "short"
        assert rm2.total_pnl == rm.total_pnl
        assert rm2.total_fees == rm.total_fees
