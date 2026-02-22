"""
Tests for MarketImpactModel — realistic execution cost modeling.
Covers: slippage, market impact, spread, partial fills, stress scenarios.
"""

import pytest

from market_impact import ExecutionResult, MarketImpactModel, StressScenario


@pytest.fixture()
def model():
    """Standard model with stress disabled for deterministic tests."""
    return MarketImpactModel(fee_pct=0.001, enable_partial_fills=True, enable_stress=False)


@pytest.fixture()
def stress_model():
    """Model with stress scenarios enabled."""
    return MarketImpactModel(fee_pct=0.001, enable_stress=True)


# ── Basic Execution ───────────────────────────────────────────────


class TestBasicExecution:
    def test_returns_execution_result(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="long",
            is_entry=True,
        )
        assert isinstance(result, ExecutionResult)

    def test_full_fill_small_order(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=0.001,
            side="long",
            is_entry=True,
            avg_volume=1000,
            bar_volume=100,
        )
        assert result.fill_rate == 1.0
        assert result.is_partial_fill is False

    def test_entry_price_adverse_for_long(self, model):
        """Buying should get a worse (higher) fill price."""
        result = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="long",
            is_entry=True,
        )
        assert result.average_fill_price > 50000

    def test_entry_price_adverse_for_short(self, model):
        """Shorting entry should get a worse (lower) fill price."""
        result = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="short",
            is_entry=True,
        )
        assert result.average_fill_price < 50000

    def test_exit_price_direction(self, model):
        """Closing a long should get a slightly worse (lower) price."""
        result = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="long",
            is_entry=False,
        )
        assert result.average_fill_price < 50000

    def test_fees_calculated(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="long",
            is_entry=True,
        )
        assert result.fees_paid > 0

    def test_total_cost_positive(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="long",
            is_entry=True,
        )
        assert result.total_cost_pct > 0


# ── Volatility Impact ─────────────────────────────────────────────


class TestVolatilityImpact:
    def test_higher_atr_more_slippage(self, model):
        low_vol = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="long",
            is_entry=True,
            atr_pct=0.005,
        )
        high_vol = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="long",
            is_entry=True,
            atr_pct=0.05,
        )
        assert high_vol.slippage_pct > low_vol.slippage_pct


# ── Market Impact (order size) ────────────────────────────────────


class TestMarketImpact:
    def test_larger_order_more_impact(self, model):
        small = model.simulate_execution(
            price=50000,
            quantity=0.01,
            side="long",
            is_entry=True,
            avg_volume=1000,
        )
        large = model.simulate_execution(
            price=50000,
            quantity=10.0,
            side="long",
            is_entry=True,
            avg_volume=1000,
        )
        assert large.market_impact_pct > small.market_impact_pct

    def test_impact_capped_at_5pct(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=10000,
            side="long",
            is_entry=True,
            avg_volume=10,
        )
        assert result.market_impact_pct <= 0.05

    def test_zero_volume_handled(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=1,
            side="long",
            is_entry=True,
            avg_volume=0,
        )
        assert result.market_impact_pct == 0.001


# ── Partial Fills ─────────────────────────────────────────────────


class TestPartialFills:
    def test_small_order_full_fill(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=0.001,
            side="long",
            is_entry=True,
            avg_volume=1000,
            bar_volume=100,
        )
        assert result.fill_rate == 1.0

    def test_large_order_partial_fill(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=50,
            side="long",
            is_entry=True,
            avg_volume=100,
            bar_volume=100,
        )
        assert result.fill_rate < 1.0
        assert result.is_partial_fill is True
        assert result.filled_quantity < 50

    def test_partial_fills_disabled(self):
        model = MarketImpactModel(enable_partial_fills=False, enable_stress=False)
        result = model.simulate_execution(
            price=50000,
            quantity=50,
            side="long",
            is_entry=True,
            avg_volume=100,
            bar_volume=100,
        )
        assert result.fill_rate == 1.0


# ── Stress Scenarios ──────────────────────────────────────────────


class TestStressScenarios:
    def test_no_stress_initially(self, model):
        assert model.get_active_stress() is None

    def test_advance_bar_with_stress_disabled(self, model):
        for _ in range(100):
            model.advance_bar()
        assert model.get_active_stress() is None

    def test_manual_stress_activation(self, model):
        scenario = StressScenario(
            name="test_crash",
            price_shock_pct=-0.1,
            liquidity_mult=0.1,
            spread_mult=5.0,
            latency_ms=1000,
            duration_bars=3,
            probability=1.0,
        )
        model._active_stress = scenario
        model._stress_bars_remaining = 3
        info = model.get_active_stress()
        assert info["name"] == "test_crash"
        assert info["bars_remaining"] == 3

    def test_stress_expires_after_bars(self, model):
        scenario = StressScenario(
            name="short_stress",
            price_shock_pct=0,
            liquidity_mult=0.5,
            spread_mult=2.0,
            latency_ms=0,
            duration_bars=2,
            probability=0,
        )
        model._active_stress = scenario
        model._stress_bars_remaining = 2
        model.advance_bar()
        assert model._stress_bars_remaining == 1
        model.advance_bar()
        assert model.get_active_stress() is None


# ── Execution Stats ───────────────────────────────────────────────


class TestExecutionStats:
    def test_empty_stats(self, model):
        stats = model.get_execution_stats()
        assert stats["executions"] == 0

    def test_stats_after_executions(self, model):
        for _ in range(5):
            model.simulate_execution(
                price=50000,
                quantity=0.1,
                side="long",
                is_entry=True,
            )
        stats = model.get_execution_stats()
        assert stats["executions"] == 5
        assert "avg_slippage_pct" in stats
        assert "avg_fill_rate" in stats


class TestToDict:
    def test_to_dict_structure(self, model):
        result = model.simulate_execution(
            price=50000,
            quantity=0.1,
            side="long",
            is_entry=True,
        )
        d = result.to_dict()
        assert "filled_qty" in d
        assert "fill_rate" in d
        assert "total_cost_pct" in d
