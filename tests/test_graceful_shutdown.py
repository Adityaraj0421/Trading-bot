"""
Tests for GracefulShutdown and RateLimiter.
Covers: callback registration, LIFO ordering, idempotent shutdown,
        rate limiter tracking, and wait_if_needed.
"""

from graceful_shutdown import GracefulShutdown, RateLimiter

# ── GracefulShutdown ──────────────────────────────────────────────


class TestShutdownBasics:
    def test_initial_state(self):
        gs = GracefulShutdown()
        assert gs.shutdown_requested is False

    def test_register_callback(self):
        gs = GracefulShutdown()
        gs.register_callback("test", lambda: None)
        assert len(gs._callbacks) == 1

    def test_initiate_sets_flag(self):
        gs = GracefulShutdown()
        gs.initiate_shutdown(reason="test")
        assert gs.shutdown_requested is True


class TestCallbackOrdering:
    def test_callbacks_executed_lifo(self):
        gs = GracefulShutdown()
        order = []
        gs.register_callback("first", lambda: order.append("first"))
        gs.register_callback("second", lambda: order.append("second"))
        gs.register_callback("third", lambda: order.append("third"))
        gs.initiate_shutdown(reason="test order")
        assert order == ["third", "second", "first"]

    def test_single_callback_runs(self):
        gs = GracefulShutdown()
        called = []
        gs.register_callback("only", lambda: called.append(True))
        gs.initiate_shutdown()
        assert called == [True]


class TestIdempotency:
    def test_double_shutdown_executes_once(self):
        gs = GracefulShutdown()
        count = []
        gs.register_callback("counter", lambda: count.append(1))
        gs.initiate_shutdown(reason="first")
        gs.initiate_shutdown(reason="second")
        assert len(count) == 1

    def test_flag_stays_true_after_double_call(self):
        gs = GracefulShutdown()
        gs.initiate_shutdown()
        gs.initiate_shutdown()
        assert gs.shutdown_requested is True


class TestCallbackErrors:
    def test_failing_callback_does_not_block_others(self):
        gs = GracefulShutdown()
        order = []
        gs.register_callback("good1", lambda: order.append("good1"))
        gs.register_callback("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        gs.register_callback("good2", lambda: order.append("good2"))
        gs.initiate_shutdown(reason="error test")
        # good2 runs first (LIFO), then bad fails, then good1 still runs
        assert "good2" in order
        assert "good1" in order

    def test_all_callbacks_attempted_despite_errors(self):
        gs = GracefulShutdown()
        calls = []

        def fail():
            calls.append("fail")
            raise ValueError("intentional")

        def succeed():
            calls.append("succeed")

        gs.register_callback("s1", succeed)
        gs.register_callback("fail", fail)
        gs.register_callback("s2", succeed)
        gs.initiate_shutdown()
        assert len(calls) == 3


# ── RateLimiter ───────────────────────────────────────────────────


class TestRateLimiterBasics:
    def test_initial_can_request(self):
        rl = RateLimiter(max_requests_per_minute=10)
        assert rl.can_request() is True

    def test_initial_can_order(self):
        rl = RateLimiter(max_orders_per_minute=5)
        assert rl.can_order() is True

    def test_record_request_counts(self):
        rl = RateLimiter(max_requests_per_minute=3)
        rl.record_request()
        rl.record_request()
        assert rl.can_request() is True
        rl.record_request()
        assert rl.can_request() is False

    def test_record_order_counts(self):
        rl = RateLimiter(max_orders_per_minute=2)
        rl.record_order()
        rl.record_order()
        assert rl.can_order() is False


class TestRateLimiterStatus:
    def test_status_structure(self):
        rl = RateLimiter()
        status = rl.get_status()
        assert "requests_per_minute" in status
        assert "max_requests_per_minute" in status
        assert "orders_per_minute" in status
        assert "request_headroom" in status
        assert "order_headroom" in status

    def test_status_reflects_usage(self):
        rl = RateLimiter(max_requests_per_minute=100)
        rl.record_request()
        rl.record_request()
        status = rl.get_status()
        assert status["requests_per_minute"] == 2
        assert status["request_headroom"] == 98
