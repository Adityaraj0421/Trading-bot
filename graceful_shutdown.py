"""
Graceful Shutdown Handler
===========================
Ensures the agent shuts down cleanly when receiving SIGINT, SIGTERM,
or when a kill switch is triggered.

On shutdown:
  1. Stop accepting new trades
  2. Optionally close all open positions
  3. Persist state to disk
  4. Send notification
  5. Close all connections (websocket, exchange, DB)
  6. Exit cleanly
"""

import signal
import sys
import logging
import threading
from datetime import datetime
from typing import Any, Callable

_log = logging.getLogger(__name__)


class GracefulShutdown:
    """Manages clean shutdown of all agent components."""

    def __init__(self) -> None:
        self._shutdown_requested = False
        self._callbacks: list[tuple[str, Callable[[], None]]] = []
        self._lock = threading.Lock()

        # Register signal handlers (only works in the main thread;
        # when running inside FastAPI's daemon thread, skip silently)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            _log.info("Skipping signal handlers (not main thread)")

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def register_callback(self, name: str, callback: Callable[[], None]) -> None:
        """Register a shutdown callback. Called in LIFO order."""
        self._callbacks.append((name, callback))
        _log.debug("Registered shutdown callback: %s", name)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle OS signals."""
        sig_name = signal.Signals(signum).name
        _log.info("Received %s, initiating graceful shutdown...", sig_name)
        print(f"\n  [Shutdown] Received {sig_name}, shutting down gracefully...")
        self.initiate_shutdown(reason=f"Signal {sig_name}")

    def initiate_shutdown(self, reason: str = "Manual", close_positions: bool = False) -> None:
        """Begin graceful shutdown sequence."""
        with self._lock:
            if self._shutdown_requested:
                _log.warning("Shutdown already in progress")
                return
            self._shutdown_requested = True

        print(f"\n{'='*50}")
        print(f"  GRACEFUL SHUTDOWN INITIATED")
        print(f"  Reason: {reason}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        # Execute callbacks in reverse order (LIFO)
        for name, callback in reversed(self._callbacks):
            try:
                print(f"  [Shutdown] Running: {name}...")
                callback()
                print(f"  [Shutdown] Done: {name}")
            except Exception as e:
                _log.error("Shutdown callback '%s' failed: %s", name, e)
                print(f"  [Shutdown] FAILED: {name} ({e})")

        print(f"\n  [Shutdown] Complete. Goodbye!")


class RateLimiter:
    """
    Exchange API rate limiter to prevent bans.
    Tracks requests per endpoint and enforces configurable limits.
    """

    def __init__(self, max_requests_per_minute: int = 1200,
                 max_orders_per_minute: int = 10) -> None:
        self._max_rpm = max_requests_per_minute
        self._max_opm = max_orders_per_minute
        self._request_times: list[float] = []
        self._order_times: list[float] = []
        self._lock = threading.Lock()

    def can_request(self) -> bool:
        """Check if we can make a general API request."""
        return self._check_limit(self._request_times, self._max_rpm)

    def can_order(self) -> bool:
        """Check if we can place an order."""
        return self._check_limit(self._order_times, self._max_opm)

    def record_request(self) -> None:
        """Record a general API request."""
        with self._lock:
            import time
            self._request_times.append(time.time())

    def record_order(self) -> None:
        """Record an order placement."""
        with self._lock:
            import time
            self._order_times.append(time.time())

    def _check_limit(self, times: list, max_count: int) -> bool:
        """Check if the rate limit would be exceeded."""
        import time
        now = time.time()
        cutoff = now - 60  # 1 minute window
        with self._lock:
            # Remove old entries
            while times and times[0] < cutoff:
                times.pop(0)
            return len(times) < max_count

    def wait_if_needed(self, is_order: bool = False) -> None:
        """Block until rate limit allows the next request."""
        import time
        check = self.can_order if is_order else self.can_request
        while not check():
            time.sleep(0.1)

    def get_status(self) -> dict[str, int]:
        """Get current rate limit status."""
        import time
        now = time.time()
        cutoff = now - 60
        with self._lock:
            recent_requests = sum(1 for t in self._request_times if t >= cutoff)
            recent_orders = sum(1 for t in self._order_times if t >= cutoff)
        return {
            "requests_per_minute": recent_requests,
            "max_requests_per_minute": self._max_rpm,
            "orders_per_minute": recent_orders,
            "max_orders_per_minute": self._max_opm,
            "request_headroom": self._max_rpm - recent_requests,
            "order_headroom": self._max_opm - recent_orders,
        }
