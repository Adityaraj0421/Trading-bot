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

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

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
        """Whether a shutdown has been requested.

        Returns:
            True once :meth:`initiate_shutdown` has been called or a
            SIGINT/SIGTERM signal has been received.
        """
        return self._shutdown_requested

    def register_callback(self, name: str, callback: Callable[[], None]) -> None:
        """Register a shutdown callback. Called in LIFO order.

        Args:
            name: Human-readable label used in shutdown progress output.
            callback: Zero-argument callable invoked during shutdown.
        """
        self._callbacks.append((name, callback))
        _log.debug("Registered shutdown callback: %s", name)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle OS signals by initiating a graceful shutdown.

        Args:
            signum: Signal number received (e.g. ``signal.SIGINT``).
            frame: Current stack frame (unused).
        """
        sig_name = signal.Signals(signum).name
        _log.info("Received %s, initiating graceful shutdown...", sig_name)
        print(f"\n  [Shutdown] Received {sig_name}, shutting down gracefully...")
        self.initiate_shutdown(reason=f"Signal {sig_name}")

    def initiate_shutdown(self, reason: str = "Manual", close_positions: bool = False) -> None:
        """Begin graceful shutdown sequence.

        Executes all registered callbacks in LIFO order.  If shutdown has
        already been requested this method returns immediately.

        Args:
            reason: Human-readable description of why the shutdown was
                triggered (used in the console banner).
            close_positions: Reserved for future use; currently unused by
                the callback mechanism itself.
        """
        with self._lock:
            if self._shutdown_requested:
                _log.warning("Shutdown already in progress")
                return
            self._shutdown_requested = True

        print(f"\n{'=' * 50}")
        print("  GRACEFUL SHUTDOWN INITIATED")
        print(f"  Reason: {reason}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 50}")

        # Execute callbacks in reverse order (LIFO)
        for name, callback in reversed(self._callbacks):
            try:
                print(f"  [Shutdown] Running: {name}...")
                callback()
                print(f"  [Shutdown] Done: {name}")
            except Exception as e:
                _log.error("Shutdown callback '%s' failed: %s", name, e)
                print(f"  [Shutdown] FAILED: {name} ({e})")

        print("\n  [Shutdown] Complete. Goodbye!")


class RateLimiter:
    """
    Exchange API rate limiter to prevent bans.
    Tracks requests per endpoint and enforces configurable limits.
    """

    def __init__(self, max_requests_per_minute: int = 1200, max_orders_per_minute: int = 10) -> None:
        self._max_rpm = max_requests_per_minute
        self._max_opm = max_orders_per_minute
        self._request_times: list[float] = []
        self._order_times: list[float] = []
        self._lock = threading.Lock()

    def can_request(self) -> bool:
        """Check if a general API request can be made without exceeding the rate limit.

        Returns:
            True if the number of requests in the last 60 seconds is below
            ``max_requests_per_minute``.
        """
        return self._check_limit(self._request_times, self._max_rpm)

    def can_order(self) -> bool:
        """Check if an order can be placed without exceeding the order rate limit.

        Returns:
            True if the number of orders in the last 60 seconds is below
            ``max_orders_per_minute``.
        """
        return self._check_limit(self._order_times, self._max_opm)

    def record_request(self) -> None:
        """Record the timestamp of a general API request for rate-limit tracking."""
        with self._lock:
            import time

            self._request_times.append(time.time())

    def record_order(self) -> None:
        """Record the timestamp of an order placement for rate-limit tracking."""
        with self._lock:
            import time

            self._order_times.append(time.time())

    def _check_limit(self, times: list[float], max_count: int) -> bool:
        """Check whether the rate limit for a given time window would be exceeded.

        Prunes entries older than 60 seconds before comparing the remaining
        count against *max_count*.

        Args:
            times: Mutable list of UNIX timestamps for past events.
                Old entries are removed in-place.
            max_count: Maximum number of events allowed within the 60-second
                sliding window.

        Returns:
            True if a new event can be accepted without breaching the limit.
        """
        import time

        now = time.time()
        cutoff = now - 60  # 1 minute window
        with self._lock:
            # Remove old entries
            while times and times[0] < cutoff:
                times.pop(0)
            return len(times) < max_count

    def wait_if_needed(self, is_order: bool = False) -> None:
        """Block the calling thread until the rate limit allows the next request.

        Polls every 100 ms until :meth:`can_order` or :meth:`can_request`
        returns True.

        Args:
            is_order: When True, checks the order rate limit; otherwise
                checks the general request rate limit.
        """
        import time

        check = self.can_order if is_order else self.can_request
        while not check():
            time.sleep(0.1)

    def get_status(self) -> dict[str, int]:
        """Return a snapshot of current rate-limit usage.

        Returns:
            Dictionary with keys: ``requests_per_minute``,
            ``max_requests_per_minute``, ``orders_per_minute``,
            ``max_orders_per_minute``, ``request_headroom``, and
            ``order_headroom``.
        """
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
