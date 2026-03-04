"""
Trade execution module.
Handles paper trading simulation and live order placement via CCXT.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import ccxt

from config import Config

_log = logging.getLogger(__name__)


class PaperExecutor:
    """Simulates trade execution without touching real money."""

    def __init__(self) -> None:
        """Initialize the paper executor with an empty in-memory orders list."""
        self.orders: list[dict[str, Any]] = []

    def place_order(self, symbol: str, side: str, quantity: float, price: float) -> dict[str, Any]:
        """Simulate placing an order. Fills instantly at current price.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.
            side: Order direction — ``"long"`` (buy) or ``"short"`` (sell).
            quantity: Asset quantity to trade.
            price: Simulated fill price.

        Returns:
            Order dict with keys ``id``, ``symbol``, ``side``, ``quantity``,
            ``price``, ``status``, ``timestamp``, and ``mode``.
        """
        order = {
            "id": f"paper_{len(self.orders) + 1}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER",
        }
        self.orders.append(order)
        _log.info("[Paper] %s %.6f %s @ $%,.2f", side.upper(), quantity, symbol, price)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order (always succeeds).

        Args:
            order_id: Identifier of the order to cancel.

        Returns:
            Always ``True`` for paper trading.
        """
        return True

    def partial_close(
        self,
        position: Any,
        fraction: float,
        current_price: float,
        reason: str = "partial_tp",
    ) -> dict[str, Any]:
        """Simulate a partial position close.

        Reduces ``position.quantity`` by ``fraction`` and records the
        closed portion as a filled order. PnL is calculated for the
        closed quantity only.

        Args:
            position: Open Position object (quantity is mutated in-place).
            fraction: Fraction to close, e.g. ``0.50`` for 50%.
            current_price: Simulated fill price.
            reason: Tag for the audit trail (e.g. ``"partial_tp"``).

        Returns:
            Order dict with keys: id, symbol, side, quantity, price, pnl,
            reason, status, timestamp, mode.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        closed_qty = position.quantity * fraction
        if position.side == "long":
            pnl = (current_price - position.entry_price) * closed_qty
        else:
            pnl = (position.entry_price - current_price) * closed_qty

        position.quantity -= closed_qty

        order = {
            "id": f"paper_partial_{len(self.orders) + 1}",
            "symbol": position.symbol,
            "side": "sell" if position.side == "long" else "buy",
            "quantity": closed_qty,
            "price": current_price,
            "pnl": pnl,
            "reason": reason,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER",
        }
        self.orders.append(order)
        _log.info(
            "[Paper] Partial close %.0f%% of %s @ $%,.2f (PnL: $%.2f)",
            fraction * 100,
            position.symbol,
            current_price,
            pnl,
        )
        return order


class PerpPaperExecutor:
    """Simulates leveraged perpetual futures in paper mode.

    Creates ``Position`` objects with leverage, margin, and liquidation
    price pre-computed. Funding costs are accrued via ``apply_funding()``.

    Args:
        leverage: Integer leverage multiplier (e.g. ``3`` for 3×).
    """

    def __init__(self, leverage: int = 3) -> None:
        """Initialize the perp paper executor.

        Args:
            leverage: Integer leverage multiplier (e.g. ``3`` for 3×).
        """
        self.leverage = leverage
        self.orders: list[dict[str, Any]] = []

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict[str, Any]:
        """Simulate an instant fill at the given price.

        Args:
            symbol: Trading pair symbol.
            side: ``"long"`` (buy) or ``"short"`` (sell).
            quantity: Asset quantity.
            price: Simulated fill price.

        Returns:
            Order dict with ``status="filled"`` and ``mode="PERP_PAPER"``.
        """
        order = {
            "id": f"perp_paper_{len(self.orders) + 1}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "filled",
            "mode": "PERP_PAPER",
            "timestamp": datetime.now().isoformat(),
        }
        self.orders.append(order)
        _log.info("[PerpPaper] %s %.6f %s @ $%,.2f", side.upper(), quantity, symbol, price)
        return order

    def open_position(self, **kwargs: Any) -> Any:
        """Create a Position with perp-specific fields computed.

        Accepts the same keyword arguments as ``Position.__init__``.
        Sets ``leverage``, ``margin_used``, and ``liquidation_price``
        automatically.

        Returns:
            ``Position`` instance with perp fields populated.
        """
        from risk_manager import Position

        pos = Position(**kwargs)
        pos.leverage = self.leverage
        pos.margin_used = pos.entry_price * pos.quantity / self.leverage
        if pos.side == "long":
            pos.liquidation_price = pos.entry_price * (1 - 1 / self.leverage)
        else:
            pos.liquidation_price = pos.entry_price * (1 + 1 / self.leverage)
        return pos

    def apply_funding(self, position: Any, funding_rate: float) -> None:
        """Accrue synthetic funding cost to ``position.funding_pnl``.

        Called every 8h by the agent. Positive ``funding_rate`` means
        longs pay shorts (reduces long PnL).

        Args:
            position: Open ``Position`` with ``leverage > 1``.
            funding_rate: 8h funding rate as a decimal (e.g. ``0.0001``).
        """
        notional = position.entry_price * position.quantity
        cost = notional * funding_rate
        if position.side == "long":
            position.funding_pnl -= cost
        else:
            position.funding_pnl += cost

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        """No-op in paper mode — always returns True.

        Args:
            order_id: Identifier of the order to cancel (ignored).
            symbol: Trading pair symbol (ignored in paper mode).

        Returns:
            Always ``True``.
        """
        return True

    def partial_close(
        self,
        position: Any,
        fraction: float,
        current_price: float,
        reason: str = "partial_tp",
    ) -> dict[str, Any]:
        """Simulate a partial close. Identical logic to PaperExecutor.

        Args:
            position: Open Position object (quantity is mutated in-place).
            fraction: Fraction to close, e.g. ``0.50`` for 50%.
            current_price: Simulated fill price.
            reason: Tag for the audit trail (e.g. ``"partial_tp"``).

        Returns:
            Order dict with keys: id, symbol, side, quantity, price, pnl,
            reason, status, timestamp, mode.

        Raises:
            ValueError: If ``fraction`` is not strictly between 0 and 1.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        closed_qty = position.quantity * fraction
        if position.side == "long":
            pnl = (current_price - position.entry_price) * closed_qty
        else:
            pnl = (position.entry_price - current_price) * closed_qty
        position.quantity -= closed_qty
        order = {
            "id": f"perp_paper_partial_{len(self.orders) + 1}",
            "symbol": position.symbol,
            "side": "sell" if position.side == "long" else "buy",
            "quantity": closed_qty,
            "price": current_price,
            "pnl": pnl,
            "reason": reason,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
            "mode": "PERP_PAPER",
        }
        self.orders.append(order)
        return order


class LiveExecutor:
    """Places real orders on the exchange via CCXT."""

    def __init__(self, exchange: ccxt.Exchange) -> None:
        """Wrap a CCXT exchange instance for live order placement.

        Args:
            exchange: An authenticated CCXT exchange object (e.g.
                ``ccxt.binance({"apiKey": ..., "secret": ...})``).
                All order and cancel calls are delegated to this instance.
        """
        self.exchange = exchange

    def place_order(self, symbol: str, side: str, quantity: float, price: float) -> dict[str, Any]:
        """Place a limit order on the exchange.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.
            side: Order direction — ``"long"`` (buy) or ``"short"`` (sell).
            quantity: Asset quantity to trade.
            price: Limit price for the order.

        Returns:
            CCXT order dict on success, or a dict with an ``"error"`` key on
            ``InsufficientFunds`` / ``ExchangeError``.

        Raises:
            ValueError: If ``side`` is not ``"long"`` or ``"short"``.
            ccxt.NetworkError, ccxt.RequestTimeout: Unexpected CCXT exceptions
                propagate to the caller unwrapped.
        """
        if side not in ("long", "short"):
            raise ValueError(f"Invalid side '{side}': must be 'long' or 'short'")
        try:
            if side == "long":
                order = self.exchange.create_limit_buy_order(symbol, quantity, price)
            else:
                order = self.exchange.create_limit_sell_order(symbol, quantity, price)

            _log.info(
                "[Live] %s %.6f %s @ $%,.2f (order: %s)",
                side.upper(),
                quantity,
                symbol,
                price,
                order["id"],
            )
            return order
        except ccxt.InsufficientFunds as e:
            _log.error("[Live] Insufficient funds: %s", e)
            return {"error": str(e)}
        except ccxt.ExchangeError as e:
            _log.error("[Live] Exchange error: %s", e)
            return {"error": str(e)}

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        """Cancel an open order on the exchange.

        Args:
            order_id: Exchange-assigned order identifier.
            symbol: Trading pair symbol required by some exchanges. Falls back
                to ``Config.TRADING_PAIR`` when ``None`` and logs a warning.

        Returns:
            ``True`` if the order was cancelled successfully, ``False`` on any
            exchange or network error.
        """
        effective_symbol = symbol or Config.TRADING_PAIR
        if symbol is None:
            _log.warning(
                "[Live] cancel_order called without symbol — falling back to primary pair %s",
                effective_symbol,
            )
        try:
            self.exchange.cancel_order(order_id, effective_symbol)
            return True
        except Exception as e:
            _log.error("[Live] Cancel error: %s", e)
            return False

    def partial_close(
        self,
        position: Any,
        fraction: float,
        current_price: float,
        reason: str = "partial_tp",
    ) -> dict[str, Any]:
        """Place a market order to close a fraction of the position.

        Args:
            position: Open Position object (quantity is mutated in-place).
            fraction: Fraction to close, e.g. ``0.50`` for 50%.
            current_price: Reference price for PnL estimation (actual fill
                may differ).
            reason: Tag for the audit trail (e.g. ``"partial_tp"``).

        Returns:
            Order dict with keys: id, symbol, side, quantity, price, pnl,
            reason, status, timestamp, mode.
        """
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction must be in (0, 1), got {fraction}")
        closed_qty = position.quantity * fraction
        # For a long: sell to reduce; for a short: buy to reduce
        order_side = "sell" if position.side == "long" else "buy"
        try:
            raw = self.exchange.create_market_order(
                position.symbol, order_side, closed_qty
            )
            filled_price = float(raw.get("average") or current_price)
        except Exception as e:
            _log.error("[Live] partial_close failed: %s", e)
            return {"error": str(e)}

        if position.side == "long":
            pnl = (filled_price - position.entry_price) * closed_qty
        else:
            pnl = (position.entry_price - filled_price) * closed_qty

        position.quantity -= closed_qty

        order = {
            "id": raw.get("id", "unknown"),
            "symbol": position.symbol,
            "side": order_side,
            "quantity": closed_qty,
            "price": filled_price,
            "pnl": pnl,
            "reason": reason,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
            "mode": "LIVE",
        }
        _log.info(
            "[Live] Partial close %.0f%% of %s @ $%,.2f (PnL: $%.2f)",
            fraction * 100,
            position.symbol,
            filled_price,
            pnl,
        )
        return order
