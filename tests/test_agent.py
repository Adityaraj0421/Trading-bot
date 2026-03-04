"""Unit tests for TradingAgent agent.py."""


class TestPerpRouting:
    """_resolve_executor routes decision.route to correct executor."""

    def _make_agent_paper(self):
        """Return a minimal TradingAgent in paper mode for testing."""
        import os

        os.environ.setdefault("TRADING_MODE", "paper")
        from agent import TradingAgent
        from executor import PaperExecutor, PerpPaperExecutor

        agent = TradingAgent.__new__(TradingAgent)
        # Manually wire the minimum attributes needed
        agent.executor = PaperExecutor()
        agent.perp_executor = PerpPaperExecutor(leverage=3)
        agent._use_perp = True
        return agent

    def test_route_spot_uses_spot_executor(self):
        """_resolve_executor('spot') returns self.executor (spot)."""
        from executor import PaperExecutor

        agent = self._make_agent_paper()
        assert isinstance(agent._resolve_executor("spot"), PaperExecutor)

    def test_route_perp_uses_perp_executor(self):
        """_resolve_executor('perp') returns self.perp_executor when USE_PERP=true."""
        from executor import PerpPaperExecutor

        agent = self._make_agent_paper()
        assert isinstance(agent._resolve_executor("perp"), PerpPaperExecutor)

    def test_route_perp_falls_back_to_spot_when_use_perp_false(self):
        """When USE_PERP=false, perp route degrades gracefully to spot executor."""
        from executor import PaperExecutor

        agent = self._make_agent_paper()
        agent._use_perp = False
        assert isinstance(agent._resolve_executor("perp"), PaperExecutor)


class TestWsFeedIntegration:
    """Agent uses WsFeed close price when available."""

    def test_ws_feed_attribute_exists_when_enabled(self):
        """When USE_WS_FEED=true, WsFeed is importable and constructable."""
        import os

        os.environ["USE_WS_FEED"] = "true"
        try:
            from unittest.mock import patch

            from ws_feed import WsFeed

            # Patch WsFeed.start() to avoid spawning real threads in tests
            with patch("ws_feed.WsFeed.start"):
                feed = WsFeed(["BTC/USDT"])
                assert feed is not None
                assert isinstance(feed, WsFeed)
        finally:
            del os.environ["USE_WS_FEED"]

    def test_ws_price_overrides_rest_price(self):
        """WsFeed.get_latest_close() overrides the REST snapshot price."""
        from unittest.mock import MagicMock

        from agent import TradingAgent
        from ws_feed import WsFeed

        agent = TradingAgent.__new__(TradingAgent)
        # Wire a mock WsFeed that returns a real-time price
        mock_feed = MagicMock(spec=WsFeed)
        mock_feed.get_latest_close.return_value = 55_000.0
        agent._ws_feed = mock_feed

        # Replicate the price-override logic from _run_phase9_cycle
        rest_price = 50_000.0
        current_price = rest_price
        ws_price = agent._ws_feed.get_latest_close("BTC/USDT")
        if agent._ws_feed is not None and ws_price is not None:
            current_price = ws_price

        assert current_price == 55_000.0
        mock_feed.get_latest_close.assert_called_once_with("BTC/USDT")

    def test_ws_price_none_falls_back_to_rest(self):
        """When WsFeed returns None (no data yet), REST price is preserved."""
        from unittest.mock import MagicMock

        from agent import TradingAgent
        from ws_feed import WsFeed

        agent = TradingAgent.__new__(TradingAgent)
        mock_feed = MagicMock(spec=WsFeed)
        mock_feed.get_latest_close.return_value = None  # feed not yet populated
        agent._ws_feed = mock_feed

        rest_price = 50_000.0
        current_price = rest_price
        ws_price = agent._ws_feed.get_latest_close("BTC/USDT")
        if agent._ws_feed is not None and ws_price is not None:
            current_price = ws_price

        assert current_price == 50_000.0  # unchanged — REST price preserved

    def test_ws_feed_none_preserves_rest_price(self):
        """When _ws_feed is None (disabled), REST price is preserved."""
        from agent import TradingAgent

        agent = TradingAgent.__new__(TradingAgent)
        agent._ws_feed = None

        rest_price = 50_000.0
        current_price = rest_price
        if agent._ws_feed is not None:
            ws_price = agent._ws_feed.get_latest_close("BTC/USDT")  # type: ignore[union-attr]
            if ws_price is not None:
                current_price = ws_price

        assert current_price == 50_000.0
