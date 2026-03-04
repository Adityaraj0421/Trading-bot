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
