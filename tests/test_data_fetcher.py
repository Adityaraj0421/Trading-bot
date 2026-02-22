"""
Unit tests for data_fetcher.py — DataFetcher with CCXT exchange + demo fallback.

Mocks the ccxt exchange class so no real network calls are made.
"""

from unittest.mock import MagicMock

import ccxt
import pandas as pd
import pytest

from config import Config
from data_fetcher import DataFetcher


@pytest.fixture()
def mock_exchange():
    """Create a mock exchange with typical methods."""
    exchange = MagicMock()
    exchange.fetch_ohlcv.return_value = [
        [1704067200000, 42000, 42500, 41800, 42200, 1000],
        [1704070800000, 42200, 42700, 42100, 42500, 1200],
        [1704074400000, 42500, 42600, 42000, 42100, 800],
        [1704078000000, 42100, 42300, 41900, 42250, 950],
        [1704081600000, 42250, 42800, 42200, 42700, 1100],
    ]
    exchange.fetch_ticker.return_value = {
        "last": 42700,
        "bid": 42699,
        "ask": 42701,
        "baseVolume": 5000,
    }
    exchange.fetch_order_book.return_value = {
        "bids": [[42699, 1.5], [42698, 2.0]],
        "asks": [[42701, 1.0], [42702, 3.0]],
    }
    exchange.markets = {"BTC/USDT": {}, "ETH/USDT": {}, "SOL/USDT": {}}
    exchange.load_markets.return_value = exchange.markets
    return exchange


@pytest.fixture()
def fetcher(mock_exchange, monkeypatch):
    """Patch ccxt.binance to return our mock exchange."""
    mock_class = MagicMock(return_value=mock_exchange)
    monkeypatch.setattr(ccxt, Config.EXCHANGE_ID, mock_class)
    return DataFetcher()


# ---------------------------------------------------------------------------
# fetch_ohlcv
# ---------------------------------------------------------------------------


class TestFetchOhlcv:
    def test_success_returns_dataframe(self, fetcher):
        df = fetcher.fetch_ohlcv()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_timestamp_index(self, fetcher):
        df = fetcher.fetch_ohlcv()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"

    def test_filters_zero_volume(self, fetcher):
        fetcher.exchange.fetch_ohlcv.return_value = [
            [1704067200000, 42000, 42500, 41800, 42200, 1000],
            [1704070800000, 42200, 42700, 42100, 42500, 0],  # zero volume
            [1704074400000, 42500, 42600, 42000, 42100, 800],
        ]
        df = fetcher.fetch_ohlcv()
        assert len(df) == 2  # zero-volume row filtered out

    def test_network_error_fallback(self, fetcher):
        fetcher.exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("timeout")
        df = fetcher.fetch_ohlcv()
        assert isinstance(df, pd.DataFrame)
        assert fetcher.using_demo is True

    def test_exchange_error_fallback(self, fetcher):
        fetcher.exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("nonce error")
        df = fetcher.fetch_ohlcv()
        assert isinstance(df, pd.DataFrame)
        assert fetcher.using_demo is True

    def test_generic_error_fallback(self, fetcher):
        fetcher.exchange.fetch_ohlcv.side_effect = Exception("unexpected")
        df = fetcher.fetch_ohlcv()
        assert isinstance(df, pd.DataFrame)
        assert fetcher.using_demo is True


# ---------------------------------------------------------------------------
# fetch_ticker
# ---------------------------------------------------------------------------


class TestFetchTicker:
    def test_success(self, fetcher):
        result = fetcher.fetch_ticker()
        assert result["last"] == 42700

    def test_fallback_simulated(self, fetcher):
        fetcher.exchange.fetch_ticker.side_effect = Exception("fail")
        result = fetcher.fetch_ticker()
        assert "last" in result
        assert "bid" in result
        assert "ask" in result

    def test_fallback_structure(self, fetcher):
        fetcher.exchange.fetch_ticker.side_effect = Exception("fail")
        result = fetcher.fetch_ticker()
        # bid < last < ask (0.9999/1.0001 spread)
        assert result["bid"] < result["last"] < result["ask"]


# ---------------------------------------------------------------------------
# fetch_order_book
# ---------------------------------------------------------------------------


class TestFetchOrderBook:
    def test_success(self, fetcher):
        result = fetcher.fetch_order_book()
        assert result["bid"] == 42699
        assert result["ask"] == 42701
        assert result["spread"] == 2  # 42701 - 42699
        assert result["bid_volume"] == 3.5
        assert result["ask_volume"] == 4.0

    def test_error_returns_empty(self, fetcher):
        fetcher.exchange.fetch_order_book.side_effect = Exception("fail")
        result = fetcher.fetch_order_book()
        assert result == {}

    def test_empty_bids_handling(self, fetcher):
        fetcher.exchange.fetch_order_book.return_value = {
            "bids": [],
            "asks": [[42701, 1.0]],
        }
        result = fetcher.fetch_order_book()
        assert result["bid"] is None
        assert result["spread"] is None


# ---------------------------------------------------------------------------
# get_available_pairs
# ---------------------------------------------------------------------------


class TestGetAvailablePairs:
    def test_returns_market_list(self, fetcher):
        pairs = fetcher.get_available_pairs()
        assert isinstance(pairs, list)
        assert "BTC/USDT" in pairs
        assert len(pairs) == 3

    def test_error_returns_empty_list(self, fetcher):
        fetcher.exchange.load_markets.side_effect = Exception("fail")
        pairs = fetcher.get_available_pairs()
        assert pairs == []


# ---------------------------------------------------------------------------
# _generate_demo_data
# ---------------------------------------------------------------------------


class TestGenerateDemoData:
    def test_returns_valid_dataframe(self, fetcher):
        df = fetcher._generate_demo_data()
        assert isinstance(df, pd.DataFrame)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_timeframe_mapping(self, fetcher):
        df = fetcher._generate_demo_data(periods=50, timeframe="4h")
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=4)).all()


# ---------------------------------------------------------------------------
# _init_exchange
# ---------------------------------------------------------------------------


class TestInitExchange:
    def test_paper_mode_no_keys(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "paper")
        monkeypatch.setattr(Config, "API_KEY", "")
        monkeypatch.setattr(Config, "API_SECRET", "")
        mock_class = MagicMock()
        monkeypatch.setattr(ccxt, Config.EXCHANGE_ID, mock_class)
        DataFetcher()
        call_args = mock_class.call_args[0][0]
        assert "apiKey" not in call_args

    def test_live_mode_with_keys(self, monkeypatch):
        monkeypatch.setattr(Config, "TRADING_MODE", "live")
        monkeypatch.setattr(Config, "API_KEY", "test_key")
        monkeypatch.setattr(Config, "API_SECRET", "test_secret")
        # v9.0: Must clear Ed25519 key path — it takes priority over API_SECRET
        monkeypatch.setattr(Config, "API_PRIVATE_KEY_PATH", "")
        mock_class = MagicMock()
        monkeypatch.setattr(ccxt, Config.EXCHANGE_ID, mock_class)
        DataFetcher()
        call_args = mock_class.call_args[0][0]
        assert call_args["apiKey"] == "test_key"
        assert call_args["secret"] == "test_secret"
