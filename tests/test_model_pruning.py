"""Tests for TradingModel feature pruning methods and the /model/feature-importance endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from indicators import FEATURE_COLUMNS
from model import TradingModel


def _make_trained_model() -> TradingModel:
    """Create a mock TradingModel that appears trained with fake importances."""
    m = TradingModel()
    m.is_trained = True
    m._tier = 3
    m.feature_cols = list(FEATURE_COLUMNS)
    # Mock get_feature_importance to return sorted dict
    importance = {col: float(i) / len(FEATURE_COLUMNS) for i, col in enumerate(reversed(FEATURE_COLUMNS))}
    m.get_feature_importance = MagicMock(return_value=importance)
    return m


class TestGetTopFeatures:
    """Tests for TradingModel.get_top_features()."""

    def test_returns_empty_list_when_untrained(self) -> None:
        m = TradingModel()
        assert m.get_top_features() == []

    def test_returns_k_features_after_training(self) -> None:
        m = _make_trained_model()
        result = m.get_top_features(k=5)
        assert len(result) == 5

    def test_all_returned_features_are_valid_feature_columns(self) -> None:
        m = _make_trained_model()
        result = m.get_top_features(k=10)
        for feat in result:
            assert feat in FEATURE_COLUMNS

    def test_k_clamped_to_feature_count(self) -> None:
        m = _make_trained_model()
        result = m.get_top_features(k=9999)
        assert len(result) == len(FEATURE_COLUMNS)

    def test_default_k_is_14(self) -> None:
        m = _make_trained_model()
        result = m.get_top_features()
        assert len(result) == 14

    def test_features_sorted_by_importance_descending(self) -> None:
        m = _make_trained_model()
        result = m.get_top_features(k=5)
        importance = m.get_feature_importance()
        expected = list(importance.keys())[:5]
        assert result == expected


class TestPruneAndRetrain:
    """Tests for TradingModel.prune_and_retrain()."""

    def _minimal_df(self, n: int = 200) -> pd.DataFrame:
        """Build a minimal DataFrame with all FEATURE_COLUMNS."""
        data = {col: np.random.randn(n) for col in FEATURE_COLUMNS}
        data["close"] = np.random.uniform(100, 200, n)
        data["open"] = data["close"] * np.random.uniform(0.99, 1.01, n)
        data["high"] = data["close"] * np.random.uniform(1.0, 1.02, n)
        data["low"] = data["close"] * np.random.uniform(0.98, 1.0, n)
        data["volume"] = np.random.uniform(1e6, 1e7, n)
        return pd.DataFrame(data)

    def test_returns_error_when_untrained(self) -> None:
        m = TradingModel()
        df = self._minimal_df()
        result = m.prune_and_retrain(df, k=10)
        assert "error" in result
        assert result["error"] == "not_trained"

    def test_k_minimum_floor_is_5(self) -> None:
        m = _make_trained_model()
        m.train = MagicMock(return_value={"cv_accuracy": 0.55, "samples": 100})
        m.prune_and_retrain(m.feature_cols, k=2)  # k=2 should be floored to 5
        assert len(m.feature_cols) >= 5

    def test_prediction_cache_invalidated(self) -> None:
        m = _make_trained_model()
        m._pred_cache_key = "some_stale_key"
        m._pred_cache_result = {"signal": "BUY"}
        m.train = MagicMock(return_value={"cv_accuracy": 0.55, "samples": 100})
        m.prune_and_retrain(None, k=10)
        assert m._pred_cache_key is None
        assert m._pred_cache_result is None

    def test_returns_error_on_train_failure(self) -> None:
        m = _make_trained_model()
        m.train = MagicMock(side_effect=RuntimeError("train failed"))
        result = m.prune_and_retrain(None, k=10)
        assert "error" in result

    def test_original_features_restored_on_failure(self) -> None:
        m = _make_trained_model()
        original = list(m.feature_cols)
        m.train = MagicMock(side_effect=RuntimeError("train failed"))
        m.prune_and_retrain(None, k=10)
        assert m.feature_cols == original

    def test_no_feature_importance_returns_error(self) -> None:
        m = TradingModel()
        m.is_trained = True
        m.get_feature_importance = MagicMock(return_value={})
        result = m.prune_and_retrain(None, k=10)
        assert result == {"error": "no_feature_importance"}


class TestSaveLoadFeatureCols:
    """Tests that feature_cols is persisted in save/load."""

    def test_save_model_state_includes_feature_cols(self) -> None:
        m = TradingModel()
        state = m.save_model_state()
        assert "feature_cols" in state
        assert state["feature_cols"] == list(FEATURE_COLUMNS)

    def test_load_model_state_restores_feature_cols(self) -> None:
        m = TradingModel()
        pruned = FEATURE_COLUMNS[:10]
        state = m.save_model_state()
        state["feature_cols"] = pruned
        m.load_model_state(state)
        assert m.feature_cols == pruned

    def test_load_model_state_defaults_to_feature_columns_when_missing(self) -> None:
        m = TradingModel()
        state = m.save_model_state()
        state.pop("feature_cols", None)
        m.load_model_state(state)
        assert m.feature_cols == list(FEATURE_COLUMNS)


class TestModelAPIEndpoint:
    """Tests for GET /model/feature-importance endpoint."""

    def test_returns_503_when_no_model(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.data_store import DataStore
        from api.routes.model import create_router

        app = FastAPI()
        store = DataStore()
        app.include_router(create_router(store))
        client = TestClient(app)

        response = client.get("/model/feature-importance")
        assert response.status_code == 503

    def test_returns_not_trained_status(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.data_store import DataStore
        from api.routes.model import create_router

        app = FastAPI()
        store = DataStore()
        mock_model = MagicMock()
        mock_model.is_trained = False
        mock_model.feature_cols = list(FEATURE_COLUMNS)
        store.set_model(mock_model)
        app.include_router(create_router(store))
        client = TestClient(app)

        response = client.get("/model/feature-importance")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_trained"

    def test_returns_feature_importance_when_trained(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from api.data_store import DataStore
        from api.routes.model import create_router

        app = FastAPI()
        store = DataStore()
        mock_model = _make_trained_model()
        store.set_model(mock_model)
        app.include_router(create_router(store))
        client = TestClient(app)

        response = client.get("/model/feature-importance")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "feature_importance" in data
        assert "top_features" in data
        assert isinstance(data["top_features"], list)
        assert len(data["top_features"]) <= 14
