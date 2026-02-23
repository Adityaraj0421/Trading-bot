"""
Unit tests for model.py — Signal enum, TradingModel train/predict/cache.

Uses demo_data + Indicators for real training data; tests pure methods directly.
"""

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from demo_data import generate_ohlcv
from indicators import FEATURE_COLUMNS, Indicators
from model import Signal, TradingModel


@pytest.fixture(autouse=True)
def _clear_indicator_cache():
    """Ensure indicator cache is clean before and after each test."""
    Indicators.invalidate_cache()
    yield
    Indicators.invalidate_cache()


@pytest.fixture()
def sample_df():
    return generate_ohlcv(periods=200, seed=42)


@pytest.fixture()
def sample_indicators(sample_df):
    return Indicators.add_all(sample_df)


@pytest.fixture()
def trained_model(sample_df, sample_indicators):
    model = TradingModel()
    model.train(sample_df, df_ind=sample_indicators)
    return model


# ---------------------------------------------------------------------------
# Signal enum
# ---------------------------------------------------------------------------


class TestSignalEnum:
    def test_buy_value(self):
        assert Signal.BUY == "BUY"

    def test_sell_value(self):
        assert Signal.SELL == "SELL"

    def test_hold_value(self):
        assert Signal.HOLD == "HOLD"


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestTradingModelInit:
    def test_tier_is_set(self):
        """v9.0: Model selects tier based on available libraries."""
        model = TradingModel()
        assert model._tier in (1, 2, 3)

    def test_rf_model_only_tier3(self):
        """rf_model only exists in tier 3 (sklearn fallback)."""
        model = TradingModel()
        if model._tier == 3:
            assert isinstance(model.rf_model, RandomForestClassifier)
        else:
            assert not hasattr(model, "rf_model")

    def test_gb_model_only_tier3(self):
        """gb_model only exists in tier 3 (sklearn fallback)."""
        model = TradingModel()
        if model._tier == 3:
            assert isinstance(model.gb_model, GradientBoostingClassifier)
        else:
            assert not hasattr(model, "gb_model")

    def test_scaler_type(self):
        model = TradingModel()
        assert isinstance(model.scaler, StandardScaler)

    def test_not_trained_initially(self):
        model = TradingModel()
        assert model.is_trained is False

    def test_accuracy_zero_initially(self):
        model = TradingModel()
        assert model.last_train_accuracy == 0.0

    def test_feature_cols_match(self):
        model = TradingModel()
        assert model.feature_cols == FEATURE_COLUMNS
        # v9.1: FEATURE_COLUMNS expanded to 24 (added williams_r, cci)
        assert len(model.feature_cols) == 24


# ---------------------------------------------------------------------------
# _create_labels
# ---------------------------------------------------------------------------


class TestCreateLabels:
    def test_positive_return_buy(self):
        model = TradingModel()
        s = pd.Series([0.05])
        labels = model._create_labels(s)
        assert labels[0] == "BUY"

    def test_negative_return_sell(self):
        model = TradingModel()
        s = pd.Series([-0.05])
        labels = model._create_labels(s)
        assert labels[0] == "SELL"

    def test_neutral_return_hold(self):
        model = TradingModel()
        s = pd.Series([0.005])
        labels = model._create_labels(s)
        assert labels[0] == "HOLD"

    def test_mixed_returns(self):
        model = TradingModel()
        s = pd.Series([0.05, -0.05, 0.005, 0.02, -0.02])
        labels = model._create_labels(s)
        assert list(labels) == ["BUY", "SELL", "HOLD", "BUY", "SELL"]

    def test_boundary_exactly_at_threshold_is_hold(self):
        """Exactly ±threshold should be HOLD (strict > and < operators)."""
        from config import Config

        threshold = Config.ML_LABEL_THRESHOLD
        model = TradingModel()
        s = pd.Series([threshold, -threshold])
        labels = model._create_labels(s)
        assert labels[0] == "HOLD"
        assert labels[1] == "HOLD"


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrain:
    def test_successful_training_returns_keys(self, sample_df, sample_indicators):
        model = TradingModel()
        result = model.train(sample_df, df_ind=sample_indicators)
        assert "cv_accuracy" in result
        assert "samples" in result
        assert "class_distribution" in result

    def test_sets_is_trained_true(self, sample_df, sample_indicators):
        model = TradingModel()
        model.train(sample_df, df_ind=sample_indicators)
        assert model.is_trained is True

    def test_insufficient_data_error(self):
        df = generate_ohlcv(periods=50, seed=42)
        df_ind = Indicators.add_all(df)
        model = TradingModel()
        result = model.train(df, df_ind=df_ind)
        assert result == {"error": "insufficient_data"}
        assert model.is_trained is False

    def test_insufficient_class_variety(self):
        """All-constant prices yield only HOLD labels (1 class)."""
        df = generate_ohlcv(periods=200, seed=42, volatility=0.0)
        df_ind = Indicators.add_all(df)
        model = TradingModel()
        result = model.train(df, df_ind=df_ind)
        # Either insufficient_data or insufficient_class_variety
        assert "error" in result
        assert model.is_trained is False

    def test_invalidates_pred_cache(self, sample_df, sample_indicators):
        model = TradingModel()
        model._pred_cache_key = ("fake", "key")
        model.train(sample_df, df_ind=sample_indicators)
        assert model._pred_cache_key is None


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_untrained_returns_hold(self):
        model = TradingModel()
        signal, confidence = model.predict()
        assert signal == Signal.HOLD
        assert confidence == 0.0

    def test_trained_returns_valid_signal(self, trained_model, sample_indicators):
        signal, confidence = trained_model.predict(df_ind=sample_indicators)
        assert signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert 0 < confidence <= 1.0

    def test_cache_hit_same_data(self, trained_model, sample_indicators):
        result1 = trained_model.predict(df_ind=sample_indicators)
        result2 = trained_model.predict(df_ind=sample_indicators)
        assert result1 == result2

    def test_cache_miss_new_candle(self, trained_model, sample_indicators):
        trained_model.predict(df_ind=sample_indicators)
        old_key = trained_model._pred_cache_key

        # Modify the last close to simulate new candle
        modified = sample_indicators.copy()
        modified.iloc[-1, modified.columns.get_loc("close")] += 100.0
        trained_model.predict(df_ind=modified)
        new_key = trained_model._pred_cache_key
        assert old_key != new_key


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------


class TestGetFeatureImportance:
    def test_untrained_empty(self):
        model = TradingModel()
        assert model.get_feature_importance() == {}

    def test_trained_returns_all_features(self, trained_model):
        """v9.1: 24 features (was 22; added williams_r, cci)."""
        imp = trained_model.get_feature_importance()
        assert len(imp) == 24
        # In tier 1 (LSTM+XGB), raw feature importances may not sum to 1.0
        # because LSTM embeddings share the importance budget.
        if trained_model._tier >= 2:
            assert pytest.approx(sum(imp.values()), abs=0.01) == 1.0
        else:
            assert sum(imp.values()) > 0
        # Should be sorted descending
        values = list(imp.values())
        assert values == sorted(values, reverse=True)
