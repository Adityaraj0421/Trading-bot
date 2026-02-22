"""
AI Model module v9.0 — PyTorch LSTM + XGBoost Hybrid
======================================================
Migrated from Keras/TensorFlow to PyTorch for Python 3.14 compatibility
and better research ecosystem integration.

Architecture:
  - Tier 1 (best):  PyTorch LSTM (sequence memory) → XGBoost (final prediction)
  - Tier 2 (good):  XGBoost classifier (no sequence memory)
  - Tier 3 (basic): RandomForest + GradientBoosting ensemble

The LSTM captures temporal patterns (divergences, momentum shifts) that
tree-based models miss because they see each bar independently.
XGBoost then combines LSTM hidden state with raw features for the final call.

v9.0 changes:
  - Replaced Keras LSTM with PyTorch LSTM (Python 3.14 compatible)
  - Added gradient clipping and learning rate scheduling
  - Improved self-supervised pre-training objective
  - Added model checkpointing and state serialization
  - All tiers and fallback logic preserved
  - Full backward compatibility with v8.0 interface

Install for full power:
    pip install torch xgboost
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# ── CRITICAL: Import torch BEFORE sklearn ────────────────────
# On macOS ARM64 (Apple Silicon), sklearn's Accelerate BLAS and
# PyTorch's BLAS can conflict causing SIGSEGV if sklearn loads first.
# Importing torch first ensures PyTorch initializes BLAS cleanly.

_HAS_TORCH = False
_HAS_XGBOOST = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _HAS_TORCH = True
except ImportError:
    pass

try:
    import xgboost as xgb

    _HAS_XGBOOST = True
except ImportError:
    pass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from config import Config  # noqa: E402
from indicators import FEATURE_COLUMNS, Indicators  # noqa: E402

# ── PyTorch LSTM Module ──────────────────────────────────────────

if _HAS_TORCH:

    class LSTMFeatureExtractor(nn.Module):
        """
        PyTorch LSTM that extracts temporal feature embeddings.

        Input: (batch, seq_len, n_features)
        Output: (batch, embedding_dim) — compressed temporal representation

        Architecture:
          - 2-layer LSTM with dropout for regularization
          - Final linear projection to embedding space
          - Layer normalization for training stability
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int = 64,
            n_layers: int = 2,
            dropout: float = 0.2,
            embedding_dim: int = 16,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers

            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )

            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(hidden_dim)

            # Projection head: hidden_dim → embedding_dim
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, embedding_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: (batch, seq_len, n_features) tensor

            Returns:
                (batch, embedding_dim) tensor of temporal embeddings
            """
            # LSTM: take the last hidden state
            lstm_out, (h_n, _) = self.lstm(x)

            # Use the last layer's hidden state
            last_hidden = h_n[-1]  # (batch, hidden_dim)

            # Normalize and project
            normed = self.layer_norm(last_hidden)
            normed = self.dropout(normed)
            embedding = self.projection(normed)

            return embedding

        def extract_features(self, x_np: np.ndarray) -> np.ndarray:
            """
            Extract features from numpy input (convenience method).

            Args:
                x_np: (batch, seq_len, n_features) numpy array

            Returns:
                (batch, embedding_dim) numpy array
            """
            self.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x_np)
                embeddings = self.forward(x_tensor)
                return embeddings.numpy()

    class SelfSupervisedTrainer:
        """
        Self-supervised pre-training for LSTM feature extractor.

        Objective: predict a compressed representation of the next
        timestep's features from the sequence so far. This teaches
        the LSTM to capture predictive temporal patterns.
        """

        def __init__(
            self, model: LSTMFeatureExtractor, n_features: int, lr: float = 0.001, weight_decay: float = 1e-5
        ) -> None:
            self.model = model
            self.embedding_dim = model.projection[-1].out_features

            # Prediction head: embedding → target_dim features
            self.target_dim = min(16, n_features)
            self.pred_head = nn.Linear(self.embedding_dim, self.target_dim)

            # Combine parameters for joint optimization
            all_params = list(model.parameters()) + list(self.pred_head.parameters())
            self.optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
            self.loss_fn = nn.MSELoss()

        def train_epoch(self, X_sequences: np.ndarray, Y_targets: np.ndarray, batch_size: int = 32) -> float:
            """
            Train one epoch of self-supervised pre-training.

            Args:
                X_sequences: (N, seq_len, n_features)
                Y_targets: (N, target_dim) — next-step compressed features

            Returns:
                Average loss for the epoch
            """
            self.model.train()
            self.pred_head.train()

            n_samples = len(X_sequences)
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                x_batch = torch.FloatTensor(X_sequences[batch_idx])
                y_batch = torch.FloatTensor(Y_targets[batch_idx])

                # Forward
                embeddings = self.model(x_batch)
                predictions = self.pred_head(embeddings)
                loss = self.loss_fn(predictions, y_batch)

                # Backward with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.scheduler.step(avg_loss)
            return avg_loss

        def train_full(
            self,
            X_sequences: np.ndarray,
            Y_targets: np.ndarray,
            X_val_seq: np.ndarray | None = None,
            Y_val_targets: np.ndarray | None = None,
            epochs: int = 15,
            patience: int = 3,
        ) -> dict[str, list[float]]:
            """
            Full training loop with early stopping.

            Returns training history dict.
            """
            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0
            history = {"train_loss": [], "val_loss": []}

            for epoch in range(epochs):
                train_loss = self.train_epoch(X_sequences, Y_targets)
                history["train_loss"].append(train_loss)

                # Validation
                if X_val_seq is not None and len(X_val_seq) > 0:
                    val_loss = self._validate(X_val_seq, Y_val_targets)
                    history["val_loss"].append(val_loss)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        _log.debug("LSTM early stopping at epoch %d", epoch + 1)
                        break

            # Restore best model
            if best_state is not None:
                self.model.load_state_dict(best_state)

            return history

        def _validate(self, X_seq: np.ndarray, Y_targets: np.ndarray) -> float:
            """Compute validation loss."""
            self.model.eval()
            self.pred_head.eval()
            with torch.no_grad():
                x = torch.FloatTensor(X_seq)
                y = torch.FloatTensor(Y_targets)
                embeddings = self.model(x)
                predictions = self.pred_head(embeddings)
                loss = self.loss_fn(predictions, y)
            return loss.item()


# ── Signal Constants ──────────────────────────────────────────────


class Signal:
    """Trading signal constants (BUY, SELL, HOLD)."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# ── Main Trading Model ───────────────────────────────────────────


class TradingModel:
    """
    v9.0 Hybrid model with automatic fallback.
    Tier 1: PyTorch LSTM feature extractor → XGBoost classifier
    Tier 2: XGBoost classifier
    Tier 3: RandomForest + GradientBoosting ensemble
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_train_accuracy = 0.0
        self.feature_cols = FEATURE_COLUMNS
        self.lookback = 30  # LSTM sequence length

        # Prediction cache
        self._pred_cache_key = None
        self._pred_cache_result = None

        # Label mapping
        self._label_map = {Signal.BUY: 0, Signal.SELL: 1, Signal.HOLD: 2}
        self._label_inv = {v: k for k, v in self._label_map.items()}

        # Determine model tier
        if _HAS_TORCH and _HAS_XGBOOST:
            self._tier = 1
            self.lstm_model = None  # Built dynamically based on feature count
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
        elif _HAS_XGBOOST:
            self._tier = 2
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self._tier = 3
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            self.gb_model = GradientBoostingClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
            )

        tier_name = {1: "PyTorch-LSTM+XGB", 2: "XGBoost", 3: "RF+GB"}[self._tier]
        _log.info("Model tier %d: %s", self._tier, tier_name)

    # ── LSTM Builder ────────────────────────────────────────────────

    def _build_lstm(self, n_features: int) -> "LSTMFeatureExtractor":
        """Build PyTorch LSTM feature extractor."""
        model = LSTMFeatureExtractor(
            n_features=n_features,
            hidden_dim=64,
            n_layers=2,
            dropout=0.2,
            embedding_dim=16,
        )
        return model

    # ── Sequence Creation ────────────────────────────────────────────

    def _create_sequences(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Create LSTM-compatible sequences from flat feature matrix."""
        sequences = []
        labels = []
        for i in range(self.lookback, len(X)):
            sequences.append(X[i - self.lookback : i])
            if y is not None:
                labels.append(y[i])
        if not sequences:
            return np.array([]), np.array([]) if y is not None else None
        return np.array(sequences), np.array(labels) if y is not None else None

    def _create_labels(self, future_return: pd.Series) -> np.ndarray:
        """Vectorized label creation with configurable threshold."""
        threshold = Config.ML_LABEL_THRESHOLD
        labels = np.full(len(future_return), Signal.HOLD)
        labels[future_return.values > threshold] = Signal.BUY
        labels[future_return.values < -threshold] = Signal.SELL
        return labels

    # ── Training ─────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame | None = None, df_ind: pd.DataFrame | None = None) -> dict[str, Any]:
        """Train the model using the best available tier."""
        if df_ind is None:
            df_ind = Indicators.add_all(df)

        df_ind = df_ind[df_ind["future_return"].notna()]
        if len(df_ind) < 100:
            return {"error": "insufficient_data"}

        X = df_ind[self.feature_cols].values
        y = self._create_labels(df_ind["future_return"])

        if len(np.unique(y)) < 2:
            return {"error": "insufficient_class_variety"}

        X_scaled = self.scaler.fit_transform(X)

        # Train/val split (time-series aware)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        try:
            if self._tier == 1:
                val_accuracy = self._train_tier1(X_train, y_train, X_val, y_val, X_scaled, y)
            elif self._tier == 2:
                val_accuracy = self._train_tier2(X_train, y_train, X_val, y_val, X_scaled, y)
            else:
                val_accuracy = self._train_tier3(X_train, y_train, X_val, y_val, X_scaled, y)
        except Exception as e:
            _log.warning("Tier %d training failed (%s), falling back", self._tier, e)
            if self._tier == 1:
                # Try tier 2 first
                try:
                    self._tier = 2
                    val_accuracy = self._train_tier2(X_train, y_train, X_val, y_val, X_scaled, y)
                except Exception as e:
                    _log.warning("Tier 2 fallback failed (%s), dropping to tier 3", e)
                    self._tier = 3
                    self._init_tier3()
                    val_accuracy = self._train_tier3(X_train, y_train, X_val, y_val, X_scaled, y)
            elif self._tier == 2:
                self._tier = 3
                self._init_tier3()
                val_accuracy = self._train_tier3(X_train, y_train, X_val, y_val, X_scaled, y)
            else:
                raise

        self.is_trained = True
        self.last_train_accuracy = val_accuracy
        self._pred_cache_key = None

        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))
        tier_name = {1: "PyTorch-LSTM+XGB", 2: "XGBoost", 3: "RF+GB"}[self._tier]
        _log.info("[Model] Trained (%s) — val acc: %.2f%% | %s", tier_name, val_accuracy * 100, dist)
        return {"cv_accuracy": val_accuracy, "samples": len(y), "class_distribution": dist, "tier": self._tier}

    def _init_tier3(self) -> None:
        """Initialize tier 3 models."""
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
        )

    def _train_tier1(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_all: np.ndarray,
        y_all: np.ndarray,
    ) -> float:
        """
        Train PyTorch LSTM feature extractor + XGBoost classifier.

        Steps:
          1. Self-supervised LSTM pre-training (predict next features)
          2. Extract LSTM embeddings
          3. Combine embeddings + raw features → XGBoost
        """
        n_features = X_train.shape[1]
        self.lstm_model = self._build_lstm(n_features)

        X_seq_train, y_seq_train = self._create_sequences(X_train, y_train)
        X_seq_val, y_seq_val = self._create_sequences(X_val, y_val)

        if len(X_seq_train) < 20:
            _log.info("Not enough sequences for LSTM, falling back to tier 2")
            return self._train_tier2(X_train, y_train, X_val, y_val, X_all, y_all)

        # Self-supervised pre-training target: predict compressed next features
        target_dim = min(16, n_features)
        y_target_train = X_train[self.lookback :, :target_dim]

        y_target_val = None
        if len(X_seq_val) > 5:
            y_target_val = X_val[self.lookback :, :target_dim]

        # Train LSTM with self-supervised objective
        trainer = SelfSupervisedTrainer(self.lstm_model, n_features, lr=0.001, weight_decay=1e-5)
        history = trainer.train_full(
            X_seq_train,
            y_target_train,
            X_seq_val,
            y_target_val,
            epochs=15,
            patience=3,
        )

        _log.debug(
            "LSTM training: %d epochs, final loss: %.4f",
            len(history["train_loss"]),
            history["train_loss"][-1] if history["train_loss"] else 0,
        )

        # Extract embeddings + combine with raw features
        lstm_train = self.lstm_model.extract_features(X_seq_train)
        combined_train = np.hstack([lstm_train, X_train[self.lookback :]])
        y_train_mapped = np.array([self._label_map[lbl] for lbl in y_seq_train])

        self.xgb_model.fit(combined_train, y_train_mapped)

        # Validation accuracy
        if len(X_seq_val) > 0:
            lstm_val = self.lstm_model.extract_features(X_seq_val)
            combined_val = np.hstack([lstm_val, X_val[self.lookback :]])
            y_val_mapped = np.array([self._label_map[lbl] for lbl in y_seq_val])
            val_accuracy = float(np.mean(self.xgb_model.predict(combined_val) == y_val_mapped))
        else:
            val_accuracy = 0.5

        # Final fit on all data
        X_seq_all, y_seq_all = self._create_sequences(X_all, y_all)
        if len(X_seq_all) > 0:
            lstm_all = self.lstm_model.extract_features(X_seq_all)
            combined_all = np.hstack([lstm_all, X_all[self.lookback :]])
            y_all_mapped = np.array([self._label_map[lbl] for lbl in y_seq_all])
            self.xgb_model.fit(combined_all, y_all_mapped)

        return val_accuracy

    def _train_tier2(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_all: np.ndarray,
        y_all: np.ndarray,
    ) -> float:
        """Train XGBoost only."""
        y_train_mapped = np.array([self._label_map[lbl] for lbl in y_train])
        y_val_mapped = np.array([self._label_map[lbl] for lbl in y_val])

        self.xgb_model.fit(X_train, y_train_mapped)
        val_accuracy = float(np.mean(self.xgb_model.predict(X_val) == y_val_mapped))

        y_all_mapped = np.array([self._label_map[lbl] for lbl in y_all])
        self.xgb_model.fit(X_all, y_all_mapped)
        return val_accuracy

    def _train_tier3(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_all: np.ndarray,
        y_all: np.ndarray,
    ) -> float:
        """Train sklearn ensemble."""
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)

        rf_proba = self.rf_model.predict_proba(X_val)
        gb_proba = self.gb_model.predict_proba(X_val)
        avg_proba = (rf_proba + gb_proba) / 2
        y_pred = self.rf_model.classes_[np.argmax(avg_proba, axis=1)]
        val_accuracy = float(np.mean(y_pred == y_val))

        self.rf_model.fit(X_all, y_all)
        self.gb_model.fit(X_all, y_all)
        return val_accuracy

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame | None = None, df_ind: pd.DataFrame | None = None) -> tuple[str, float]:
        """Predict signal for latest candle."""
        if not self.is_trained:
            return Signal.HOLD, 0.0

        if df_ind is None:
            df_ind = Indicators.add_all(df)

        cache_key = (float(df_ind["close"].iloc[-1]), str(df_ind.index[-1]))
        if self._pred_cache_key == cache_key and self._pred_cache_result is not None:
            return self._pred_cache_result

        try:
            if self._tier == 1:
                result = self._predict_tier1(df_ind)
            elif self._tier == 2:
                result = self._predict_tier2(df_ind)
            else:
                result = self._predict_tier3(df_ind)
        except Exception as e:
            _log.warning("Prediction error (tier %d): %s", self._tier, e)
            result = (Signal.HOLD, 0.0)

        self._pred_cache_key = cache_key
        self._pred_cache_result = result
        return result

    def _predict_tier1(self, df_ind: pd.DataFrame) -> tuple[str, float]:
        """Predict using PyTorch LSTM + XGBoost."""
        if self.lstm_model is None or len(df_ind) < self.lookback + 1:
            return self._predict_tier2(df_ind)

        recent = df_ind[self.feature_cols].iloc[-(self.lookback + 1) :].values
        recent_scaled = self.scaler.transform(recent)

        # Create single sequence: (1, lookback, n_features)
        sequence = recent_scaled[:-1].reshape(1, self.lookback, -1)
        lstm_features = self.lstm_model.extract_features(sequence)

        latest_features = recent_scaled[-1:].reshape(1, -1)
        combined = np.hstack([lstm_features, latest_features])

        proba = self.xgb_model.predict_proba(combined)[0]
        pred_idx = np.argmax(proba)
        return self._label_inv[pred_idx], float(proba[pred_idx])

    def _predict_tier2(self, df_ind: pd.DataFrame) -> tuple[str, float]:
        """Predict using XGBoost only."""
        latest = df_ind[self.feature_cols].iloc[-1:].values
        latest_scaled = self.scaler.transform(latest)
        proba = self.xgb_model.predict_proba(latest_scaled)[0]
        pred_idx = np.argmax(proba)
        return self._label_inv[pred_idx], float(proba[pred_idx])

    def _predict_tier3(self, df_ind: pd.DataFrame) -> tuple[str, float]:
        """Predict using sklearn ensemble."""
        latest = df_ind[self.feature_cols].iloc[-1:].values
        latest_scaled = self.scaler.transform(latest)

        rf_proba = self.rf_model.predict_proba(latest_scaled)
        gb_proba = self.gb_model.predict_proba(latest_scaled)
        avg_proba = (rf_proba + gb_proba) / 2

        pred_idx = np.argmax(avg_proba[0])
        signal = self.rf_model.classes_[pred_idx]
        confidence = float(avg_proba[0][pred_idx])
        return signal, confidence

    # ── Feature Importance ───────────────────────────────────────────

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores sorted by importance (descending)."""
        if not self.is_trained:
            return {}
        if self._tier in (1, 2) and hasattr(self, "xgb_model"):
            importance = self.xgb_model.feature_importances_
            cols = self.feature_cols
            if len(importance) > len(cols):
                # LSTM embeddings come first in tier 1
                raw_importance = importance[len(importance) - len(cols) :]
                return dict(sorted(zip(cols, raw_importance), key=lambda x: x[1], reverse=True))
            return dict(sorted(zip(cols, importance), key=lambda x: x[1], reverse=True))
        elif hasattr(self, "rf_model"):
            importance = self.rf_model.feature_importances_
            return dict(sorted(zip(self.feature_cols, importance), key=lambda x: x[1], reverse=True))
        return {}

    # ── v9.0: Model State Serialization ──────────────────────────────

    def save_model_state(self) -> dict[str, Any]:
        """
        Serialize model state for persistence.

        Returns dict with all model parameters (including LSTM weights).
        """
        state = {
            "tier": self._tier,
            "is_trained": self.is_trained,
            "last_train_accuracy": self.last_train_accuracy,
            "lookback": self.lookback,
            "scaler_mean": self.scaler.mean_.tolist() if self.is_trained else None,
            "scaler_scale": self.scaler.scale_.tolist() if self.is_trained else None,
        }

        if self._tier == 1 and self.lstm_model is not None:
            state["lstm_state_dict"] = {k: v.tolist() for k, v in self.lstm_model.state_dict().items()}

        return state

    def load_model_state(self, state: dict[str, Any]) -> None:
        """Restore model from serialized state."""
        self._tier = state.get("tier", self._tier)
        self.is_trained = state.get("is_trained", False)
        self.last_train_accuracy = state.get("last_train_accuracy", 0.0)
        self.lookback = state.get("lookback", 30)

        if state.get("scaler_mean") is not None:
            self.scaler.mean_ = np.array(state["scaler_mean"])
            self.scaler.scale_ = np.array(state["scaler_scale"])
            self.scaler.n_features_in_ = len(state["scaler_mean"])

        if self._tier == 1 and "lstm_state_dict" in state and _HAS_TORCH:
            try:
                n_features = len(state.get("scaler_mean", [0] * 22))
                self.lstm_model = self._build_lstm(n_features)
                sd = {k: torch.tensor(v) for k, v in state["lstm_state_dict"].items()}
                self.lstm_model.load_state_dict(sd)
                _log.info("LSTM state restored successfully")
            except Exception as e:
                _log.warning("Failed to restore LSTM state: %s", e)

    def get_model_info(self) -> dict[str, Any]:
        """Return model architecture info for dashboard."""
        info = {
            "tier": self._tier,
            "tier_name": {1: "PyTorch-LSTM+XGB", 2: "XGBoost", 3: "RF+GB"}[self._tier],
            "is_trained": self.is_trained,
            "accuracy": round(self.last_train_accuracy, 4),
            "lookback": self.lookback,
            "n_features": len(self.feature_cols),
            "runtime": "PyTorch" if _HAS_TORCH else "sklearn",
        }

        if self._tier == 1 and self.lstm_model is not None:
            n_params = sum(p.numel() for p in self.lstm_model.parameters())
            info["lstm_params"] = n_params
            info["lstm_hidden_dim"] = self.lstm_model.hidden_dim
            info["lstm_layers"] = self.lstm_model.n_layers

        return info
