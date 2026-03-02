"""Model feature importance API routes.

Endpoints:
    GET /model/feature-importance — XGBoost feature importances sorted
    descending, plus metadata about the current model state.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.data_store import DataStore


def create_router(store: DataStore) -> APIRouter:
    """Create the model router.

    Args:
        store: The shared DataStore instance used by the agent and API.

    Returns:
        Configured ``APIRouter`` with prefix ``/model``.
    """
    router = APIRouter(prefix="/model", tags=["model"])

    @router.get("/feature-importance")
    async def get_feature_importance() -> Any:
        """Return current XGBoost feature importances sorted descending.

        Returns the live model's feature importance scores along with
        metadata about feature columns, model tier, and training status.
        Useful for diagnosing which signals drive predictions and for
        deciding pruning thresholds.

        Returns:
            JSON response with ``status``, ``tier``, ``n_features``,
            ``feature_cols``, ``feature_importance`` (sorted descending),
            and ``top_features`` (top 14 by importance).  Returns 503
            when the model is not available.
        """
        trading_model = store.get_model()
        if trading_model is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "message": "Model not available — agent not running or DataStore not wired",
                },
            )

        if not trading_model.is_trained:
            return {
                "status": "not_trained",
                "message": "Model has not been trained yet. Wait for the first training cycle.",
                "feature_importance": {},
                "top_features": [],
                "n_features": len(trading_model.feature_cols),
                "feature_cols": trading_model.feature_cols,
            }

        importance = trading_model.get_feature_importance()
        top_features = list(importance.keys())[:14]

        return {
            "status": "ok",
            "tier": trading_model._tier,
            "n_features": len(trading_model.feature_cols),
            "feature_cols": trading_model.feature_cols,
            "feature_importance": importance,
            "top_features": top_features,
        }

    return router
