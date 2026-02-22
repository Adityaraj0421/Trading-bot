"""
Autonomous mode routes — status, events, kill switch, manual override.
"""

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from api.data_store import DataStore


class HaltRequest(BaseModel):
    reason: str | None = Field(
        default="Manual kill switch via API",
        max_length=500,
        description="Reason for halting (stored in event log)",
    )


def create_router(store: DataStore) -> APIRouter:
    """Create autonomous-mode routes (status, events, kill switch, alerts)."""
    router = APIRouter(prefix="/autonomous", tags=["autonomous"])

    @router.get("/status")
    def autonomous_status() -> dict[str, Any]:
        """Return autonomous trading mode status and state."""
        snapshot = store.get_snapshot()
        autonomous = snapshot.get("autonomous", {})
        if not autonomous:
            return {"status": "not_running", "autonomous": {}}
        return autonomous

    @router.get("/events")
    def autonomous_events(limit: int = Query(default=50, ge=1)) -> dict[str, Any]:
        """Return recent autonomous-mode events."""
        events = store.get_events(limit=limit)
        return {"events": events, "count": len(events)}

    # --- Production safeguards ---

    @router.post("/halt")
    def emergency_halt(req: HaltRequest) -> dict[str, Any]:
        """Kill switch — immediately halt all trading."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"status": "error", "message": "Agent not running"}
        decision.emergency_halt(req.reason)
        return {"status": "halted", "reason": req.reason}

    @router.post("/resume")
    def emergency_resume() -> dict[str, Any]:
        """Resume trading after a kill switch halt."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"status": "error", "message": "Agent not running"}
        decision.emergency_resume()
        return {"status": "resumed"}

    @router.post("/force-close")
    def force_close_all() -> dict[str, Any]:
        """Force close all open positions immediately."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"status": "error", "message": "Agent not running"}
        decision.force_close_all_positions()
        return {"status": "force_close_signaled"}

    @router.get("/alerts")
    def get_alerts(unacknowledged: bool = False) -> dict[str, Any]:
        """Get alerts from the decision engine."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"alerts": [], "count": 0}
        alerts = decision.get_alerts(unacknowledged_only=unacknowledged)
        return {"alerts": alerts, "count": len(alerts)}

    @router.post("/alerts/acknowledge")
    def acknowledge_alerts() -> dict[str, Any]:
        """Acknowledge all alerts."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"status": "error"}
        decision.acknowledge_alerts()
        return {"status": "acknowledged"}

    return router
