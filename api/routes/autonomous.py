"""
Autonomous mode routes — status, events, kill switch, manual override.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import Optional

from api.data_store import DataStore


class HaltRequest(BaseModel):
    reason: Optional[str] = Field(
        default="Manual kill switch via API",
        max_length=500,
        description="Reason for halting (stored in event log)",
    )


def create_router(store: DataStore) -> APIRouter:
    router = APIRouter(prefix="/autonomous", tags=["autonomous"])

    @router.get("/status")
    def autonomous_status():
        snapshot = store.get_snapshot()
        autonomous = snapshot.get("autonomous", {})
        if not autonomous:
            return {"status": "not_running", "autonomous": {}}
        return autonomous

    @router.get("/events")
    def autonomous_events(limit: int = Query(default=50, ge=1)):
        events = store.get_events(limit=limit)
        return {"events": events, "count": len(events)}

    # --- Production safeguards ---

    @router.post("/halt")
    def emergency_halt(req: HaltRequest):
        """Kill switch — immediately halt all trading."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"status": "error", "message": "Agent not running"}
        decision.emergency_halt(req.reason)
        return {"status": "halted", "reason": req.reason}

    @router.post("/resume")
    def emergency_resume():
        """Resume trading after a kill switch halt."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"status": "error", "message": "Agent not running"}
        decision.emergency_resume()
        return {"status": "resumed"}

    @router.post("/force-close")
    def force_close_all():
        """Force close all open positions immediately."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"status": "error", "message": "Agent not running"}
        decision.force_close_all_positions()
        return {"status": "force_close_signaled"}

    @router.get("/alerts")
    def get_alerts(unacknowledged: bool = False):
        """Get alerts from the decision engine."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"alerts": [], "count": 0}
        alerts = decision.get_alerts(unacknowledged_only=unacknowledged)
        return {"alerts": alerts, "count": len(alerts)}

    @router.post("/alerts/acknowledge")
    def acknowledge_alerts():
        """Acknowledge all alerts."""
        decision = store.get_decision_engine()
        if decision is None:
            return {"status": "error"}
        decision.acknowledge_alerts()
        return {"status": "acknowledged"}

    return router
