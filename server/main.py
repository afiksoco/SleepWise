"""
SleepWise Backend — storage & analytics service.

Phase 2 scope:
- Server is no longer the inference engine. Sleep-stage prediction happens
  on-device (TFLite GRU in the Android app); the backend's role is to
  persist completed sleep sessions and surface aggregate insights.
- Per-device bearer-token auth (see server/auth.py). user_id is server-
  assigned at /devices/register and resolved from the token on every
  authed request — the client cannot spoof another user's id.
- Storage is SQLAlchemy-backed; DATABASE_URL selects sqlite (local dev) vs
  managed Postgres (Railway).
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .auth import current_user_id, mint_device
from .db import SleepSession, get_session, init_db
from .logging_config import RequestIdMiddleware, configure_logging


# ─── Models ───────────────────────────────────────────────────────────────────

class StageTick(BaseModel):
    t: str = Field(..., description="ISO 8601 timestamp of the tick")
    stage: str = Field(..., description="Predicted stage, e.g. 'Light' or 'Deep'")
    conf: float = Field(..., ge=0.0, le=1.0)
    stable: bool


class SessionUpload(BaseModel):
    # `user_id` is accepted for backward compat with the existing client DTO
    # but is *ignored* — the server overwrites it with the token-resolved id.
    user_id: Optional[str] = None
    window_start: str
    window_end: str
    started_at: str
    ended_at: Optional[str] = None
    fired_at: Optional[str] = None
    fired_reason: Optional[str] = None
    stages: list[StageTick] = []


class SessionRecord(BaseModel):
    id: int
    user_id: str
    window_start: str
    window_end: str
    started_at: str
    ended_at: Optional[str] = None
    fired_at: Optional[str] = None
    fired_reason: Optional[str] = None
    stages: list[StageTick] = []
    created_at: str


class WeeklyReport(BaseModel):
    user_id: str
    sessions: list[SessionRecord]
    fired_count: int
    favorable_count: int
    fallback_count: int
    avg_window_minutes: float


class DeviceRegisterResponse(BaseModel):
    user_id: str
    token: str


# ─── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    init_db()
    logging.getLogger("sleepwise").info("startup_complete")
    yield


app = FastAPI(
    title="SleepWise Backend",
    description="Storage & analytics for SleepWise sleep sessions",
    version="2.1.0",
    lifespan=lifespan,
)
app.add_middleware(RequestIdMiddleware)


@app.get("/")
def health(session: Session = Depends(get_session)) -> dict:
    n = session.execute(select(func.count()).select_from(SleepSession)).scalar_one()
    return {"service": "SleepWise Backend", "status": "running", "sessions_stored": n}


# ─── Auth handshake ───────────────────────────────────────────────────────────

@app.post("/devices/register", response_model=DeviceRegisterResponse, status_code=status.HTTP_201_CREATED)
def register_device(session: Session = Depends(get_session)) -> DeviceRegisterResponse:
    """Mint a new (user_id, token) pair. Unauthenticated — first-launch handshake."""
    device = mint_device(session)
    return DeviceRegisterResponse(user_id=device.user_id, token=device.token)


# ─── Sessions endpoints ───────────────────────────────────────────────────────

@app.post("/sessions", response_model=SessionRecord, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: SessionUpload,
    user_id: str = Depends(current_user_id),
    session: Session = Depends(get_session),
) -> SessionRecord:
    # Token-resolved user_id wins; any value in the request body is ignored.
    row = SleepSession(
        user_id=user_id,
        window_start=payload.window_start,
        window_end=payload.window_end,
        started_at=payload.started_at,
        ended_at=payload.ended_at,
        fired_at=payload.fired_at,
        fired_reason=payload.fired_reason,
        stages_json=json.dumps([s.model_dump() for s in payload.stages]),
    )
    session.add(row)
    session.flush()
    return _row_to_record(row)


@app.get("/sessions/{path_user_id}", response_model=list[SessionRecord])
def list_sessions(
    path_user_id: str,
    limit: int = 50,
    user_id: str = Depends(current_user_id),
    session: Session = Depends(get_session),
) -> list[SessionRecord]:
    _require_owner(path_user_id, user_id)
    rows = session.execute(
        select(SleepSession)
        .where(SleepSession.user_id == user_id)
        .order_by(SleepSession.started_at.desc())
        .limit(limit)
    ).scalars().all()
    return [_row_to_record(r) for r in rows]


@app.get("/sessions/{path_user_id}/weekly", response_model=WeeklyReport)
def weekly_report(
    path_user_id: str,
    user_id: str = Depends(current_user_id),
    session: Session = Depends(get_session),
) -> WeeklyReport:
    _require_owner(path_user_id, user_id)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    rows = session.execute(
        select(SleepSession)
        .where(SleepSession.user_id == user_id, SleepSession.started_at >= cutoff)
        .order_by(SleepSession.started_at.desc())
    ).scalars().all()

    sessions = [_row_to_record(r) for r in rows]
    fired = [s for s in sessions if s.fired_at]
    favorable = [s for s in fired if s.fired_reason == "favorable"]
    fallback = [s for s in fired if s.fired_reason == "fallback"]

    total_minutes = 0.0
    for s in sessions:
        try:
            ws = datetime.fromisoformat(s.window_start.replace("Z", "+00:00"))
            we = datetime.fromisoformat(s.window_end.replace("Z", "+00:00"))
            total_minutes += max(0.0, (we - ws).total_seconds() / 60.0)
        except ValueError:
            pass
    avg_window = (total_minutes / len(sessions)) if sessions else 0.0

    return WeeklyReport(
        user_id=user_id,
        sessions=sessions,
        fired_count=len(fired),
        favorable_count=len(favorable),
        fallback_count=len(fallback),
        avg_window_minutes=avg_window,
    )


@app.delete("/sessions/{path_user_id}")
def clear_user(
    path_user_id: str,
    user_id: str = Depends(current_user_id),
    session: Session = Depends(get_session),
) -> dict:
    _require_owner(path_user_id, user_id)
    result = session.execute(
        SleepSession.__table__.delete().where(SleepSession.user_id == user_id)
    )
    return {"deleted": result.rowcount or 0, "user_id": user_id}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_owner(path_user_id: str, authed_user_id: str) -> None:
    """Reject access to other users' data — even with a valid token."""
    if path_user_id != authed_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")


def _row_to_record(row: SleepSession) -> SessionRecord:
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    stages = [StageTick(**s) for s in json.loads(row.stages_json)]
    return SessionRecord(
        id=row.id,
        user_id=row.user_id,
        window_start=row.window_start,
        window_end=row.window_end,
        started_at=row.started_at,
        ended_at=row.ended_at,
        fired_at=row.fired_at,
        fired_reason=row.fired_reason,
        stages=stages,
        created_at=row.created_at,
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "5000"))
    print("=" * 56)
    print("  SleepWise Backend — storage & analytics")
    print("=" * 56)
    print("\n  Endpoints:")
    print("    GET    /                              health + count")
    print("    POST   /devices/register              mint (user_id, token)")
    print("    POST   /sessions                      upload a session [auth]")
    print("    GET    /sessions/{user_id}            list sessions   [auth]")
    print("    GET    /sessions/{user_id}/weekly     7-day rollup    [auth]")
    print("    DELETE /sessions/{user_id}            wipe user data  [auth]")
    print(f"\n  API docs: http://localhost:{port}/docs")
    print("=" * 56 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
