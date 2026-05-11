"""
Bearer-token auth.

Flow:
1. First launch, Android calls POST /devices/register (no auth) — server mints
   a uuid4 user_id and a 32-byte url-safe token, stores them, returns both.
2. Android persists both, sends `Authorization: Bearer <token>` on every
   /sessions/* request.
3. `current_user_id` dependency resolves the bearer token to a user_id via
   the `devices` table.
"""
from __future__ import annotations

import secrets
import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .db import Device, get_session

_bearer = HTTPBearer(auto_error=False)


def mint_device(session: Session) -> Device:
    """Create a new (user_id, token) pair and persist it."""
    device = Device(
        token=secrets.token_urlsafe(32),
        user_id=str(uuid.uuid4()),
    )
    session.add(device)
    session.flush()
    return device


def current_user_id(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    session: Session = Depends(get_session),
) -> str:
    """Resolve the bearer token to a user_id. 401 on any failure."""
    if creds is None or (creds.scheme or "").lower() != "bearer" or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    device = session.get(Device, creds.credentials)
    if device is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Stash on request.state so the access log can include it.
    request.state.user_id = device.user_id
    return device.user_id
