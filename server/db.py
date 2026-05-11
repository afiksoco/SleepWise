"""
SQLAlchemy 2.0 (sync) layer for SleepWise.

Connection target is controlled by the DATABASE_URL env var:
- unset → local SQLite at server/sleepwise.db (dev default)
- postgres://... or postgresql://... → managed Postgres (Railway, etc.)

Railway hands out URLs starting with `postgres://` for legacy reasons;
SQLAlchemy 2.x requires `postgresql://`, so we rewrite the scheme.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from sqlalchemy import Index, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


_DEFAULT_SQLITE_PATH = Path(__file__).parent / "sleepwise.db"


def _resolve_database_url() -> str:
    raw = os.getenv("DATABASE_URL")
    if not raw:
        return f"sqlite:///{_DEFAULT_SQLITE_PATH}"
    if raw.startswith("postgres://"):
        return "postgresql://" + raw[len("postgres://") :]
    return raw


DATABASE_URL = _resolve_database_url()

_engine_kwargs: dict = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Device(Base):
    """Authenticated device → user_id mapping. One row per registered device."""

    __tablename__ = "devices"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=_utcnow_iso)


class SleepSession(Base):
    """One uploaded sleep session. Matches the original sqlite schema 1:1."""

    __tablename__ = "sleep_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[str] = mapped_column(String(32), nullable=False)
    window_end: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[str] = mapped_column(String(32), nullable=False)
    ended_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fired_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fired_reason: Mapped[str | None] = mapped_column(String(16), nullable=True)
    stages_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, default=_utcnow_iso)

    __table_args__ = (Index("idx_sessions_user", "user_id", "started_at"),)


def init_db() -> None:
    if DATABASE_URL.startswith("sqlite"):
        _DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency — yields a session and commits on success."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
