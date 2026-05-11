"""
JSON access logging + X-Request-ID middleware.

Stdlib only — no python-json-logger dep. Every HTTP request emits one log
line on completion with: ts, method, path, status, latency_ms, request_id,
and user_id (when the auth dep ran successfully).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


_BASE_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record. Any `extra={...}` keys are merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _BASE_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Replace any existing handlers with a single JSON handler on stdout."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    # Uvicorn's own loggers default to plain text; route them through ours.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(noisy)
        logger.handlers.clear()
        logger.propagate = True
    # The default uvicorn.access logger duplicates what our middleware emits.
    logging.getLogger("uvicorn.access").disabled = True


access_logger = logging.getLogger("sleepwise.access")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Generate/echo X-Request-ID and emit a structured access log."""

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
            access_logger.exception(
                "request_failed",
                extra={
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": latency_ms,
                    "user_id": getattr(request.state, "user_id", None),
                },
            )
            raise
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        response.headers["X-Request-ID"] = rid
        access_logger.info(
            "request",
            extra={
                "request_id": rid,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
                "user_id": getattr(request.state, "user_id", None),
            },
        )
        return response
