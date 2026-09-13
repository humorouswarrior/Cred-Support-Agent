"""Task 12 - structured JSON-lines request logging.

One JSON object per line, per request, with a trace id and timing metadata.

Exactly one entry per request, however the request arrives. The HTTP middleware
opens a request log for every HTTP call; the service layer, when it runs inside
that request, enriches the same entry instead of writing a second one. Callers
with no HTTP request around them (WebSocket turns, the terminal chat, scripts)
get an entry opened by the service layer itself. The
request text written to the log is the *masked* text - the same masking applied
to the model-visible input - so a PAN, Aadhaar or bank account number a customer
pastes into chat never lands on disk.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

from cred_support_agent.config import LOG_PATH
from cred_support_agent.safety.guardrails import mask_pii

_WRITE_LOCK = threading.Lock()


def new_trace_id() -> str:
    return f"trc_{uuid.uuid4().hex[:16]}"


@dataclass
class RequestLog:
    """Accumulates one request's log entry; written once on exit."""

    trace_id: str
    endpoint: str
    session_id: str = "default"
    started_at: float = field(default_factory=time.time)
    _t0: float = field(default_factory=time.perf_counter)
    fields: Dict[str, Any] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    _stage_t0: float = field(default_factory=time.perf_counter)

    def set(self, **kwargs: Any) -> None:
        self.fields.update(kwargs)

    def set_request_text(self, raw_text: str) -> None:
        """Store only the masked form of the request text. Never the raw form."""
        masked = mask_pii(raw_text)
        self.fields["request_text"] = masked.text
        self.fields["request_chars"] = len(raw_text)
        self.fields["pii_masked"] = masked.findings

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.stages.append({"stage": stage, "ms": round((now - self._stage_t0) * 1000, 3)})
        self._stage_t0 = now

    def to_entry(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.started_at)) + "Z",
            "endpoint": self.endpoint,
            "session_id": self.session_id,
            "latency_ms": round((time.perf_counter() - self._t0) * 1000, 3),
            "stages": self.stages,
        }
        entry.update(self.fields)
        return entry


def write_entry(entry: Dict[str, Any]) -> None:
    """Append one JSON-lines record. Never raises into the request path."""
    line = json.dumps(entry, default=str, ensure_ascii=False)
    try:
        with _WRITE_LOCK:
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError:  # pragma: no cover - logging must never break a request
        pass


_active_log: contextvars.ContextVar[RequestLog | None] = contextvars.ContextVar(
    "cred_active_request_log", default=None
)


def current_log() -> RequestLog | None:
    return _active_log.get()


@contextmanager
def request_log(endpoint: str, session_id: str = "default", trace_id: str | None = None) -> Iterator[RequestLog]:
    """Log exactly one JSON-lines entry for this request, success or failure.

    If a request log is already open in this context (an HTTP request), the same
    entry is yielded and enriched, and the outer owner writes it. Otherwise a new
    entry is opened here and written on exit.
    """
    existing = _active_log.get()
    if existing is not None:
        existing.set(operation=endpoint)
        if session_id != "default":
            existing.session_id = session_id
        try:
            yield existing
        except Exception as exc:
            existing.set(error_type=type(exc).__name__, error=str(exc)[:500])
            raise
        return

    log = RequestLog(trace_id=trace_id or new_trace_id(), endpoint=endpoint, session_id=session_id)
    token = _active_log.set(log)
    try:
        yield log
    except Exception as exc:
        # Keep a more specific status (e.g. "rejected") if the caller set one.
        log.fields.setdefault("status", "error")
        log.set(error_type=type(exc).__name__, error=str(exc)[:500])
        write_entry(log.to_entry())
        raise
    else:
        log.fields.setdefault("status", "ok")
        write_entry(log.to_entry())
    finally:
        _active_log.reset(token)


@contextmanager
def http_request_log(method: str, path: str) -> Iterator[RequestLog]:
    """Owner of the single log entry for one HTTP request (used by middleware).

    The entry is always written - with the response status code set by the
    caller - including for requests rejected by validation or by the budget cap.
    """
    log = RequestLog(trace_id=new_trace_id(), endpoint=f"{method} {path}")
    token = _active_log.set(log)
    try:
        yield log
    except Exception as exc:
        log.set(status="error", http_status=500, error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        _active_log.reset(token)
        write_entry(log.to_entry())


def read_log(limit: int | None = None) -> List[Dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with LOG_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:] if limit else rows


def clear_log() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()
