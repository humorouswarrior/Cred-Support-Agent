"""Task 11 - FastAPI deployment.

HTTP
  POST /ask           - one support question through the full pipeline
  POST /add-document  - index a new policy document into both collections
  GET  /health        - system state
  GET  /governance    - risk classification, autonomy matrix, budget caps
  GET  /logs          - recent structured log entries (masked)
  POST /session/{id}/reset - clear one conversation's memory

WebSocket
  /ws/chat            - multi-turn chat on one session, tolerant of disconnects

Pydantic models are used for every request and response body, and every HTTP
request - whatever its outcome - writes exactly one JSON-lines log entry. The WebSocket
handler catches ``WebSocketDisconnect`` so one client dropping mid-conversation
never takes the server - or any other client's session - down with it.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

import os

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from cred_support_agent.config import LLM_BACKEND
from cred_support_agent.safety.governance import (
    LEAST_AUTONOMY_EXPLANATION,
    BudgetExceeded,
    RequestBudget,
    autonomy_matrix,
    risk_classification,
)
from cred_support_agent.observability import http_request_log, new_trace_id, read_log
from cred_support_agent.schemas import (
    AddDocumentRequest,
    AddDocumentResponse,
    AskRequest,
    AskResponse,
    GovernanceResponse,
    HealthResponse,
    LogsResponse,
    ResetResponse,
)
from cred_support_agent.service import add_document, ask, conversation_turns, health, reset_conversation

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Warm up at start-up instead of on the first customer's request.

    Loads the local embedding model, builds both ChromaDB collections if they are
    empty, and loads (or measures) the calibrated threshold. Without this, a fresh
    server reports empty collections on /health and the first question pays the
    ~10 s model load.
    """
    from cred_support_agent.retrieval.calibration import load_calibration
    from cred_support_agent.retrieval.indexing import ensure_indexes

    await asyncio.to_thread(ensure_indexes)
    await asyncio.to_thread(load_calibration)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Cred Domain Support Agent",
    version="1.0.0",
    description=(
        "Banking & FinTech (Cred) support agent: grounded loan-policy Q&A and loan "
        "application status lookup, running on a deterministic MOCK_LLM with no API "
        "keys and no outbound network calls."
    ),
)


@app.middleware("http")
async def log_every_request(request: Request, call_next: Any) -> Any:
    """Task 12: exactly one JSON-lines entry for every HTTP request.

    The entry is opened here, enriched by the service layer when the request runs
    the pipeline, and written once the response status is known - including
    requests rejected by validation (422) or the budget cap (413). The trace id is
    returned to the caller in the ``X-Trace-Id`` header.
    """
    with http_request_log(request.method, request.url.path) as log:
        response = await call_next(request)
        log.set(http_status=response.status_code)
        if response.status_code == 413:
            log.fields.setdefault("status", "rejected")
        log.fields.setdefault("status", "ok" if response.status_code < 400 else "error")
        response.headers["X-Trace-Id"] = log.trace_id
        log.mark("response")
        return response


class ChatTurn(BaseModel):
    """One inbound WebSocket frame."""

    query: str
    review: bool = True


class ChatReply(BaseModel):
    """One outbound WebSocket frame."""

    type: str
    trace_id: str | None = None
    session_id: str | None = None
    turn: int | None = None
    answer: str | None = None
    answer_type: str | None = None
    citations: List[str] = []
    tools_used: List[str] = []
    escalation_recommended: bool = False
    guardrail_flags: List[str] = []
    review_approved: bool | None = None
    resolved_question: str | None = None
    detail: Dict[str, Any] | None = None


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(**health())


@app.get("/governance", response_model=GovernanceResponse)
def get_governance() -> GovernanceResponse:
    return GovernanceResponse(
        risk=risk_classification(),
        least_autonomy=autonomy_matrix(),
        least_autonomy_explanation=LEAST_AUTONOMY_EXPLANATION,
        runtime_budget_caps=RequestBudget().summary()["caps"],
        llm_backend=LLM_BACKEND,
        telemetry={
            key: os.environ.get(key, "")
            for key in ("CREWAI_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED", "CREWAI_TRACING_ENABLED")
        },
    )


@app.get("/logs", response_model=LogsResponse)
def get_logs(limit: int = 20) -> LogsResponse:
    return LogsResponse(entries=read_log(limit=max(1, min(limit, 200))))


@app.post("/ask", response_model=AskResponse)
def post_ask(payload: AskRequest) -> AskResponse:
    try:
        result = ask(
            payload.query,
            session_id=payload.session_id,
            review=payload.review,
            endpoint="POST /ask",
        )
    except BudgetExceeded as exc:
        # Runtime governance: reject, never silently exceed the cap.
        raise HTTPException(status_code=413, detail=exc.detail) from exc

    return AskResponse(
        trace_id=result["trace_id"],
        session_id=result["session_id"],
        response=result["response"],
        review=result["review"],
        cached=result["cached"],
        latency_ms=result["latency_ms"],
    )


@app.post("/add-document", response_model=AddDocumentResponse)
def post_add_document(payload: AddDocumentRequest) -> AddDocumentResponse:
    result = add_document(payload.model_dump())
    return AddDocumentResponse(**result)


@app.post("/session/{session_id}/reset", response_model=ResetResponse)
def post_reset(session_id: str) -> ResetResponse:
    reset_conversation(session_id)
    return ResetResponse(session_id=session_id, turns=conversation_turns(session_id))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Multi-turn chat over one session.

    Every failure mode is contained: a malformed frame is answered with an error
    frame and the socket stays open, and a disconnect (at any point, including
    mid-send) ends only this connection. Other clients are unaffected.
    """
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or f"ws_{new_trace_id()}"
    try:
        await websocket.send_text(
            ChatReply(type="ready", session_id=session_id, turn=0).model_dump_json()
        )
        while True:
            raw = await websocket.receive_text()
            try:
                turn = ChatTurn.model_validate_json(raw)
            except ValidationError as exc:
                await websocket.send_text(
                    ChatReply(
                        type="error",
                        session_id=session_id,
                        detail={"validation_errors": json.loads(exc.json())},
                    ).model_dump_json()
                )
                continue

            try:
                # CrewAI refuses a synchronous kickoff() from inside a running
                # event loop, and the review stage runs its own asyncio loop, so
                # the whole blocking pipeline is handed to a worker thread. This
                # also keeps one slow turn from stalling other WebSocket clients.
                result = await asyncio.to_thread(
                    ask,
                    turn.query,
                    session_id=session_id,
                    review=turn.review,
                    endpoint="WS /ws/chat",
                )
            except BudgetExceeded as exc:
                await websocket.send_text(
                    ChatReply(
                        type="rejected", session_id=session_id, detail=exc.detail
                    ).model_dump_json()
                )
                continue

            response = result["response"]
            await websocket.send_text(
                ChatReply(
                    type="answer",
                    trace_id=result["trace_id"],
                    session_id=session_id,
                    turn=conversation_turns(session_id) // 2,
                    answer=response.answer,
                    answer_type=response.answer_type,
                    citations=response.citations,
                    tools_used=response.tools_used,
                    escalation_recommended=response.escalation_recommended,
                    guardrail_flags=response.guardrail_flags,
                    review_approved=(result["review"] or {}).get("approved"),
                    resolved_question=result.get("resolved_question"),
                ).model_dump_json()
            )
    except WebSocketDisconnect:
        # The client went away. Log nothing alarming and keep serving everyone else.
        return
    except Exception as exc:  # pragma: no cover - defensive
        try:
            await websocket.send_text(
                ChatReply(type="error", session_id=session_id, detail={"error": str(exc)}).model_dump_json()
            )
        except (WebSocketDisconnect, RuntimeError):
            pass
        return


@app.exception_handler(BudgetExceeded)
async def budget_handler(request: Any, exc: BudgetExceeded) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=413, content={"detail": exc.detail})
