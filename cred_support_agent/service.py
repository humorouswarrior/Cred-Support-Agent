"""The application service: one entry point that ties every layer together.

    budget admission -> input guardrails -> session memory -> CrewAI crew
    -> schema validation -> output guardrail -> Autogen review -> JSON log

Both the FastAPI app and the demo scripts call through here, so the HTTP path and
the scripted path exercise exactly the same pipeline.
"""

from __future__ import annotations

import contextvars
import time
from typing import Any, Dict, List

from cred_support_agent.agents.crew import CrewRunResult, answer_question
from cred_support_agent.safety.governance import BudgetExceeded, RequestBudget
from cred_support_agent.agents.memory import ConversationService, reset_session, session_turn_count
from cred_support_agent.safety.guardrails import mask_pii
from cred_support_agent.observability import new_trace_id, request_log
from cred_support_agent.retrieval.pipeline import CACHE
from cred_support_agent.schemas import SupportResponse


# Whether the review stage runs is a per-request decision, but the memory layer
# sits between ask() and the crew and does not forward it. A context variable
# carries it across that boundary without a module-global that concurrent
# WebSocket turns (which run in worker threads) would race on.
_review_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cred_review_enabled", default=True
)


def _answer_fn(question: str, history: List[Any], session_id: str) -> tuple[str, CrewRunResult]:
    result = answer_question(question, review=_review_enabled.get())
    return result.response.answer, result


_CONVERSATION = ConversationService(_answer_fn)


def ask(
    question: str,
    session_id: str = "default",
    *,
    review: bool = True,
    endpoint: str = "ask",
    trace_id: str | None = None,
) -> Dict[str, Any]:
    """Answer one customer turn, with memory, governance and a log entry."""
    trace_id = trace_id or new_trace_id()
    started = time.perf_counter()
    _review_enabled.set(review)

    # Mask at the very front door. Everything downstream - session memory, the
    # crew, the review stage and the log - only ever sees the masked text, so a
    # PAN, Aadhaar or account number a customer pastes into chat is never stored
    # in the conversation history or written to disk.
    front_door = mask_pii(question)
    masked_question = front_door.text

    with request_log(endpoint, session_id=session_id, trace_id=trace_id) as log:
        # Inside an HTTP request the middleware owns the entry and its trace id.
        trace_id = log.trace_id
        log.set_request_text(question)
        log.mark("intake")
        try:
            turn = _CONVERSATION.ask(masked_question, session_id=session_id)
        except BudgetExceeded as exc:
            log.set(
                status="rejected",
                rejection="budget_exceeded",
                budget_detail=exc.detail,
                answer_type="refusal",
            )
            log.mark("budget_rejected")
            raise
        log.mark("crew")

        result: CrewRunResult = turn["response"]
        # The crew's own input guardrail only ever sees already-masked text, so it
        # cannot know masking happened. Carry the front-door findings through, or
        # the response would claim no guardrail fired on a turn that was redacted.
        if front_door.fired:
            merged = list(dict.fromkeys(front_door.flags + result.guardrail_flags))
            result.guardrail_flags = merged
            result.response.guardrail_flags = merged
        response: SupportResponse = result.response
        log.set(
            answer_type=response.answer_type,
            grounded=response.grounded,
            citations=response.citations,
            tools_used=result.tools_invoked,
            guardrail_flags=result.guardrail_flags,
            blocked=result.blocked,
            record_id=response.record_id,
            escalation_recommended=response.escalation_recommended,
            confidence=response.confidence,
            crew_model_calls=result.llm_calls,
            budget=result.budget,
            review_approved=(result.review or {}).get("approved"),
            resolved_question=turn.get("resolved_question"),
            carried_record_id=turn.get("carried_record_id"),
            history_length=turn.get("history_length"),
            cache_hit=bool((result.retrieval or {}).get("cached")),
            cache=CACHE.stats(),
        )

    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "response": response,
        "review": result.review,
        "result": result,
        "resolved_question": turn.get("resolved_question"),
        "carried_record_id": turn.get("carried_record_id"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        # Whether *this* request's policy answer came from cache.
        "cached": bool((result.retrieval or {}).get("cached")),
    }


def add_document(doc: Dict[str, Any], *, trace_id: str | None = None) -> Dict[str, Any]:
    """Index a new policy document into both collections and drop stale cache."""
    from cred_support_agent.retrieval.indexing import COLLECTION_BY_STRATEGY, index_document
    from cred_support_agent.data.knowledge_base import DOCS_BY_ID, KNOWLEDGE_BASE

    trace_id = trace_id or new_trace_id()
    with request_log("add_document", trace_id=trace_id) as log:
        trace_id = log.trace_id
        log.set_request_text(doc.get("text", ""))
        existing = DOCS_BY_ID.get(doc["doc_id"])
        if existing is not None:
            existing.update(doc)
        else:
            KNOWLEDGE_BASE.append(doc)
            DOCS_BY_ID[doc["doc_id"]] = doc
        counts = index_document(doc)
        # New evidence can change any previous answer: the cache must not serve
        # a pre-update reply.
        CACHE.clear()
        log.set(doc_id=doc["doc_id"], indexed_chunks=counts, cache_cleared=True)

    return {
        "trace_id": trace_id,
        "doc_id": doc["doc_id"],
        "indexed_chunks": counts,
        "collections": list(COLLECTION_BY_STRATEGY.values()),
    }


def remove_document(doc_id: str) -> Dict[str, Any]:
    """Undo add_document: drop the document from the knowledge base, both
    collections and the cache. Used so demonstrations leave no state behind."""
    from cred_support_agent.data.knowledge_base import DOCS_BY_ID, KNOWLEDGE_BASE
    from cred_support_agent.retrieval.indexing import remove_document as remove_chunks

    DOCS_BY_ID.pop(doc_id, None)
    KNOWLEDGE_BASE[:] = [d for d in KNOWLEDGE_BASE if d["doc_id"] != doc_id]
    removed = remove_chunks(doc_id)
    CACHE.clear()
    return {"doc_id": doc_id, "removed_chunks": removed}


def reset_conversation(session_id: str) -> None:
    reset_session(session_id)


def conversation_turns(session_id: str) -> int:
    return session_turn_count(session_id)


def health() -> Dict[str, Any]:
    from cred_support_agent.retrieval.calibration import load_calibration
    from cred_support_agent.config import LLM_BACKEND
    from cred_support_agent.data.dataset import LOAN_APPLICATIONS
    from cred_support_agent.retrieval.embeddings import backend_name
    from cred_support_agent.safety.governance import RISK_LEVEL
    from cred_support_agent.retrieval.indexing import COLLECTION_BY_STRATEGY, ensure_indexes, get_client
    from cred_support_agent.data.knowledge_base import KNOWLEDGE_BASE

    ensure_indexes()  # report the real index state, not "empty until first question"
    collections = {}
    for strategy, name in COLLECTION_BY_STRATEGY.items():
        try:
            collections[strategy] = get_client().get_collection(name).count()
        except Exception:
            collections[strategy] = 0
    calibration = load_calibration()
    return {
        "status": "ok",
        "llm_backend": LLM_BACKEND,
        "embedding_backend": backend_name(),
        "documents": len(KNOWLEDGE_BASE),
        "loan_applications": len(LOAN_APPLICATIONS),
        "collections": collections,
        "active_threshold": calibration.get("active_threshold"),
        "risk_level": RISK_LEVEL,
        "cache": CACHE.stats(),
    }


def budget_probe(text: str) -> Dict[str, Any]:
    """Check a request against the runtime cap without running the crew."""
    budget = RequestBudget()
    try:
        tokens = budget.admit(text)
    except BudgetExceeded as exc:
        return {"admitted": False, "detail": exc.detail, "message": str(exc)}
    return {"admitted": True, "estimated_tokens": tokens, "caps": budget.summary()["caps"]}
