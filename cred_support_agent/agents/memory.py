"""Task 8 - LangChain session memory.

``InMemoryChatMessageHistory`` holds the turns of one conversation and
``RunnableWithMessageHistory`` binds that history to the crew call, keyed by
``session_id``. State lives for the life of the process: two turns in the same
session share context, and a fresh ``session_id`` starts genuinely empty.

Memory is not decorative here - it does work. ``resolve_references`` reads the
history to resolve an anaphoric follow-up ("is *it* escalated?") back to the
record id the customer named on an earlier turn, which is the single most common
multi-turn pattern on a lending support desk.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain 1.x emits a LangChainDeprecationWarning for RunnableWithMessageHistory,
# pointing to LangGraph persistence. The brief names this primitive, notes that the
# warning is expected, and says it does not need to be silenced - so it is left
# visible. The class works correctly for in-process session memory.


_RECORD_ID = re.compile(r"\b(CRED-LN-\d{4})\b", re.IGNORECASE)

_ANAPHORA = re.compile(
    r"\b(it|that|this|the same|that one|this one|my application|the application|"
    r"same application|it\'s|its)\b",
    re.IGNORECASE,
)

_SESSIONS: Dict[str, InMemoryChatMessageHistory] = {}
_LOCK = threading.Lock()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (creating on first use) the history for one conversation."""
    with _LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = InMemoryChatMessageHistory()
        return _SESSIONS[session_id]


def reset_session(session_id: str) -> None:
    with _LOCK:
        _SESSIONS.pop(session_id, None)


def clear_all_sessions() -> None:
    with _LOCK:
        _SESSIONS.clear()


def session_turn_count(session_id: str) -> int:
    with _LOCK:
        history = _SESSIONS.get(session_id)
        return len(history.messages) if history else 0


def last_record_id(messages: List[Any]) -> str | None:
    """Most recently mentioned application id anywhere in the conversation."""
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        found = _RECORD_ID.findall(content)
        if found:
            return found[-1].upper()
    return None


def resolve_references(question: str, history_messages: List[Any]) -> tuple[str, str | None]:
    """Rewrite an anaphoric follow-up using the conversation history.

    Returns ``(resolved_question, carried_record_id)``. The question is left
    untouched when it already names a record id or carries no back-reference.
    """
    if _RECORD_ID.search(question):
        return question, None
    if not history_messages or not _ANAPHORA.search(question):
        return question, None
    carried = last_record_id(history_messages)
    if not carried:
        return question, None
    return f"{question.rstrip('?.! ')} (application {carried})?", carried


def build_conversational_chain(answer_fn: Any) -> RunnableWithMessageHistory:
    """Wrap the crew pipeline in LangChain session memory.

    ``answer_fn(question, history, session_id) -> (answer_text, SupportResponse)``
    """

    def _invoke(payload: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        question = payload["input"]
        history = payload.get("history") or []
        session_id = ((config or {}).get("configurable") or {}).get("session_id", "default")
        resolved, carried = resolve_references(question, history)
        answer_text, response = answer_fn(resolved, history, session_id)
        return {
            "output": answer_text,
            "response": response,
            "resolved_question": resolved,
            "carried_record_id": carried,
            "history_length": len(history),
        }

    runnable = RunnableLambda(_invoke)
    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )


class ConversationService:
    """Convenience wrapper: one call per turn, memory handled underneath."""

    def __init__(self, answer_fn: Any) -> None:
        self._chain = build_conversational_chain(answer_fn)

    def ask(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        return self._chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )

    @staticmethod
    def turns(session_id: str) -> int:
        return session_turn_count(session_id)

    @staticmethod
    def reset(session_id: str) -> None:
        reset_session(session_id)
