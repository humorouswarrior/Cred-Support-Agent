"""Task 6 / Task 7 - the two crew tools.

``check_loan_application_status`` is the status tool with the designed
escalation score. ``search_loan_policy_kb`` is the RAG tool. Both are exposed
twice: as a plain Python function (used by demos, the API and the tests) and as
a CrewAI ``BaseTool`` with an explicit ``args_schema``, which is what lets the
MOCK_LLM dispatch by declared argument schema rather than by tool name.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from cred_support_agent.config import (
    ESCALATION_PERCENTILE,
    ESCALATION_WEIGHT_AGE,
    ESCALATION_WEIGHT_FRAUD,
    MAX_DAYS_SINCE_CREATED,
)
from cred_support_agent.data.dataset import LOAN_APPLICATIONS, get_application, percentile
from cred_support_agent.safety.governance import LOOKUP_TOOL_NAME, RAG_TOOL_NAME, assert_tool_invocation

# --------------------------------------------------------------------------
# Escalation score design (Task 6)
# --------------------------------------------------------------------------
#
#   age_norm         = min(days_since_created, 30) / 30          -> [0, 1]
#   escalation_score = 0.5 * flagged_for_fraud_review + 0.5 * age_norm
#
# Two independent signals, equally weighted, producing a continuous ranking
# score rather than a boolean: a fraud flag contributes a fixed 0.5, and queue
# age contributes up to a further 0.5 in proportion to how long the application
# has been sitting.
#
# The cutoff is derived from the generated data rather than picked by feel:
#
#   ESCALATION_THRESHOLD = 0.5 * (p80(days_since_created) / 30)
#
# With the shipped seed p80 = 26 days, so the cutoff is 0.4333. Read as policy
# that means:
#
#   * escalation is recommended when the score is strictly ABOVE the cutoff;
#   * every fraud-flagged application escalates (its 0.5 floor is above 0.4333),
#     which matches KB-014: fraud review goes to the financial crime desk
#     immediately;
#   * an unflagged application escalates only once its age passes the 80th
#     percentile of the current queue (more than 26 days) - exactly the ageing
#     rule in KB-014 - so only the oldest part of the unflagged queue reaches a
#     supervisor, which is a workload a human desk can actually absorb.
#
FORMULA = (
    "escalation_score = 0.5 * flagged_for_fraud_review + 0.5 * "
    "(min(days_since_created, 30) / 30)"
)


def _p80_days() -> float:
    return percentile(
        [float(r["days_since_created"]) for r in LOAN_APPLICATIONS], ESCALATION_PERCENTILE
    )


def escalation_threshold() -> float:
    """Data-derived cutoff: the 80th percentile of queue age, on the score scale."""
    return round(ESCALATION_WEIGHT_AGE * (_p80_days() / MAX_DAYS_SINCE_CREATED), 4)


ESCALATION_THRESHOLD = escalation_threshold()


def compute_escalation_score(days_since_created: int, flagged_for_fraud_review: bool) -> float:
    age_norm = min(int(days_since_created), MAX_DAYS_SINCE_CREATED) / MAX_DAYS_SINCE_CREATED
    score = ESCALATION_WEIGHT_FRAUD * float(bool(flagged_for_fraud_review))
    score += ESCALATION_WEIGHT_AGE * age_norm
    return round(score, 4)


def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    """Look up one Cred loan application and score it for escalation.

    Returns ``status``, ``loan_amount_inr`` and ``escalation_score`` (plus the
    supporting fields a support agent needs to explain the score).
    """
    assert_tool_invocation(LOOKUP_TOOL_NAME)
    record = get_application(record_id)
    threshold = escalation_threshold()
    if record is None:
        missing = {
            "record_id": (record_id or "").strip().upper(),
            "found": False,
            "status": "Not Found",
            "loan_amount_inr": 0,
            "escalation_score": 0.0,
            "escalation_threshold": threshold,
            "escalation_recommended": False,
            "category": "Unknown",
            "days_since_created": 0,
            "flagged_for_fraud_review": False,
            "formula": FORMULA,
        }
        _record(LOOKUP_TOOL_NAME, {"record_id": record_id}, missing)
        return missing
    score = compute_escalation_score(
        record["days_since_created"], record["flagged_for_fraud_review"]
    )
    result = {
        "record_id": record["record_id"],
        "found": True,
        "status": record["status"],
        "loan_amount_inr": record["loan_amount_inr"],
        "escalation_score": score,
        "escalation_threshold": threshold,
        "escalation_recommended": score > threshold,
        "category": record["category"],
        "days_since_created": record["days_since_created"],
        "flagged_for_fraud_review": record["flagged_for_fraud_review"],
        "formula": FORMULA,
    }
    _record(LOOKUP_TOOL_NAME, {"record_id": record_id}, result)
    return result


def escalation_distribution() -> Dict[str, Any]:
    """Score the whole generated dataset - the evidence behind the cutoff."""
    scored = [
        {
            "record_id": r["record_id"],
            "score": compute_escalation_score(
                r["days_since_created"], r["flagged_for_fraud_review"]
            ),
            "flagged": r["flagged_for_fraud_review"],
            "days": r["days_since_created"],
        }
        for r in LOAN_APPLICATIONS
    ]
    threshold = escalation_threshold()
    above = [s for s in scored if s["score"] > threshold]
    return {
        "formula": FORMULA,
        "p80_days_since_created": _p80_days(),
        "threshold": threshold,
        "threshold_derivation": (
            f"0.5 * (p80(days_since_created)={_p80_days()} / {MAX_DAYS_SINCE_CREATED})"
        ),
        "total_records": len(scored),
        "above_threshold": len(above),
        "above_threshold_pct": round(100.0 * len(above) / len(scored), 2),
        "flagged_above_threshold": sum(1 for s in above if s["flagged"]),
        "unflagged_above_threshold": sum(1 for s in above if not s["flagged"]),
        "min_score": min(s["score"] for s in scored),
        "max_score": max(s["score"] for s in scored),
    }


# --------------------------------------------------------------------------
# RAG tool
# --------------------------------------------------------------------------


def search_loan_policy_kb(query: str) -> Dict[str, Any]:
    """Retrieve grounded policy context for a natural-language question."""
    assert_tool_invocation(RAG_TOOL_NAME)
    # Goes through the cached grounded-generation step, so a repeated question
    # to the live crew is answered from cache instead of re-running retrieval and
    # composition. Status lookups are deliberately never cached: a record can
    # change between two identical questions.
    from cred_support_agent.retrieval.pipeline import grounded_answer  # local import avoids an import cycle

    result = grounded_answer(query)
    _record(RAG_TOOL_NAME, {"query": query}, result)
    return result


# --------------------------------------------------------------------------
# CrewAI tool wrappers. The args_schema is the contract the MOCK_LLM dispatches
# on: one tool declares {record_id: str}, the other declares {query: str}.
# --------------------------------------------------------------------------


class LoanStatusArgs(BaseModel):
    # The `pattern` here is not decoration: it is the declared constraint the
    # MOCK_LLM dispatches on. A free-text question cannot satisfy it, so the
    # status tool is simply not dispatchable unless a real record id is present.
    record_id: str = Field(
        ...,
        pattern=r"^CRED-LN-\d{4}$",
        description=(
            "Cred loan application record identifier, formatted CRED-LN-0000."
        ),
    )


class PolicySearchArgs(BaseModel):
    query: str = Field(
        ...,
        min_length=8,
        description=(
            "Natural-language question about Cred lending, card or account policy."
        ),
    )


class LoanStatusTool(BaseTool):
    name: str = LOOKUP_TOOL_NAME
    description: str = (
        "Look up the current status, requested amount and escalation score of one "
        "Cred loan application by its record id."
    )
    args_schema: Type[BaseModel] = LoanStatusArgs

    def _run(self, record_id: str) -> str:
        return json.dumps(check_loan_application_status(record_id))


class PolicySearchTool(BaseTool):
    name: str = RAG_TOOL_NAME
    description: str = (
        "Search the Cred policy knowledge base and return the retrieved passages "
        "that ground an answer about loans, cards, KYC, accounts or fees."
    )
    args_schema: Type[BaseModel] = PolicySearchArgs

    def _run(self, query: str) -> str:
        return json.dumps(search_loan_policy_kb(query))


def report() -> str:
    dist = escalation_distribution()
    lines = ["=" * 72, "ESCALATION SCORE DESIGN", "=" * 72, f"formula: {dist['formula']}"]
    lines.append(f"p80(days_since_created) = {dist['p80_days_since_created']}")
    lines.append(f"threshold = {dist['threshold_derivation']} = {dist['threshold']}")
    lines.append(
        f"records above threshold: {dist['above_threshold']}/{dist['total_records']} "
        f"({dist['above_threshold_pct']}%)  "
        f"[flagged={dist['flagged_above_threshold']}, "
        f"unflagged={dist['unflagged_above_threshold']}]"
    )
    lines.append(f"score range across dataset: {dist['min_score']} - {dist['max_score']}")
    lines.append("")
    lines.append("sample lookups:")
    for record_id in ("CRED-LN-0001", "CRED-LN-0007", "CRED-LN-0042", "CRED-LN-9999"):
        result = check_loan_application_status(record_id)
        lines.append(
            f"  {result['record_id']:<14} found={str(result['found']):<5} "
            f"status={result['status']:<13} amount={result['loan_amount_inr']:>9,} "
            f"score={result['escalation_score']:<7} escalate={result['escalation_recommended']}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())


# --------------------------------------------------------------------------
# Tool tracing. The crew needs to know what its tools actually returned - the
# output-side groundedness guardrail checks the answer against the very context
# the retrieval tool produced, and the observability layer records which tools
# ran. A context-local trace keeps that per-request and concurrency-safe.
# --------------------------------------------------------------------------

import contextvars  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from typing import Iterator  # noqa: E402

_tool_trace: contextvars.ContextVar[List[Dict[str, Any]] | None] = contextvars.ContextVar(
    "cred_tool_trace", default=None
)


@contextmanager
def tool_trace_scope() -> Iterator[List[Dict[str, Any]]]:
    """Collect every tool invocation made inside this block."""
    trace: List[Dict[str, Any]] = []
    token = _tool_trace.set(trace)
    try:
        yield trace
    finally:
        _tool_trace.reset(token)


def _record(tool_name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
    trace = _tool_trace.get()
    if trace is not None:
        trace.append({"tool": tool_name, "args": args, "result": result})


def trace_lookup(trace: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for entry in reversed(trace):
        if entry["tool"] == LOOKUP_TOOL_NAME and entry["result"].get("found"):
            return entry["result"]
    for entry in reversed(trace):
        if entry["tool"] == LOOKUP_TOOL_NAME:
            return entry["result"]
    return None


def trace_retrieval(trace: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for entry in reversed(trace):
        if entry["tool"] == RAG_TOOL_NAME:
            return entry["result"]
    return None
