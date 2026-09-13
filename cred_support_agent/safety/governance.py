"""Task 15 - four-layer AI governance for the Cred support agent.

Layer 1 (Data)        : dataset and KB are fabricated; fixed-format PII is masked
                        before it reaches the model or the logs (guardrails.py,
                        observability.py).
Layer 2 (Model)       : deterministic MOCK_LLM by default, no keys, no network;
                        a real backend is opt-in behind CRED_LLM_BACKEND.
Layer 3 (Application) : least autonomy - exactly one agent role may hold and call
                        the loan-status tool; wiring it anywhere else raises.
Layer 4 (Runtime)     : per-request token and cost caps; oversized requests are
                        rejected before any model call, never truncated silently.

This module implements layers 3 and 4 and carries the written risk
classification that layers 1-2 are argued against.
"""

from __future__ import annotations

import contextvars
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Set

from cred_support_agent.config import (
    COST_PER_1K_TOKENS_INR,
    MAX_COST_PER_REQUEST_INR,
    MAX_REQUEST_TOKENS,
    MAX_RESPONSE_TOKENS,
    MAX_TOTAL_TOKENS_PER_REQUEST,
)

# --------------------------------------------------------------------------
# Application layer: least autonomy
# --------------------------------------------------------------------------

LOOKUP_TOOL_NAME = "check_loan_application_status"
RAG_TOOL_NAME = "search_loan_policy_kb"

#: Role names match the brief's own vocabulary so the least-autonomy rule reads
#: exactly as specified: only the Lookup Agent may call the lookup tool.
RETRIEVAL_AGENT_ROLE = "Retrieval Agent"
LOOKUP_AGENT_ROLE = "Lookup Agent"
COMPOSER_AGENT_ROLE = "Response Composer"

#: The whole least-autonomy policy in one table. A tool may only be wired to,
#: and invoked by, the roles listed here.
TOOL_AUTONOMY_POLICY: Dict[str, Set[str]] = {
    LOOKUP_TOOL_NAME: {LOOKUP_AGENT_ROLE},
    RAG_TOOL_NAME: {RETRIEVAL_AGENT_ROLE},
}

ALL_AGENT_ROLES = [RETRIEVAL_AGENT_ROLE, LOOKUP_AGENT_ROLE, COMPOSER_AGENT_ROLE]


class AutonomyViolation(PermissionError):
    """Raised when a tool is wired to, or invoked by, a role that may not hold it."""


_current_agent_role: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cred_current_agent_role", default=None
)


@contextmanager
def agent_scope(role: str) -> Iterator[None]:
    """Mark the agent role that is currently executing (runtime enforcement)."""
    token = _current_agent_role.set(role)
    try:
        yield
    finally:
        _current_agent_role.reset(token)


def current_agent_role() -> str | None:
    return _current_agent_role.get()


def assert_tool_wiring(tool_name: str, agent_role: str) -> None:
    """Build-time check: refuse to attach a restricted tool to the wrong role."""
    allowed = TOOL_AUTONOMY_POLICY.get(tool_name)
    if allowed is None:
        return
    if agent_role not in allowed:
        raise AutonomyViolation(
            f"Least-autonomy policy violation: agent role {agent_role!r} may not hold "
            f"tool {tool_name!r}. Permitted roles: {sorted(allowed)}."
        )


def assert_tool_invocation(tool_name: str) -> None:
    """Runtime check: refuse a call made from an unauthorised agent scope.

    An unset scope means the tool was called directly by trusted orchestration
    code (a demo script or the API layer), which is permitted; what is blocked is
    a *different* crew agent reaching for a tool it does not own.
    """
    allowed = TOOL_AUTONOMY_POLICY.get(tool_name)
    if allowed is None:
        return
    role = current_agent_role()
    if role is None:
        return
    if role not in allowed:
        raise AutonomyViolation(
            f"Least-autonomy policy violation: agent role {role!r} attempted to invoke "
            f"tool {tool_name!r}. Permitted roles: {sorted(allowed)}."
        )


def autonomy_matrix() -> List[Dict[str, Any]]:
    """Rendered permission matrix used by the governance demo and the README."""
    rows: List[Dict[str, Any]] = []
    for tool, allowed in sorted(TOOL_AUTONOMY_POLICY.items()):
        for role in ALL_AGENT_ROLES:
            rows.append(
                {"tool": tool, "agent_role": role, "permitted": role in allowed}
            )
    return rows


# --------------------------------------------------------------------------
# Application layer: risk classification
# --------------------------------------------------------------------------

#: The brief's scheme: Low = summarization/transcription; Medium = code
#: generation/customer support tickets; High = medical data, hiring decisions,
#: financial data.
RISK_SCHEME = {
    "Low": "summarization, transcription",
    "Medium": "code generation, customer support tickets",
    "High": "medical data, hiring decisions, financial data",
}

RISK_LEVEL = "High"

RISK_JUSTIFICATION = (
    "This system is classified High risk. By its function alone it would look like the "
    "scheme's Medium category, customer support, but the scheme classifies a system by "
    "the data it handles as well as the task it performs, and the highest category that "
    "applies governs. This agent works directly with financial data: it reads loan "
    "application records carrying requested amounts, application status and fraud-review "
    "flags, it states lending policy on interest rates, fees and penalties that a member "
    "may act on financially, and its input can carry PAN, Aadhaar and bank account "
    "numbers. A wrong rate, an invented fee, a leaked identifier or a disclosed fraud flag "
    "would each cause real financial or personal harm, which is exactly what places "
    "financial data in the High category. The controls in this repository are the ones a "
    "High classification demands, and they are designed on that basis rather than used "
    "as grounds for a lower rating: answers must clear a measured retrieval threshold and "
    "an output groundedness gate, every draft is reviewed by an independent Autogen team "
    "before delivery, fixed-format identifiers are masked before the model, the memory or "
    "the log sees them, the lookup tool is read-only and restricted to one agent, and "
    "every request runs under a hard token and cost budget. Those controls reduce the "
    "likelihood of harm; they do not change the fact that the system handles financial "
    "data, so the classification stays High."
)

#: One-paragraph explanation of the least-autonomy guard (Task 15).
LEAST_AUTONOMY_EXPLANATION = (
    "Only the Lookup Agent may call check_loan_application_status, and that rule is "
    "enforced in two independent places from one permission table, "
    "TOOL_AUTONOMY_POLICY. At build time, every tool is attached to an agent through "
    "assert_tool_wiring, which raises AutonomyViolation if the agent's role is not "
    "listed for that tool, so a crew that gives the lookup tool to the Retrieval Agent "
    "or the Response Composer cannot be constructed; in the shipped crew the tool is "
    "wired to the Lookup Agent alone and the Composer holds no tools at all. At run "
    "time, a CrewAI before-tool-call hook consults the same table with the agent that "
    "is actually executing and blocks the call if that agent does not own the tool, so "
    "even a crew mis-wired by bypassing the build-time check cannot reach the record: "
    "the attempt is refused by the framework before the tool function runs, and the "
    "violation is recorded. The guard does not depend on any agent's prompt or on the "
    "model choosing to behave, because neither check can be influenced by what an agent "
    "says."
)


def risk_classification() -> Dict[str, Any]:
    return {
        "risk_level": RISK_LEVEL,
        "scheme": RISK_SCHEME,
        "justification": RISK_JUSTIFICATION,
    }


# --------------------------------------------------------------------------
# Runtime layer: token / cost budget
# --------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Raised when a request would exceed the per-request token or cost cap."""

    def __init__(self, message: str, *, detail: Dict[str, Any]) -> None:
        super().__init__(message)
        self.detail = detail


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate (~4 characters per token, min 1 per word).

    A heuristic is the right tool here: MOCK_LLM has no tokenizer, and the cap
    only has to be consistent and conservative, not byte-exact.
    """
    if not text:
        return 0
    return max(math.ceil(len(text) / 4), len(text.split()))


@dataclass
class RequestBudget:
    """Per-request ledger enforcing the runtime cap.

    ``admit`` runs *before* any model call so an oversized request is rejected
    outright rather than being silently truncated or allowed to overrun.
    """

    max_request_tokens: int = MAX_REQUEST_TOKENS
    max_response_tokens: int = MAX_RESPONSE_TOKENS
    max_total_tokens: int = MAX_TOTAL_TOKENS_PER_REQUEST
    max_cost_inr: float = MAX_COST_PER_REQUEST_INR
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)
    exhausted: bool = False
    breach: Dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_inr(self) -> float:
        return round(self.total_tokens / 1000.0 * COST_PER_1K_TOKENS_INR, 6)

    def admit(self, request_text: str) -> int:
        """Gate an incoming request. Raises BudgetExceeded before any model call."""
        tokens = estimate_tokens(request_text)
        if tokens > self.max_request_tokens:
            raise BudgetExceeded(
                f"Request rejected: {tokens} estimated prompt tokens exceeds the "
                f"per-request cap of {self.max_request_tokens}.",
                detail={
                    "reason": "request_token_cap",
                    "estimated_tokens": tokens,
                    "cap": self.max_request_tokens,
                },
            )
        projected_cost = round(
            (tokens + self.max_response_tokens) / 1000.0 * COST_PER_1K_TOKENS_INR, 6
        )
        if projected_cost > self.max_cost_inr:
            raise BudgetExceeded(
                f"Request rejected: projected cost INR {projected_cost} exceeds the "
                f"per-request cap of INR {self.max_cost_inr}.",
                detail={
                    "reason": "request_cost_cap",
                    "projected_cost_inr": projected_cost,
                    "cap_inr": self.max_cost_inr,
                },
            )
        return tokens

    def charge(self, prompt_text: str, completion_text: str, *, label: str = "llm") -> None:
        """Record a model call and enforce the running token AND cost caps.

        Every language-model stage charges this one ledger: the three crew agents,
        the grounded-generation model and both Autogen reviewers.

        Once the cap is breached the ledger is sealed: later calls raise without
        being charged. Orchestration frameworks retry a failing step, and a
        budget that kept billing those retries would both misreport the spend and
        let a capped request keep consuming.
        """
        if self.exhausted:
            raise BudgetExceeded(
                f"Request already over budget at {self.total_tokens} tokens; "
                "no further model calls are permitted.",
                detail=self.breach or {"reason": "total_token_cap"},
            )
        prompt = estimate_tokens(prompt_text)
        completion = estimate_tokens(completion_text)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        self.events.append(
            {
                "label": label,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "running_total": self.total_tokens,
            }
        )
        if self.total_tokens > self.max_total_tokens:
            self._seal(
                "total_token_cap",
                f"Request aborted mid-flight: {self.total_tokens} total tokens exceeds "
                f"the per-request cap of {self.max_total_tokens}.",
                total_tokens=self.total_tokens,
                cap=self.max_total_tokens,
            )
        if self.cost_inr > self.max_cost_inr:
            self._seal(
                "total_cost_cap",
                f"Request aborted mid-flight: running cost INR {self.cost_inr} exceeds the "
                f"per-request cap of INR {self.max_cost_inr}.",
                cost_inr=self.cost_inr,
                cap_inr=self.max_cost_inr,
            )

    def _seal(self, reason: str, message: str, **detail: Any) -> None:
        """Record the breach, seal the ledger and reject the request."""
        self.exhausted = True
        self.breach = {
            "reason": reason,
            **detail,
            "breached_on_call": self.calls,
            "events": list(self.events),
        }
        raise BudgetExceeded(message, detail=self.breach)

    def summary(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_inr": self.cost_inr,
            "exhausted": self.exhausted,
            "caps": {
                "max_request_tokens": self.max_request_tokens,
                "max_total_tokens": self.max_total_tokens,
                "max_cost_inr": self.max_cost_inr,
            },
        }


_current_budget: contextvars.ContextVar[RequestBudget | None] = contextvars.ContextVar(
    "cred_request_budget", default=None
)


@contextmanager
def budget_scope(budget: RequestBudget) -> Iterator[RequestBudget]:
    token = _current_budget.set(budget)
    try:
        yield budget
    finally:
        _current_budget.reset(token)


def current_budget() -> RequestBudget | None:
    return _current_budget.get()


# --------------------------------------------------------------------------
# Framework-level enforcement of least autonomy.
#
# Wiring checks (assert_tool_wiring) stop a restricted tool being attached to
# the wrong role at build time. This hook is the runtime counterpart: CrewAI
# calls it before *every* tool execution, with the agent that is actually
# executing, so a tool reached for by a role that does not own it is blocked by
# the framework itself rather than by anything the agent could talk its way past.
# --------------------------------------------------------------------------

_violations: List[Dict[str, Any]] = []


def recorded_violations() -> List[Dict[str, Any]]:
    return list(_violations)


def clear_violations() -> None:
    _violations.clear()


def _least_autonomy_hook(context: Any) -> bool:
    """CrewAI before-tool-call hook.

    CrewAI's contract here is inverted from the obvious reading: returning
    ``False`` *blocks* the tool call (it is mapped to ``HookAborted``), and any
    other return value lets it through. So permitted calls return True.
    """
    tool_name = getattr(context, "tool_name", "")
    allowed = TOOL_AUTONOMY_POLICY.get(tool_name)
    if allowed is None:
        return True
    agent = getattr(context, "agent", None)
    role = getattr(agent, "role", None)
    if role is None or role in allowed:
        return True
    _violations.append(
        {
            "tool": tool_name,
            "attempted_by": role,
            "permitted_roles": sorted(allowed),
            "action": "blocked",
        }
    )
    return False  # blocks execution


_hook_installed = False


def install_least_autonomy_hook() -> None:
    """Idempotently register the runtime least-autonomy enforcement hook."""
    global _hook_installed
    if _hook_installed:
        return
    from crewai.hooks import register_before_tool_call_hook

    register_before_tool_call_hook(_least_autonomy_hook)
    _hook_installed = True
