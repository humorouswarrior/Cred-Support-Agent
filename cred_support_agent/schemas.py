"""Task 9 - structured output contracts.

``SupportResponse`` is the single schema every crew response is validated
against, and it is also what the crew's LLM is asked to emit via
``response_format``. Validation happens in code (``validate_support_response``)
on every path - crew, API and evaluation - so a malformed answer fails loudly
instead of being surfaced to a customer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SupportResponse(BaseModel):
    """The one response shape the Cred support crew is allowed to emit."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1, description="Customer-facing answer text.")
    answer_type: Literal["policy", "status", "policy_and_status", "refusal"] = Field(
        ..., description="Which capability produced the answer."
    )
    grounded: bool = Field(
        ..., description="True when every claim is supported by retrieved context."
    )
    citations: List[str] = Field(
        default_factory=list, description="Knowledge-base doc ids backing the answer."
    )
    tools_used: List[str] = Field(
        default_factory=list, description="Tool names actually invoked for this turn."
    )
    record_id: str | None = Field(
        default=None, description="Loan application id when a status lookup ran."
    )
    escalation_recommended: bool = Field(
        default=False, description="True when the escalation score is above cutoff."
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Top-1 retrieval similarity."
    )
    guardrail_flags: List[str] = Field(
        default_factory=list, description="Guardrails that fired on this turn."
    )

    @field_validator("citations", "tools_used", "guardrail_flags", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value


class ReviewVerdict(BaseModel):
    """Task 14 - structured verdict emitted by the Autogen Final-Editor."""

    model_config = ConfigDict(extra="forbid")

    approved: bool = Field(..., description="True when the draft is accepted unchanged.")
    final_answer: str = Field(..., min_length=1, description="Answer to deliver.")
    reason: str = Field(..., min_length=1, description="Why it was approved or revised.")


class LookupResult(BaseModel):
    """Task 6 - return contract of check_loan_application_status."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    found: bool
    status: str
    loan_amount_inr: int
    escalation_score: float = Field(..., ge=0.0, le=1.0)
    escalation_threshold: float
    escalation_recommended: bool
    category: str
    days_since_created: int
    flagged_for_fraud_review: bool
    formula: str


class AskRequest(BaseModel):
    """POST /ask request body."""

    model_config = ConfigDict(extra="forbid")

    # Deliberately generous: the binding limit on request size is the runtime
    # governance budget (Task 15), not this schema. A request between these two
    # limits must be rejected by the budget cap with a 413 and a stated reason,
    # not silently by a 422 from the request model.
    query: str = Field(..., min_length=1, max_length=20000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    review: bool = Field(default=True, description="Run the Autogen review stage.")


class AskResponse(BaseModel):
    """POST /ask response body."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    session_id: str
    response: SupportResponse
    review: Dict[str, Any] | None = None
    cached: bool = False
    latency_ms: float = 0.0


class AddDocumentRequest(BaseModel):
    """POST /add-document request body."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., min_length=3, max_length=64)
    topic: str = Field(..., min_length=3, max_length=64)
    title: str = Field(..., min_length=3, max_length=200)
    text: str = Field(..., min_length=40, max_length=4000)


class AddDocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    doc_id: str
    indexed_chunks: Dict[str, int]
    collections: List[str]


def validate_support_response(payload: Any) -> SupportResponse:
    """Validate any candidate crew output against the schema.

    Accepts a model instance, a mapping or a JSON string; raises
    ``pydantic.ValidationError`` when the payload does not conform.
    """
    if isinstance(payload, SupportResponse):
        return SupportResponse.model_validate(payload.model_dump())
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        return SupportResponse.model_validate_json(payload)
    return SupportResponse.model_validate(payload)


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str
    llm_backend: str
    embedding_backend: str
    documents: int
    loan_applications: int
    collections: Dict[str, int]
    active_threshold: float | None
    risk_level: str
    cache: Dict[str, Any]


class AutonomyRule(BaseModel):
    tool: str
    agent_role: str
    permitted: bool


class GovernanceResponse(BaseModel):
    """GET /governance response body."""

    risk: Dict[str, Any]
    least_autonomy: List[AutonomyRule]
    least_autonomy_explanation: str
    runtime_budget_caps: Dict[str, Any]
    llm_backend: str
    telemetry: Dict[str, str]


class LogsResponse(BaseModel):
    """GET /logs response body."""

    entries: List[Dict[str, Any]]


class ResetResponse(BaseModel):
    """POST /session/{session_id}/reset response body."""

    session_id: str
    turns: int
