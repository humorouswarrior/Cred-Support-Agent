"""Task 7 - the CrewAI crew, plus the request pipeline that wraps it.

Three agents, executed sequentially:

  1. Retrieval Agent   - holds ``search_loan_policy_kb``
  2. Lookup Agent      - holds ``check_loan_application_status``
  3. Response Composer - holds no tools at all

Tool wiring goes through ``governance.assert_tool_wiring`` (Task 15, least
autonomy), so attaching a restricted tool to the wrong role raises before the
crew is ever built. The crew is executed with ``kickoff()``.

``answer_question`` is the full request pipeline around the crew: budget
admission, input guardrails, crew kickoff, schema validation, the Autogen review
stage, and the output-side groundedness gate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

# Importing config must precede importing crewai: config sets the telemetry
# environment variables, and crewai reads them at import time.
from cred_support_agent import config as _config

assert (
    os.environ.get("CREWAI_DISABLE_TELEMETRY") == "true"
    and os.environ.get("CREWAI_TRACING_ENABLED") == "false"
    and _config.PROJECT_ROOT.exists()
), (
    "Telemetry must be disabled before CrewAI is imported; import "
    "cred_support_agent.config first."
)

from crewai import Agent, Crew, Process, Task  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from cred_support_agent.retrieval.generation import (  # noqa: E402
    compose_status_answer,
    extract_citations,
    is_refusal,
)
from cred_support_agent.safety.governance import (  # noqa: E402
    COMPOSER_AGENT_ROLE,
    LOOKUP_AGENT_ROLE,
    RETRIEVAL_AGENT_ROLE,
    RequestBudget,
    assert_tool_wiring,
    budget_scope,
    install_least_autonomy_hook,
)
from cred_support_agent.safety.guardrails import apply_input_guardrails, check_groundedness  # noqa: E402
from cred_support_agent.agents.llm import QUESTION_CLOSE, QUESTION_OPEN, build_llm  # noqa: E402
from cred_support_agent.schemas import SupportResponse, validate_support_response  # noqa: E402
from cred_support_agent.agents.tools import (  # noqa: E402
    LoanStatusTool,
    PolicySearchTool,
    tool_trace_scope,
    trace_lookup,
    trace_retrieval,
)


def _wire(tool: Any, role: str) -> Any:
    """Attach a tool to a role only if the least-autonomy policy permits it."""
    assert_tool_wiring(tool.name, role)
    return tool


def build_composer_llm() -> Any:
    """The Response Composer's model, declared with ``response_format`` (Task 9).

    The two specialist agents answer with tool calls and intermediate notes; only
    the Composer produces the customer-facing reply, so only its model is bound to
    the ``SupportResponse`` schema.
    """
    return build_llm(response_format=SupportResponse)


def build_agents(llm: Any, composer_llm: Any | None = None) -> Dict[str, Agent]:
    """Construct the three crew agents with policy-checked tool wiring."""
    composer_llm = composer_llm or build_composer_llm()
    retrieval = Agent(
        role=RETRIEVAL_AGENT_ROLE,
        goal=(
            "Find the Cred policy passages that answer the customer's question, and "
            "say plainly when the knowledge base does not cover it."
        ),
        backstory=(
            "You are a Cred lending-policy researcher. You never state policy from "
            "memory; every claim you make is traceable to a retrieved passage."
        ),
        tools=[_wire(PolicySearchTool(), RETRIEVAL_AGENT_ROLE)],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
    lookup = Agent(
        role=LOOKUP_AGENT_ROLE,
        goal=(
            "Retrieve the current state of a named Cred loan application and report its "
            "escalation score against the cutoff."
        ),
        backstory=(
            "You are the only member of the crew cleared to read loan application "
            "records. You report what the record says and never speculate about "
            "approval odds."
        ),
        tools=[_wire(LoanStatusTool(), LOOKUP_AGENT_ROLE)],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=3,
    )
    composer = Agent(
        role=COMPOSER_AGENT_ROLE,
        goal=(
            "Merge the policy findings and the application status into one accurate "
            "customer reply that matches the required response schema."
        ),
        backstory=(
            "You write the message the customer actually receives. You hold no tools "
            "of your own and may only use what the specialist agents supplied."
        ),
        tools=[],  # least autonomy: the composer is deliberately tool-free
        llm=composer_llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )
    return {"retrieval": retrieval, "lookup": lookup, "composer": composer}


def _question_block(question: str) -> str:
    return f"{QUESTION_OPEN}\n{question}\n{QUESTION_CLOSE}"


def build_crew(
    question: str, llm: Any, composer_llm: Any | None = None
) -> tuple[Crew, Dict[str, Agent]]:
    """Assemble the three-agent crew for one customer question."""
    agents = build_agents(llm, composer_llm)
    block = _question_block(question)

    retrieval_task = Task(
        description=(
            "Search the Cred policy knowledge base for evidence that answers the "
            f"customer's question.\n{block}\n"
            "Use your knowledge-base search tool. Report only what the retrieved "
            "passages support."
        ),
        expected_output=(
            "A JSON object describing the retrieved policy evidence and the grounded "
            "answer it supports."
        ),
        agent=agents["retrieval"],
    )
    lookup_task = Task(
        description=(
            "If the customer's question names a Cred loan application record id, look "
            f"up that application's current state.\n{block}\n"
            "If no record id is present, report that no lookup was required."
        ),
        expected_output=(
            "A JSON object with the application status, amount and escalation score, or "
            "a short statement that no record id was supplied."
        ),
        agent=agents["lookup"],
    )
    compose_task = Task(
        description=(
            "Merge the policy evidence and the application status into a single reply "
            f"for the customer.\n{block}\n"
            "Use only what the other agents provided. Do not add policy of your own."
        ),
        expected_output="A SupportResponse JSON object.",
        agent=agents["composer"],
        context=[retrieval_task, lookup_task],
        # Task 9: the Composer's draft must parse into SupportResponse. Its model
        # is also declared with response_format=SupportResponse (see
        # build_composer_llm), and run_crew validates the result again in code.
        output_pydantic=SupportResponse,
    )

    crew = Crew(
        agents=[agents["retrieval"], agents["lookup"], agents["composer"]],
        tasks=[retrieval_task, lookup_task, compose_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        cache=False,
    )
    return crew, agents


# --------------------------------------------------------------------------
# Request pipeline
# --------------------------------------------------------------------------


@dataclass
class CrewRunResult:
    response: SupportResponse
    question: str
    masked_question: str
    tools_invoked: List[str] = field(default_factory=list)
    llm_calls: int = 0
    retrieval: Dict[str, Any] | None = None
    lookup: Dict[str, Any] | None = None
    guardrail_flags: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    review: Dict[str, Any] | None = None
    #: The Composer's draft exactly as it entered the review stage.
    draft_answer: str | None = None
    #: True when a test/demo hook deliberately altered the draft (Task 14 demo).
    draft_altered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "masked_question": self.masked_question,
            "response": self.response.model_dump(),
            "tools_invoked": self.tools_invoked,
            "llm_calls": self.llm_calls,
            "guardrail_flags": self.guardrail_flags,
            "budget": self.budget,
            "blocked": self.blocked,
            "review": self.review,
            "retrieval_top1": (self.retrieval or {}).get("top1_similarity"),
            "lookup_record_id": (self.lookup or {}).get("record_id"),
        }


def record_grounded_contexts(
    retrieval: Dict[str, Any] | None, lookup: Dict[str, Any] | None
) -> List[Dict[str, Any]]:
    """Evidence for the output guardrail: retrieved policy plus the record lookup.

    A status answer is grounded in the application record, not in policy prose,
    so the lookup is supplied as evidence whenever one ran - *including* when the
    record was not found. Without that, an honest "I couldn't find that
    application" fails the groundedness check and the customer who mistyped their
    reference is told the policy knowledge base can't support an answer instead.
    """
    contexts = list((retrieval or {}).get("contexts", []))
    if lookup:
        contexts.append(
            {
                "doc_id": "RECORD",
                "text": f"{compose_status_answer(lookup)} {json.dumps(lookup)}",
            }
        )
    return contexts


def _blocked_response(text: str, flags: List[str]) -> SupportResponse:
    return SupportResponse(
        answer=text,
        answer_type="refusal",
        grounded=True,
        citations=[],
        tools_used=[],
        guardrail_flags=flags,
    )


SCHEMA_REFUSAL = (
    "I'm not able to give you a reliable answer right now. My draft reply did not pass "
    "an internal format check, so it has been withheld rather than sent. Please try "
    "again, or contact a Cred support specialist."
)


def run_crew(
    question: str, llm: Any | None = None, composer_llm: Any | None = None
) -> CrewRunResult:
    """Kick off the crew for one (already guardrailed) question."""
    llm = llm or build_llm()
    composer_llm = composer_llm or build_composer_llm()
    crew, _ = build_crew(question, llm, composer_llm)

    # Runtime least-autonomy enforcement is installed at the framework level:
    # CrewAI consults it before every tool execution, with the agent that is
    # actually running, so per-agent permissions are enforced during kickoff().
    install_least_autonomy_hook()

    with tool_trace_scope() as trace:
        output = crew.kickoff()

    raw = getattr(output, "raw", str(output))
    pydantic_output = getattr(output, "pydantic", None)
    blocked = False
    try:
        # Task 9: every crew response is validated against the schema in code.
        response = validate_support_response(pydantic_output or raw)
    except (ValidationError, ValueError):
        # A reply that fails the schema is never delivered as if it were valid.
        response = _blocked_response(SCHEMA_REFUSAL, ["schema:validation_failed"])
        blocked = True

    retrieval = trace_retrieval(trace)
    lookup = trace_lookup(trace)
    tools_invoked = list(dict.fromkeys(entry["tool"] for entry in trace))
    response.tools_used = tools_invoked

    models = {id(m): m for m in (llm, composer_llm)}.values()
    return CrewRunResult(
        response=response,
        question=question,
        masked_question=question,
        tools_invoked=tools_invoked,
        llm_calls=sum(getattr(m, "call_count", 0) for m in models),
        retrieval=retrieval,
        lookup=lookup,
        guardrail_flags=list(response.guardrail_flags),
        blocked=blocked,
    )


def _apply_verdict(result: CrewRunResult, verdict: Dict[str, Any]) -> None:
    """Deliver the review team's revision in place of the Composer's draft."""
    final = verdict.get("final_answer") or result.response.answer
    result.response.answer = final
    result.response.citations = extract_citations(final)
    if is_refusal(final):
        result.response.answer_type = "refusal"


def answer_question(
    question: str,
    *,
    review: bool = True,
    budget: RequestBudget | None = None,
    llm: Any | None = None,
    composer_llm: Any | None = None,
    draft_transform: Callable[[str], str] | None = None,
) -> CrewRunResult:
    """The full production path.

    budget gate -> input guardrails -> crew (Part 2) -> Autogen review (Part 4)
    -> output groundedness gate (Part 2) -> deliver

    The review team sees the Composer's draft first and may revise it; the
    groundedness gate then checks whatever is about to be delivered, so nothing
    reaches the customer unsupported even if a revision were itself faulty.

    ``draft_transform`` is a fault-injection hook for demonstrations and tests: it
    alters the Composer's draft before review (for example to plant an ungrounded
    claim, as the Task 14 demonstration does). Production callers never pass it.
    """
    budget = budget or RequestBudget()
    flags: List[str] = []

    # Runtime layer: reject an oversized request before any model call.
    budget.admit(question)

    guarded = apply_input_guardrails(question)
    flags.extend(guarded.flags)
    masked_question = guarded.detail.get("masked_input", guarded.text)

    if not guarded.allowed:
        response = _blocked_response(guarded.text, flags)
        return CrewRunResult(
            response=response,
            question=question,
            masked_question=masked_question,
            guardrail_flags=flags,
            budget=budget.summary(),
            blocked=True,
        )

    with budget_scope(budget):
        result = run_crew(guarded.text, llm=llm, composer_llm=composer_llm)
    result.question = question
    result.masked_question = guarded.text
    result.guardrail_flags = flags + result.guardrail_flags

    if result.blocked:  # schema validation failed; nothing to review or ground
        result.response.guardrail_flags = list(result.guardrail_flags)
        result.budget = budget.summary()
        return result

    contexts = record_grounded_contexts(result.retrieval, result.lookup)

    if draft_transform is not None:
        result.response.answer = draft_transform(result.response.answer)
        result.draft_altered = True
    result.draft_answer = result.response.answer

    # Part 4 / Task 14: the review team takes the Composer's draft plus the
    # retrieved context, and approves it unchanged or revises it.
    if review:
        from cred_support_agent.review.team import review_draft  # keeps autogen optional

        with budget_scope(budget):
            verdict = review_draft(result.response.answer, contexts)
        result.review = verdict
        if not verdict.get("approved", True):
            _apply_verdict(result, verdict)

    # Part 2 / Task 10: output-side groundedness is the last gate before delivery.
    grounded_check = check_groundedness(result.response.answer, contexts)
    if not grounded_check.allowed:
        result.guardrail_flags.extend(grounded_check.flags)
        result.response = _blocked_response(grounded_check.text, result.guardrail_flags)
        result.blocked = True
    result.response.guardrail_flags = list(result.guardrail_flags)

    result.budget = budget.summary()
    return result
