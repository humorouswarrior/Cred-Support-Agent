"""Task 14 - Autogen review stage after the CrewAI draft.

A two-agent ``RoundRobinGroupChat`` - Policy-Compliance-Reviewer, then
Final-Editor, bounded at ``max_turns=2`` - inspects the Response Composer's draft
together with the context it was built from, and either approves it unchanged or
revises it. The Final-Editor emits a structured ``ReviewVerdict`` via
``output_content_type``, and the team is constructed with
``custom_message_types=[StructuredMessage[ReviewVerdict]]``; without that
registration the run crashes with "Message type ... is not registered".

What the reviewer checks, sentence by sentence:

* **Evidence** - is the sentence, including every figure in it, supported by the
  retrieved policy context or the application record?
* **Compliance** - does it break one of Cred's own support rules (KB-014): quoting
  approval odds or guaranteeing approval, promising a disbursement date, or
  telling a member their application is flagged for fraud?
* **Citations** - does every ``[source: ...]`` document actually appear in the
  retrieved context?

The Final-Editor then *revises* rather than discards: it removes only the
sentences that failed, keeps everything that passed, and repairs the citation
list. A draft is replaced with a referral only when nothing in it survives.

The model client is a deterministic, keyless ``ChatCompletionClient`` built for
MOCK_LLM. The checks it performs are real; it simply calls nothing over a network.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel

from cred_support_agent.retrieval.generation import extract_citations, is_refusal
from cred_support_agent.safety.guardrails import check_groundedness
from cred_support_agent.schemas import ReviewVerdict
from cred_support_agent.text_utils import split_sentences

REVIEWER_NAME = "Policy_Compliance_Reviewer"
EDITOR_NAME = "Final_Editor"

REVIEW_TAG = "COMPLIANCE_FINDING"

REFERRAL_ANSWER = (
    "I can't confirm that from Cred's policy knowledge base. The draft answer made "
    "claims the retrieved policy text does not support, so it has been withheld. Please "
    "contact a Cred support specialist, who can confirm the position on your account "
    "directly."
)

#: Cred's own support-conduct rules, taken from KB-014 (escalation policy).
COMPLIANCE_RULES: List[Dict[str, Any]] = [
    {
        "rule": "no_approval_promises",
        "description": "support agents never quote an approval probability or guarantee approval",
        "pattern": re.compile(
            r"\bguarantee[ds]?\b|\bwill (?:definitely|certainly|surely) be (?:approved|sanctioned)\b"
            r"|\b(?:approval|sanction) is (?:certain|assured|guaranteed)\b"
            r"|\b\d{1,3}\s*(?:%|percent)\s+(?:chance|likelihood|probability)\b",
            re.IGNORECASE,
        ),
    },
    {
        "rule": "no_disbursement_date_promises",
        "description": "support agents never promise a disbursement date before the sanction letter",
        "pattern": re.compile(
            r"\bwill be disbursed\b|\bfunds will (?:reach|arrive|be credited)\b",
            re.IGNORECASE,
        ),
    },
    {
        "rule": "no_fraud_flag_disclosure",
        "description": "a member is told only that additional verification is running",
        "pattern": re.compile(
            r"\b(?:your|this) (?:application|account|loan) (?:is|has been|was) flagged for fraud",
            re.IGNORECASE,
        ),
    },
]

_CITATION_BLOCK = re.compile(r"\s*\[source:[^\]]*\]\s*$")


def review_sentences(draft: str, contexts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The Policy-Compliance-Reviewer's assessment of one draft."""
    if is_refusal(draft):
        return {"refusal": True, "sentences": [], "invalid_citations": []}

    body = _CITATION_BLOCK.sub("", draft).strip()
    context_ids = {c.get("doc_id") for c in contexts}
    sentences: List[Dict[str, Any]] = []
    for text in split_sentences(body):
        evidence = check_groundedness(text, contexts)
        violations = [r["rule"] for r in COMPLIANCE_RULES if r["pattern"].search(text)]
        problems = []
        if not evidence.allowed:
            problems.append("not supported by the retrieved context")
        if evidence.detail.get("unsupported_numbers"):
            problems.append(f"figures absent from context: {evidence.detail['unsupported_numbers']}")
        problems.extend(f"breaks compliance rule '{v}'" for v in violations)
        sentences.append(
            {
                "text": text,
                "keep": not problems,
                "support": evidence.detail.get("support"),
                "problems": problems,
            }
        )
    citations = extract_citations(draft)
    return {
        "refusal": False,
        "sentences": sentences,
        "citations": citations,
        "invalid_citations": [c for c in citations if c not in context_ids],
    }


def edit_draft(
    draft: str, finding: Dict[str, Any], contexts: Sequence[Dict[str, Any]]
) -> ReviewVerdict:
    """The Final-Editor's verdict: approve unchanged, or revise."""
    if finding.get("refusal"):
        return ReviewVerdict(
            approved=True,
            final_answer=draft,
            reason=(
                "Policy-Compliance-Reviewer found the draft is an explicit refusal that "
                "asserts no policy, so there is nothing to contradict the context. "
                "Delivered unchanged."
            ),
        )

    sentences = finding.get("sentences") or []
    removed = [s for s in sentences if not s["keep"]]
    invalid = finding.get("invalid_citations") or []
    if not removed and not invalid:
        supports = [s["support"] for s in sentences if s["support"] is not None]
        return ReviewVerdict(
            approved=True,
            final_answer=draft,
            reason=(
                f"Policy-Compliance-Reviewer checked all {len(sentences)} sentence(s): each "
                "is supported by the retrieved Cred context"
                + (f" (lowest support {min(supports)})" if supports else "")
                + ", none breaks a KB-014 support rule, and every citation is to a "
                "retrieved document. Delivered unchanged."
            ),
        )

    kept = [s["text"] for s in sentences if s["keep"]]
    notes = [f"removed \"{s['text']}\" ({'; '.join(s['problems'])})" for s in removed]
    if invalid:
        notes.append(f"dropped citation(s) to documents that were not retrieved: {invalid}")

    if not kept:
        return ReviewVerdict(
            approved=False,
            final_answer=REFERRAL_ANSWER,
            reason=(
                "Policy-Compliance-Reviewer found no sentence in the draft that could be "
                "delivered: " + "; ".join(notes) + ". Replaced with a referral to a specialist."
            ),
        )

    # Keep a citation only if it was retrieved and still backs a surviving sentence.
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for context in contexts:
        by_doc.setdefault(context.get("doc_id"), []).append(context)
    citations = [
        c
        for c in (finding.get("citations") or [])
        if c in by_doc and any(check_groundedness(t, by_doc[c]).allowed for t in kept)
    ]
    revised = " ".join(kept) + (f" [source: {', '.join(citations)}]" if citations else "")
    return ReviewVerdict(
        approved=False,
        final_answer=revised,
        reason=(
            f"Policy-Compliance-Reviewer passed {len(kept)} of {len(sentences)} sentence(s) "
            "and the Final-Editor revised the draft: " + "; ".join(notes) + "."
        ),
    )


class MockReviewClient(ChatCompletionClient):
    """Deterministic Autogen model client - no key, no network, no randomness.

    The reviewer turn produces a sentence-by-sentence compliance finding; the
    editor turn turns that finding into a ``ReviewVerdict``.
    """

    def __init__(self) -> None:
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.call_count = 0

    # -- payload plumbing --------------------------------------------------

    @staticmethod
    def _text(message: LLMMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)

    @classmethod
    def _task_payload(cls, messages: Sequence[LLMMessage]) -> Dict[str, Any]:
        """Recover the review packet from the first user turn."""
        for message in messages:
            if isinstance(message, UserMessage):
                text = cls._text(message)
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
        return {}

    @classmethod
    def _finding(cls, messages: Sequence[LLMMessage]) -> Dict[str, Any] | None:
        """Recover the Policy_Compliance_Reviewer's finding from the conversation.

        Autogen hands a teammate's message to the next agent's model as a
        ``UserMessage`` carrying the teammate's name, not as an ``AssistantMessage``,
        so both are searched. This is what makes the Final_Editor act on the
        reviewer's actual output rather than re-running the review itself.
        """
        for message in reversed(messages):
            if isinstance(message, (UserMessage, AssistantMessage)):
                text = cls._text(message)
                if REVIEW_TAG in text:
                    try:
                        return json.loads(text.split(REVIEW_TAG, 1)[1].strip())
                    except json.JSONDecodeError:
                        return None
        return None

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        self.call_count += 1
        packet = self._task_payload(messages)
        draft = packet.get("draft_answer", "")
        contexts = packet.get("contexts") or []

        wants_structured = isinstance(json_output, type) and issubclass(json_output, BaseModel)
        if wants_structured:
            # Final-Editor turn: act on the finding the reviewer actually sent.
            finding = self._finding(messages)
            if finding is None:  # pragma: no cover - reviewer turn missing
                finding = review_sentences(draft, contexts)
            content = edit_draft(draft, finding, contexts).model_dump_json()
        else:
            # Policy-Compliance-Reviewer turn: report the finding.
            content = f"{REVIEW_TAG} {json.dumps(review_sentences(draft, contexts))}"

        # Task 15: the review team's model calls count against the same per-request
        # budget as the crew's.
        from cred_support_agent.safety.governance import current_budget

        budget = current_budget()
        if budget is not None:
            budget.charge(
                "\n".join(self._text(m) for m in messages),
                content,
                label="autogen-review:" + ("final_editor" if wants_structured else "policy_compliance_reviewer"),
            )

        usage = RequestUsage(
            prompt_tokens=sum(len(self._text(m)) // 4 for m in messages),
            completion_tokens=len(content) // 4,
        )
        self._total = RequestUsage(
            prompt_tokens=self._total.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total.completion_tokens + usage.completion_tokens,
        )
        self._usage = usage
        return CreateResult(
            finish_reason="stop", content=content, usage=usage, cached=False
        )

    async def create_stream(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        result = await self.create(*args, **kwargs)
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._total

    def count_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return sum(len(self._text(m)) // 4 for m in messages)

    def remaining_tokens(self, messages: Sequence[LLMMessage], **kwargs: Any) -> int:
        return 8192 - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelInfo:  # pragma: no cover - legacy alias
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            structured_output=True,
            family="unknown",
            multiple_system_messages=True,
        )


def build_review_team() -> tuple[RoundRobinGroupChat, MockReviewClient]:
    """Two agents, bounded at max_turns=2, with the structured message registered."""
    client = MockReviewClient()

    reviewer = AssistantAgent(
        name=REVIEWER_NAME,
        model_client=client,
        description="Checks a draft Cred support answer against the retrieved policy context.",
        system_message=(
            "You are a Cred policy compliance reviewer. Given a draft answer and the "
            "policy context it was built from, assess the draft sentence by sentence: "
            "is each claim, including every figure, supported by that context; does "
            "any sentence quote approval odds, promise a disbursement date or disclose "
            "a fraud flag (all forbidden by Cred's support rules); and is every cited "
            "document one that was actually retrieved? Report the finding only; do not "
            "rewrite the answer."
        ),
    )
    editor = AssistantAgent(
        name=EDITOR_NAME,
        model_client=client,
        description="Issues the final structured verdict on a Cred support answer.",
        system_message=(
            "You are the final editor for Cred support replies. Using the compliance "
            "finding, approve the draft unchanged if every sentence passed. Otherwise "
            "revise it: remove only the sentences that failed, keep the rest, and "
            "repair the citations; if nothing survives, refer the member to a "
            "specialist. Always return the structured verdict."
        ),
        output_content_type=ReviewVerdict,
    )

    team = RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,  # one reviewer turn, one editor turn - the chat is bounded
        # Without registering the structured message type the run crashes.
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    return team, client


async def review_draft_async(
    draft_answer: str, contexts: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run the bounded review chat over one draft and return the verdict."""
    team, client = build_review_team()
    packet = json.dumps(
        {
            "draft_answer": draft_answer,
            "contexts": [
                {"doc_id": c.get("doc_id", ""), "text": c.get("text", "")} for c in contexts
            ],
        }
    )
    result = await team.run(task=packet)

    verdict: ReviewVerdict | None = None
    transcript: List[Dict[str, str]] = []
    for message in result.messages:
        content = getattr(message, "content", None)
        source = getattr(message, "source", "user")
        if isinstance(content, ReviewVerdict):
            verdict = content
            transcript.append({"source": source, "content": content.model_dump_json()})
        else:
            transcript.append({"source": source, "content": str(content)[:600]})

    if verdict is None:  # pragma: no cover - defensive
        verdict = ReviewVerdict(
            approved=True, final_answer=draft_answer, reason="No verdict emitted; draft kept."
        )

    return {
        "approved": verdict.approved,
        "final_answer": verdict.final_answer,
        "reason": verdict.reason,
        "turns": len(result.messages),
        "stop_reason": result.stop_reason,
        "model_calls": client.call_count,
        "transcript": transcript,
    }


def review_draft(draft_answer: str, contexts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Synchronous wrapper used by the crew pipeline."""
    return asyncio.run(review_draft_async(draft_answer, contexts))


def plant_claim(claim: str):
    """Build a draft transform that inserts ``claim`` before the citation block.

    Used only to demonstrate and test the revise path: it plants a deliberately
    ungrounded sentence into a real Response Composer draft.
    """

    def transform(draft: str) -> str:
        body, sep, rest = draft.rpartition(" [source:")
        return f"{body} {claim}{sep}{rest}" if sep else f"{draft} {claim}"

    return transform


#: Task 14 demonstration: real sample queries through the full crew pipeline.
REVIEW_DEMO_CASES: List[Dict[str, Any]] = [
    {
        "label": "approve unchanged",
        "query": "Is there a prepayment penalty if I foreclose my loan early?",
        "planted_claim": None,
    },
    {
        "label": "revise: planted ungrounded claim",
        "query": "What is the minimum balance I must maintain in a Cred savings account?",
        "planted_claim": (
            "Cred also waives the minimum balance entirely for customers who hold a Cred "
            "credit card and pays 7 percent interest on it."
        ),
    },
    {
        "label": "revise: planted compliance breach",
        "query": "What is the status of CRED-LN-0005?",
        "planted_claim": (
            "Your application is guaranteed to be approved and the funds will be "
            "disbursed by Friday."
        ),
    },
]


def report() -> str:
    """Run the review stage on real crew drafts: one approved, two revised."""
    from cred_support_agent.agents.crew import answer_question

    lines = [
        "=" * 78,
        "TASK 14 - AUTOGEN REVIEW STAGE (RoundRobinGroupChat, max_turns=2)",
        "=" * 78,
        "Each case runs a real sample query through the CrewAI crew. The Response",
        "Composer's draft plus the retrieved context go to the review team. In the",
        "revise cases a deliberately ungrounded sentence is planted into the draft first.",
    ]
    for case in REVIEW_DEMO_CASES:
        transform = plant_claim(case["planted_claim"]) if case["planted_claim"] else None
        result = answer_question(case["query"], draft_transform=transform)
        verdict = result.review or {}
        lines.append(f"\n--- case: {case['label']} ---")
        lines.append(f"  query          : {case['query']}")
        if case["planted_claim"]:
            lines.append(f"  planted claim  : {case['planted_claim']}")
        lines.append(f"  composer draft : {result.draft_answer}")
        lines.append(f"  approved       : {verdict.get('approved')}")
        lines.append(f"  reason         : {verdict.get('reason')}")
        lines.append(f"  final_answer   : {verdict.get('final_answer')}")
        lines.append(f"  delivered      : {result.response.answer}")
        lines.append(
            f"  chat           : {verdict.get('turns')} messages, stop_reason="
            f"{verdict.get('stop_reason')!r}"
        )
        for entry in verdict.get("transcript", []):
            lines.append(f"    [{entry['source']}] {entry['content'][:200]}")
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
