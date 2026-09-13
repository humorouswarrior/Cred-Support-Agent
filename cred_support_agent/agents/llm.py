"""MOCK_LLM - a real custom CrewAI LLM built by extending ``BaseLLM``.

This is a proper LLM implementation, not an interception shim: CrewAI drives it
through the ordinary ``BaseLLM.call`` contract, it advertises native function
calling, it emits tool calls, it consumes the tool results CrewAI feeds back,
and it returns the final answer. It simply happens to be deterministic.

Two pitfalls called out in the brief are avoided explicitly:

1. **Tool output is detected from model-generated content, never from the system
   prompt template.** ``_collect_observations`` reads only ``role="tool"``
   messages - the actual tool results CrewAI appends - and never inspects the
   system prompt, and never greps for scaffolding tokens like "Observation:".
   See ``_conversation_view``.

2. **Tool dispatch is driven by the declared argument schema, never by matching
   tool-name substrings.** ``_plan_tool_calls`` reads each tool's JSON-Schema
   ``parameters`` block and fills every ``required`` property with a candidate
   value that actually validates against that property's declared ``type``,
   ``pattern``, ``minLength`` and ``enum``. A tool whose required arguments
   cannot be satisfied from the request is simply not dispatchable. The tool's
   *name* is used only as the identifier to call back with.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, PrivateAttr

from cred_support_agent.retrieval.generation import (
    FALLBACK_ANSWER,
    compose_grounded_answer,
    compose_status_answer,
    extract_citations,
    merge_answers,
)
from cred_support_agent.safety.governance import current_budget

#: Delimiters the crew uses to hand the customer's turn to the model. Reading
#: these is reading the *user* message, not the framework's system template.
QUESTION_OPEN = "<<<CUSTOMER_QUESTION>>>"
QUESTION_CLOSE = "<<<END_CUSTOMER_QUESTION>>>"

#: Envelope kinds the specialist agents emit and the composer consumes.
KIND_POLICY = "policy_context"
KIND_STATUS = "loan_status"

_RECORD_ID = re.compile(r"\b([A-Z]{2,6}-[A-Z]{2,4}-\d{3,6})\b", re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


# --------------------------------------------------------------------------
# Candidate values extracted from the request, before any tool is considered
# --------------------------------------------------------------------------


class Candidate:
    """A value the model believes it can pass as a tool argument."""

    __slots__ = ("value", "kind", "specificity")

    def __init__(self, value: Any, kind: str, specificity: float) -> None:
        self.value = value
        self.kind = kind
        self.specificity = specificity

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Candidate({self.kind}={self.value!r}, spec={self.specificity})"


def extract_candidates(request_text: str) -> List[Candidate]:
    """Pull typed candidate argument values out of the customer's turn."""
    candidates: List[Candidate] = []
    for match in _RECORD_ID.finditer(request_text):
        # Normalise the way a real model would before emitting a tool call.
        candidates.append(Candidate(match.group(1).upper(), "identifier", 1.0))
    cleaned = " ".join(request_text.split()).strip()
    if cleaned:
        candidates.append(Candidate(cleaned, "free_text", 0.2))
    return candidates


def _satisfies_property(value: Any, prop: Dict[str, Any]) -> bool:
    """Does ``value`` validate against this declared JSON-Schema property?"""
    declared_type = prop.get("type")
    if isinstance(declared_type, list):
        types = set(declared_type)
    elif declared_type:
        types = {declared_type}
    else:
        types = set()

    if types and "string" in types:
        if not isinstance(value, str):
            return False
    elif types and "integer" in types:
        if not isinstance(value, int):
            return False
    elif types and "number" in types:
        if not isinstance(value, (int, float)):
            return False

    if isinstance(value, str):
        pattern = prop.get("pattern")
        if pattern and not re.fullmatch(pattern, value):
            return False
        min_length = prop.get("minLength")
        if min_length is not None and len(value) < int(min_length):
            return False
        max_length = prop.get("maxLength")
        if max_length is not None and len(value) > int(max_length):
            return False

    enum = prop.get("enum")
    if enum is not None and value not in enum:
        return False
    return True


def _is_shape_constrained(prop: Dict[str, Any]) -> bool:
    """Does the property declare the *shape* of its value, not just its size?

    ``pattern``/``enum``/``format`` say "this argument looks like X". ``minLength``
    only says "this argument is at least this long", which a free-text question
    satisfies as readily as an identifier does.
    """
    return bool(prop.get("pattern") or prop.get("enum") or prop.get("format"))


def _affinity(candidate: "Candidate", prop: Dict[str, Any]) -> float:
    """How well does this candidate suit what the property declares it wants?

    A shape-constrained property wants the most specific value that satisfies its
    shape (an identifier). An unconstrained string property is a free-text slot,
    so the customer's question is the natural fill and an identifier is a poor
    substitute - which is what stops the policy-search tool being handed a bare
    record id as its search query.
    """
    if _is_shape_constrained(prop):
        return candidate.specificity
    return 1.0 if candidate.kind == "free_text" else 0.3


def _property_strictness(prop: Dict[str, Any]) -> float:
    """How constrained is this declared property? Drives dispatch preference."""
    score = 0.0
    if prop.get("pattern"):
        score += 1.0
    if prop.get("enum"):
        score += 0.8
    if prop.get("minLength"):
        score += 0.2
    if prop.get("format"):
        score += 0.2
    return score


def plan_tool_calls(
    request_text: str, tools: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Choose a tool purely from declared argument schemas.

    Returns CrewAI-native tool-call dicts ``{"id", "name", "input"}``. Returns an
    empty list when no declared schema can be satisfied from the request.
    """
    candidates = extract_candidates(request_text)
    if not candidates:
        return []

    scored: List[Tuple[float, str, Dict[str, Any]]] = []
    for tool in tools:
        function = tool.get("function", tool) or {}
        name = function.get("name")
        if not name:
            continue
        parameters = function.get("parameters") or {}
        properties: Dict[str, Any] = parameters.get("properties") or {}
        required: List[str] = list(parameters.get("required") or properties.keys())

        args: Dict[str, Any] = {}
        fit = 0.0
        satisfiable = True
        for prop_name in required:
            prop = properties.get(prop_name) or {}
            best: Candidate | None = None
            best_affinity = -1.0
            for candidate in candidates:
                if not _satisfies_property(candidate.value, prop):
                    continue
                affinity = _affinity(candidate, prop)
                if affinity > best_affinity:
                    best, best_affinity = candidate, affinity
            if best is None:
                satisfiable = False
                break
            args[prop_name] = best.value
            fit += best_affinity * (1.0 + _property_strictness(prop))

        if not satisfiable:
            continue
        # Optional properties are filled only when a candidate clearly fits.
        for prop_name, prop in properties.items():
            if prop_name in args:
                continue
            for candidate in candidates:
                if candidate.specificity >= 1.0 and _satisfies_property(candidate.value, prop):
                    args[prop_name] = candidate.value
                    break
        scored.append((fit, name, args))

    if not scored:
        return []
    scored.sort(key=lambda row: (-row[0], row[1]))
    _, name, args = scored[0]
    # Derived from the call itself rather than random, so reruns are byte-identical.
    digest = hashlib.sha1(f"{name}:{json.dumps(args, sort_keys=True)}".encode("utf-8")).hexdigest()
    return [{"id": f"call_{digest[:12]}", "name": name, "input": args}]


# --------------------------------------------------------------------------
# The LLM
# --------------------------------------------------------------------------


class MockLLM(BaseLLM):
    """Deterministic, keyless, network-free CrewAI LLM.

    Set ``CRED_LLM_BACKEND=mock`` (the default) and nothing here ever touches a
    network socket or an API key.
    """

    llm_type: str = "cred-mock-llm"
    answer_sentence_limit: int = 3
    #: Pydantic model the reply must conform to (Task 9). Set on the Response
    #: Composer's model; the specialist agents answer with tool calls instead.
    response_format: Any = None

    _call_count: int = PrivateAttr(default=0)
    _tool_call_count: int = PrivateAttr(default=0)
    _last_request: str = PrivateAttr(default="")
    _tools_invoked: List[str] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any) -> None:
        data.setdefault("model", "mock-llm/cred-deterministic-v1")
        data.setdefault("temperature", 0.0)
        super().__init__(**data)

    # -- CrewAI capability advertisement ----------------------------------

    def supports_function_calling(self) -> bool:
        """Native tool calling: CrewAI hands us JSON-Schema tools directly."""
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    # -- introspection used by demos and tests ----------------------------

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def tool_call_count(self) -> int:
        return self._tool_call_count

    @property
    def tools_invoked(self) -> List[str]:
        return list(self._tools_invoked)

    def reset_counters(self) -> None:
        self._call_count = 0
        self._tool_call_count = 0
        self._tools_invoked = []

    # -- message handling --------------------------------------------------

    @staticmethod
    def _conversation_view(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Split the conversation into (request text, tool observations).

        Only ``user`` turns and ``tool`` results are read. The ``system`` turn -
        CrewAI's prompt template, which is where the "Observation:" scaffolding
        lives - is deliberately never inspected, because detecting tool output by
        pattern-matching the template is exactly the failure mode this project is
        required to avoid.
        """
        request_parts: List[str] = []
        observations: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user" and isinstance(content, str):
                request_parts.append(content)
            elif role == "tool":
                observations.append(
                    {
                        "tool": message.get("name", "unknown"),
                        "content": content if isinstance(content, str) else str(content),
                    }
                )
        request_text = "\n".join(request_parts)
        return request_text, observations

    @staticmethod
    def _customer_question(request_text: str) -> str:
        """Read the customer turn out of the delimited block the crew supplies."""
        start = request_text.find(QUESTION_OPEN)
        end = request_text.find(QUESTION_CLOSE)
        if start != -1 and end > start:
            return request_text[start + len(QUESTION_OPEN) : end].strip()
        return request_text.strip()

    # -- payload classification (by shape, not by tool name) ---------------

    @staticmethod
    def _classify_payload(payload: Any) -> str | None:
        """Identify a tool result by the fields it carries, not by its source."""
        if not isinstance(payload, dict):
            return None
        keys = set(payload)
        if {"escalation_score", "status", "loan_amount_inr"} <= keys:
            return KIND_STATUS
        if {"contexts", "top1_similarity"} <= keys:
            return KIND_POLICY
        if payload.get("kind") in (KIND_STATUS, KIND_POLICY):
            return str(payload["kind"])
        return None

    @classmethod
    def _decode_observations(cls, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        decoded: List[Dict[str, Any]] = []
        for observation in observations:
            for blob in cls._json_blobs(observation["content"]):
                kind = cls._classify_payload(blob)
                if kind:
                    decoded.append({"kind": kind, "payload": blob, "tool": observation["tool"]})
        return decoded

    @staticmethod
    def _json_blobs(text: str) -> List[Dict[str, Any]]:
        blobs: List[Dict[str, Any]] = []
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return [parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        for match in _JSON_OBJECT.finditer(text):
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                blobs.append(parsed)
        return blobs

    # -- answer composition ------------------------------------------------

    def _envelope_from_observations(
        self, question: str, decoded: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Specialist-agent output: a structured envelope the composer can merge."""
        for item in decoded:
            if item["kind"] == KIND_STATUS:
                payload = item["payload"]
                return {
                    "kind": KIND_STATUS,
                    "answer": compose_status_answer(payload),
                    "record_id": payload.get("record_id"),
                    "status": payload.get("status"),
                    "loan_amount_inr": payload.get("loan_amount_inr"),
                    "escalation_score": payload.get("escalation_score"),
                    "escalation_recommended": bool(payload.get("escalation_recommended")),
                    "found": bool(payload.get("found", True)),
                    "tool": item["tool"],
                }
        for item in decoded:
            if item["kind"] == KIND_POLICY:
                payload = item["payload"]
                contexts = payload.get("contexts") or []
                if isinstance(payload.get("answer"), str):
                    # The tool already ran grounded generation (possibly from
                    # cache); re-composing would throw that work away.
                    answer = payload["answer"]
                elif payload.get("in_scope"):
                    answer = compose_grounded_answer(
                        question, contexts, limit=self.answer_sentence_limit
                    )
                else:
                    answer = FALLBACK_ANSWER
                return {
                    "kind": KIND_POLICY,
                    "answer": answer,
                    "citations": extract_citations(answer),
                    "in_scope": bool(payload.get("in_scope")),
                    "top1_similarity": payload.get("top1_similarity", 0.0),
                    "contexts": contexts,
                    "tool": item["tool"],
                }
        return {"kind": KIND_POLICY, "answer": FALLBACK_ANSWER, "citations": [], "in_scope": False}

    def _compose_from_context(self, question: str, request_text: str) -> Dict[str, Any]:
        """Composer-agent output: merge the envelopes found in the task context."""
        envelopes = [
            blob
            for blob in self._json_blobs(request_text)
            if blob.get("kind") in (KIND_POLICY, KIND_STATUS)
        ]
        policy = next((e for e in envelopes if e["kind"] == KIND_POLICY), None)
        status = next((e for e in envelopes if e["kind"] == KIND_STATUS), None)

        policy_answer = policy.get("answer") if policy else None
        status_answer = status.get("answer") if status else None
        if policy_answer == FALLBACK_ANSWER and status_answer:
            policy_answer = None  # a status answer stands on its own

        merged = merge_answers(policy_answer, status_answer)
        # Label by what actually made it into the reply, not by which envelopes
        # arrived: a policy envelope whose answer was dropped contributes nothing.
        if policy_answer is None:
            policy = None
        if policy and status:
            answer_type = "policy_and_status"
        elif status:
            answer_type = "status"
        elif policy:
            answer_type = "refusal" if not policy.get("in_scope") else "policy"
        else:
            answer_type = "refusal"
        if merged == FALLBACK_ANSWER:
            answer_type = "refusal"

        citations = list(policy.get("citations") or []) if policy else []
        tools_used = [e["tool"] for e in envelopes if e.get("tool")]
        return {
            "answer": merged,
            "answer_type": answer_type,
            "grounded": bool(policy and policy.get("in_scope")) or bool(status),
            "citations": citations,
            "tools_used": tools_used,
            "record_id": status.get("record_id") if status else None,
            "escalation_recommended": bool(status.get("escalation_recommended")) if status else False,
            "confidence": float(policy.get("top1_similarity", 0.0)) if policy else 0.0,
            "guardrail_flags": [],
        }

    # -- the contract ------------------------------------------------------

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> Any:
        formatted = self._format_messages(messages)
        self._call_count += 1
        request_text, observations = self._conversation_view(formatted)
        question = self._customer_question(request_text)

        # Step 1: no tool results yet and tools are on the table -> emit a call,
        # chosen by declared argument schema.
        if tools and not observations:
            planned = plan_tool_calls(question, tools)
            if planned:
                self._tool_call_count += 1
                self._tools_invoked.extend(call["name"] for call in planned)
                self._charge(request_text, json.dumps(planned))
                return planned

        # Step 1b: tools were offered but no declared schema could be satisfied
        # from this request. Say so plainly rather than inventing a tool call or
        # answering from nowhere.
        if tools and not observations:
            output = (
                "No Cred loan application record id was present in this request, so no "
                "status lookup was performed."
            )
            self._charge(request_text, output)
            return output

        # Step 2: tool results are present -> summarise them into an envelope.
        if observations:
            decoded = self._decode_observations(observations)
            if decoded:
                envelope = self._envelope_from_observations(question, decoded)
                output = json.dumps(envelope)
                self._charge(request_text, output)
                return self._finalise(output, envelope, response_model)

        # Step 3: no tools of our own -> merge whatever context we were handed.
        merged = self._compose_from_context(question, request_text)
        target = response_model or self.response_format
        if target is not None and isinstance(target, type) and issubclass(target, BaseModel):
            instance = target.model_validate(self._project(merged, target))
            output = instance.model_dump_json()
            self._charge(request_text, output)
            return output
        output = merged["answer"]
        self._charge(request_text, output)
        return output

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _project(merged: Dict[str, Any], target: type[BaseModel]) -> Dict[str, Any]:
        """Keep only the fields the target schema declares."""
        fields = set(target.model_fields)
        projected = {k: v for k, v in merged.items() if k in fields}
        for name, field in target.model_fields.items():
            if name not in projected and field.is_required():
                projected[name] = "" if field.annotation is str else None
        return projected

    def _finalise(self, output: str, envelope: Dict[str, Any], response_model: Any) -> Any:
        if (
            response_model is not None
            and isinstance(response_model, type)
            and issubclass(response_model, BaseModel)
        ):
            return response_model.model_validate(self._project(envelope, response_model))
        return output

    def _charge(self, prompt_text: str, completion_text: str) -> None:
        """Bill this call against the active per-request runtime budget."""
        budget = current_budget()
        if budget is not None:
            budget.charge(prompt_text, completion_text, label=self.llm_type)


def build_llm(**overrides: Any) -> BaseLLM:
    """Return the configured LLM. MOCK_LLM unless a backend is opted into."""
    from cred_support_agent.config import LLM_BACKEND, is_mock_backend

    if is_mock_backend():
        return MockLLM(**overrides)

    # Optional, opt-in only. Every acceptance check in this repo passes without
    # ever reaching this branch.
    from crewai import LLM  # pragma: no cover - requires a key and a network

    return LLM(model=LLM_BACKEND, temperature=0.0, **overrides)  # pragma: no cover
