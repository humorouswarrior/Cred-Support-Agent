"""The grounded-generation model (Task 4), as a real custom ``BaseLLM``.

The brief lists grounded generation as one of the four language-model stages
that must default to MOCK_LLM. Routing it through a ``BaseLLM`` subclass, rather
than calling the composition function directly, makes it a genuine model call
with a real prompt - which is also what lets the Task 16 cache demonstrate that a
hit avoids a redundant LLM call, not just a function call.

The prompt carries the question and the retrieved passages; the mock model reads
them back from the model-visible user turn and answers extractively from the
passages alone.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr

from cred_support_agent.retrieval.generation import FALLBACK_ANSWER, compose_grounded_answer

GROUNDED_GENERATION_SYSTEM_PROMPT = """\
You answer questions for Cred support using ONLY the retrieved policy passages
provided. Do not use outside knowledge. Quote or closely follow the passages, and
end the answer with the ids of the passages you used, as [source: KB-xxx, ...].
If the passages do not answer the question, reply exactly:
"I don't know." followed by a short explanation.
"""

PACKET_MARKER = "GROUNDING_PACKET:"


def render_generation_prompt(question: str, contexts: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    packet = {
        "question": question,
        "passages": [{"doc_id": c.get("doc_id"), "text": c.get("text"), "similarity": c.get("similarity", 0.0)} for c in contexts],
    }
    return [
        {"role": "system", "content": GROUNDED_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"{PACKET_MARKER}\n{json.dumps(packet)}"},
    ]


class MockGenerationLLM(BaseLLM):
    """Deterministic, keyless grounded-generation model."""

    llm_type: str = "cred-mock-generation"
    _call_count: int = PrivateAttr(default=0)

    def __init__(self, **data: Any) -> None:
        data.setdefault("model", "mock-llm/cred-grounded-generation-v1")
        data.setdefault("temperature", 0.0)
        super().__init__(**data)

    @property
    def call_count(self) -> int:
        return self._call_count

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    def call(self, messages: Any, tools: Any = None, callbacks: Any = None,
             available_functions: Any = None, from_task: Any = None,
             from_agent: Any = None, response_model: Any = None) -> str:
        self._call_count += 1
        # Read the packet from the model-visible user turn, never the system prompt.
        for message in reversed(self._format_messages(messages)):
            content = message.get("content") or ""
            if message.get("role") == "user" and PACKET_MARKER in content:
                packet = json.loads(content.split(PACKET_MARKER, 1)[1])
                contexts = packet.get("passages") or []
                answer = compose_grounded_answer(packet.get("question", ""), contexts)
                self._charge(content, answer)
                return answer
        return FALLBACK_ANSWER

    def _charge(self, prompt_text: str, completion_text: str) -> None:
        """Bill this call to the active per-request budget (Task 15 runtime layer)."""
        from cred_support_agent.safety.governance import current_budget

        budget = current_budget()
        if budget is not None:
            budget.charge(prompt_text, completion_text, label=self.llm_type)
