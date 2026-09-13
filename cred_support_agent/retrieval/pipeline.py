"""Task 4 (grounded generation) + Task 16 (response caching).

Retrieval pulls top-k chunks from a ChromaDB collection, the calibrated
threshold decides whether the evidence is strong enough to answer at all, and
generation is a call to the grounded-generation model (a MOCK_LLM ``BaseLLM``)
that answers only from the retrieved context. An in-memory cache keyed on the
normalised query short-circuits the whole step. Separate counters for vector
retrievals and model calls are the before/after evidence that a hit avoided both.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from cred_support_agent.retrieval.calibration import active_threshold
from cred_support_agent.config import DEFAULT_RETRIEVAL_STRATEGY, DEFAULT_TOP_K
from cred_support_agent.retrieval.generation import FALLBACK_ANSWER, extract_citations
from cred_support_agent.retrieval.generator_llm import MockGenerationLLM, render_generation_prompt
from cred_support_agent.retrieval.indexing import query as index_query
from cred_support_agent.text_utils import normalize_query, split_sentences

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def expand_to_sentences(chunk_text: str, doc_text: str) -> str:
    """Widen a chunk to the complete sentences it overlaps in its parent document.

    Fixed-size chunks cut sentences in half. Retrieval still ranks them well, but
    the generator will not quote half a sentence to a customer, so without this
    step the most relevant passage could be discarded in favour of a complete but
    irrelevant sentence from a lower-ranked chunk. Similarity scores are unchanged:
    the chunk is what was matched; the widened text is what is used as evidence.
    """
    start = doc_text.find(chunk_text)
    if start == -1:
        return chunk_text
    end = start + len(chunk_text)
    pieces: List[str] = []
    cursor = 0
    for sentence in split_sentences(doc_text):
        position = doc_text.find(sentence, cursor)
        if position == -1:
            continue
        cursor = position + len(sentence)
        if position < end and cursor > start:
            pieces.append(sentence)
    return " ".join(pieces) or chunk_text


def retrieve_context(
    text: str, strategy: str = DEFAULT_RETRIEVAL_STRATEGY, top_k: int = DEFAULT_TOP_K
) -> Dict[str, Any]:
    """Retrieve top-k chunks and decide whether they clear the answer threshold."""
    from cred_support_agent.data.knowledge_base import DOCS_BY_ID

    hits = index_query(text, strategy=strategy, top_k=top_k)
    threshold = active_threshold(strategy)
    top1 = hits[0].similarity if hits else 0.0
    in_scope = bool(hits) and top1 >= threshold

    contexts: List[Dict[str, Any]] = []
    for hit in hits:
        context = hit.as_dict()
        parent = DOCS_BY_ID.get(hit.doc_id)
        if parent is not None:
            context["chunk_text"] = hit.text
            context["text"] = expand_to_sentences(hit.text, " ".join(parent["text"].split()))
        contexts.append(context)

    return {
        "query": text,
        "strategy": strategy,
        "threshold": threshold,
        "top1_similarity": round(top1, 4),
        "in_scope": in_scope,
        "contexts": contexts,
        "doc_ids": list(dict.fromkeys(h.doc_id for h in hits)),
    }


# --------------------------------------------------------------------------
# Cache (Task 16)
# --------------------------------------------------------------------------


@dataclass
class GenerationCache:
    """In-memory cache for the grounded-generation step, keyed by normalised query.

    ``generation_calls`` counts how many times the expensive path (retrieval +
    composition) actually ran. A cache hit leaves the counter untouched, which is
    the before/after evidence Task 16 asks for.
    """

    _store: Dict[Tuple[str, str, int], Dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    hits: int = 0
    misses: int = 0
    #: Cache misses that ran the grounded-generation step.
    generation_calls: int = 0
    #: Vector-store retrievals actually performed.
    retrieval_calls: int = 0
    #: Grounded-generation model calls actually made.
    llm_calls: int = 0

    @staticmethod
    def make_key(text: str, strategy: str, top_k: int) -> Tuple[str, str, int]:
        return (normalize_query(text), strategy, top_k)

    def get(self, key: Tuple[str, str, int]) -> Dict[str, Any] | None:
        with self._lock:
            value = self._store.get(key)
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            return dict(value)

    def put(self, key: Tuple[str, str, int], value: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = dict(value)

    def note_generation(self) -> None:
        with self._lock:
            self.generation_calls += 1

    def note_retrieval(self) -> None:
        with self._lock:
            self.retrieval_calls += 1

    def note_llm_call(self) -> None:
        with self._lock:
            self.llm_calls += 1

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
                "lookups": total,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "generation_calls": self.generation_calls,
                "retrieval_calls": self.retrieval_calls,
                "llm_calls": self.llm_calls,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
            self.generation_calls = 0
            self.retrieval_calls = 0
            self.llm_calls = 0


CACHE = GenerationCache()

#: The grounded-generation model. MOCK_LLM: deterministic, keyless, offline.
GENERATOR_LLM = MockGenerationLLM()


# --------------------------------------------------------------------------
# Grounded generation
# --------------------------------------------------------------------------


def grounded_answer(
    text: str,
    strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    top_k: int = DEFAULT_TOP_K,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Answer strictly from retrieved context, or refuse when evidence is weak."""
    key = GenerationCache.make_key(text, strategy, top_k)
    started = time.perf_counter()

    if use_cache:
        cached = CACHE.get(key)
        if cached is not None:
            cached["cached"] = True
            cached["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return cached

    CACHE.note_generation()
    CACHE.note_retrieval()
    retrieval = retrieve_context(text, strategy=strategy, top_k=top_k)

    if not retrieval["in_scope"]:
        # Below the calibrated threshold there is nothing to ground an answer in,
        # so the model is not called at all.
        answer = FALLBACK_ANSWER
        citations: List[str] = []
    else:
        CACHE.note_llm_call()
        answer = GENERATOR_LLM.call(render_generation_prompt(text, retrieval["contexts"]))
        # Cite only what the answer actually quotes, not everything retrieved.
        citations = extract_citations(answer)

    payload = {
        "query": text,
        "answer": answer,
        "in_scope": retrieval["in_scope"],
        "refused": not retrieval["in_scope"],
        "citations": citations,
        "contexts": retrieval["contexts"],
        "top1_similarity": retrieval["top1_similarity"],
        "threshold": retrieval["threshold"],
        "strategy": strategy,
        "cached": False,
    }
    if use_cache:
        CACHE.put(key, payload)
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return payload


def report() -> str:
    """Task 4 demonstration: 5+ in-scope queries plus a fallback query."""
    # The exact five queries Task 5 scores, taken from the one shared definition.
    from cred_support_agent.evaluation.benchmarks import (
        OUT_OF_SCOPE_DEMO_QUERY,
        RETRIEVAL_EVAL_QUERIES,
    )

    in_scope_demo = [query for query, _ in RETRIEVAL_EVAL_QUERIES]
    out_of_scope_demo = OUT_OF_SCOPE_DEMO_QUERY

    lines = ["=" * 78, "GROUNDED GENERATION UNDER MOCK_LLM", "=" * 78]
    lines.append(
        f"collection: {DEFAULT_RETRIEVAL_STRATEGY} (recommended by Task 5)   "
        f"threshold in force: {active_threshold(DEFAULT_RETRIEVAL_STRATEGY)}"
    )
    lines.append(f"generation model: {GENERATOR_LLM.model} (MOCK_LLM, answers only from retrieved context)")
    for text in in_scope_demo:
        result = grounded_answer(text)
        lines.append("")
        lines.append(f"Q: {text}")
        lines.append(f"   top1_similarity={result['top1_similarity']}  in_scope={result['in_scope']}")
        lines.append(f"   citations={result['citations']}")
        lines.append(f"   A: {result['answer']}")
    result = grounded_answer(out_of_scope_demo)
    lines.append("")
    lines.append("--- deliberate out-of-scope query (must trigger fallback) ---")
    lines.append(f"Q: {out_of_scope_demo}")
    lines.append(f"   top1_similarity={result['top1_similarity']}  in_scope={result['in_scope']}")
    lines.append(f"   refused={result['refused']}")
    lines.append(f"   A: {result['answer']}")
    lines.append("")
    lines.append(f"cache stats: {CACHE.stats()}")
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
