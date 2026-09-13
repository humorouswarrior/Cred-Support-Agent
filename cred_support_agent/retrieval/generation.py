"""Deterministic MOCK_LLM generation.

MOCK_LLM answers are produced *extractively* from retrieved evidence: sentences
are scored against the query and the highest-scoring ones are stitched into an
answer that carries its own ``[KB-xxx]`` citations. That design is what makes
the whole system gradeable with zero keys - the answer is a pure function of the
retrieved context, so groundedness checks, judge scores and cache hits are all
reproducible run to run.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from cred_support_agent.text_utils import content_words, split_sentences

FALLBACK_ANSWER = (
    "I don't know. I could not find anything in the Cred policy knowledge base that "
    "supports an answer to this question, so I am not going to guess. Please rephrase "
    "the question or contact a Cred support specialist."
)

REFUSAL_PREFIX = "I don't know."

#: Openings of every fixed refusal the system can deliver: retrieval fallback,
#: injection block, review-team referral, groundedness refusal, schema refusal.
#: A refusal asserts no policy, so the groundedness checks exempt it.
REFUSAL_PREFIXES = (
    REFUSAL_PREFIX,
    "I can't act on that request.",
    "I can't confirm that from Cred's policy knowledge base.",
    "I'm not able to answer that from the Cred policy knowledge base.",
    "I'm not able to give you a reliable answer right now.",
)


def is_refusal(text: str) -> bool:
    """Is this one of the system's fixed refusal messages?"""
    return (text or "").strip().startswith(REFUSAL_PREFIXES)

#: A supporting sentence must score at least this share of the best sentence's
#: score to be included. Chosen by comparing answers at 0.62, 0.75 and 0.85: the
#: 15-query evaluation scored identically at all three, and 0.85 was the value that
#: kept off-topic second sentences out of the answer.
RELATIVE_SENTENCE_CUTOFF = 0.85


def is_complete_sentence(sentence: str) -> bool:
    """Is this a whole sentence, or a fragment left by a chunk boundary?

    Fixed-size chunking slices mid-sentence, so the first and last sentence of a
    chunk are often partial. Quoting one to a customer produces an answer that
    trails off ("...must be"), which reads as a broken system even though the
    retrieval was correct. The overlap window means the complete form of the
    sentence usually survives in a neighbouring chunk.
    """
    text = sentence.strip()
    if len(text) < 12:
        return False
    if not text.endswith((".", "!", "?")):
        return False
    first = text[0]
    return first.isupper() or first.isdigit() or first in "\"'(["


#: Weight of exact query-term overlap relative to semantic similarity. Overlap only
#: breaks near-ties; on its own it prefers sentences that repeat the question's
#: words ("Fixed-rate Personal Loans ... charge") over the sentence that answers it.
LEXICAL_WEIGHT = 0.15


def rank_evidence_sentences(
    query: str, contexts: Sequence[Dict[str, Any]], limit: int = 3
) -> List[Dict[str, Any]]:
    """Rank every complete sentence in the retrieved context by relevance to the query.

    Relevance is the cosine similarity between the query and the sentence, using the
    same local SentenceTransformers model that ranks chunks during retrieval, plus a
    small exact-term-overlap bonus. Sentence-level semantic ranking matters because a
    retrieved chunk usually carries several sentences and only one of them answers the
    question; ranking by word overlap alone picks whichever sentence repeats the most
    of the question's words.
    """
    from cred_support_agent.retrieval.embeddings import cosine_similarity, embed

    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rank, ctx in enumerate(contexts):
        for sentence in split_sentences(ctx["text"]):
            key = sentence.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"sentence": sentence.strip(), "doc_id": ctx.get("doc_id", "KB-UNKNOWN"), "rank": rank})
    if not candidates:
        return []

    # Quote whole sentences only; fall back to fragments only if nothing else exists.
    complete = [c for c in candidates if is_complete_sentence(c["sentence"])] or candidates

    query_terms = set(content_words(query))
    query_vector = embed([query])[0]
    sentence_vectors = embed([c["sentence"] for c in complete])
    for candidate, vector in zip(complete, sentence_vectors):
        overlap = len(query_terms & set(content_words(candidate["sentence"])))
        lexical = overlap / max(len(query_terms), 1)
        candidate["score"] = round(cosine_similarity(query_vector, vector) + LEXICAL_WEIGHT * lexical, 4)

    complete.sort(key=lambda c: (-c["score"], c["rank"], c["doc_id"], c["sentence"]))
    return [{k: c[k] for k in ("sentence", "doc_id", "score")} for c in complete[:limit]]


def compose_grounded_answer(
    query: str, contexts: Sequence[Dict[str, Any]], limit: int = 3
) -> str:
    """Compose a policy answer using only sentences present in ``contexts``."""
    if not contexts:
        return FALLBACK_ANSWER
    top = rank_evidence_sentences(query, contexts, limit=limit)
    if not top or top[0]["score"] <= 0.0:
        return FALLBACK_ANSWER
    # Keep only sentences that stay close to the best one, so a loosely related
    # second or third sentence from the same chunk doesn't dilute a precise answer.
    cutoff = top[0]["score"] * RELATIVE_SENTENCE_CUTOFF
    top = [item for item in top if item["score"] >= cutoff] or top[:1]
    body = " ".join(item["sentence"] for item in top)
    citations = []
    for item in top:
        if item["doc_id"] not in citations:
            citations.append(item["doc_id"])
    return f"{body} [source: {', '.join(citations)}]"


def compose_status_answer(lookup: Dict[str, Any]) -> str:
    """Render a status-lookup payload as a support-desk sentence."""
    if lookup.get("found") is False:
        return (
            f"I could not find a Cred loan application with record id "
            f"{lookup.get('record_id', 'the supplied id')}. Please re-check the "
            "application reference and try again."
        )
    escalate = lookup.get("escalation_recommended")
    escalation_line = (
        "This application is above the escalation cutoff, so it is flagged for manual "
        "review by a Cred specialist."
        if escalate
        else "This application is below the escalation cutoff and is progressing on the "
        "normal service track."
    )
    return (
        f"Application {lookup['record_id']} ({lookup['category']}) is currently "
        f"'{lookup['status']}' for a requested amount of INR "
        f"{lookup['loan_amount_inr']:,}. It was created "
        f"{lookup['days_since_created']} day(s) ago and carries an escalation score of "
        f"{lookup['escalation_score']} against a cutoff of "
        f"{lookup['escalation_threshold']}. {escalation_line}"
    )


def merge_answers(policy_answer: str | None, status_answer: str | None) -> str:
    """Response Composer behaviour: merge the two agent outputs into one reply."""
    parts = [p.strip() for p in (status_answer, policy_answer) if p and p.strip()]
    if not parts:
        return FALLBACK_ANSWER
    unique: List[str] = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    return " ".join(unique)


_CITATION_BLOCK = re.compile(r"\[source:\s*([^\]]+)\]\s*$")


def extract_citations(answer: str) -> List[str]:
    """Read back the doc ids the composed answer actually cites."""
    match = _CITATION_BLOCK.search(answer.strip())
    if not match:
        return []
    return [c.strip() for c in match.group(1).split(",") if c.strip()]
