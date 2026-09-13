"""Task 2 - Cred domain knowledge base.

The documents themselves live as plain files in ``data/knowledge_base/``, one
Markdown file per document, so they can be read, reviewed and edited as documents
rather than as Python string literals. Each file carries a small front-matter block
(``doc_id``, ``topic``, ``title``) followed by the document text.

Fourteen original policy documents cover the twelve mandatory topics plus two
operational extras (loan application lifecycle, escalation policy) that the support
crew needs in order to answer status questions in policy language. Every document
is 2-5 sentences of original wording written for this brief. All names, amounts and
reference numbers are fabricated; no real customer data appears anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from cred_support_agent.config import KNOWLEDGE_BASE_DIR
from cred_support_agent.text_utils import count_sentences

_REQUIRED_FIELDS = ("doc_id", "topic", "title")


def parse_document(path: Path) -> Dict[str, Any]:
    """Parse one knowledge-base file: a ``---`` front-matter block, then the text."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path.name}: missing front-matter block")
    try:
        close = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path.name}: unterminated front-matter block") from exc

    meta: Dict[str, Any] = {}
    for line in lines[1:close]:
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    missing = [f for f in _REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise ValueError(f"{path.name}: front matter missing {missing}")

    # Wrapped lines are rejoined into one normalised string, which is exactly the
    # form the chunkers and the embedding model consume.
    text = " ".join(" ".join(lines[close + 1 :]).split())
    if not text:
        raise ValueError(f"{path.name}: empty document body")
    return {"doc_id": meta["doc_id"], "topic": meta["topic"], "title": meta["title"], "text": text}


def load_knowledge_base(directory: Path = KNOWLEDGE_BASE_DIR) -> List[Dict[str, Any]]:
    """Load every ``*.md`` document in the knowledge-base folder, ordered by doc id."""
    documents = [parse_document(path) for path in sorted(directory.glob("*.md"))]
    if not documents:
        raise FileNotFoundError(f"no knowledge-base documents found in {directory}")
    return sorted(documents, key=lambda d: d["doc_id"])


KNOWLEDGE_BASE: List[Dict[str, Any]] = load_knowledge_base()

REQUIRED_TOPICS = [
    "loan_eligibility_by_type",
    "emi_calculation_rules",
    "credit_card_fee_structure",
    "kyc_document_requirements",
    "fraud_dispute_resolution",
    "account_closure_process",
    "interest_rate_slabs",
    "prepayment_penalty_rules",
    "minimum_balance_requirements",
    "credit_score_impact_factors",
    "joint_account_rules",
    "nri_account_eligibility",
]

DOCS_BY_ID: Dict[str, Dict[str, Any]] = {d["doc_id"]: d for d in KNOWLEDGE_BASE}




def validate_knowledge_base(docs: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    rows = docs if docs is not None else KNOWLEDGE_BASE
    topics = {d["topic"] for d in rows}
    sentence_counts = {d["doc_id"]: count_sentences(d["text"]) for d in rows}
    checks = {
        "at_least_12_documents": len(rows) >= 12,
        "all_required_topics_covered": all(t in topics for t in REQUIRED_TOPICS),
        "every_doc_2_to_5_sentences": all(2 <= n <= 5 for n in sentence_counts.values()),
        "doc_ids_unique": len({d["doc_id"] for d in rows}) == len(rows),
    }
    return {
        "document_count": len(rows),
        "topics": sorted(topics),
        "missing_topics": [t for t in REQUIRED_TOPICS if t not in topics],
        "sentence_counts": sentence_counts,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def report() -> str:
    result = validate_knowledge_base()
    lines = [
        "=" * 72,
        "CRED KNOWLEDGE BASE",
        "=" * 72,
        f"documents: {result['document_count']} (minimum 12)",
        f"topics covered: {len(result['topics'])}",
        f"missing required topics: {result['missing_topics'] or 'none'}",
        "",
        "sentence counts per document (must be 2-5):",
    ]
    for doc_id, count in result["sentence_counts"].items():
        lines.append(f"  {doc_id} {DOCS_BY_ID[doc_id]['topic']:<32} {count}")
    lines.append("")
    for key, value in result["checks"].items():
        lines.append(f"  [{'PASS' if value else 'FAIL'}] {key}")
    lines.append(f"\nALL CHECKS PASSED: {result['all_passed']}")
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
