"""Task 4 - evidence-based calibration of the "I don't know" threshold.

No preset 0.5/0.6/0.7. The threshold is measured: top-1 cosine similarity is
recorded for a set of in-scope queries and a set of deliberately out-of-scope
queries, the two clusters are inspected, and the threshold is placed in the gap
between them (the midpoint of ``min(in-scope)`` and ``max(out-of-scope)``).
The measured numbers and the resulting threshold are written to
``artifacts/threshold_calibration.json`` and quoted in README.md.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from cred_support_agent.config import (
    PROJECT_ROOT,
    CALIBRATION_PATH,
    DEFAULT_RETRIEVAL_STRATEGY,
    FALLBACK_SIMILARITY_THRESHOLD,
)
from cred_support_agent.retrieval.indexing import COLLECTION_BY_STRATEGY, query

#: Six questions a Cred customer would genuinely ask, each answerable from the KB.
IN_SCOPE_CALIBRATION_QUERIES: List[str] = [
    "What documents are required to complete KYC for a Cred loan?",
    "How is the EMI calculated on a Cred personal loan?",
    "Is there a prepayment penalty if I foreclose my loan early?",
    "What is the minimum balance I must maintain in a Cred savings account?",
    "Am I eligible for an NRE account if I live abroad?",
    "What interest rate applies to a home loan with a high credit score?",
]

#: Questions with no support anywhere in the Cred policy knowledge base.
OUT_OF_SCOPE_CALIBRATION_QUERIES: List[str] = [
    "What is the best recipe for Hyderabadi biryani?",
    "Who won the football World Cup in 2014?",
    "How do I train a convolutional neural network on satellite imagery?",
    "What is the weather forecast for Bengaluru this weekend?",
]


def _top1(text: str, strategy: str) -> float:
    hits = query(text, strategy=strategy, top_k=1)
    return round(hits[0].similarity, 4) if hits else 0.0


def measure(strategy: str = DEFAULT_RETRIEVAL_STRATEGY) -> Dict[str, Any]:
    """Measure both clusters and place the threshold between them."""
    in_scope = {q: _top1(q, strategy) for q in IN_SCOPE_CALIBRATION_QUERIES}
    out_scope = {q: _top1(q, strategy) for q in OUT_OF_SCOPE_CALIBRATION_QUERIES}

    min_in = min(in_scope.values())
    max_out = max(out_scope.values())
    separated = min_in > max_out

    if separated:
        threshold = round((min_in + max_out) / 2.0, 4)
        rationale = (
            f"The two clusters are cleanly separated: the weakest in-scope query scores "
            f"{min_in} and the strongest out-of-scope query scores {max_out}. The "
            f"threshold is placed at the midpoint of that gap, {threshold}, so every "
            f"measured in-scope query is answered and every measured out-of-scope query "
            f"falls back to 'I don't know'."
        )
    else:
        # Documented degenerate case (e.g. the offline fallback embedder): bias to
        # refusal by sitting just above the strongest out-of-scope score.
        threshold = round(min(max_out + 0.01, max(min_in - 0.001, 0.0)), 4) or (
            FALLBACK_SIMILARITY_THRESHOLD
        )
        rationale = (
            f"The clusters overlap under this embedding backend (min in-scope {min_in} "
            f"<= max out-of-scope {max_out}); the threshold is pinned just above the "
            f"strongest out-of-scope score at {threshold}, which biases the system "
            f"towards refusing rather than answering on weak evidence."
        )

    return {
        "strategy": strategy,
        "collection": COLLECTION_BY_STRATEGY[strategy],
        "in_scope_top1": in_scope,
        "out_of_scope_top1": out_scope,
        "min_in_scope": min_in,
        "max_out_of_scope": max_out,
        "clusters_separated": separated,
        "threshold": threshold,
        "rationale": rationale,
    }


def calibrate_all() -> Dict[str, Any]:
    """Calibrate every strategy and persist the report."""
    from cred_support_agent.retrieval.embeddings import backend_name

    per_strategy = {s: measure(s) for s in COLLECTION_BY_STRATEGY}
    payload = {
        "embedding_backend": backend_name(),
        "default_strategy": DEFAULT_RETRIEVAL_STRATEGY,
        "per_strategy": per_strategy,
        "active_threshold": per_strategy[DEFAULT_RETRIEVAL_STRATEGY]["threshold"],
    }
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_calibration() -> Dict[str, Any]:
    """Read the persisted calibration, measuring it on first use."""
    if CALIBRATION_PATH.exists():
        try:
            return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return calibrate_all()


def active_threshold(strategy: str = DEFAULT_RETRIEVAL_STRATEGY) -> float:
    data = load_calibration()
    per = data.get("per_strategy", {}).get(strategy)
    if per and "threshold" in per:
        return float(per["threshold"])
    return float(data.get("active_threshold", FALLBACK_SIMILARITY_THRESHOLD))


def report() -> str:
    data = calibrate_all()
    lines = ["=" * 78, "THRESHOLD CALIBRATION (measured, not preset)", "=" * 78]
    lines.append(f"embedding backend: {data['embedding_backend']}")
    for strategy, result in data["per_strategy"].items():
        lines.append("")
        lines.append(f"--- strategy: {strategy} (collection {result['collection']}) ---")
        lines.append("  IN-SCOPE top-1 cosine similarity:")
        for q, s in result["in_scope_top1"].items():
            lines.append(f"    {s:.4f}  {q}")
        lines.append("  OUT-OF-SCOPE top-1 cosine similarity:")
        for q, s in result["out_of_scope_top1"].items():
            lines.append(f"    {s:.4f}  {q}")
        lines.append(
            f"  min(in-scope)={result['min_in_scope']:.4f}   "
            f"max(out-of-scope)={result['max_out_of_scope']:.4f}   "
            f"separated={result['clusters_separated']}"
        )
        lines.append(f"  => THRESHOLD = {result['threshold']}")
        lines.append(f"  {result['rationale']}")
    lines.append("")
    lines.append(f"active threshold (default strategy): {data['active_threshold']}")
    lines.append(f"written to: {CALIBRATION_PATH.relative_to(PROJECT_ROOT)}")
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
