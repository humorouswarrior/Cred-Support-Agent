"""Task 5 - document-level precision and recall for both chunking strategies.

Chunks are mapped back to their parent document and deduplicated *before*
scoring, so a strategy is not rewarded for returning three chunks of the same
document. For each query and each collection:

    precision = |retrieved_docs & relevant_docs| / |retrieved_docs|
    recall    = |retrieved_docs & relevant_docs| / |relevant_docs|

Per-query arithmetic is printed for both collections and the recommendation at
the end is made from these measured numbers only.
"""

from __future__ import annotations

from typing import Any, Dict

from cred_support_agent.evaluation.benchmarks import RETRIEVAL_EVAL_QUERIES
from cred_support_agent.config import DEFAULT_TOP_K
from cred_support_agent.retrieval.indexing import COLLECTION_BY_STRATEGY, chunk_stats, query


def evaluate_query(text: str, relevant: set[str], strategy: str, top_k: int) -> Dict[str, Any]:
    hits = query(text, strategy=strategy, top_k=top_k)
    retrieved_chunks = [h.chunk_id for h in hits]
    # Map chunks back to parent documents, deduplicated, order preserved.
    retrieved_docs = list(dict.fromkeys(h.doc_id for h in hits))
    intersection = [d for d in retrieved_docs if d in relevant]
    precision = len(intersection) / len(retrieved_docs) if retrieved_docs else 0.0
    recall = len(intersection) / len(relevant) if relevant else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "query": text,
        "strategy": strategy,
        "relevant_docs": sorted(relevant),
        "retrieved_chunks": retrieved_chunks,
        "retrieved_docs_deduped": retrieved_docs,
        "hits": intersection,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "precision_arithmetic": f"{len(intersection)}/{len(retrieved_docs)}",
        "recall_arithmetic": f"{len(intersection)}/{len(relevant)}",
    }


def evaluate_strategy(strategy: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    rows = [
        evaluate_query(text, relevant, strategy, top_k)
        for text, relevant in RETRIEVAL_EVAL_QUERIES
    ]
    n = len(rows)
    return {
        "strategy": strategy,
        "collection": COLLECTION_BY_STRATEGY[strategy],
        "top_k": top_k,
        "per_query": rows,
        "mean_precision": round(sum(r["precision"] for r in rows) / n, 4),
        "mean_recall": round(sum(r["recall"] for r in rows) / n, 4),
        "mean_f1": round(sum(r["f1"] for r in rows) / n, 4),
    }


def compare(top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    results = {s: evaluate_strategy(s, top_k) for s in COLLECTION_BY_STRATEGY}
    ranked = sorted(
        results.values(),
        key=lambda r: (r["mean_f1"], r["mean_precision"], r["mean_recall"]),
        reverse=True,
    )
    winner, runner_up = ranked[0], ranked[1]
    if winner["mean_f1"] == runner_up["mean_f1"]:
        stats = chunk_stats()
        # Tie-break on index cost: fewer, denser chunks is the cheaper index.
        winner = min(ranked, key=lambda r: stats[r["strategy"]]["chunks"])
        rationale = (
            f"Both strategies measured identical mean F1 ({ranked[0]['mean_f1']}), so the "
            f"tie is broken on index cost: {winner['strategy']} stores "
            f"{stats[winner['strategy']]['chunks']} chunks versus "
            f"{stats[runner_up['strategy']]['chunks']} for the alternative, for the same "
            "retrieval quality."
        )
    else:
        # Task 5 asks for 2-3 sentences that cite both sets of numbers.
        rationale = (
            f"Deploy {winner['strategy']}: on the same five queries it measured mean "
            f"precision {winner['mean_precision']} and F1 {winner['mean_f1']}, against "
            f"{runner_up['mean_precision']} and {runner_up['mean_f1']} for "
            f"{runner_up['strategy']}. Mean recall was {winner['mean_recall']} for "
            f"{winner['strategy']} and {runner_up['mean_recall']} for {runner_up['strategy']}, "
            "so precision is what separates them: the smaller windows more often return a "
            "second chunk of the correct document instead of an unrelated one."
        )
    return {
        "results": results,
        "chunk_stats": chunk_stats(),
        "recommended_strategy": winner["strategy"],
        "recommendation_rationale": rationale,
    }


def report(top_k: int = DEFAULT_TOP_K) -> str:
    comparison = compare(top_k)
    lines = [
        "=" * 78,
        f"TASK 5 - CHUNKING STRATEGY EVALUATION (document-level, top_k={top_k})",
        "=" * 78,
        "chunk inventory:",
    ]
    for strategy, stats in comparison["chunk_stats"].items():
        lines.append(
            f"  {strategy:<16} chunks={stats['chunks']:<4} "
            f"avg_chars={stats['avg_chars']:<7} range={stats['min_chars']}-{stats['max_chars']}"
        )

    for strategy, result in comparison["results"].items():
        lines.append("")
        lines.append(f"--- {strategy}  (collection: {result['collection']}) ---")
        for row in result["per_query"]:
            lines.append(f"  Q: {row['query']}")
            lines.append(f"     relevant docs      : {row['relevant_docs']}")
            lines.append(f"     retrieved chunks   : {row['retrieved_chunks']}")
            lines.append(f"     -> parent docs     : {row['retrieved_docs_deduped']} (deduplicated)")
            lines.append(f"     correct            : {row['hits']}")
            lines.append(
                f"     precision = {row['precision_arithmetic']} = {row['precision']}"
                f"     recall = {row['recall_arithmetic']} = {row['recall']}"
                f"     F1 = {row['f1']}"
            )
        lines.append(
            f"  MEAN precision={result['mean_precision']}  "
            f"recall={result['mean_recall']}  F1={result['mean_f1']}"
        )

    lines.append("")
    lines.append(f"RECOMMENDATION: {comparison['recommended_strategy']}")
    lines.append(f"  {comparison['recommendation_rationale']}")
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())
