"""Task 3 - two chunking strategies, two separate ChromaDB collections.

Strategy A ``fixed_overlap``  : 240-character windows with 60-character overlap.
Strategy B ``sentence``       : consecutive-sentence windows, 2 sentences each.

Both strategies embed every chunk with the same local SentenceTransformers model
so the comparison in Task 5 isolates the chunking decision rather than the
embedding model. Each strategy lives in its own persistent Chroma collection and
is written with ``upsert()`` so re-running the indexer is idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import chromadb
from chromadb.config import Settings

from cred_support_agent.config import (
    CHROMA_DIR,
    DEFAULT_TOP_K,
    FIXED_CHUNK_COLLECTION,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    SENTENCE_CHUNK_COLLECTION,
)
from cred_support_agent.retrieval.embeddings import backend_name, embed
from cred_support_agent.data.knowledge_base import KNOWLEDGE_BASE
from cred_support_agent.text_utils import fixed_size_chunks, sentence_chunks

FIXED_STRATEGY = "fixed_overlap"
SENTENCE_STRATEGY = "sentence"

COLLECTION_BY_STRATEGY = {
    FIXED_STRATEGY: FIXED_CHUNK_COLLECTION,
    SENTENCE_STRATEGY: SENTENCE_CHUNK_COLLECTION,
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    topic: str
    title: str
    text: str
    strategy: str
    position: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    topic: str
    title: str
    text: str
    strategy: str
    similarity: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "topic": self.topic,
            "title": self.title,
            "text": self.text,
            "strategy": self.strategy,
            "similarity": round(self.similarity, 4),
        }


def build_chunks(strategy: str, docs: Sequence[Dict[str, Any]] | None = None) -> List[Chunk]:
    """Chunk every knowledge-base document under the named strategy."""
    rows = list(docs if docs is not None else KNOWLEDGE_BASE)
    chunks: List[Chunk] = []
    for doc in rows:
        if strategy == FIXED_STRATEGY:
            pieces = fixed_size_chunks(doc["text"], FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP)
        elif strategy == SENTENCE_STRATEGY:
            pieces = sentence_chunks(doc["text"], max_sentences=2)
        else:
            raise ValueError(f"unknown chunking strategy: {strategy!r}")
        for position, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}::{strategy}::{position:02d}",
                    doc_id=doc["doc_id"],
                    topic=doc["topic"],
                    title=doc["title"],
                    text=piece,
                    strategy=strategy,
                    position=position,
                )
            )
    return chunks


_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    """Persistent, telemetry-free local Chroma client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _client


#: HNSW settings for both collections. The corpus is small (under 100 chunks), so
#: the index is built single-threaded, which is deterministic, and searched with a
#: breadth far above the collection size, which is effectively exact. ChromaDB's
#: defaults (multi-threaded build, search breadth 10) could occasionally miss a
#: true nearest neighbour, changing an answer between otherwise identical runs.
HNSW_SETTINGS: Dict[str, Any] = {
    "hnsw:space": "cosine",
    "hnsw:num_threads": 1,
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 200,
    "hnsw:M": 16,
}


def get_collection(strategy: str):
    """Open (or create) one strategy's collection with the deterministic settings."""
    return get_client().get_or_create_collection(
        name=COLLECTION_BY_STRATEGY[strategy],
        metadata={**HNSW_SETTINGS, "strategy": strategy},
    )


def index_strategy(strategy: str) -> Dict[str, Any]:
    """Embed and ``upsert()`` every chunk of one strategy into its own collection."""
    chunks = build_chunks(strategy)
    collection = get_collection(strategy)
    vectors = embed([c.text for c in chunks])
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "topic": c.topic,
                "title": c.title,
                "strategy": c.strategy,
                "position": c.position,
            }
            for c in chunks
        ],
    )
    return {
        "strategy": strategy,
        "collection": COLLECTION_BY_STRATEGY[strategy],
        "chunk_count": len(chunks),
        "stored_count": collection.count(),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / max(len(chunks), 1), 1),
        "embedding_backend": backend_name(),
    }


def build_all_indexes() -> List[Dict[str, Any]]:
    return [index_strategy(FIXED_STRATEGY), index_strategy(SENTENCE_STRATEGY)]


def ensure_indexes() -> None:
    """Index on first use; cheap no-op when both collections are already populated."""
    client = get_client()
    for strategy, name in COLLECTION_BY_STRATEGY.items():
        try:
            collection = client.get_collection(name)
            if collection.count() > 0:
                continue
        except Exception:
            pass
        index_strategy(strategy)


def query(
    text: str, strategy: str = SENTENCE_STRATEGY, top_k: int = DEFAULT_TOP_K
) -> List[RetrievedChunk]:
    """Top-k nearest chunks from one collection, as cosine *similarity*."""
    ensure_indexes()
    collection = get_client().get_collection(COLLECTION_BY_STRATEGY[strategy])
    result = collection.query(
        query_embeddings=embed([text]),
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    retrieved: List[RetrievedChunk] = []
    ids = result["ids"][0]
    for i, chunk_id in enumerate(ids):
        meta = result["metadatas"][0][i]
        # Chroma reports cosine *distance*; similarity is 1 - distance.
        similarity = 1.0 - float(result["distances"][0][i])
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=str(meta["doc_id"]),
                topic=str(meta["topic"]),
                title=str(meta["title"]),
                text=result["documents"][0][i],
                strategy=strategy,
                similarity=similarity,
            )
        )
    return retrieved


def chunk_stats() -> Dict[str, Any]:
    stats = {}
    for strategy in COLLECTION_BY_STRATEGY:
        chunks = build_chunks(strategy)
        lengths = [len(c.text) for c in chunks]
        stats[strategy] = {
            "chunks": len(chunks),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "avg_chars": round(sum(lengths) / len(lengths), 1),
        }
    return stats


def report() -> str:
    infos = build_all_indexes()
    lines = ["=" * 72, "CHUNKING STRATEGIES + CHROMADB COLLECTIONS", "=" * 72]
    for info in infos:
        lines.append(
            f"{info['strategy']:<16} collection={info['collection']:<26} "
            f"chunks={info['chunk_count']:<4} stored={info['stored_count']:<4} "
            f"avg_chars={info['avg_chunk_chars']}"
        )
    lines.append(f"embedding backend: {infos[0]['embedding_backend']}")
    lines.append("")
    lines.append("sample queries against BOTH collections:")
    for sample in (
        "What documents do I need for KYC?",
        "Is there a penalty if I foreclose my personal loan early?",
    ):
        lines.append(f"\n  query: {sample}")
        for strategy in COLLECTION_BY_STRATEGY:
            hits = query(sample, strategy=strategy, top_k=2)
            lines.append(f"    [{strategy}]")
            for hit in hits:
                lines.append(
                    f"      {hit.doc_id} sim={hit.similarity:.4f} :: {hit.text[:88]}..."
                )
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(report())


def index_document(doc: Dict[str, Any]) -> Dict[str, int]:
    """Chunk and upsert a single new document into *both* collections.

    Used by ``POST /add-document``. Because ``upsert()`` is idempotent, posting
    the same doc_id twice replaces its chunks rather than duplicating them.
    """
    counts: Dict[str, int] = {}
    for strategy in COLLECTION_BY_STRATEGY:
        chunks = build_chunks(strategy, [doc])
        if not chunks:
            counts[strategy] = 0
            continue
        collection = get_collection(strategy)
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embed([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "topic": c.topic,
                    "title": c.title,
                    "strategy": c.strategy,
                    "position": c.position,
                }
                for c in chunks
            ],
        )
        counts[strategy] = len(chunks)
    return counts


def remove_document(doc_id: str) -> Dict[str, int]:
    """Delete every chunk of one document from both collections.

    Used by demonstrations that add a document so they leave the store exactly as
    they found it.
    """
    removed: Dict[str, int] = {}
    for strategy, name in COLLECTION_BY_STRATEGY.items():
        collection = get_collection(strategy)
        existing = collection.get(where={"doc_id": doc_id})
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)
        removed[strategy] = len(ids)
    return removed
