"""Local, keyless embedding layer.

Primary backend is a free local SentenceTransformers model (``all-MiniLM-L6-v2``)
which is downloaded once during setup and then served from the on-disk HF cache
with no network traffic. If the model weights are genuinely unavailable, the
module falls back to a deterministic hashed character n-gram embedder so that
every acceptance check still runs offline and without keys. Which backend is
active is reported explicitly rather than silently swapped.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from typing import List, Sequence

from cred_support_agent.config import EMBEDDING_MODEL_NAME, MODEL_DIR

_HASHED_DIM = 512
_NGRAM = re.compile(r"[a-z0-9]+")


class HashedNgramEmbedder:
    """Deterministic offline fallback: L2-normalised hashed word + 4-gram bag.

    This is not a neural embedder, but it is stable, dependency-free and keeps
    lexical similarity ordering intact, which is all the retrieval threshold
    calibration needs in order to stay meaningful when the model cache is cold.
    """

    name = "hashed-ngram-fallback"
    dimension = _HASHED_DIM

    @staticmethod
    def _features(text: str) -> List[str]:
        tokens = _NGRAM.findall(text.lower())
        features: List[str] = list(tokens)
        for token in tokens:
            padded = f"#{token}#"
            features.extend(padded[i : i + 4] for i in range(max(len(padded) - 3, 1)))
        return features

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vec = [0.0] * _HASHED_DIM
            for feature in self._features(text):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % _HASHED_DIM
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[bucket] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class SentenceTransformerEmbedder:
    """Free local SentenceTransformers backend (no API key, no paid service)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer  # local import: heavy

        # Load from the project's own models/ folder only (config points the
        # Hugging Face cache there) and never contact the Hugging Face Hub.
        self._model = SentenceTransformer(model_name, local_files_only=True)
        self.name = f"sentence-transformers/{model_name}"
        try:
            self.dimension = int(self._model.get_embedding_dimension())
        except AttributeError:  # older sentence-transformers releases
            self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]


_embedder = None
_lock = threading.Lock()


def get_embedder():
    """Process-wide singleton embedder; loads the local model at most once.

    If the model has not been downloaded, this stops with a clear error rather than
    quietly switching embedders: a silent fallback would produce a run that looks
    healthy but gives different retrieval results and different numbers from the
    README. The deterministic fallback is available on request with
    CRED_ALLOW_FALLBACK_EMBEDDER=1.
    """
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                try:
                    _embedder = SentenceTransformerEmbedder()
                except Exception as exc:
                    if os.environ.get("CRED_ALLOW_FALLBACK_EMBEDDER") != "1":
                        raise RuntimeError(
                            f"The embedding model '{EMBEDDING_MODEL_NAME}' is not in {MODEL_DIR}. "
                            "Run `python bootstrap.py` once (it downloads the model into the "
                            "project), then try again. To run anyway with the deterministic "
                            "offline fallback embedder, whose results differ from the README, "
                            "set CRED_ALLOW_FALLBACK_EMBEDDER=1."
                        ) from exc
                    print(
                        f"[embeddings] Model not found in {MODEL_DIR} ({type(exc).__name__}); "
                        "CRED_ALLOW_FALLBACK_EMBEDDER=1 is set, so using the deterministic "
                        "fallback embedder. Similarity figures will differ from the README."
                    )
                    _embedder = HashedNgramEmbedder()
    return _embedder


def embed(texts: Sequence[str]) -> List[List[float]]:
    return get_embedder().encode(texts)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return float(dot / (na * nb))


def backend_name() -> str:
    return get_embedder().name
