"""Shared text handling: sentence segmentation, chunking and normalisation.

The sentence splitter is deliberately decimal-aware. Cred policy text is full of
figures like "11.5 percent" and "INR 2,00,000", and a naive ``text.split('.')``
both miscounts sentences and produces garbage sentence chunks.
"""

from __future__ import annotations

import re
from typing import List

# A sentence boundary is terminal punctuation followed by whitespace and then
# the start of a new sentence. Requiring the whitespace is what keeps "11.5"
# and "INR 1,00,000." from being split mid-number.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z])")

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """a an and are as at be been by for from has have how in is it its of on or
    that the this to was were what when where which who will with your you my me
    i do does can could should would there their them they he she his her""".split()
)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences without breaking on decimals or amounts."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(cleaned) if s.strip()]


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fixed-size character chunks with a sliding overlap window.

    Chunk boundaries are nudged to the nearest preceding whitespace so a chunk
    never ends mid-word, which would otherwise poison the embedding.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            space = cleaned.rfind(" ", start + step, end)
            if space > start:
                end = space
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        next_start = max(end - overlap, start + 1)
        # Snap the start of the next window forward to a word boundary. Without
        # this the overlap slices mid-word ("...ive account") and the fragment
        # embeds badly, which measurably hurts retrieval ranking.
        space = cleaned.find(" ", next_start)
        if 0 <= space < end:
            next_start = space + 1
        start = next_start
    return chunks


def sentence_chunks(text: str, max_sentences: int = 2) -> List[str]:
    """Sentence-based chunks: consecutive sentences grouped into small windows.

    Grouping two sentences at a time keeps each chunk large enough to carry a
    complete policy statement (rule + qualifier) while staying far smaller than
    the fixed-size window.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    return [
        " ".join(sentences[i : i + max_sentences])
        for i in range(0, len(sentences), max_sentences)
    ]


def normalize_query(text: str) -> str:
    """Cache key normalisation: casefold, collapse whitespace, drop end punctuation."""
    collapsed = " ".join(text.split()).strip().casefold()
    return collapsed.rstrip(" ?.!,;:")


#: Conservative suffix stripping, longest suffix first. This is not a full
#: stemmer - it exists so that "close" and "closure", or "requirement" and
#: "required", count as the same concept when scoring overlap. A suffix is only
#: removed if at least four characters survive, which keeps short banking terms
#: like "cred" and "loan" intact.
_SUFFIXES = (
    "ization", "ational", "iveness", "fulness", "ements", "ations", "ingly",
    "ement", "ation", "ition", "ities", "ently", "ance", "ence", "ical", "ing",
    "ies", "ied", "ure", "ers", "est", "ity", "ive", "ous", "ed", "es", "ly", "s", "e",
)


def stem(word: str) -> str:
    """Reduce a token to a crude stem so morphological variants align."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> List[str]:
    """Lowercase, stemmed content tokens with stopwords removed.

    Used by the sentence scorer, the groundedness guardrail and the judge's
    completeness metric, so all three agree on what "the same word" means.
    """
    return [
        stem(w)
        for w in _WORD.findall(text.lower())
        if w not in _STOPWORDS and len(w) > 2
    ]
