"""Shared helpers for the demonstration scripts."""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Never crash on output encoding. Windows writes redirected output in the console
# code page (often cp1252), where some characters a library may print can't be
# encoded; replace them instead of raising UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cred_support_agent.config import TRANSCRIPT_DIR  # noqa: E402


class _Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def transcript(name: str) -> Iterator[Path]:
    """Mirror everything printed in this block into transcripts/<name>."""
    path = TRANSCRIPT_DIR / name
    with path.open("w", encoding="utf-8") as handle:
        original = sys.stdout
        sys.stdout = _Tee(original, handle)
        try:
            yield path
        finally:
            sys.stdout = original
    print(f"\n[transcript written] {path.relative_to(PROJECT_ROOT)}")


def banner(title: str) -> None:
    print()
    print("#" * 78)
    print(f"# {title}")
    print("#" * 78)
