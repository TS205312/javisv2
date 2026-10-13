"""Abstract RAG engine interface and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class RetrievalResult:
    """A single retrieval result from a RAG engine.

    Attributes:
        text: The retrieved chunk text.
        score: Similarity score (higher is better for cosine/IP).
        source: Source file path as string.
        page: Page number for multi-page documents, else None.
        chunk_id: Sequential chunk index within the source file.
        metadata: Additional metadata dict.
    """

    text: str
    score: float
    source: str
    page: Optional[int] = None
    chunk_id: int = 0
    metadata: Optional[dict] = field(default_factory=dict)


class RAGEngine:
    """Abstract base class for RAG engines.

    Subclasses must implement ``health``, ``ingest``, ``query`` and ``close``.
    """

    def health(self) -> bool:
        """Return True if the engine is ready to serve queries."""
        raise NotImplementedError

    def ingest(self, paths: List[Path]) -> int:
        """Ingest documents from paths. Returns number of new chunks."""
        raise NotImplementedError

    def query(self, text: str, k: int = 4) -> List[RetrievalResult]:
        """Query for similar chunks. Returns results sorted by score desc."""
        raise NotImplementedError

    def close(self) -> None:
        """Release engine resources."""
        raise NotImplementedError


class UnavailableRAGEngine(RAGEngine):
    """Stub engine returned when a real engine cannot be constructed.

    All operations are no-ops; ``health`` returns False.
    """

    def __init__(self, reason: str = "") -> None:
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason

    def health(self) -> bool:
        return False

    def ingest(self, paths: List[Path]) -> int:
        return 0

    def query(self, text: str, k: int = 4) -> List[RetrievalResult]:
        return []

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"UnavailableRAGEngine(reason={self._reason!r})"


__all__ = [
    "RAGEngine",
    "RetrievalResult",
    "UnavailableRAGEngine",
]
