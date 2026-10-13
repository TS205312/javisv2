"""Factory for creating RAG engines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import RAGEngine, UnavailableRAGEngine
from .faiss_engine import FAISSRAGEngine, create_faiss_engine


def create_rag_engine(
    config: Optional[object] = None,
    paths: Optional[object] = None,
    engine_type: str = "faiss",
) -> RAGEngine:
    """Create a RAG engine instance from config and paths.

    Args:
        config: Config-like object with rag_model, rag_chunk_size, rag_overlap attributes.
        paths: PathManager-like object for resolving faiss directory.
        engine_type: Engine type ("faiss" supported).

    Returns:
        RAGEngine instance or UnavailableRAGEngine if dependencies missing.
    """
    faiss_dir = None
    chunk_size = 512
    overlap = 80
    model_name = None

    if config is not None:
        model_name = getattr(config, "rag_model", None)
        chunk_size = getattr(config, "rag_chunk_size", chunk_size)
        overlap = getattr(config, "rag_overlap", overlap)

    if paths is not None:
        try:
            faiss_dir = paths.get("faiss")
        except Exception:
            pass

    if engine_type.lower() == "faiss":
        engine = create_faiss_engine(faiss_dir, chunk_size, overlap, model_name)
        # Do NOT call health() here - it triggers lazy imports which may fail
        # Caller should call health() when ready to use
        return engine

    return UnavailableRAGEngine("Unknown engine type: {0}".format(engine_type))


__all__ = ["create_rag_engine", "create_faiss_engine"]