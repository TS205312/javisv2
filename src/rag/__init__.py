"""Jarvis RAG (Retrieval-Augmented Generation) package."""

from .base import RAGEngine, RetrievalResult, UnavailableRAGEngine
from .chunker import Chunker, Chunk
from .faiss_engine import FAISSRAGEngine
from .factory import create_rag_engine

__all__ = [
    "RAGEngine",
    "RetrievalResult",
    "UnavailableRAGEngine",
    "Chunker",
    "Chunk",
    "FAISSRAGEngine",
    "create_rag_engine",
]
