"""FAISS-based RAG engine implementation."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List, Optional

from .base import RAGEngine, RetrievalResult, UnavailableRAGEngine
from .chunker import Chunker, Chunk


class FAISSRAGEngine(RAGEngine):
    """FAISS-backed RAG engine with sentence-transformers embeddings.

    Uses IndexFlatIP with cosine normalization for similarity search.
    Persists index and metadata to D:\\AI_Brain\\faiss.
    Lazy-loads model and dependencies on first use.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"
    _EXPECTED_DIM = 384
    _MAX_BATCH = 16
    _INDEX_FILENAME = "index.faiss"
    _META_FILENAME = "metadata.jsonl"

    def __init__(
        self,
        faiss_dir: Optional[Path] = None,
        chunk_size: int = 512,
        overlap: int = 80,
        model_name: Optional[str] = None,
    ) -> None:
        # Set HF cache env vars BEFORE any imports
        self._setup_hf_cache()

        self._faiss_dir = faiss_dir or Path(r"D:\AI_Brain\faiss")
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._model_name = model_name or self._MODEL_NAME

        self._index = None
        self._metadata: List[dict] = []
        self._model = None
        self._faiss_module = None
        self._st_module = None
        self._lock = threading.RLock()
        self._initialized = False
        self._init_error: Optional[str] = None

    def _setup_hf_cache(self) -> None:
        """Set HuggingFace cache directories before importing transformers."""
        hf_home = Path(r"D:\AI_Brain\huggingface")
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = str(hf_home)
        if "SENTENCE_TRANSFORMERS_HOME" not in os.environ:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(hf_home)
        # Ensure directory exists
        try:
            hf_home.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def _lazy_import(self) -> None:
        """Lazy import heavy dependencies."""
        if self._faiss_module is not None:
            return

        try:
            import faiss
            self._faiss_module = faiss
        except Exception as exc:
            raise RuntimeError(
                "FAISS not available. Install with: pip install faiss-cpu"
            ) from exc

        try:
            from sentence_transformers import SentenceTransformer
            self._st_module = SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            ) from exc

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self._model is not None:
            return

        self._lazy_import()
        assert self._st_module is not None

        try:
            self._model = self._st_module(self._model_name, device="cpu")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load model {0!r}. "
                "Ensure model is cached in D:\\AI_Brain\\huggingface or run:\n"
                "  pip install sentence-transformers\n"
                "  python -c \"from sentence_transformers import SentenceTransformer; "
                "SentenceTransformer(\"{1}\")\"".format(self._model_name, self._model_name)
            ) from exc

    def _ensure_index_dir(self) -> None:
        try:
            self._faiss_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError("Cannot create FAISS directory {0}: {1}".format(self._faiss_dir, exc))

    def _index_path(self) -> Path:
        return self._faiss_dir / self._INDEX_FILENAME

    def _meta_path(self) -> Path:
        return self._faiss_dir / self._META_FILENAME

    def _initialize(self) -> None:
        """Load existing index and metadata if present."""
        with self._lock:
            if self._initialized:
                return
            self._initialized = True

            self._lazy_import()
            self._ensure_index_dir()
            index_path = self._index_path()
            meta_path = self._meta_path()

            if not index_path.exists() or not meta_path.exists():
                # Fresh start
                self._index = self._faiss_module.IndexFlatIP(self._EXPECTED_DIM)
                self._metadata = []
                return

            # Load existing index
            try:
                self._index = self._faiss_module.read_index(str(index_path))
            except Exception as exc:
                raise RuntimeError("Failed to load FAISS index: {0}".format(exc)) from exc

            # Verify dimension
            if self._index.d != self._EXPECTED_DIM:
                raise RuntimeError(
                    "Index dimension mismatch: expected {0}, "
                    "got {1}. Delete {2} to rebuild.".format(
                        self._EXPECTED_DIM, self._index.d, index_path
                    )
                )

            # Load metadata
            self._metadata = []
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            self._metadata.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError("Failed to load metadata: {0}".format(exc)) from exc

            # Verify metadata count matches index
            if len(self._metadata) != self._index.ntotal:
                raise RuntimeError(
                    "Metadata count ({0}) != index count ({1}). "
                    "Delete {2} and {3} to rebuild.".format(
                        len(self._metadata), self._index.ntotal, index_path, meta_path
                    )
                )

    def _persist(self) -> None:
        """Persist index and metadata to disk."""
        self._ensure_index_dir()
        index_path = self._index_path()
        meta_path = self._meta_path()

        try:
            self._faiss_module.write_index(self._index, str(index_path))
        except Exception as exc:
            raise RuntimeError("Failed to write FAISS index: {0}".format(exc)) from exc

        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                for meta in self._metadata:
                    fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        except Exception as exc:
            raise RuntimeError("Failed to write metadata: {0}".format(exc)) from exc

    def _encode_texts(self, texts: List[str]):
        """Encode texts to embeddings with batching."""
        import numpy as np

        self._load_model()
        assert self._model is not None

        all_embeddings = []
        for i in range(0, len(texts), self._MAX_BATCH):
            batch = texts[i : i + self._MAX_BATCH]
            emb = self._model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            all_embeddings.append(emb)

        if not all_embeddings:
            return np.empty((0, self._EXPECTED_DIM), dtype=np.float32)
        return np.vstack(all_embeddings).astype(np.float32)

    def health(self) -> bool:
        """Return True if the engine is ready to serve queries."""
        try:
            self._initialize()
            return True
        except Exception:
            return False

    def ingest(self, paths: List[Path]) -> int:
        """Ingest documents from paths.

        Args:
            paths: List of file or directory paths.

        Returns:
            Number of new chunks ingested.

        Raises:
            RuntimeError: If dependencies unavailable or initialization fails.
        """
        with self._lock:
            self._initialize()
            self._load_model()

            chunker = Chunker(chunk_size=self._chunk_size, overlap=self._overlap)
            all_chunks: List[Chunk] = []

            for path in paths:
                path = Path(path)
                if path.is_file():
                    try:
                        all_chunks.extend(chunker.chunk_file(path))
                    except (ValueError, OSError):
                        continue
                elif path.is_dir():
                    all_chunks.extend(chunker.chunk_directory(path))
                # Skip non-existent silently

            if not all_chunks:
                return 0

            # Encode new chunks
            texts = [c.text for c in all_chunks]
            embeddings = self._encode_texts(texts)

            if embeddings.shape[0] == 0:
                return 0

            # Add to index
            self._index.add(embeddings)

            # Extend metadata
            for chunk in all_chunks:
                self._metadata.append({
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "metadata": chunk.metadata,
                })

            self._persist()
            return len(all_chunks)

    def query(self, text: str, k: int = 4) -> List[RetrievalResult]:
        """Query for similar chunks.

        Args:
            text: Query text.
            k: Number of results (clamped to [1, index_size]).

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        with self._lock:
            if not text or not text.strip():
                return []

            try:
                self._initialize()
            except Exception:
                return []

            if self._index is None or self._index.ntotal == 0:
                return []

            self._load_model()

            # Clamp k
            k = max(1, min(k, self._index.ntotal))

            # Encode query
            query_emb = self._encode_texts([text.strip()])
            if query_emb.shape[0] == 0:
                return []

            # Search
            scores, indices = self._index.search(query_emb, k)

            results: List[RetrievalResult] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                meta = self._metadata[idx]
                results.append(
                    RetrievalResult(
                        text=meta["text"],
                        score=float(score),
                        source=meta["source"],
                        page=meta.get("page"),
                        chunk_id=meta.get("chunk_id", 0),
                        metadata=meta.get("metadata"),
                    )
                )
            return results

    def close(self) -> None:
        """Release model and index resources."""
        with self._lock:
            self._model = None
            self._index = None
            self._metadata = []
            self._initialized = False

    def __repr__(self) -> str:
        return "FAISSRAGEngine(faiss_dir={0!r}, model={1!r})".format(self._faiss_dir, self._model_name)


def create_faiss_engine(
    faiss_dir: Optional[Path] = None,
    chunk_size: int = 512,
    overlap: int = 80,
    model_name: Optional[str] = None,
) -> FAISSRAGEngine:
    """Factory function to create a FAISS RAG engine.

    Construction never raises; use ``health()`` to check availability.
    """
    return FAISSRAGEngine(faiss_dir, chunk_size, overlap, model_name)


__all__ = ["FAISSRAGEngine", "create_faiss_engine"]