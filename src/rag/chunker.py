"""Document chunking utilities for RAG ingestion."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Chunk:
    """A text chunk with metadata.

    Attributes:
        text: The chunk text content.
        source: Source file path as string.
        page: Page number for multi-page documents (PDF), else None.
        chunk_id: Sequential chunk index within the source file.
        metadata: Additional metadata (e.g., file extension, char range).
    """

    text: str
    source: str
    page: Optional[int]
    chunk_id: int
    metadata: dict


class Chunker:
    """Split documents into overlapping chunks for embedding.

    Supports plain text (.txt, .md) and PDF (if pypdf is available).
    Uses UTF-8 encoding throughout.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 80,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._pdf_available = self._check_pdf_support()

    @staticmethod
    def _check_pdf_support() -> bool:
        try:
            import pypdf  # noqa: F401
            return True
        except Exception:
            return False

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap

    @property
    def supports_pdf(self) -> bool:
        return self._pdf_available

    def chunk_file(self, path: Path) -> List[Chunk]:
        """Chunk a single file.

        Args:
            path: Path to the file.

        Returns:
            List of Chunk objects.

        Raises:
            ValueError: If file extension is not supported.
            OSError: If file cannot be read.
        """
        suffix = path.suffix.lower()
        if suffix in {'.txt', '.md'}:
            return self._chunk_text_file(path)
        if suffix == '.pdf':
            if not self._pdf_available:
                raise ValueError("PDF support not available: install pypdf")
            return self._chunk_pdf_file(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def chunk_files(self, paths: Sequence[Path]) -> List[Chunk]:
        """Chunk multiple files, skipping unsupported ones.

        Args:
            paths: Sequence of file paths.

        Returns:
            List of all chunks from supported files.
        """
        all_chunks: List[Chunk] = []
        for path in paths:
            try:
                chunks = self.chunk_file(path)
                all_chunks.extend(chunks)
            except (ValueError, OSError):
                # Skip unsupported or unreadable files silently
                continue
        return all_chunks

    def chunk_directory(self, directory: Path, recursive: bool = True) -> List[Chunk]:
        """Chunk all supported files in a directory.

        Args:
            directory: Directory to scan.
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of all chunks from supported files.
        """
        if not directory.is_dir():
            return []

        patterns = ('*.txt', '*.md', '*.pdf')
        files: List[Path] = []
        for pattern in patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))

        return self.chunk_files(files)

    def _chunk_text_file(self, path: Path) -> List[Chunk]:
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback: replace invalid chars
            text = path.read_text(encoding='utf-8', errors='replace')

        return self._create_chunks(text, str(path), page=None)

    def _chunk_pdf_file(self, path: Path) -> List[Chunk]:
        import pypdf

        chunks: List[Chunk] = []
        try:
            with open(path, 'rb') as fh:
                reader = pypdf.PdfReader(fh)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ''
                    if text.strip():
                        page_chunks = self._create_chunks(text, str(path), page=page_num)
                        chunks.extend(page_chunks)
        except Exception as exc:
            raise OSError(f"Failed to read PDF {path}: {exc}") from exc
        return chunks

    def _create_chunks(self, text: str, source: str, page: Optional[int]) -> List[Chunk]:
        """Create overlapping chunks from text with deduplication.

        Args:
            text: Full text to chunk.
            source: Source identifier (file path).
            page: Page number or None.

        Returns:
            List of unique chunks.
        """
        if not text.strip():
            return []

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        chunks: List[Chunk] = []
        seen_hashes: set[str] = set()
        chunk_id = 0
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            chunk_text = text[start:end]

            # Deduplicate by content hash
            content_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                metadata = {
                    'char_start': start,
                    'char_end': end,
                    'file_extension': Path(source).suffix.lower(),
                }
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=source,
                        page=page,
                        chunk_id=chunk_id,
                        metadata=metadata,
                    )
                )
                chunk_id += 1

            # Advance by chunk_size - overlap
            if end == text_len:
                break
            start = end - self._overlap
            if start < 0:
                start = 0

        return chunks
