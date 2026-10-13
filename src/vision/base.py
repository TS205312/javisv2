"""Abstract interface for Jarvis vision engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VisionEngine(ABC):
    """Base interface implemented by capture and image-analysis engines."""

    @abstractmethod
    def analyze(self, image: Path | bytes | Any, prompt: str) -> str:
        """Analyze an image and return a text response."""

    @abstractmethod
    def capture(self, prompt: str, camera_index: int = 0) -> str:
        """Capture a frame and either preprocess it or analyze it."""

    @property
    @abstractmethod
    def health(self) -> bool:
        """Return whether the engine is currently operational."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the engine."""

    def __enter__(self) -> "VisionEngine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


__all__ = ["VisionEngine"]