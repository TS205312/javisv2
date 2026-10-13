"""Base abstract class for all STT engines in Jarvis.

Provides the abstract interface that every STT engine must implement:
transcribe, listen, health, and close.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


class STTEngine(ABC):
    """Abstract base class for speech-to-text engines."""

    @abstractmethod
    def transcribe(self, audio: Union[bytes, Path]) -> str:
        """Transcribe audio data (raw PCM bytes or a WAV file path) into text."""
        ...

    @abstractmethod
    def listen(self, duration: float, prompt: str = "") -> str:
        """Record from the microphone for *duration* seconds and transcribe.

        Returns the transcribed text.  *prompt* is an optional guiding
        instruction for the recognition (engine-specific).
        """
        ...

    @property
    @abstractmethod
    def health(self) -> bool:
        """Return True if the engine is operational."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the engine."""
        ...


__all__ = ["STTEngine"]
