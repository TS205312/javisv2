"""Base abstract class for all TTS engines in Jarvis."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for text-to-speech engines."""

    @property
    @abstractmethod
    def health(self) -> bool:
        """Return True if the engine is ready to synthesize/speak."""

    @property
    @abstractmethod
    def available_voices(self) -> List[str]:
        """Return a list of available voice identifiers/names."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw audio bytes.

        Returns audio data as bytes (e.g. WAV PCM).  May return b""
        if the engine cannot produce raw audio directly.
        """

    @abstractmethod
    def play(self, text: str) -> Union[bool, str]:
        """Synthesize and play text.

        Returns True if playback succeeded, or a str path to the
        saved audio file when direct playback is unavailable.
        """

    @abstractmethod
    def speak(self, text: str) -> bool:
        """Speak text directly (real-time playback).

        Returns True on success, False on failure.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the engine."""


def _play_wav_file(path: Path) -> bool:
    """Play a WAV file using sounddevice + soundfile if available.

    Returns True if playback succeeded, False otherwise (including
    when the optional dependencies are not installed).  Never raises
    or crashes for missing dependencies.
    """
    try:
        import soundfile as sf
        import sounddevice as sd
    except ImportError:
        logger.debug("sounddevice/soundfile not installed; skipping playback")
        return False

    try:
        data, sample_rate = sf.read(str(path), dtype="float32")
        sd.play(data, sample_rate)
        sd.wait()
        return True
    except Exception as exc:
        logger.warning("Failed to play WAV file %s: %s", path, exc)
        return False
