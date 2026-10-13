"""Speech-to-Text module for Jarvis.

Provides Vosk-based offline speech recognition with microphone
capture via sounddevice. Engines are created via the factory
in ``stt.factory``.
"""

from __future__ import annotations

from .base import STTEngine
from .vosk_engine import VoskEngine
from .factory import STTEngineStub, create_stt_engine

__all__ = [
    "STTEngine",
    "VoskEngine",
    "STTEngineStub",
    "create_stt_engine",
]
