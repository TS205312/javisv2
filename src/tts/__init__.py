"""Text-to-Speech module for Jarvis.

Provides a hybrid TTS engine that prefers online (edge-tts) synthesis
with offline (pyttsx3) fallback, plus a factory for creating engine
instances from configuration.
"""

from __future__ import annotations

from .base import TTSEngine
from .pyttsx3_engine import Pyttsx3Engine
from .edge_tts_engine import EdgeTTSEngine
from .hybrid_engine import HybridTTSEngine
from .factory import create_tts_engine, list_available_engines

__all__ = [
    "TTSEngine",
    "Pyttsx3Engine",
    "EdgeTTSEngine",
    "HybridTTSEngine",
    "create_tts_engine",
    "list_available_engines",
]
