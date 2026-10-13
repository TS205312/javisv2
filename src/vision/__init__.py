"""Vision module for Jarvis.

Provides vision engines for frame capture (OpenCV) and image analysis
(Ollama with moondream model). Engines are created via the factory
in ``vision.factory``.
"""

from __future__ import annotations

from .base import VisionEngine
from .opencv_engine import OpenCVFrameCapture
from .moondream_engine import OllamaMoondreamEngine
from .factory import (
    VisionEngineStub,
    create_vision_engines,
    create_opencv_engine,
    create_moondream_engine,
    get_default,
    list_available_engines,
    UnavailableVisionEngine,
)

__all__ = [
    "VisionEngine",
    "OpenCVFrameCapture",
    "OllamaMoondreamEngine",
    "VisionEngineStub",
    "create_vision_engines",
    "create_opencv_engine",
    "create_moondream_engine",
    "get_default",
    "list_available_engines",
    "UnavailableVisionEngine",
]
