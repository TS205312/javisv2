"""Vosk-based speech-to-text engine for Jarvis.

Uses Vosk for offline speech recognition with sounddevice.RawInputStream
for microphone capture.  Heavy dependencies (vosk, sounddevice, numpy)
are imported lazily so this module remains importable without them.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Union

from .base import STTEngine

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCKSIZE = 1024
DEFAULT_DURATION = 5.0
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "int16"


class VoskEngine(STTEngine):
    """Offline speech-to-text engine using Vosk.

    Records audio via ``sounddevice.RawInputStream`` (callback-based) and
    transcribes with Vosk ``KaldiRecognizer``.

    Args:
        model_path: Path to the Vosk model directory.  If ``None`` the
            factory or PathManager should supply a valid path.
        sample_rate: Audio sample rate in Hz (must match model).
        blocksize: Frames per callback buffer.
        channels: Number of audio channels.
    """

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        channels: int = DEFAULT_CHANNELS,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._channels = channels
        self._dtype = DEFAULT_DTYPE
        self._model: Any = None
        self._health = False
        self._lock = Lock()

    def _ensure_model(self) -> None:
        """Load the Vosk model if not already loaded."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import vosk

                if self._model_path is None:
                    raise RuntimeError(
                        "Vosk model path not configured. "
                        "Set model_path or JARVIS_STT_MODEL_PATH."
                    )
                self._model = vosk.Model(model_path=str(self._model_path))
                logger.info("Vosk model loaded from %s", self._model_path)
            except Exception as exc:
                logger.error("Failed to load Vosk model: %s", exc)
                raise

    @property
    def health(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: Union[bytes, Path]) -> str:
        """Transcribe raw PCM audio bytes or a WAV file path into text."""
        self._ensure_model()
        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "sounddevice is required for transcribe; "
                "install sounddevice numpy soundfile"
            )

        if isinstance(audio, Path):
            try:
                import soundfile as sf

                data, sr = sf.read(str(audio), dtype="float32")
                pcm = (data * 32767).astype("int16").tobytes()
            except ImportError:
                raise RuntimeError(
                    "soundfile is required to read WAV files; "
                    "install soundfile"
                )
        else:
            pcm = audio

        recognizer = self._create_recognizer()
        pcm_data = pcm if isinstance(pcm, bytes) else bytes(pcm)
        recognizer.AcceptWaveform(pcm_data)
        result_json = recognizer.Result()
        result = json.loads(result_json)
        return result.get("text", "")

    def listen(self, duration: float, prompt: str = "") -> str:
        """Record from the microphone for *duration* seconds and transcribe.

        Uses ``sounddevice.RawInputStream`` with a callback that accumulates
        raw PCM frames into a thread-safe buffer.  After recording, the
        accumulated audio is fed to Vosk for transcription.

        Args:
            duration: Recording duration in seconds.
            prompt: Optional guiding prompt (stored, engine-specific).

        Returns:
            Transcribed text string (may be empty if no speech detected).
        """
        if prompt:
            logger.debug("Listen prompt: %s", prompt)

        try:
            import sounddevice as sd
        except ImportError:
            raise RuntimeError(
                "sounddevice is required for microphone listening; "
                "install sounddevice"
            )

        self._ensure_model()

        buffers: List[bytes] = []

        def _callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            if status.input_overflow:
                logger.warning("Audio input overflow detected")
            if status.input_underflow:
                logger.warning("Audio input underflow detected")
            buffers.append(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self._blocksize,
                channels=self._channels,
                dtype=self._dtype,
                callback=_callback,
            ) as stream:
                elapsed = 0.0
                while elapsed < duration:
                    chunk = min(self._blocksize / self._sample_rate, duration - elapsed)
                    time.sleep(chunk)
                    elapsed += chunk
        except Exception as exc:
            logger.error("Microphone recording failed: %s", exc)
            raise RuntimeError(
                "Microphone recording failed: {0}".format(exc)
            ) from exc

        pcm_data = b"".join(buffers)
        if not pcm_data:
            logger.warning("No audio captured during listen")
            return ""

        recognizer = self._create_recognizer()
        recognizer.AcceptWaveform(pcm_data)
        result_json = recognizer.Result()
        result = json.loads(result_json)
        text = result.get("text", "")
        logger.info("Listen result: %r", text)
        return text

    def _create_recognizer(self) -> Any:
        """Create a Vosk KaldiRecognizer for the current model."""
        import vosk

        return vosk.KaldiRecognizer(self._model, self._sample_rate)

    def close(self) -> None:
        """Release the Vosk model and any resources."""
        with self._lock:
            self._model = None
        logger.info("VoskEngine closed")


__all__ = ["VoskEngine"]
