r"""Edge TTS online engine for Jarvis.

Uses edge-tts (Microsoft Edge Online TTS) via WebSocket.  Audio chunks are
collected from the streaming generator and persisted under D:\AI_System\audio
through the configured PathManager (never on the system C: drive).  All async
operations are wrapped so they never block an existing event loop when called
from synchronous code, and a defensive timeout guards every network call.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .base import TTSEngine, _play_wav_file

logger = logging.getLogger(__name__)

_EDGE_TTS_HOST = "eastus2.tts.speech.microsoft.com"
_DEFAULT_TIMEOUT = 30.0
_HEALTH_TTL = 5.0
_VOICES_TTL = 120.0
_FALLBACK_DIR = Path(r"D:\AI_System\audio")


class EdgeTTSEngine(TTSEngine):
    r"""Online TTS engine using Microsoft Edge TTS (edge-tts).

    Features:
        - Lazy import of edge_tts (not required at import time).
        - Voice defaults to `vi-VN-NamMinhNeural`.
        - All audio output/temp files written under D:\AI_System\audio.
        - Network timeout and cancellation support.
        - Sync wrappers avoid blocking an existing event loop.
        - Short-lived caches for health and voice listing.
    """

    def __init__(
        self,
        config=None,
        paths=None,
        voice: str = "vi-VN-NamMinhNeural",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._config = config
        self._paths = paths
        # Use config voice if provided
        if config is not None:
            self._voice = getattr(config, "tts_voice", voice)
        else:
            self._voice = voice
        self._timeout = max(5.0, float(timeout))
        self._client = None
        self._last_file: Optional[Path] = None
        self._closed = False
        self._health_cached: Optional[bool] = None
        self._health_ts = 0.0
        self._voices_cache: Optional[List[str]] = None
        self._voices_ts = 0.0

    def _ensure_client(self):
        if self._client is None:
            try:
                import edge_tts
            except ImportError as exc:
                raise ImportError(
                    "edge-tts is not installed; install with: pip install edge-tts"
                ) from exc
            self._client = edge_tts
        return self._client

    def _fallback_voices(self) -> List[str]:
        return [
            "vi-VN-NamMinhNeural",
            "vi-VN-NamNeural",
            "vi-VN-HoangNeural",
            "vi-VN-TuanNeural",
        ]

    def _audio_dir(self) -> Path:
        if self._paths is not None:
            try:
                return self._paths.get("audio")
            except Exception:
                pass
        return _FALLBACK_DIR

    def _ensure_audio_dir(self) -> Path:
        audio_dir = self._audio_dir()
        try:
            audio_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Cannot create audio dir %s: %s", audio_dir, exc)
        return audio_dir

    def _cleanup_previous(self) -> None:
        if self._last_file is not None:
            try:
                self._last_file.unlink(missing_ok=True)
            except Exception as exc:
                logger.debug("Cannot remove previous audio file %s: %s", self._last_file, exc)
            self._last_file = None

    @property
    def health(self) -> bool:
        if self._closed:
            return False
        now = time.monotonic()
        if self._health_cached is not None and (now - self._health_ts) < _HEALTH_TTL:
            return self._health_cached
        self._health_cached = self._check_network()
        self._health_ts = now
        return self._health_cached

    @property
    def available_voices(self) -> List[str]:
        try:
            client = self._ensure_client()
        except ImportError:
            return self._fallback_voices()
        now = time.monotonic()
        if self._voices_cache is not None and (now - self._voices_ts) < _VOICES_TTL:
            return list(self._voices_cache)
        voices: List[str] = []
        try:
            async def _fetch() -> List[Dict]:
                return await asyncio.wait_for(
                    asyncio.shield(client.list_voices()),
                    timeout=self._timeout,
                )
            raw = self._run_async(_fetch())
            for v in raw or []:
                if not isinstance(v, dict):
                    continue
                try:
                    name = v.get("ShortName") or v.get("Name") or ""
                except Exception:
                    continue
                if name:
                    voices.append(name)
        except Exception as exc:
            logger.debug("Cannot list edge-tts voices: %s", exc)
        if voices:
            self._voices_cache = list(voices)
            self._voices_ts = now
            return voices
        return self._fallback_voices()

    def _check_network(self) -> bool:
        try:
            import socket
            sock = socket.create_connection((_EDGE_TTS_HOST, 443), timeout=5.0)
            sock.close()
            return True
        except Exception:
            return False

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=self._timeout * 3)

    def _create_communicate(self, text: str):
        client = self._ensure_client()
        # edge_tts >=7 removed the `format` kwarg and hard-codes mp3 output;
        # passing `voice` only keeps this compatible across versions.
        return client.Communicate(text, self._voice)

    @staticmethod
    def _parse_stream_message(message) -> Tuple[Optional[str], object]:
        if isinstance(message, dict):
            return message.get("type"), message.get("data", b"")
        try:
            msg_type, data = message  # type: ignore[misc]
            return msg_type, data
        except Exception:
            return None, None

    async def _async_synthesize_bytes(self, text: str) -> bytes:
        comm = self._create_communicate(text)
        chunks: List[bytes] = []
        try:
            async for message in comm.stream():
                msg_type, data = self._parse_stream_message(message)
                if msg_type == "audio":
                    if isinstance(data, (bytes, bytearray)):
                        chunks.append(bytes(data))
                elif msg_type == "error":
                    raise RuntimeError("edge-tts error: {0}".format(data))
                elif msg_type == "End":
                    break
        except asyncio.CancelledError:
            logger.warning("edge-tts synthesis cancelled")
            raise
        except Exception as exc:
            logger.error("edge-tts stream failed: %s", exc)
            raise
        return b"".join(chunks)

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            return b""
        if self._closed:
            logger.warning("EdgeTTSEngine is closed")
            return b""
        try:
            return self._run_async(self._async_synthesize_bytes(text))
        except Exception as exc:
            logger.error("edge-tts synthesize failed: %s", exc)
            return b""

    async def _async_save_to_file(self, text: str, output_path: Path) -> Path:
        comm = self._create_communicate(text)
        await asyncio.wait_for(comm.save(str(output_path)), timeout=self._timeout)
        return output_path

    def _save_to_file(self, text: str, output_path: Path) -> Path:
        return self._run_async(self._async_save_to_file(text, output_path))

    def play(self, text: str, output_path: Optional[Path] = None) -> Union[bool, str]:
        if not text or not text.strip():
            return False
        if self._closed:
            logger.warning("EdgeTTSEngine is closed")
            return False
        auto = output_path is None
        try:
            if auto:
                self._cleanup_previous()
                self._ensure_audio_dir()
            if output_path is None:
                audio_dir = self._audio_dir()
                output_path = audio_dir / "edge_{0}.wav".format(hash(text) & 0xFFFFFFFF)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_to_file(text, output_path)
            if auto:
                self._last_file = output_path
            if _play_wav_file(output_path):
                return True
            return str(output_path)
        except Exception as exc:
            logger.error("edge-tts play failed: %s", exc)
            if not auto and output_path is not None and output_path.exists():
                return str(output_path)
            return False

    def speak(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        if self._closed:
            logger.warning("EdgeTTSEngine is closed")
            return False
        try:
            result = self.play(text)
            if isinstance(result, bool):
                return result
            return False
        except Exception as exc:
            logger.error("edge-tts speak failed: %s", exc)
            return False

    def close(self) -> None:
        self._cleanup_previous()
        self._client = None
        self._health_cached = False
        self._closed = True
        logger.info("EdgeTTSEngine closed")


__all__ = ["EdgeTTSEngine"]