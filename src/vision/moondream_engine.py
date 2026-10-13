"""Ollama Moondream image-analysis engine.

Images are resized and JPEG-compressed before being base64-encoded and sent to
Ollama.  Heavy dependencies are imported only when an operation needs them.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from .base import VisionEngine

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path(r"D:\AI_System")
_DEFAULT_OUTPUT_DIR = _DEFAULT_ROOT / "vision"
_FALLBACK_OUTPUT_DIR = _DEFAULT_ROOT / "audio"


class OllamaMoondreamEngine(VisionEngine):
    """Analyze images with Ollama's ``moondream:latest`` model."""

    def __init__(
        self,
        config: Any = None,
        paths: Any = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_width: int | None = None,
        jpeg_quality: int | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self._config = config
        self._paths = paths
        self.base_url = self._resolve_base_url(base_url).rstrip("/")
        self.model = self._resolve_model(model)
        self.timeout = self._positive_float(
            self._config_value("vision_timeout", "ollama_timeout", "timeout", default=timeout),
            90.0,
        )
        self._max_width = self._positive_int(
            self._config_value(
                "vision_max_width", "image_max_width", "max_width", default=max_width
            ),
            640,
        )
        self._jpeg_quality = self._quality(
            self._config_value(
                "vision_jpeg_quality", "image_jpeg_quality", "jpeg_quality", default=jpeg_quality
            ),
            80,
        )
        self._output_dir = self._resolve_output_dir(output_dir)
        self._model = self.model
        self._timeout = self.timeout
        self._client: Any = None
        self._closed = False
        self._available = False
        self._last_frame: Any = None
        self._last_image_bytes: bytes | None = None
        self._last_path: Path | None = None
        self._ollama_host = self.base_url

    @property
    def health(self) -> bool:
        """Check Ollama liveness and, when possible, the configured model."""
        if self._closed:
            return False
        try:
            httpx = self._import_httpx()
            client = self._get_client()
            tags_url = f"{self.base_url}/api/tags"
            try:
                response = client.get(tags_url, timeout=min(5.0, self.timeout))
            except Exception as exc:
                logger.debug("Ollama tags check failed: %s", exc)
                response = None
            if response is not None and getattr(response, "status_code", None) == 200:
                try:
                    data = response.json()
                except Exception:
                    data = {}
                if self._model_present(data):
                    self._available = True
                    return True

            version_url = f"{self.base_url}/api/version"
            try:
                response = client.get(version_url, timeout=min(5.0, self.timeout))
                self._available = getattr(response, "status_code", None) == 200
                return self._available
            except Exception as exc:
                logger.debug("Ollama version check failed: %s", exc)
                self._available = False
                return False
        except Exception as exc:
            logger.debug("Ollama health check failed: %s", exc)
            self._available = False
            return False

    @property
    def ollama_host(self) -> str:
        return self.base_url

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def last_frame(self) -> Any:
        return self._last_frame

    @property
    def last_image_bytes(self) -> bytes | None:
        return self._last_image_bytes

    @property
    def last_path(self) -> Path | None:
        return self._last_path

    def analyze(self, image: Path | bytes | Any, prompt: str = "") -> str:
        """Send one base64 JPEG to Ollama and return the model response."""
        if self._closed:
            return "Error: OllamaMoondreamEngine is closed."
        if not prompt or not prompt.strip():
            prompt = "Describe this image."
        encoded_image = self._image_to_base64(image)
        if not encoded_image:
            return "Error: could not prepare or encode image for analysis."

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False,
        }
        try:
            client = self._get_client()
            endpoint = self._analysis_endpoint()
            response = client.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.debug("Ollama image analysis failed: %s", exc)
            return "Error: Ollama request failed: {0}".format(exc)

        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            detail = getattr(response, "text", "")
            if len(detail) > 500:
                detail = detail[:500] + "..."
            return "Error: Ollama API returned status {0}: {1}".format(status_code, detail)
        try:
            data = response.json()
        except Exception as exc:
            return "Error: invalid JSON response from Ollama: {0}".format(exc)
        text = self._response_text(data)
        if text:
            self._available = True
            return text
        self._available = False
        return "Error: empty response from Ollama."

    def capture(self, prompt: str = "", camera_index: int = 0) -> str:
        """Capture one camera frame and analyze it with Moondream."""
        if self._closed:
            return "Error: OllamaMoondreamEngine is closed."
        try:
            cv2, np = self._import_cv2()
        except Exception as exc:
            return "Error: OpenCV is required for camera capture: {0}".format(exc)

        cap = None
        try:
            cap = cv2.VideoCapture(int(camera_index))
            if not cap.isOpened():
                return "Error: camera {0} could not be opened.".format(camera_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                return "Error: failed to read frame from camera {0}.".format(camera_index)

            processed = self.preprocess_frame(frame, cv2=cv2, np=np)
            success, encoded = cv2.imencode(
                ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not success or encoded is None:
                return "Error: failed to encode captured frame as JPEG."
            jpeg_bytes = encoded.tobytes()
            self._last_frame = processed
            self._last_image_bytes = jpeg_bytes
            self._last_path = self._save_bytes(jpeg_bytes, int(camera_index))
            return self.analyze(jpeg_bytes, prompt)
        except Exception as exc:
            logger.debug("Moondream camera capture failed: %s", exc)
            return "Error: {0}".format(exc)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def preprocess_frame(self, frame: Any, *, cv2: Any = None, np: Any = None) -> Any:
        """Resize a BGR frame while preserving its aspect ratio."""
        if cv2 is None or np is None:
            cv2, np = self._import_cv2()
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame must be a non-empty image array")
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            raise ValueError("frame dimensions must be positive")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have one, three, or four channels")
        if w > self._max_width:
            scale = self._max_width / float(w)
            new_w = self._max_width
            new_h = max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def preprocess(self, image: Path | bytes | Any) -> bytes:
        """Load an image and return a resized JPEG payload."""
        return self._image_to_jpeg_bytes(image)

    def preprocess_image(self, image: Path | bytes | Any) -> bytes:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def compress_image(self, image: Path | bytes | Any) -> bytes:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def close(self) -> None:
        """Close the lazy HTTP client and release cached resources."""
        self._closed = True
        self._available = False
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._client = None
        self._last_frame = None
        self._last_image_bytes = None
        self._last_path = None

    def _import_httpx(self) -> Any:
        try:
            import httpx as _httpx
        except ImportError as exc:
            raise RuntimeError(
                "OllamaMoondreamEngine requires httpx; install it with: pip install httpx"
            ) from exc
        return _httpx

    def _import_base64(self) -> Any:
        import base64 as _base64
        return _base64

    def _import_cv2(self) -> tuple[Any, Any]:
        try:
            import cv2 as _cv2
            import numpy as _np
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV and numpy are required for image preprocessing"
            ) from exc
        return _cv2, _np

    def _get_client(self) -> Any:
        if self._client is None:
            httpx = self._import_httpx()
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _image_to_base64(self, image: Path | bytes | Any) -> str | None:
        """Return a raw base64 image payload, or ``None`` when unreadable."""
        try:
            image_bytes = self._image_to_jpeg_bytes(image)
            return self._import_base64().b64encode(image_bytes).decode("ascii")
        except Exception:
            return None

    def _image_to_jpeg_bytes(self, image: Path | bytes | Any) -> bytes:
        if isinstance(image, str) and image.lstrip().startswith("data:"):
            header, separator, payload = image.partition(",")
            if separator and "base64" in header.lower():
                image = self._import_base64().b64decode(payload)
            else:
                image = Path(image)

        if isinstance(image, (bytes, bytearray, memoryview)):
            raw = bytes(image)
            try:
                cv2, np = self._import_cv2()
                decoded = self._decode_bytes(cv2, np, raw)
                if decoded is not None:
                    return self._compress_frame(cv2, np, decoded)
            except Exception:
                pass
            return raw

        if isinstance(image, (Path, str)):
            path = Path(image).expanduser()
            try:
                cv2, np = self._import_cv2()
                decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if decoded is not None:
                    return self._compress_frame(cv2, np, decoded)
            except Exception:
                pass
            return path.read_bytes()

        if isinstance(image, dict) and "base64" in image:
            value = image["base64"]
            if isinstance(value, str):
                return self._image_to_jpeg_bytes(value)
            return self._image_to_jpeg_bytes(value)

        try:
            cv2, np = self._import_cv2()
            if isinstance(image, np.ndarray):
                return self._compress_frame(cv2, np, image)
        except Exception:
            pass

        read = getattr(image, "read", None)
        if callable(read):
            return self._image_to_jpeg_bytes(read())
        raise TypeError("unsupported image input: {0}".format(type(image).__name__))

    def _image_to_bytes(self, image: Path | bytes | Any) -> bytes | None:
        """Compatibility wrapper returning ``None`` instead of raising."""
        try:
            return self._image_to_jpeg_bytes(image)
        except Exception:
            return None

    def _decode_bytes(self, cv2: Any, np: Any, raw: bytes) -> Any:
        buffer = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def _compress_frame(self, cv2: Any, np: Any, frame: Any) -> bytes:
        processed = self.preprocess_frame(frame, cv2=cv2, np=np)
        success, encoded = cv2.imencode(
            ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not success or encoded is None:
            raise RuntimeError("failed to encode image as JPEG")
        return encoded.tobytes()

    def _response_text(self, data: Any) -> str:
        if isinstance(data, dict):
            response = data.get("response")
            if response:
                return str(response).strip()
            message = data.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if content:
                    return str(content).strip()
            for key in ("text", "output"):
                if data.get(key):
                    return str(data[key]).strip()
        if isinstance(data, str):
            return data.strip()
        return ""

    def _analysis_endpoint(self) -> str:
        endpoint = self._config_value(
            "vision_endpoint", "ollama_vision_endpoint", default="/api/generate"
        )
        if not endpoint:
            endpoint = "/api/generate"
        endpoint = str(endpoint).strip()
        return endpoint if endpoint.startswith("/") else "/" + endpoint

    def _model_present(self, data: Any) -> bool:
        if not isinstance(data, dict):
            return False
        models = data.get("models", data.get("data", []))
        if not isinstance(models, list):
            return False
        wanted = self.model.lower()
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", item.get("model", ""))).lower()
            if name == wanted or name.startswith(wanted + "@") or wanted.startswith(name + "@"):
                return True
        return False

    def _resolve_base_url(self, explicit: str | None) -> str:
        if explicit:
            return str(explicit)
        value = self._config_value("ollama_host", "ollama_url", "base_url", default=None)
        if value:
            return str(value)
        value = os.environ.get("JARVIS_OLLAMA_HOST")
        return value or "http://localhost:11434"

    def _resolve_model(self, explicit: str | None) -> str:
        if explicit:
            return str(explicit)
        value = self._config_value("vision_model", "moondream_model", default=None)
        if value:
            return str(value)
        return os.environ.get("JARVIS_VISION_MODEL", "moondream:latest")

    def _resolve_output_dir(self, explicit: str | Path | None) -> Path:
        if explicit is not None:
            return Path(explicit).expanduser()
        configured = self._config_value("vision_output_dir", "vision_dir", default=None)
        if configured:
            return Path(configured).expanduser()
        env_dir = os.environ.get("JARVIS_VISION_OUTPUT_DIR")
        if env_dir:
            return Path(env_dir).expanduser()
        if self._paths is not None:
            for key in ("vision", "vision_dir", "vision_output", "capture"):
                try:
                    value = self._paths.get(key)
                except Exception:
                    value = None
                if value:
                    return Path(value).expanduser()
            base_dir = getattr(self._paths, "base_dir", None)
            if base_dir:
                return Path(base_dir).expanduser() / "vision"
            root = getattr(self._paths, "root", None)
            if root:
                return Path(root).expanduser() / "vision"
        return _DEFAULT_OUTPUT_DIR

    def _ensure_output_dir(self) -> Path | None:
        candidates = [self._output_dir]
        if self._output_dir != _DEFAULT_OUTPUT_DIR:
            candidates.append(_DEFAULT_OUTPUT_DIR)
        if self._output_dir != _FALLBACK_OUTPUT_DIR:
            candidates.append(_FALLBACK_OUTPUT_DIR)
        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
            except OSError as exc:
                logger.debug("Cannot create vision directory %s: %s", candidate, exc)
        return None

    def _save_bytes(self, data: bytes, camera_index: int) -> Path | None:
        directory = self._ensure_output_dir()
        if directory is None:
            return None
        stamp = int(time.time())
        path = directory / "frame_{0}_{1}.jpg".format(camera_index, stamp)
        suffix = 1
        while path.exists():
            path = directory / "frame_{0}_{1}_{2}.jpg".format(camera_index, stamp, suffix)
            suffix += 1
        try:
            path.write_bytes(data)
            return path
        except OSError as exc:
            logger.debug("Cannot save captured frame %s: %s", path, exc)
            return None

    def _config_value(self, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(self._config, dict) and name in self._config:
                return self._config[name]
            if self._config is not None and hasattr(self._config, name):
                return getattr(self._config, name)
        return default

    @staticmethod
    def _positive_int(value: Any, fallback: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return fallback
        return result if result > 0 else fallback

    @staticmethod
    def _positive_float(value: Any, fallback: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return fallback
        return result if result > 0 else fallback

    @staticmethod
    def _quality(value: Any, fallback: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return fallback
        return min(100, max(1, result))


__all__ = ["OllamaMoondreamEngine"]