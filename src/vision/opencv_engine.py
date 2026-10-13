"""OpenCV frame capture and lightweight image preprocessing.

The module intentionally imports cv2 and numpy only when an operation needs
them, so the vision package remains usable on machines without OpenCV.
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


class OpenCVFrameCapture(VisionEngine):
    """Capture, resize, encode, save, and inspect camera frames."""

    def __init__(
        self,
        config: Any = None,
        paths: Any = None,
        *,
        max_width: int | None = None,
        jpeg_quality: int | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self._config = config
        self._paths = paths
        self._max_width = self._positive_int(
            self._config_value(
                "vision_max_width", "capture_max_width", "max_width", default=max_width
            ),
            640,
        )
        self._jpeg_quality = self._quality(
            self._config_value(
                "vision_jpeg_quality", "capture_jpeg_quality", "jpeg_quality", default=jpeg_quality
            ),
            85,
        )
        self._output_dir = self._resolve_output_dir(output_dir)
        self._camera: Any = None
        self._camera_index: int = -1
        self._captured_image: Any = None
        self._last_frame: Any = None
        self._last_image_bytes: bytes | None = None
        self._last_path: Path | None = None
        self._closed = False

    @property
    def health(self) -> bool:
        """Return True when OpenCV is available and a camera can be opened."""
        if self._closed:
            return False
        try:
            cv2, _ = self._import_cv2()
            cap = cv2.VideoCapture(0)
            try:
                return bool(cap.isOpened())
            finally:
                cap.release()
        except Exception:
            return False

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def captured_image(self) -> Any:
        return self._captured_image

    @property
    def captured_frame(self) -> Any:
        return self._captured_image

    @property
    def last_frame(self) -> Any:
        return self._last_frame

    @property
    def capture_path(self) -> Path | None:
        return self._last_path

    @property
    def last_image_bytes(self) -> bytes | None:
        return self._last_image_bytes

    @property
    def last_path(self) -> Path | None:
        return self._last_path

    def analyze(self, image: Path | bytes | Any, prompt: str = "") -> str:
        """Load an image, run lightweight preprocessing, and describe its edges."""
        if self._closed:
            return "Error: OpenCVFrameCapture is closed."
        try:
            cv2, np = self._import_cv2()
            img = self._load_image(cv2, np, image)
            if img is None:
                return "Error: could not load image for analysis."
            processed = self.preprocess_frame(img, cv2=cv2, np=np)
            h, w = processed.shape[:2]
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_count = int(np.count_nonzero(edges))
            result = "Image size: {0}x{1}, edges: {2}".format(w, h, edge_count)
            if prompt:
                result = "[{0}] {1}".format(prompt, result)
            return result
        except Exception as exc:
            logger.debug("OpenCV image analysis failed: %s", exc)
            return "Error: {0}".format(exc)

    def capture(self, prompt: str = "", camera_index: int = 0) -> str:
        """Capture one frame, resize it, persist JPEG data, and return its path."""
        if self._closed:
            return "Error: OpenCVFrameCapture is closed."
        cv2 = None
        np = None
        cap = None
        try:
            cv2, np = self._import_cv2()
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
                return "Error: failed to encode frame as JPEG."
            jpeg_bytes = encoded.tobytes()
            save_dir = self._ensure_output_dir()
            if save_dir is None:
                return "Error: no writable vision output directory is available."
            save_path = self._next_capture_path(save_dir, int(camera_index))
            try:
                save_path.write_bytes(jpeg_bytes)
            except OSError as exc:
                return "Error: failed to save frame to {0}: {1}".format(save_path, exc)

            self._captured_image = processed
            self._last_frame = processed
            self._last_image_bytes = jpeg_bytes
            self._last_path = save_path
            self._camera_index = int(camera_index)
            result = "Captured frame {0}x{1}, saved to {2}".format(
                processed.shape[1], processed.shape[0], save_path
            )
            if prompt:
                result = "[{0}] {1}".format(prompt, result)
            return result
        except Exception as exc:
            logger.debug("OpenCV frame capture failed: %s", exc)
            return "Error: {0}".format(exc)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def preprocess_frame(self, frame: Any, *, cv2: Any = None, np: Any = None) -> Any:
        """Resize a BGR frame while preserving aspect ratio."""
        if cv2 is None or np is None:
            cv2, np = self._import_cv2()
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise ValueError("frame must be a non-empty image array")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have one, three, or four channels")
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            raise ValueError("frame dimensions must be positive")
        if w > self._max_width:
            scale = self._max_width / float(w)
            new_w = self._max_width
            new_h = max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame

    def preprocess(self, image: Path | bytes | Any) -> Any:
        """Load and preprocess an image without performing model inference."""
        cv2, np = self._import_cv2()
        img = self._load_image(cv2, np, image)
        if img is None:
            raise ValueError("could not load image")
        return self.preprocess_frame(img, cv2=cv2, np=np)

    def preprocess_image(self, image: Path | bytes | Any) -> Any:
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(image)

    def save_frame(self, frame: Any, filename: str | None = None) -> Path:
        """Encode and save a BGR frame, returning the resulting path."""
        cv2, np = self._import_cv2()
        processed = self.preprocess_frame(frame, cv2=cv2, np=np)
        success, encoded = cv2.imencode(
            ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not success or encoded is None:
            raise RuntimeError("failed to encode frame as JPEG")
        save_dir = self._ensure_output_dir()
        if save_dir is None:
            raise OSError("no writable vision output directory is available")
        save_path = save_dir / (filename or self._capture_filename(0))
        save_path.write_bytes(encoded.tobytes())
        self._captured_image = processed
        self._last_frame = processed
        self._last_image_bytes = encoded.tobytes()
        self._last_path = save_path
        return save_path

    def close(self) -> None:
        """Release a persistent camera handle, if one is held."""
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None
            self._camera_index = -1
        self._captured_image = None
        self._closed = True

    def _import_cv2(self) -> tuple[Any, Any]:
        try:
            import cv2 as _cv2
            import numpy as _np
        except ImportError as exc:
            raise ImportError(
                "OpenCVFrameCapture requires opencv-python and numpy"
            ) from exc
        return _cv2, _np

    def _load_image(
        self, cv2: Any, np: Any, image: Path | bytes | Any
    ) -> Any:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, (bytes, bytearray, memoryview)):
            buffer = np.frombuffer(bytes(image), dtype=np.uint8)
            return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if isinstance(image, (Path, str)):
            return cv2.imread(str(image), cv2.IMREAD_COLOR)
        return None

    def _resolve_output_dir(self, explicit: str | Path | None) -> Path:
        if explicit is not None:
            return Path(explicit).expanduser()
        configured = self._config_value(
            "vision_output_dir", "vision_dir", "capture_dir", default=None
        )
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

    def _next_capture_path(self, directory: Path, camera_index: int) -> Path:
        stamp = int(time.time())
        path = directory / self._capture_filename(camera_index, stamp)
        suffix = 1
        while path.exists():
            path = directory / self._capture_filename(camera_index, stamp, suffix)
            suffix += 1
        return path

    @staticmethod
    def _capture_filename(camera_index: int, stamp: int, suffix: int | None = None) -> str:
        name = "frame_{0}_{1}".format(camera_index, stamp)
        if suffix:
            name += "_{0}".format(suffix)
        return name + ".jpg"

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
    def _quality(value: Any, fallback: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return fallback
        return min(100, max(1, result))

    def _get_save_dir(self) -> str:
        """Compatibility helper returning the selected output directory."""
        return str(self._ensure_output_dir() or self._output_dir)


__all__ = ["OpenCVFrameCapture"]