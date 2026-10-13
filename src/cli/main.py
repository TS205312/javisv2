"""Command-line interface for the standalone Jarvis assistant."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Optional

from .console import Console, render_value
from src.core.config import Config
from src.core.paths import PathManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_VOICE_DURATION = 5.0
DEFAULT_VOICE_PAUSE = 0.5
FALLBACK_VOICE_TEXT = "Xin loi, toi chua nghe ro. Ban co the nhap text khong?"


class CLIError(RuntimeError):
    """An expected CLI or assistant error that should not show a traceback."""


class AssistantRuntime:
    """Load configuration and own the future assistant orchestrator."""

    def __init__(
        self,
        config_path: Path,
        base_dir: Optional[str] = None,
        brain_dir: Optional[str] = None,
    ) -> None:
        resolved_config = (
            config_path
            if config_path is not None
            else DEFAULT_CONFIG_PATH
        ).expanduser().resolve()
        self.config_path = resolved_config
        self.config = Config.from_yaml(str(resolved_config))
        self.paths = PathManager(base_dir=base_dir, brain_dir=brain_dir)
        self.orchestrator: Any = None

    def get_orchestrator(self) -> Any:
        if self.orchestrator is None:
            orchestrator_class = load_orchestrator_class()
            self.orchestrator = orchestrator_class(
                config=self.config,
                paths=self.paths,
            )
        return self.orchestrator

    def close(self) -> None:
        if self.orchestrator is None:
            return
        close = getattr(self.orchestrator, "close", None)
        if callable(close):
            close()
        self.orchestrator = None

    def __enter__(self) -> "AssistantRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class AssistantApp:
    """Dispatch CLI commands against one runtime instance."""

    def __init__(self, args: argparse.Namespace, console: Optional[Console] = None) -> None:
        self.args = args
        self.console = console or Console(enabled=not bool(args.no_rich))
        self.runtime = AssistantRuntime(
            config_path=getattr(args, "config_path", None),
            base_dir=getattr(args, "base_dir", None),
            brain_dir=getattr(args, "brain_dir", None),
        )

    def run(self) -> int:
        try:
            if self.args.command == "doctor":
                return self.run_doctor()
            if self.args.command == "config":
                return self.run_config()

            orchestrator = self.runtime.get_orchestrator()
            handlers = {
                "chat": self.run_chat,
                "voice": self.run_voice,
                "vision": self.run_vision,
                "ingest": self.run_ingest,
                "status": self.run_status,
            }
            return handlers[self.args.command](orchestrator)
        except KeyboardInterrupt:
            self.console.warning("Da dung lai by Ctrl-C.")
            return 130
        except CLIError as exc:
            self.console.error(str(exc))
            return 2
        except (OSError, ValueError, RuntimeError) as exc:
            self.console.error(str(exc))
            return 1
        except Exception as exc:
            self.console.error("Loi: {0}".format(exc))
            return 1
        finally:
            self.runtime.close()

    def run_chat(self, orchestrator: Any) -> int:
        text = " ".join(self.args.text).strip()
        if not text:
            raise CLIError("Chat requires text.")
        result = orchestrator.chat(text=text, speak=self.args.speak)
        display_command_result(self.console, result)
        return 0

    def run_voice(self, orchestrator: Any) -> int:
        duration = self.args.duration
        speak = self.args.speak
        turns = 0

        while True:
            try:
                result = orchestrator.voice_turn(duration=duration, speak=speak)
                normalized = normalize_result(result)
                if normalized.response.strip():
                    display_command_result(self.console, result)
                else:
                    self.console.warning("Khong phat hien duoc noi dung.")
                    fallback_text = self.args.fallback_text or FALLBACK_VOICE_TEXT
                    fallback_result = orchestrator.chat(
                        text=fallback_text,
                        speak=speak,
                    )
                    display_command_result(self.console, fallback_result)

                turns += 1
                if self.args.once or (
                    self.args.max_turns is not None and turns >= self.args.max_turns
                ):
                    return 0
                time.sleep(self.args.pause)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.console.error("Voice error: {0}".format(exc))
                if self.args.once:
                    return 1
                time.sleep(self.args.pause)

    def run_vision(self, orchestrator: Any) -> int:
        prompt = " ".join(self.args.prompt).strip()
        if not prompt:
            raise CLIError("Vision requires a prompt.")
        image: Optional[Path] = None
        if self.args.image is not None:
            image = self.args.image.expanduser()
            if not image.is_file():
                raise CLIError("Image file does not exist: {0}".format(image))
        result = orchestrator.vision(
            prompt=prompt,
            image=image,
            camera_index=self.args.camera_index,
            speak=self.args.speak,
        )
        display_command_result(self.console, result)
        return 0

    def run_ingest(self, orchestrator: Any) -> int:
        paths = [path.expanduser().resolve() for path in self.args.paths]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise CLIError("Input path does not exist: {0}".format(missing[0]))
        result = orchestrator.ingest(paths=paths)
        display_command_result(self.console, result)
        return 0

    def run_status(self, orchestrator: Any) -> int:
        result = orchestrator.status()
        self.console.print(render_value(result))
        return 0

    def run_config(self) -> int:
        data = self.runtime.config.as_dict()
        key = self.args.key
        if key:
            if key not in data:
                raise CLIError("Unknown config key: {0}".format(key))
            value: Any = data[key]
            if self.args.json:
                self.console.print(json.dumps(value, ensure_ascii=False, indent=2))
            else:
                self.console.print(value)
            return 0

        if self.args.json:
            self.console.print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0

        self.console.rows(
            ((name, value) for name, value in data.items()),
            title="Jarvis configuration",
            columns=("Key", "Value"),
        )
        return 0

    def run_doctor(self) -> int:
        checks = Doctor(self.runtime.config, self.runtime.paths).run()
        self.console.rows(
            checks,
            title="Jarvis doctor",
            columns=("Check", "Status", "Detail"),
        )
        failed = any(str(row[1]).upper() == "FAIL" for row in checks)
        return 1 if failed else 0


class Doctor:
    """Run dependency and path checks without loading assistant models."""

    def __init__(self, config: Config, paths: PathManager) -> None:
        self.config = config
        self.paths = paths

    def run(self) -> list[tuple[str, str, str]]:
        checks: list[tuple[str, str, str]] = []
        checks.extend(self._check_paths())
        checks.append(self._check_ollama())
        checks.append(self._check_stt_model())
        checks.append(self._check_tts())
        checks.append(self._check_rag())
        checks.append(self._check_camera())
        return checks

    def _check_paths(self) -> list[tuple[str, str, str]]:
        checks: list[tuple[str, str, str]] = []
        required = (
            ("Base directory", self.paths.base_dir),
            ("Brain directory", self.paths.brain_dir),
            ("Models directory", self.paths.get("models")),
            ("Cache directory", self.paths.get("cache")),
            ("Logs directory", self.paths.get("logs")),
            ("FAISS directory", self.paths.get("faiss")),
            ("Audio directory", self.paths.get("audio")),
            ("Vision directory", self.paths.get("vision")),
            ("Hugging Face cache", self.paths.get("huggingface")),
        )
        for name, path in required:
            if not path.exists():
                checks.append((name, "WARN", "Missing: {0}".format(path)))
            elif not path.is_dir():
                checks.append((name, "FAIL", "Not a directory: {0}".format(path)))
            elif not os.access(str(path), os.W_OK):
                checks.append((name, "FAIL", "Not writable: {0}".format(path)))
            else:
                checks.append((name, "OK", str(path)))
        return checks

    def _check_ollama(self) -> tuple[str, str, str]:
        try:
            import httpx
        except ImportError:
            return "Ollama health", "FAIL", "httpx is not installed"

        host = str(self.config.ollama_host).rstrip("/")
        last_error = "No response"
        for endpoint in ("/api/version", "/api/tags"):
            try:
                with httpx.Client(base_url=host, timeout=3.0) as client:
                    response = client.get(endpoint)
                if response.status_code < 500:
                    return "Ollama health", "OK", "{0} returned {1}".format(
                        endpoint, response.status_code
                    )
                last_error = "HTTP {0}".format(response.status_code)
            except Exception as exc:
                last_error = str(exc)
        return "Ollama health", "FAIL", last_error

    def _check_stt_model(self) -> tuple[str, str, str]:
        model_path = self.paths.vosk_model_path(self.config.stt_model_path)
        required = ("am", "conf", "graph")
        missing = [name for name in required if not (model_path / name).is_dir()]
        if not model_path.is_dir():
            return "STT model", "FAIL", "Missing: {0}".format(model_path)
        if missing:
            return "STT model", "FAIL", "Missing model parts: {0}".format(", ".join(missing))
        return "STT model", "OK", str(model_path)

    def _check_tts(self) -> tuple[str, str, str]:
        try:
            from src.tts.factory import create_tts_engine
            engine = create_tts_engine(self.config, self.paths)
            try:
                healthy = bool(engine.health)
            finally:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            return "TTS health", "FAIL", str(exc)
        if healthy:
            return "TTS health", "OK", "At least one TTS provider is available"
        return "TTS health", "WARN", "No TTS provider is currently available"

    def _check_rag(self) -> tuple[str, str, str]:
        try:
            from src.rag.factory import create_rag_engine
            engine = create_rag_engine(self.config, self.paths)
            try:
                healthy = bool(engine.health())
            finally:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            return "RAG health", "FAIL", str(exc)
        faiss_dir = self.paths.get("faiss")
        index_ready = (faiss_dir / "index.faiss").is_file()
        metadata_ready = (faiss_dir / "metadata.jsonl").is_file()
        if healthy:
            detail = "Ready"
            if not (index_ready and metadata_ready):
                detail += "; index is empty"
            return "RAG health", "OK", detail
        return "RAG health", "WARN", "Dependencies or index are unavailable"

    def _check_camera(self) -> tuple[str, str, str]:
        try:
            from src.vision.factory import create_opencv_engine
            engine = create_opencv_engine(self.config, self.paths)
            try:
                healthy = bool(engine.health)
            finally:
                close = getattr(engine, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            return "Camera health", "FAIL", str(exc)
        if healthy:
            return "Camera health", "OK", "Camera 0 opened successfully"
        return "Camera health", "WARN", "Camera 0 is unavailable"


def load_orchestrator_class() -> Any:
    """Load the pipeline orchestrator without importing it at CLI startup."""
    module_name = os.environ.get(
        "JARVIS_ORCHESTRATOR_MODULE",
        "src.pipeline.orchestrator",
    )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CLIError(
            "AssistantOrchestrator is not available. Run setup and ensure {0} exists.".format(
                module_name
            )
        ) from exc
    orchestrator_class = getattr(module, "AssistantOrchestrator", None)
    if not callable(orchestrator_class):
        raise CLIError(
            "{0} does not define AssistantOrchestrator.".format(module_name)
        )
    return orchestrator_class


def normalize_result(result: Any) -> tuple[Optional[bool], str]:
    """Normalize CommandResult-like values returned by the pipeline API."""
    if result is None:
        return None, ""
    if isinstance(result, str):
        return None, result
    if isinstance(result, Mapping):
        handled = result.get("handled")
        response = result.get("response", result.get("text", ""))
        return _optional_bool(handled), render_value(response)
    if is_dataclass(result):
        handled = getattr(result, "handled", None)
        response = getattr(result, "response", "")
        return _optional_bool(handled), render_value(response)
    handled = getattr(result, "handled", None)
    response = getattr(result, "response", None)
    if response is None:
        response = getattr(result, "text", result)
    return _optional_bool(handled), render_value(response)


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def display_command_result(console: Console, result: Any) -> None:
    handled, response = normalize_result(result)
    if response:
        console.print(response)
        return
    if handled is False:
        console.warning("Command was not handled and returned no response.")
        return
    console.success("OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis-local",
        description="Jarvis standalone local AI assistant CLI",
    )
    _add_common_arguments(parser, suppress_defaults=False)
    parser.add_argument(
        "--version",
        action="version",
        version="jarvis-local 1.0.0",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="Send a text message")
    _add_common_arguments(chat, suppress_defaults=True)
    chat.add_argument("text", nargs="+", help="Message text")
    chat.add_argument("--speak", action="store_true", help="Speak the response")
    chat.set_defaults(func="chat", speak=False)

    voice = subparsers.add_parser("voice", help="Run a safe voice-turn loop")
    _add_common_arguments(voice, suppress_defaults=True)
    voice.add_argument(
        "--duration",
        type=_positive_float,
        default=DEFAULT_VOICE_DURATION,
        help="Recording duration in seconds (default: 5)",
    )
    voice.add_argument(
        "--pause",
        type=_nonnegative_float,
        default=DEFAULT_VOICE_PAUSE,
        help="Pause between turns in seconds (default: 0.5)",
    )
    voice.add_argument("--once", action="store_true", help="Run one voice turn")
    voice.add_argument(
        "--max-turns",
        type=_positive_int,
        help="Stop after this many completed turns",
    )
    voice.add_argument(
        "--fallback-text",
        help="Text sent to chat when voice recognition returns no response",
    )
    voice.add_argument(
        "--speak",
        dest="speak",
        action="store_true",
        default=True,
        help="Speak responses (default)",
    )
    voice.add_argument("--no-speak", dest="speak", action="store_false")
    voice.set_defaults(func="voice")

    vision = subparsers.add_parser("vision", help="Analyze an image or camera frame")
    _add_common_arguments(vision, suppress_defaults=True)
    vision.add_argument("prompt", nargs="+", help="Vision prompt")
    vision.add_argument("--image", type=Path, help="Image file to analyze")
    vision.add_argument(
        "--camera-index",
        type=_nonnegative_int,
        default=0,
        help="Camera index when --image is omitted (default: 0)",
    )
    vision.add_argument("--speak", action="store_true", help="Speak the response")
    vision.set_defaults(func="vision", speak=False)

    ingest = subparsers.add_parser("ingest", help="Ingest files or directories into RAG")
    _add_common_arguments(ingest, suppress_defaults=True)
    ingest.add_argument("paths", nargs="+", type=Path, help="Files or directories")
    ingest.set_defaults(func="ingest")

    status = subparsers.add_parser("status", help="Show assistant status")
    _add_common_arguments(status, suppress_defaults=True)
    status.set_defaults(func="status")

    doctor = subparsers.add_parser("doctor", help="Check paths and local services")
    _add_common_arguments(doctor, suppress_defaults=True)
    doctor.set_defaults(func="doctor")

    config = subparsers.add_parser("config", help="Show resolved configuration")
    _add_common_arguments(config, suppress_defaults=True)
    config.add_argument("key", nargs="?", help="Show one configuration key")
    config.add_argument("--json", action="store_true", help="Emit JSON")
    config.set_defaults(func="config", json=False)

    return parser


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool,
) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=default,
        help="Path to config.yaml",
    )
    parser.add_argument("--base-dir", default=default, help="Override D:\\AI_System")
    parser.add_argument("--brain-dir", default=default, help="Override D:\\AI_Brain")
    parser.add_argument(
        "--no-rich",
        action="store_true",
        default=default,
        help="Use plain console output even when Rich is installed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=default,
        help="Reserved for detailed diagnostics",
    )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "config_path", None) is None:
        args.config_path = DEFAULT_CONFIG_PATH
    if getattr(args, "no_rich", None) is None:
        args.no_rich = False
    console = Console(enabled=not bool(args.no_rich))
    return AssistantApp(args, console).run()


if __name__ == "__main__":
    raise SystemExit(main())
