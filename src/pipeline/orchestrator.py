"""Assistant orchestration for Jarvis pipeline."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import Config, load_config
from src.core.paths import PathManager
from src.core.events import EventBus, create_event_bus
from src.core.lazy import LazyLoader
from src.core.fallback import FallbackChain

from .commands import CommandResult, CommandRegistry, build_default_registry
from .session import SessionManager, Turn

logger = logging.getLogger(__name__)

DEFAULT_LLM_SYSTEM_PROMPT = (
    "B\u00e1n l\u00e1 m\u00f3 t\u01b0\u1eddng AI h\u1ed9i th\u01b0\u1ecdng c\u00e1ch m\u1ea1nh m\u1eeb, "
    "th\u00e2n thi\u1ec7n, h\u1ed9 tr\u1ee3 \u0111\u1ea7u v\xe0o c\xe1c c\xe2u h\u1ec7i c\u1ee7a ng\u01b0\u1eddi d\xf9ng. "
    "H\u1ec7i tr\u1ea3 l\u1eddi b\u1eb1ng ti\u1ebfng Vi\u1ec7t, ngo\u1ea4i tr\u1eeb khi ng\u01b0\u1eddi d\xf9ng "
    "y\xeau c\u1ea1r b\u1eb1ng ng\u01b0\u1eddi kh\xe1c. "
    "Gi\u1ea3i \u0111\xe1p c\xe1c c\xe2u h\u1ec7i r\u00f5 r\xe0ng, s\u01b0\u1edbi, v\xe0 ch\xednh x\xe1c."
)

_FALLBACK_RESPONSE = (
    "Xin l\u1ed7i, h\u1ec7 th\u1ed1ng kh\xf4ng th\u1ec3 ph\u1ea3n h\u1ed3i \u011f\u1ecda. "
    "Vui l\xf2ng th\u1eed l\u1ea1i sau."
)

CONTEXT_CHAR_LIMIT = 3000


class AssistantOrchestrator:
    """Main assistant orchestrator."""

    def __init__(
        self,
        config: Optional[Config] = None,
        paths: Optional[PathManager] = None,
        engines: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._closed = False

        if config is None:
            try:
                self.config = load_config()
            except Exception as exc:
                logger.warning("Failed to load config: %s; using defaults", exc)
                self.config = Config()
        else:
            self.config = config

        if paths is None:
            self.paths = PathManager()
        else:
            self.paths = paths

        self._event_bus = None
        try:
            self._event_bus = create_event_bus(queue_size=500)
        except Exception as exc:
            logger.warning("Event bus failed: %s", exc)

        session_path = self.paths.get("root") / "session.json"
        try:
            self.session = SessionManager(session_path=session_path, max_turns=12)
            self.session.load()
        except Exception as exc:
            logger.warning("Session init failed: %s", exc)
            self.session = SessionManager(session_path=session_path, max_turns=12)

        self._engines: Dict[str, Any] = {}
        if engines is None:
            engines = {}

        if "stt" in engines:
            self._engines["stt"] = engines["stt"]
        else:
            try:
                from src.stt.factory import create_stt_engine
                self._engines["stt"] = create_stt_engine(config=self.config, paths=self.paths)
            except Exception as exc:
                logger.warning("STT failed: %s", exc)
                self._engines["stt"] = _STTEngineStub()

        if "llm" in engines:
            self._engines["llm"] = engines["llm"]
        else:
            try:
                from src.llm.factory import create_llm_engine
                self._engines["llm"] = create_llm_engine(config=self.config)
            except Exception as exc:
                logger.warning("LLM failed: %s", exc)
                self._engines["llm"] = _LLMEngineStub()

        if "tts" in engines:
            self._engines["tts"] = engines["tts"]
        else:
            try:
                from src.tts.factory import create_tts_engine
                self._engines["tts"] = create_tts_engine(config=self.config, paths=self.paths)
            except Exception as exc:
                logger.warning("TTS failed: %s", exc)
                self._engines["tts"] = _TTSEngineStub()

        if "rag" in engines:
            self._engines["rag"] = engines["rag"]
        else:
            try:
                from src.rag.factory import create_rag_engine
                self._engines["rag"] = create_rag_engine(config=self.config, paths=self.paths)
            except Exception as exc:
                logger.warning("RAG failed: %s", exc)
                self._engines["rag"] = _RAGEngineStub()

        if "vision" in engines and isinstance(engines["vision"], dict):
            self._engines["vision"] = engines["vision"]
        else:
            try:
                from src.vision.factory import create_vision_engines
                self._engines["vision"] = create_vision_engines(config=self.config, paths=self.paths)
            except Exception as exc:
                logger.warning("Vision failed: %s", exc)
                self._engines["vision"] = {
                    "opencv": _VisionStub("OpenCV"),
                    "moondream": _VisionStub("Moondream"),
                }

        self._commands = build_default_registry()
        self._llm_fallback = FallbackChain[str](
            providers=[], failure_threshold=2, cooldown_seconds=10.0, cache_ttl=60.0,
        )

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def event_bus(self) -> Optional[EventBus]:
        return self._event_bus

    def chat(self, text: str, *, speak: bool = False) -> str:
        if self._closed:
            return "Tra loi khong co san vi he thong da dong."
        try:
            result: CommandResult = self._commands.match(text)
            if result.handled:
                return self._dispatch_action(result)
            messages = self._build_chat_messages(text)
            response = self._call_llm(messages)
            try:
                self.session.add_turn(user=text, assistant=response)
            except Exception as exc:
                logger.warning("Save turn failed: %s", exc)
            if speak:
                try:
                    self._speak(response)
                except Exception as exc:
                    logger.warning("TTS failed: %s", exc)
            return response
        except Exception as exc:
            logger.exception("Chat error")
            return "Xin loi, da phat sinh loi: {0}".format(exc)

    def voice_turn(self, duration: Optional[float] = None, *, speak: bool = True) -> str:
        if self._closed:
            return "Tra loi khong co san vi he thong da dong."
        if duration is None:
            duration = 5.0
        try:
            stt = self._engines.get("stt")
            if stt is None or not getattr(stt, "health", False):
                return "Khong the ghi nhan gioi noi: STT khong kha dung."
            transcribed = stt.listen(duration=duration, prompt="")
            if not transcribed or not transcribed.strip():
                return "Khong nhan duoc loi noi. Vui long thu lai."
            return self.chat(transcribed, speak=speak)
        except Exception as exc:
            logger.exception("Voice turn error")
            return "Khong the xu ly loi noi: {0}".format(exc)

    def vision(
        self,
        prompt: str,
        image: Optional[Path | bytes] = None,
        camera_index: int = 0,
        *,
        speak: bool = False,
    ) -> str:
        if self._closed:
            return "Tra loi khong co san vi he thong da dong."
        try:
            vision = self._engines.get("vision", {})
            opencv_eng = vision.get("opencv") if isinstance(vision, dict) else None
            moondream_eng = vision.get("moondream") if isinstance(vision, dict) else None

            if image is None:
                if opencv_eng is not None and getattr(opencv_eng, "health", False):
                    captured = opencv_eng.capture(prompt=prompt, camera_index=camera_index)
                    return self._post_vision_response(captured, speak)
                return "Khong the chap anh: may anh khong kha dung."

            if moondream_eng is not None and getattr(moondream_eng, "health", False):
                if isinstance(image, (Path, bytes)):
                    result = moondream_eng.analyze(image=image, prompt=prompt)
                elif isinstance(image, str):
                    result = moondream_eng.analyze(image=Path(image), prompt=prompt)
                else:
                    result = moondream_eng.analyze(image=image, prompt=prompt)
                return self._post_vision_response(result, speak)

            if opencv_eng is not None and getattr(opencv_eng, "health", False):
                captured = opencv_eng.capture(prompt=prompt, camera_index=camera_index)
                fb = "Phan tich meo (moondream khong kha dung): {0}".format(captured)
                return self._post_vision_response(fb, speak)

            return "Khong the phan tich anh: cac cong phan khong kha dung."
        except Exception as exc:
            logger.exception("Vision error")
            return "Khong the phan tich anh: {0}".format(exc)

    def ingest(self, paths: List[Path | str]) -> str:
        if self._closed:
            return "Khong the nap: he thong da dong."
        try:
            rag = self._engines.get("rag")
            if rag is None:
                return "RAG engine khong kha dung."
            path_objs = []
            for p in paths:
                if isinstance(p, Path):
                    path_objs.append(p)
                elif isinstance(p, str):
                    path_objs.append(Path(p))
                else:
                    return "Loi: duong dan khong hop le: {0}".format(p)
            result = rag.ingest(path_objs)
            return "Da nap thanh cong {0} doan van ban.".format(result)
        except Exception as exc:
            logger.exception("Ingest error")
            return "Khong the nap tai lieu: {0}".format(exc)

    def status(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "closed": self._closed,
            "session_turns": self.session.length,
            "paths": {
                "root": str(self.paths.get("root")),
                "faiss": str(self.paths.get("faiss")),
                "audio": str(self.paths.get("audio")),
                "vision": str(self.paths.get("vision")),
                "logs": str(self.paths.get("logs")),
                "huggingface": str(self.paths.get("huggingface")),
            },
            "components": {},
            "models": {},
        }
        stt = self._engines.get("stt")
        result["components"]["stt"] = getattr(stt, "health", False) if stt else False
        llm = self._engines.get("llm")
        llm_h = False
        if llm is not None:
            try:
                llm_h = llm.health() if callable(getattr(llm, "health", None)) else False
            except Exception:
                llm_h = False
        result["components"]["llm"] = llm_h
        result["models"]["llm"] = getattr(self.config, "llm_model", "unknown")
        tts = self._engines.get("tts")
        result["components"]["tts"] = getattr(tts, "health", False) if tts else False
        rag = self._engines.get("rag")
        rag_h = False
        if rag is not None:
            try:
                rag_h = rag.health() if callable(getattr(rag, "health", None)) else False
            except Exception:
                rag_h = False
        result["components"]["rag"] = rag_h
        result["models"]["rag"] = getattr(self.config, "rag_model", "unknown")
        vision = self._engines.get("vision", {})
        vd = {"opencv": False, "moondream": False}
        if isinstance(vision, dict):
            for k in ("opencv", "moondream"):
                v = vision.get(k)
                if v is not None:
                    try:
                        vd[k] = v.health() if callable(getattr(v, "health", None)) else False
                    except Exception:
                        vd[k] = False
        result["components"]["vision"] = vd
        result["components"]["event_bus"] = self._event_bus is not None
        try:
            import psutil
            mem = psutil.virtual_memory()
            result["resource_guard"] = {
                "ram_used_gb": round(mem.used / (1024 ** 3), 2),
                "ram_total_gb": round(mem.total / (1024 ** 3), 2),
                "ram_percent": mem.percent,
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "max_ram_gb": getattr(self.config, "resource_max_ram_gb", 12),
                "max_workers": getattr(self.config, "resource_max_workers", 2),
            }
        except ImportError:
            result["resource_guard"] = {"psutil": "not installed"}
        except Exception as exc:
            result["resource_guard"] = {"error": str(exc)}
        return result

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for key in ("stt", "llm", "tts", "rag"):
            eng = self._engines.get(key)
            if eng is not None:
                try:
                    eng.close()
                except Exception as exc:
                    logger.warning("Close %s failed: %s", key, exc)
        vision = self._engines.get("vision")
        if isinstance(vision, dict):
            for vk in ("opencv", "moondream"):
                ve = vision.get(vk)
                if ve is not None:
                    try:
                        ve.close()
                    except Exception as exc:
                        logger.warning("Close vision %s: %s", vk, exc)
        if self._event_bus is not None:
            try:
                self._event_bus.stop()
            except Exception as exc:
                logger.warning("Bus stop failed: %s", exc)
            self._event_bus = None
        try:
            self.session.save()
        except Exception as exc:
            logger.warning("Session save on close failed: %s", exc)

    def _dispatch_action(self, result: CommandResult) -> str:
        action = result.action
        if action == "exit":
            return result.response
        if action == "help":
            return result.response
        if action == "status":
            try:
                st = self.status()
                lines = [
                    "Trang thai he thong:",
                    "  STT:" + (" OK" if st["components"].get("stt") else " khong kha dung"),
                    "  LLM:" + (" OK" if st["components"].get("llm") else " khong kha dung"),
                    "  TTS:" + (" OK" if st["components"].get("tts") else " khong kha dung"),
                    "  RAG:" + (" OK" if st["components"].get("rag") else " khong kha dung"),
                    "  Vision:" + (" OK" if any(st["components"].get("vision", {}).values()) else " khong kha dung"),
                    "  Phien hop: {0} lan".format(st["session_turns"]),
                ]
                return "\n".join(lines)
            except Exception as exc:
                return "Khong the lay trang thai: {0}".format(exc)
        if action == "ingest":
            return self.ingest([Path(result.args.get("path", ""))])
        if action == "rag":
            return self.chat(result.args.get("query", ""), speak=False)
        if action == "vision":
            return self.vision(prompt=result.args.get("prompt", ""), speak=False)
        if action == "tts_online":
            setattr(self.config, "tts_prefer_online", True)
            return "TTS online da bat."
        if action == "tts_offline":
            setattr(self.config, "tts_prefer_online", False)
            return "TTS offline da tat."
        if action == "clear":
            self.session.clear()
            return result.response
        return result.response

    def _build_chat_messages(self, text: str) -> List[Dict[str, str]]:
        history = self.session.history_for_prompt()
        context = self._build_rag_context(text)
        messages: List[Dict[str, str]] = []
        if context:
            messages.append({"role": "system", "content": self._build_system_with_context(context)})
        else:
            messages.append({"role": "system", "content": DEFAULT_LLM_SYSTEM_PROMPT})
        messages.extend(history)
        messages.append({"role": "user", "content": text})
        return messages

    def _build_rag_context(self, query: str) -> Optional[str]:
        rag = self._engines.get("rag")
        if rag is None:
            return None
        try:
            healthy = getattr(rag, "health", lambda: False)()
            if not healthy:
                return None
            k = getattr(self.config, "rag_top_k", 4)
            results = rag.query(query, k=k)
            if not results:
                return None
            parts: List[str] = []
            for r in results:
                src = getattr(r, "source", "")
                score = getattr(r, "score", 0.0)
                txt = getattr(r, "text", "")
                part = "[{0} (score={1:.3f})]: {2}".format(src, score, txt[:500])
                parts.append(part)
            ctx = "\n\n".join(parts)
            if len(ctx) > CONTEXT_CHAR_LIMIT:
                ctx = ctx[:CONTEXT_CHAR_LIMIT] + "..."
            return ctx
        except Exception as exc:
            logger.warning("RAG context failed: %s", exc)
            return None

    def _build_system_with_context(self, context: str) -> str:
        return "{0}\n\nNoi dung tham khao tu tai lieu:\n{1}".format(
            DEFAULT_LLM_SYSTEM_PROMPT, context
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        llm = self._engines.get("llm")
        if llm is None:
            return _FALLBACK_RESPONSE
        try:
            from langchain_core.prompts import ChatPromptTemplate
        except ImportError:
            pass
        try:
            response = llm.chat(
                messages=messages,
                system_prompt=DEFAULT_LLM_SYSTEM_PROMPT,
                stream=False,
            )
            if isinstance(response, str):
                if response.strip():
                    return response
                return self._deterministic_fallback(messages)
            if hasattr(response, "__iter__"):
                result = "".join(str(c) for c in response)
                if result.strip():
                    return result
                return self._deterministic_fallback(messages)
            return str(response) if str(response).strip() else self._deterministic_fallback(messages)
        except Exception as exc:
            logger.warning("LLM failed: %s; using fallback", exc)
            return self._deterministic_fallback(messages)

    def _deterministic_fallback(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return _FALLBACK_RESPONSE
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        if last_user:
            return "Dang phan hoi ve cau hoi cua ban: \"{0}\". Xin vui long thu lai sau.".format(last_user[:100])
        return _FALLBACK_RESPONSE

    def _speak(self, text: str) -> None:
        tts = self._engines.get("tts")
        if tts is None:
            return
        try:
            tts.speak(text)
        except Exception as exc:
            logger.warning("TTS speak failed: %s", exc)

    def _post_vision_response(self, result: str, speak: bool) -> str:
        if speak:
            try:
                self._speak(result)
            except Exception:
                pass
        return result

    def __enter__(self) -> "AssistantOrchestrator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return "<AssistantOrchestrator closed={0} turns={1}>".format(
            self._closed, self.session.length
        )

class _STTEngineStub:
    @property
    def health(self) -> bool:
        return False
    def listen(self, duration: float, prompt: str = "") -> str:
        raise RuntimeError("STT khong kha dung")
    def transcribe(self, audio: Any) -> str:
        raise RuntimeError("STT khong kha dung")
    def close(self) -> None:
        return None


class _LLMEngineStub:
    def __init__(self, reason: str = "unavailable") -> None:
        self._closed = False
    def chat(self, messages: List[Dict[str, str]], *, system_prompt: Optional[str] = None, stream: bool = False) -> str:
        if self._closed:
            raise RuntimeError("Engine closed")
        return _FALLBACK_RESPONSE
    def health(self) -> bool:
        return False
    def close(self) -> None:
        self._closed = True
    def generate(self, prompt: str) -> str:
        return _FALLBACK_RESPONSE


class _TTSEngineStub:
    @property
    def health(self) -> bool:
        return False
    @property
    def available_voices(self) -> List[str]:
        return []
    def synthesize(self, text: str) -> bytes:
        return b""
    def play(self, text: str) -> bool:
        return False
    def speak(self, text: str) -> bool:
        return False
    def close(self) -> None:
        return None


class _RAGEngineStub:
    def health(self) -> bool:
        return False
    def ingest(self, paths: List[Path]) -> int:
        return 0
    def query(self, text: str, k: int = 4) -> List[Any]:
        return []
    def close(self) -> None:
        return None


class _VisionStub:
    def __init__(self, name: str = "Vision") -> None:
        self._name = name
    @property
    def health(self) -> bool:
        return False
    def analyze(self, image: Any, prompt: str) -> str:
        raise RuntimeError("{0} khong kha dung".format(self._name))
    def capture(self, prompt: str, camera_index: int = 0) -> str:
        raise RuntimeError("{0} khong kha dung".format(self._name))
    def close(self) -> None:
        return None
