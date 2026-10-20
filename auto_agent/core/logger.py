"""
Logger — ghi log JSON theo session, hỗ trợ agent_id riêng biệt cho swarm.
"""
import json
import time
import threading
from pathlib import Path
from rich.console import Console
from core.sensitive_data import redact_data, redact_text

console = Console()
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


class SessionLogger:
    """
    Ghi log sự kiện theo session.
    agent_id: nếu cung cấp, tên file log sẽ chứa suffix _agent<id>.
    """

    def __init__(self, agent_id=None, session_id=None):
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = session_id or int(time.time())
        if agent_id == "aggregator":
            suffix = "_aggregator"
        elif agent_id is not None:
            suffix = f"_agent{agent_id}"
        else:
            suffix = ""
        self.filename = str(_LOG_DIR / f"session_{ts}{suffix}.json")
        self.agent_id = agent_id
        self._lock = threading.Lock()
        self.data = {
            "session_start": time.time(),
            "agent_id": agent_id,
            "session_id": ts,
            "events": [],
        }

    def log(self, event_type, details):
        safe_event = redact_text(str(event_type))
        safe_details = redact_data(details)
        with self._lock:
            self.data["events"].append({
                "type": safe_event,
                "timestamp": time.time(),
                "details": safe_details,
            })
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)

    def print(self, prefix, message, style=None):
        tag = f"A{self.agent_id}" if self.agent_id is not None else prefix
        console.print(
            f"[{tag}] [{prefix}] {redact_text(str(message))}",
            style=style,
        )

    def print_agent_timing(self, event, timestamp, elapsed=None):
        if self.agent_id is None:
            return
        clock = time.strftime("%H:%M:%S", time.localtime(timestamp))
        message = f"[A{self.agent_id}] {event} {clock}"
        if elapsed is not None:
            message += f" ({elapsed:.1f}s)"
        console.print(message)


# Instance mặc định dùng cho chế độ đơn agent (backward compat)
session_logger = SessionLogger()
