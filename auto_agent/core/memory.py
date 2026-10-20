"""
Memory — short-term context plus shared persistent ChromaDB experience memory.
"""
import json
import logging
import uuid
from pathlib import Path

try:
    import chromadb
except ImportError:
    chromadb = None

from core.sensitive_data import redact_text
from tools.write_safety import UnsafeWritePath, prepare_workspace_path


_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_MEMORY_ROOT = Path("D:/JavisAgentMemory").resolve(strict=False)
_DEFAULT_STORAGE_PATH = (_MEMORY_ROOT / "chroma_db").resolve(strict=False)
_logger = logging.getLogger(__name__)


def _prepare_storage_path(raw_path):
    candidate = Path(raw_path).expanduser().resolve(strict=False)
    workspace_root = _WORKSPACE_ROOT.resolve()
    try:
        candidate.relative_to(workspace_root)
        return prepare_workspace_path(candidate)
    except ValueError:
        pass

    try:
        candidate.relative_to(_MEMORY_ROOT)
    except ValueError as exc:
        raise UnsafeWritePath(
            "Memory storage must stay in the workspace or D:/JavisAgentMemory."
        ) from exc
    if candidate != _DEFAULT_STORAGE_PATH:
        raise UnsafeWritePath(
            "Only the designated D:/JavisAgentMemory/chroma_db store is allowed."
        )
    return candidate


_CLIENT_CACHE = {}


class Memory:
    """
    namespace: tên collection ChromaDB riêng cho mỗi agent.
    Nếu namespace=None → dùng collection kinh nghiệm dùng chung.
    """

    def __init__(self, namespace=None, storage_path=None):
        self.short_term = []
        self.namespace = namespace or "javis_experience"
        self.storage_path = _prepare_storage_path(
            storage_path or _DEFAULT_STORAGE_PATH
        )
        self.client = None
        self.collection = None

    def _ensure_collection(self):
        if self.collection is not None:
            return self.collection
        if chromadb is None:
            raise RuntimeError("ChromaDB is not installed; persistent memory is unavailable.")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        path_str = str(self.storage_path)
        if path_str not in _CLIENT_CACHE:
            _CLIENT_CACHE[path_str] = chromadb.PersistentClient(path=path_str)
        self.client = _CLIENT_CACHE[path_str]
        self.collection = self.client.get_or_create_collection(name=self.namespace)
        return self.collection

    def add_short_term(self, role, content):
        self.short_term.append({"role": role, "content": content})

    def get_short_term_context(self, max_chars=4000):
        if not self.short_term:
            return "[]"
        raw = json.dumps(self.short_term, ensure_ascii=False)
        if len(raw) <= max_chars:
            return raw
        trimmed = self.short_term[:1] + self.short_term[-4:]
        return json.dumps(trimmed, ensure_ascii=False)

    def save_long_term(self, task_id, desc, result, success=None, tool=None, agent_id=None):
        safe_desc = redact_text(str(desc))
        safe_result = redact_text(str(result))
        doc_id = f"task_{uuid.uuid4().hex}"
        metadata = {"task_id": str(task_id)}
        if success is not None:
            metadata["success"] = bool(success)
        if tool:
            metadata["tool"] = redact_text(str(tool))
        if agent_id is not None:
            metadata["agent_id"] = int(agent_id)
        try:
            self._ensure_collection().add(
                documents=[f"Task: {safe_desc} | Experience: {safe_result}"],
                metadatas=[metadata],
                ids=[doc_id],
            )
            return True
        except Exception as exc:
            _logger.warning(
                "Persistent experience save failed (%s).", type(exc).__name__
            )
            return False

    def search_similar(self, query, n=3, success_only=True, max_distance=None):
        safe_query = redact_text(str(query))
        try:
            options = {"query_texts": [safe_query], "n_results": n}
            if success_only:
                options["where"] = {"success": True}
            results = self._ensure_collection().query(**options)
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                if max_distance is not None and "distances" in results and results["distances"]:
                    dists = results["distances"][0]
                    return [doc for doc, dist in zip(docs, dists) if dist <= max_distance]
                return docs
        except Exception as exc:
            _logger.warning(
                "Persistent experience lookup failed (%s).", type(exc).__name__
            )
        return []

    def desktop_operator_scores(self, agent_ids):
        """Return outcome counts based only on critic-verified desktop tasks."""
        scores = {
            int(agent_id): {"verified_successes": 0, "attempts": 0}
            for agent_id in agent_ids
        }
        try:
            rows = self._ensure_collection().get(
                where={"tool": "computer_tool"},
                include=["metadatas"],
                limit=500,
            )
            for metadata in (rows or {}).get("metadatas", []):
                if not isinstance(metadata, dict):
                    continue
                try:
                    agent_id = int(metadata.get("agent_id"))
                except (TypeError, ValueError):
                    continue
                if agent_id not in scores:
                    continue
                scores[agent_id]["attempts"] += 1
                if metadata.get("success") is True:
                    scores[agent_id]["verified_successes"] += 1
        except Exception as exc:
            _logger.warning("Desktop operator history unavailable (%s).", type(exc).__name__)
        return scores


# Instance mặc định dùng cho chế độ đơn agent (backward compat)
memory = Memory()
