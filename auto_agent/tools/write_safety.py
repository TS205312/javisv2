"""Workspace path policy, immutable backups, and run-owned file checkpoints."""
import contextvars
import hashlib
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from threading import RLock

from core.tool_broker import file_resource, resource_lease


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
_ACTIVE_CHECKPOINT = contextvars.ContextVar("active_file_checkpoint", default=None)
_SENSITIVE_NAMES = {
    "token", "tokens", "cookie", "cookies", "credential", "credentials",
    "secret", "secrets", "auth", "key", "apikey", "api_key",
    "private_key", "private_keys",
}
_SENSITIVE_SUFFIXES = {".pem", ".p12", ".pfx", ".key"}


class UnsafeWritePath(ValueError):
    """Raised when an agent path is not within the permitted workspace."""


def _inside_workspace(path):
    try:
        path.relative_to(WORKSPACE_ROOT.resolve())
        return True
    except ValueError:
        return False


def _path_from_input(raw_path):
    if not isinstance(raw_path, (str, os.PathLike)) or not str(raw_path).strip():
        raise UnsafeWritePath("A non-empty workspace path is required.")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    try:
        candidate = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise UnsafeWritePath("Path could not be canonicalized safely.") from exc
    if not _inside_workspace(candidate):
        raise UnsafeWritePath("Paths must remain inside the project workspace.")
    _reject_protected_path(candidate)
    return candidate


def _reject_protected_path(candidate):
    relative = candidate.relative_to(WORKSPACE_ROOT.resolve())
    for part in relative.parts:
        name = part.casefold()
        if name == ".env" or name.startswith((".env.", ".env_", ".env-")):
            raise UnsafeWritePath("Access to environment files is blocked.")
        normalized = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        name_parts = set(normalized.split("_"))
        if name_parts & _SENSITIVE_NAMES or Path(name).suffix in _SENSITIVE_SUFFIXES:
            raise UnsafeWritePath("Access to credential, token, cookie, or key paths is blocked.")


def prepare_workspace_path(raw_path, *, must_exist=False, directory=None):
    """Resolve a path and enforce workspace, protected-name, and type checks."""
    candidate = _path_from_input(raw_path)
    if must_exist and not candidate.exists():
        raise UnsafeWritePath("The requested workspace path does not exist.")
    if candidate.exists():
        if directory is True and not candidate.is_dir():
            raise UnsafeWritePath("The requested workspace path is not a directory.")
        if directory is False and not candidate.is_file():
            raise UnsafeWritePath("The requested workspace path is not a regular file.")
    return candidate


def prepare_read_path(raw_path):
    """Return an existing, regular, non-protected file under the workspace."""
    return prepare_workspace_path(raw_path, must_exist=True, directory=False)


def prepare_write_path(raw_path):
    """Resolve a write destination and retain protected-source redirection."""
    candidate = prepare_workspace_path(raw_path)
    if candidate.exists() and not candidate.is_file():
        raise UnsafeWritePath("Write destination must be a regular file.")

    relative = candidate.relative_to(WORKSPACE_ROOT.resolve())
    parts = tuple(part.casefold() for part in relative.parts)
    name = candidate.name.casefold()
    if name == "requirements.txt":
        raise UnsafeWritePath("Writing dependency configuration is blocked.")

    is_main_or_registry = len(parts) == 1 and name in {"main.py", "registry.py"}
    is_existing_source_alternate = (
        name.endswith(".new.py")
        and len(parts) >= 2
        and parts[0] in {"core", "tools"}
        and candidate.with_name(candidate.name[:-len(".new.py")] + ".py").is_file()
    )
    is_core_source = len(parts) >= 2 and parts[0] == "core" and name.endswith(".py")
    is_tools_source = (
        len(parts) >= 2
        and parts[0] == "tools"
        and name.endswith(".py")
        and not name.startswith("gh_")
    )
    if is_main_or_registry or (
        (is_core_source or is_tools_source) and not is_existing_source_alternate
    ):
        if candidate.exists():
            redirected = candidate.with_name(f"{candidate.stem}.new.py")
            _reject_protected_path(redirected)
            logger.warning("[SAFETY] Protected source write redirected to %s", redirected.name)
            return redirected
        raise UnsafeWritePath("Writing protected source files is blocked.")
    return candidate


def backup_existing_file(path):
    """Create a timestamped backup without deleting older recovery copies."""
    destination = prepare_workspace_path(path, directory=False)
    if not destination.exists():
        return None
    if not destination.is_file():
        raise UnsafeWritePath("Write destination is not a regular file.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = destination.with_name(f"{destination.name}.bak_{timestamp}")
    shutil.copy2(destination, backup)
    return backup


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            os.chmod(temporary_path, path.stat().st_mode)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


class FileCheckpoint:
    """Track the first state and final run-owned contents of each written file."""

    def __init__(self):
        self._lock = RLock()
        self._snapshots = {}
        self._closed = False

    def before_write(self, path):
        canonical = prepare_write_path(path)
        key = str(canonical)
        with self._lock:
            if self._closed:
                raise UnsafeWritePath("This run's file checkpoint is already closed.")
            if key not in self._snapshots:
                existed = canonical.is_file()
                backup = backup_existing_file(canonical) if existed else None
                self._snapshots[key] = {
                    "path": canonical,
                    "existed": existed,
                    "backup": backup,
                    "last_digest": None,
                }

    def record_write(self, path, data):
        canonical = prepare_write_path(path)
        with self._lock:
            entry = self._snapshots.get(str(canonical))
            if entry is None:
                raise UnsafeWritePath("File write is not owned by this run checkpoint.")
            entry["last_digest"] = _digest(data)

    def commit(self):
        with self._lock:
            self._closed = True
            return {"status": "committed", "files": len(self._snapshots)}

    def rollback(self):
        with self._lock:
            self._closed = True
            entries = list(self._snapshots.values())

        restored = []
        skipped = []
        for entry in reversed(entries):
            path = entry["path"]
            try:
                with resource_lease(file_resource(path)):
                    if entry["last_digest"] is None:
                        skipped.append("unchanged")
                        continue
                    current = path.read_bytes() if path.is_file() else None
                    if current is None or _digest(current) != entry["last_digest"]:
                        skipped.append("changed_after_run_write")
                        continue
                    if entry["existed"]:
                        backup = entry["backup"]
                        if backup is None or not backup.is_file():
                            skipped.append("backup_missing")
                            continue
                        _atomic_write_bytes(path, backup.read_bytes())
                    else:
                        path.unlink()
                    restored.append("restored")
            except Exception:
                skipped.append("rollback_error")
        return {
            "status": "rolled_back",
            "files_tracked": len(entries),
            "files_restored": len(restored),
            "files_skipped": len(skipped),
            "skip_reasons": sorted(set(skipped)),
        }


def set_active_checkpoint(checkpoint):
    return _ACTIVE_CHECKPOINT.set(checkpoint)


def reset_active_checkpoint(token):
    _ACTIVE_CHECKPOINT.reset(token)


def active_checkpoint():
    return _ACTIVE_CHECKPOINT.get()


def write_bytes_safely(path, data):
    """Atomically write bytes under a path lease with backup/checkpoint ownership."""
    destination = prepare_write_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with resource_lease(file_resource(destination)):
        checkpoint = active_checkpoint()
        if checkpoint is not None:
            checkpoint.before_write(destination)
        else:
            backup_existing_file(destination)
        _atomic_write_bytes(destination, data)
        if checkpoint is not None:
            checkpoint.record_write(destination, data)
    return destination
