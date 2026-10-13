"""Path restrictions and timestamped backups for agent-generated file writes."""
import logging
import shutil
from datetime import datetime
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_BACKUP_LIMIT = 5
logger = logging.getLogger(__name__)


class UnsafeWritePath(ValueError):
    """Raised when an agent attempts to write outside its permitted workspace."""


def _inside_workspace(path):
    try:
        path.relative_to(WORKSPACE_ROOT)
        return True
    except ValueError:
        return False


def prepare_write_path(raw_path):
    """Resolve a write destination and redirect protected existing Python sources."""
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    candidate = candidate.resolve()

    if not _inside_workspace(candidate):
        raise UnsafeWritePath("Writes must remain inside the project workspace.")

    relative = candidate.relative_to(WORKSPACE_ROOT)
    parts = tuple(part.casefold() for part in relative.parts)
    name = candidate.name.casefold()

    if name in {".env", "requirements.txt"}:
        raise UnsafeWritePath(f"Writing {candidate.name} is blocked.")

    is_main_or_registry = (
        len(parts) == 1 and name in {"main.py", "registry.py"}
    )
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
    if (
        is_main_or_registry
        or (is_core_source and not is_existing_source_alternate)
        or (is_tools_source and not is_existing_source_alternate)
    ):
        if candidate.exists() and candidate.is_file():
            redirected = candidate.with_name(f"{candidate.stem}.new.py")
            logger.warning(
                "[SAFETY] Không ghi đè %s; chuyển nội dung sang %s",
                candidate,
                redirected,
            )
            return redirected
        raise UnsafeWritePath(f"Writing protected source file {relative} is blocked.")

    return candidate


def backup_existing_file(path):
    """Back up an existing destination and retain only its five newest backups."""
    destination = Path(path)
    if not destination.exists():
        return None
    if not destination.is_file():
        raise UnsafeWritePath(f"Write destination is not a regular file: {destination}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = destination.with_name(f"{destination.name}.bak_{timestamp}")
    shutil.copy2(destination, backup)

    backups = sorted(
        destination.parent.glob(f"{destination.name}.bak_*"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for old_backup in backups[_BACKUP_LIMIT:]:
        old_backup.unlink()
    return backup
