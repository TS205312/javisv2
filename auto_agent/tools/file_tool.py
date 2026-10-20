from tools.write_safety import (
    UnsafeWritePath,
    prepare_read_path,
    prepare_write_path,
    write_bytes_safely,
)
from core.sensitive_data import contains_sensitive_data, redact_text
from core.tool_broker import file_resource, resource_lease


_MAX_FILE_BYTES = 2 * 1024 * 1024


def read(params):
    if not isinstance(params, dict):
        return "ERROR [SAFETY]: A path object is required."
    try:
        path = prepare_read_path(params.get("path"))
        with resource_lease(file_resource(path)):
            if path.stat().st_size > _MAX_FILE_BYTES:
                return "ERROR [SAFETY]: File exceeds the 2 MiB read limit."
            with path.open("r", encoding="utf-8") as file:
                return redact_text(file.read(_MAX_FILE_BYTES + 1))
    except UnsafeWritePath as exc:
        return f"ERROR [SAFETY]: {exc}"
    except (OSError, UnicodeError, TimeoutError) as exc:
        return f"ERROR [FILE]: Read failed ({type(exc).__name__})."

def write(params):
    if not isinstance(params, dict):
        return "ERROR [SAFETY]: A path and string content are required."
    content = params.get("content", "")
    if not isinstance(content, str):
        return "ERROR [SAFETY]: File content must be text."
    if contains_sensitive_data(content):
        return "ERROR [SAFETY]: Sensitive credential-like content cannot be written."
    if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
        return "ERROR [SAFETY]: File content exceeds the 2 MiB write limit."
    try:
        path = prepare_write_path(params.get("path"))
        write_bytes_safely(path, content.encode("utf-8"))
        return "Ghi file thành công"
    except UnsafeWritePath as exc:
        return f"ERROR [SAFETY]: {exc}"
    except (OSError, TimeoutError) as exc:
        return f"ERROR [FILE]: Write failed ({type(exc).__name__})."
