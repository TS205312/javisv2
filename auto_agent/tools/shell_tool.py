"""Bounded terminal facade using an argv allowlist and no shell interpreter."""
import subprocess

from core.safety import MAX_OUTPUT_CHARS, CommandPolicyError, prepare_command
from core.sensitive_data import redact_text
from tools.write_safety import UnsafeWritePath, prepare_workspace_path


def _truncate(value):
    value = redact_text(str(value or ""))
    if len(value) > MAX_OUTPUT_CHARS:
        return value[:MAX_OUTPUT_CHARS] + "\n[OUTPUT TRUNCATED]"
    return value


def _run_internal(spec):
    if spec["mode"] == "echo":
        return f"STDOUT:\n{_truncate(' '.join(spec['args']))}\nSTDERR:\n"
    if spec["mode"] == "pwd":
        return f"STDOUT:\n{spec['cwd']}\nSTDERR:\n"
    if spec["mode"] == "list":
        entries = []
        for item in spec["cwd"].iterdir():
            try:
                prepare_workspace_path(item, must_exist=True)
            except UnsafeWritePath:
                continue
            entries.append(item.name)
        entries.sort()
        return f"STDOUT:\n{_truncate(chr(10).join(entries))}\nSTDERR:\n"
    return None


def run(params, yolo=False):
    try:
        spec = prepare_command(params)
        internal = _run_internal(spec)
        if internal is not None:
            return internal
        result = subprocess.run(
            spec["argv"],
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=spec["timeout"],
            cwd=str(spec["cwd"]),
            check=False,
        )
        stdout = _truncate(result.stdout)
        stderr = _truncate(result.stderr)
        prefix = "" if result.returncode == 0 else f"ERROR [COMMAND exit={result.returncode}]\n"
        return f"{prefix}STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    except CommandPolicyError as exc:
        return f"ERROR [SAFETY]: {exc}"
    except subprocess.TimeoutExpired:
        return "ERROR [TIMEOUT]: Approved command exceeded its time limit."
    except OSError as exc:
        return f"ERROR [COMMAND]: Execution failed ({type(exc).__name__})."
