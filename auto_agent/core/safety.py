"""Structured allowlist for agent-initiated terminal operations."""
import os
import shlex
import shutil
import sys

from tools.write_safety import (
    UnsafeWritePath,
    WORKSPACE_ROOT,
    prepare_workspace_path,
)


MAX_COMMAND_ARGS = 12
MAX_COMMAND_LENGTH = 2048
MAX_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 4000


class CommandPolicyError(ValueError):
    pass


def _workspace_path(value, *, must_exist=False, directory=None):
    try:
        return prepare_workspace_path(
            value,
            must_exist=must_exist,
            directory=directory,
        )
    except UnsafeWritePath as exc:
        raise CommandPolicyError(str(exc)) from exc


def _argv_from_params(params):
    if not isinstance(params, dict):
        raise CommandPolicyError("Command parameters must be an object.")
    if "executable" in params:
        executable = params.get("executable")
        args = params.get("args", [])
        if not isinstance(executable, str) or not isinstance(args, list):
            raise CommandPolicyError("Use executable plus a string-array args field.")
        if any(not isinstance(argument, str) for argument in args):
            raise CommandPolicyError("Command args must contain only strings.")
        argv = [executable, *args]
    else:
        command = params.get("cmd") or params.get("command")
        if not isinstance(command, str) or not command.strip():
            raise CommandPolicyError("A non-empty command is required.")
        if len(command) > MAX_COMMAND_LENGTH:
            raise CommandPolicyError("Command exceeds the input length limit.")
        try:
            argv = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            raise CommandPolicyError("Command quoting is invalid.") from exc
    if not argv or len(argv) - 1 > MAX_COMMAND_ARGS:
        raise CommandPolicyError("Command has an invalid executable or too many arguments.")
    if any("\x00" in value or len(value) > MAX_COMMAND_LENGTH for value in argv):
        raise CommandPolicyError("Command args contain invalid or oversized values.")
    return argv


def _normalized_executable(value):
    if not isinstance(value, str) or not value.strip():
        raise CommandPolicyError("Command executable is required.")
    if "/" in value or "\\" in value:
        raise CommandPolicyError("Executable paths are not accepted; use an allowlisted command name.")
    name = value.casefold()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def prepare_command(params):
    """Return a validated command spec. YOLO flags cannot alter this policy."""
    argv = _argv_from_params(params)
    executable = _normalized_executable(argv[0])
    args = argv[1:]
    if executable not in {"echo", "pwd", "dir", "ls", "git", "python", "python3", "py"}:
        raise CommandPolicyError("Executable is not in the terminal allowlist.")

    cwd_value = params.get("cwd", str(WORKSPACE_ROOT))
    cwd = _workspace_path(cwd_value, must_exist=True, directory=True)
    timeout_value = params.get("timeout", 10)
    if isinstance(timeout_value, bool) or not isinstance(timeout_value, (int, float)):
        raise CommandPolicyError("Timeout must be a number of seconds.")
    timeout = max(1, min(int(timeout_value), MAX_TIMEOUT_SECONDS))

    if executable == "echo":
        return {"mode": "echo", "args": args, "cwd": cwd, "timeout": timeout}
    if executable == "pwd":
        if args:
            raise CommandPolicyError("pwd accepts no arguments.")
        return {"mode": "pwd", "args": [], "cwd": cwd, "timeout": timeout}
    if executable in {"dir", "ls"}:
        if len(args) > 1 or any(arg.startswith("-") for arg in args):
            raise CommandPolicyError("Directory listing accepts at most one workspace path.")
        directory = _workspace_path(
            args[0] if args else cwd,
            must_exist=True,
            directory=True,
        )
        return {"mode": "list", "args": [], "cwd": directory, "timeout": timeout}
    if executable in {"python", "python3", "py"}:
        if args != ["--version"]:
            raise CommandPolicyError("Only the non-executing Python --version command is allowed.")
        return {
            "mode": "process",
            "argv": [sys.executable, "--version"],
            "cwd": cwd,
            "timeout": timeout,
        }
    if executable == "git" and args in (["status", "--short"], ["status", "--porcelain"]):
        git_path = shutil.which("git")
        if not git_path:
            raise CommandPolicyError("Git is unavailable on PATH.")
        return {
            "mode": "process",
            "argv": [git_path, *args],
            "cwd": cwd,
            "timeout": timeout,
        }
    raise CommandPolicyError("Command and argument combination is not allowlisted.")


def is_safe(command, yolo=False):
    """Compatibility check that deliberately ignores the legacy YOLO bypass."""
    try:
        params = command if isinstance(command, dict) else {"command": command}
        prepare_command(params)
        return True, "Allowed by structured terminal policy."
    except CommandPolicyError as exc:
        return False, str(exc)
