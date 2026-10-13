"""Code-writing tool with runtime verification for generated Python scripts."""
import subprocess
import sys
from pathlib import Path

from core.llm_helper import ask_llm
from tools.file_tool import write
from tools.write_safety import UnsafeWritePath, prepare_write_path


_MAX_EXECUTIONS = 3
_EXECUTION_TIMEOUT = 60


def _extract_code(response):
    if not isinstance(response, str):
        return ""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def write_code(params):
    path = (
        params.get("path")
        or params.get("file_path")
        or params.get("filepath")
    )
    code = params.get("code")
    if code is None:
        code = params.get("content")
    if not isinstance(path, str) or not path.strip():
        return "FAIL: thiếu đường dẫn file."
    if not isinstance(code, str):
        return "FAIL: nội dung code phải là chuỗi."

    try:
        path = prepare_write_path(path)
    except UnsafeWritePath as exc:
        return f"FAIL: [SAFETY] {exc}"
    code = code.replace("TODO", "IMPLEMENTED")
    result = write({"path": str(path), "content": code})
    if result != "Ghi file thành công":
        return f"FAIL: không ghi được file: {result}"

    if path.suffix.casefold() != ".py":
        return result

    resolved_path = path.resolve()
    last_stderr = ""
    for attempt in range(1, _MAX_EXECUTIONS + 1):
        try:
            execution = subprocess.run(
                [sys.executable, str(resolved_path)],
                cwd=str(resolved_path.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_EXECUTION_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr or "Script timed out."
            )
            return (
                f"FAIL: script timed out after {_EXECUTION_TIMEOUT}s "
                f"(attempt {attempt}/{_MAX_EXECUTIONS}).\nstderr:\n{last_stderr}"
            )
        except OSError as exc:
            return f"FAIL: không chạy được script: {exc}"

        if execution.returncode == 0:
            return (
                f"PASS: Python script chạy thành công "
                f"(attempt {attempt}/{_MAX_EXECUTIONS}).\n"
                f"{execution.stdout.strip()}"
            )

        last_stderr = execution.stderr.strip() or execution.stdout.strip()
        if attempt == _MAX_EXECUTIONS:
            break

        repair_prompt = f"""Sửa Python code dưới đây để chạy thành công.
Lỗi khi chạy:
{last_stderr[:4000]}

Trả về toàn bộ file Python đã sửa, không giải thích:
```python
{code}
```"""
        repaired = _extract_code(ask_llm(repair_prompt))
        if not repaired or repaired.startswith("Error:"):
            return (
                f"FAIL: không nhận được code sửa hợp lệ sau attempt "
                f"{attempt}/{_MAX_EXECUTIONS}.\nstderr:\n{last_stderr}"
            )

        code = repaired
        result = write({"path": str(path), "content": code})
        if result != "Ghi file thành công":
            return f"FAIL: không ghi được code sửa: {result}\nstderr:\n{last_stderr}"

    return (
        f"FAIL: script vẫn lỗi sau {_MAX_EXECUTIONS} lần chạy.\n"
        f"stderr:\n{last_stderr}"
    )


def fix_code(params):
    return write_code(params)
