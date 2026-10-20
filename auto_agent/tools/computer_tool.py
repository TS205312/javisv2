"""Guarded desktop input with an exclusive lock for each action sequence."""
import json
import re
import threading
import uuid
from pathlib import Path

from core.recovery import action_fingerprint, desktop_action_summary
from core.sensitive_data import contains_sensitive_data, redact_text
from core.tool_broker import resource_lease
from tools.write_safety import prepare_write_path, write_bytes_safely


_DESKTOP_LOCK = threading.RLock()
_MAX_TEXT_LENGTH = 500
_ALLOWED_KEYS = {
    "backspace", "down", "enter", "esc", "left", "right", "space",
    "tab", "up",
}
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _desktop_modules():
    import mss
    import pyautogui

    pyautogui.FAILSAFE = True
    return pyautogui, mss


def _screenshot_path(raw_path):
    if raw_path is None:
        raw_path = "output/desktop_runs/desktop.png"
    path = prepare_write_path(raw_path)
    if path.suffix.casefold() != ".png":
        raise ValueError("Desktop screenshots must use a .png file path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _capture(path, mss_module):
    path = Path(path)
    temporary_path = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        with mss_module.mss() as capture:
            capture.shot(output=str(temporary_path))
        write_bytes_safely(path, temporary_path.read_bytes())
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return str(path)


def _validate_actions(actions, screen_size):
    if not isinstance(actions, list) or not actions:
        raise ValueError("computer_tool.perform requires a non-empty action list.")

    width, height = screen_size
    validated = []
    for index, item in enumerate(actions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Action {index} must be an object.")
        action = item.get("action")
        if action == "click":
            x, y = item.get("x"), item.get("y")
            if (
                not isinstance(x, (int, float))
                or isinstance(x, bool)
                or not isinstance(y, (int, float))
                or isinstance(y, bool)
                or not 0 <= x < width
                or not 0 <= y < height
            ):
                raise ValueError(f"Action {index} has out-of-bounds coordinates.")
            validated.append({"action": "click", "x": x, "y": y})
        elif action == "type_text":
            text = item.get("text")
            if not isinstance(text, str) or len(text) > _MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Action {index} text must be a string of at most {_MAX_TEXT_LENGTH} characters."
                )
            if contains_sensitive_data(text):
                raise ValueError("Credential-like desktop text is blocked.")
            validated.append({"action": "type_text", "text": text})
        elif action == "press":
            key = item.get("key")
            if not isinstance(key, str) or key.casefold() not in _ALLOWED_KEYS:
                raise ValueError(
                    f"Action {index} key is not allowed; use one of: "
                    f"{', '.join(sorted(_ALLOWED_KEYS))}."
                )
            validated.append({"action": "press", "key": key.casefold()})
        else:
            raise ValueError(
                f"Action {index} is unsupported. Use click, type_text, or press."
            )
    return validated


def _run_action(action, pyautogui):
    if action["action"] == "click":
        pyautogui.click(action["x"], action["y"])
    elif action["action"] == "type_text":
        pyautogui.write(action["text"], interval=0.03)
    else:
        pyautogui.press(action["key"])


def click(params):
    x = params.get("x")
    y = params.get("y")
    pyautogui, _ = _desktop_modules()
    width, height = pyautogui.size()
    if (
        not isinstance(x, (int, float))
        or isinstance(x, bool)
        or not isinstance(y, (int, float))
        or isinstance(y, bool)
        or not 0 <= x < width
        or not 0 <= y < height
    ):
        raise ValueError("Desktop click coordinates are outside the screen bounds.")
    with resource_lease("desktop:physical"), _DESKTOP_LOCK:
        pyautogui.click(x, y)
    return f"Clicked desktop coordinates ({x}, {y})."


def type_text(params):
    text = params.get("text")
    if not isinstance(text, str) or len(text) > _MAX_TEXT_LENGTH:
        raise ValueError(
            f"Desktop text must be a string of at most {_MAX_TEXT_LENGTH} characters."
        )
    pyautogui, _ = _desktop_modules()
    if contains_sensitive_data(text):
        raise ValueError("Credential-like desktop text is blocked.")
    with resource_lease("desktop:physical"), _DESKTOP_LOCK:
        pyautogui.write(text, interval=0.03)
    return f"Typed {len(text)} characters on the desktop."


def press(params):
    key = params.get("key")
    if not isinstance(key, str) or key.casefold() not in _ALLOWED_KEYS:
        raise ValueError(f"Desktop key is not allowed: {key!r}.")
    pyautogui, _ = _desktop_modules()
    with resource_lease("desktop:physical"), _DESKTOP_LOCK:
        pyautogui.press(key.casefold())
    return f"Pressed desktop key {key.casefold()}."


def screenshot(params):
    pyautogui, mss_module = _desktop_modules()
    path = _screenshot_path(params.get("path"))
    with resource_lease("desktop:physical"), _DESKTOP_LOCK:
        saved_path = _capture(path, mss_module)
    return f"Desktop screenshot saved to {saved_path}."


def observe(params):
    """Capture the current desktop and return bounded, redacted OCR evidence."""
    if not isinstance(params, dict):
        raise ValueError("computer_tool.observe requires an object of parameters.")

    pyautogui, mss_module = _desktop_modules()
    observation_id = uuid.uuid4().hex[:12]
    requested_path = params.get("path")
    if requested_path is None:
        requested_path = (
            _WORKSPACE_ROOT
            / "output"
            / "desktop_runs"
            / f"{observation_id}_observe.png"
        )
    path = _screenshot_path(requested_path)
    with resource_lease("desktop:physical"), _DESKTOP_LOCK:
        saved_path = _capture(path, mss_module)
        width, height = pyautogui.size()

    words, ocr_error = _read_screen_words(saved_path)
    result = {
        "status": "observed",
        "screenshot": saved_path,
        "screen": {"width": width, "height": height},
        "ocr_words": words,
        "ocr_error": ocr_error,
        "limitations": (
            "OCR is approximate. Inspect the current state after every action; "
            "do not infer controls or bypass login, CAPTCHA, or access protections."
        ),
    }
    return json.dumps(result, ensure_ascii=False)


def perform(params):
    """Run one small, serialized desktop transaction and capture its result."""
    if not isinstance(params, dict):
        raise ValueError("computer_tool.perform requires an object of parameters.")

    pyautogui, mss_module = _desktop_modules()
    actions = _validate_actions(params.get("actions"), pyautogui.size())
    expected_text = params.get("verify_text")
    if expected_text is not None and (
        not isinstance(expected_text, str) or contains_sensitive_data(expected_text)
    ):
        raise ValueError("Credential-like verification text is blocked.")
    fingerprint = action_fingerprint(actions)
    if params.get("reject_fingerprint") == fingerprint:
        raise ValueError("The proposed desktop action batch duplicates the previous attempt.")
    transaction_id = uuid.uuid4().hex[:12]
    agent_id = params.get("agent_id", "single")
    run_dir = _WORKSPACE_ROOT / "output" / "desktop_runs"
    before_path = _screenshot_path(run_dir / f"{transaction_id}_before.png")
    after_path = _screenshot_path(run_dir / f"{transaction_id}_after.png")
    completed = 0
    error = None

    with resource_lease("desktop:physical"), _DESKTOP_LOCK:
        _capture(before_path, mss_module)
        try:
            for action in actions:
                _run_action(action, pyautogui)
                completed += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        _capture(after_path, mss_module)

    result = {
        "status": "executed_unverified" if error is None else "partial_failure",
        "transaction_id": transaction_id,
        "agent_id": agent_id,
        "action_fingerprint": fingerprint,
        "action_summary": desktop_action_summary(actions),
        "actions_requested": len(actions),
        "actions_completed": completed,
        "error": error,
        "before_screenshot": str(before_path),
        "after_screenshot": str(after_path),
        "verification": "Visual state was captured but not semantically verified.",
        "replay_policy": "Do not replay; inspect the captured state and choose a new action.",
    }
    if expected_text is not None:
        result["verification"] = _verify_text(after_path, expected_text)
        if (
            error is None
            and completed == len(actions)
            and result["verification"].get("status") == "matched"
        ):
            result["status"] = "verified"
        elif error is None and result["verification"].get("status") == "not_found":
            result["status"] = "verification_failed"
    else:
        result["observed_text"] = _read_screen_text(after_path)
    return json.dumps(result, ensure_ascii=False)


def _read_screen_text(image_path):
    try:
        import pytesseract
        from PIL import Image

        return redact_text(pytesseract.image_to_string(Image.open(image_path))[:1200])
    except Exception as exc:
        return f"OCR unavailable: {type(exc).__name__}: {str(exc)[:160]}"


def _read_screen_words(image_path, max_words=100):
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(image_path)
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
        )
        raw_words = [str(value).strip() for value in data.get("text", [])]
        line_text = {}
        for index, raw_text in enumerate(raw_words):
            line_key = (
                data.get("block_num", [0] * len(raw_words))[index],
                data.get("par_num", [0] * len(raw_words))[index],
                data.get("line_num", [0] * len(raw_words))[index],
            )
            line_text.setdefault(line_key, []).append(raw_text)
        sensitive_lines = set()
        for line_key, values in line_text.items():
            if re.search(
                r"(?i)\b(password|passwd|passcode|secret|api.?key|token|cookie|authorization)\b",
                " ".join(values),
            ):
                sensitive_lines.add(line_key)
        words = []
        for index, raw_text in enumerate(raw_words):
            line_key = (
                data.get("block_num", [0] * len(raw_words))[index],
                data.get("par_num", [0] * len(raw_words))[index],
                data.get("line_num", [0] * len(raw_words))[index],
            )
            text = redact_text(str(raw_text).strip())
            if not text:
                continue
            if line_key in sensitive_lines:
                text = "[REDACTED]"
            try:
                confidence = float(data.get("conf", ["-1"])[index])
                left = int(data["left"][index])
                top = int(data["top"][index])
                width = int(data["width"][index])
                height = int(data["height"][index])
            except (IndexError, TypeError, ValueError):
                continue
            if confidence < 30 or width <= 0 or height <= 0:
                continue
            words.append({
                "text": text[:120],
                "confidence": round(confidence, 1),
                "bounds": {"x": left, "y": top, "width": width, "height": height},
            })
            if len(words) >= max_words:
                break
        return words, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {str(exc)[:160]}"


def _verify_text(image_path, expected_text):
    if not isinstance(expected_text, str) or not expected_text.strip():
        return {"status": "unavailable", "reason": "verify_text must be non-empty."}
    try:
        observed_text = _read_screen_text(image_path)
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"OCR unavailable: {type(exc).__name__}: {str(exc)[:160]}",
        }
    if observed_text.startswith("OCR unavailable:"):
        return {"status": "unavailable", "reason": observed_text}
    matched = expected_text.casefold() in observed_text.casefold()
    return {
        "status": "matched" if matched else "not_found",
        "expected_text": expected_text[:160],
        "observed_text": observed_text[:1200],
    }
