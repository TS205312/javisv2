"""Retry policy for bounded observe-fix-verify task recovery."""
import hashlib
import json
import re


MAX_TASK_ATTEMPTS = 3

_RETRYABLE_ACTIONS = {
    ("search_tool", "search"),
    ("browser_tool", "goto"),
    ("browser_tool", "scrape"),
    ("browser_tool", "screenshot"),
    ("file_tool", "read"),
    ("file_tool", "search"),
    ("file_tool", "write"),
    ("vision_tool", "analyze"),
    ("vision_tool", "ocr"),
    ("presentation_draft", "generate"),
    ("coder_tool", "write_code"),
    ("coder_tool", "fix_code"),
}


def retry_decision(tool_name, action, attempts, max_attempts=MAX_TASK_ATTEMPTS):
    """Return whether another attempt is safe and why.

    Actions with external or non-idempotent effects (desktop input, browser
    clicks, shell commands, publishing, and media rendering) are not replayed
    automatically. The caller can still report the observation and stop.
    """
    if attempts >= max_attempts:
        return False, f"Đã đạt giới hạn {max_attempts} lần thử."
    if not tool_name or not action:
        return False, "Không xác định được tool/action; không lặp lại mù."
    if (tool_name, action) not in _RETRYABLE_ACTIONS:
        return (
            False,
            f"{tool_name}.{action} có thể gây tác dụng phụ; cần quan sát/lập bước mới, không phát lại tự động.",
        )
    return True, "Thao tác đọc hoặc ghi có thể lặp an toàn; thử lại với sửa đổi có mục tiêu."


def action_fingerprint(actions):
    payload = json.dumps(actions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def desktop_action_summary(actions):
    summary = []
    for action in actions if isinstance(actions, list) else []:
        if not isinstance(action, dict):
            continue
        row = {key: value for key, value in action.items() if key != "text"}
        if isinstance(action.get("text"), str):
            row["text_characters"] = len(action["text"])
        summary.append(row)
    return summary


def sanitize_desktop_output(output):
    try:
        result = json.loads(str(output))
    except (TypeError, json.JSONDecodeError):
        return "Desktop action output omitted because it is not structured JSON."
    if not isinstance(result, dict):
        return "Desktop action output omitted because it is not an object."
    result.pop("observed_text", None)
    result.pop("ocr_words", None)
    verification = result.get("verification")
    if isinstance(verification, dict):
        verification.pop("observed_text", None)
        verification.pop("expected_text", None)
    return json.dumps(result, ensure_ascii=False)


def _redact_screen_text(value):
    text = str(value or "")[:1200]
    text = re.sub(
        r"(?i)\b(password|passcode|token|api[ _-]?key|secret)\b\s*[:=]\s*\S+",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(r"\bAIza[0-9A-Za-z_-]{16,}\b", "[redacted-key]", text)
    text = re.sub(r"\bAQ\.[0-9A-Za-z_-]{16,}\b", "[redacted-key]", text)
    text = re.sub(r"\bsk-[0-9A-Za-z_-]{16,}\b", "[redacted-key]", text)
    text = re.sub(r"\b[A-Za-z0-9_-]{40,}\b", "[redacted-token]", text)
    return text


def propose_desktop_recovery(goal, output, failure_reason, llm_client):
    """Ask for one new GUI action only when OCR gives a usable observation."""
    try:
        previous = json.loads(str(output))
    except (TypeError, json.JSONDecodeError):
        return None, "Desktop result is not parseable JSON."

    verification = previous.get("verification", {})
    if (
        previous.get("status") != "verification_failed"
        or not isinstance(verification, dict)
        or verification.get("status") != "not_found"
        or not isinstance(verification.get("observed_text"), str)
        or not verification.get("observed_text", "").strip()
    ):
        return None, "A reliable OCR observation is unavailable; stop rather than guess."

    safe_observation = _redact_screen_text(verification.get("observed_text", ""))
    prompt = f"""
Mục tiêu desktop ban đầu: {goal}

Chuỗi thao tác trước đó đã chạy xong nhưng kiểm chứng thất bại. Không phát lại
hoặc đảo ngược mù thao tác cũ. Hãy dùng trạng thái OCR hiện tại và chỉ đề xuất
một chuỗi sửa ngắn, khác chuỗi cũ, nếu bạn có thể suy ra bước an toàn tiếp theo.
Nếu không chắc trạng thái hoặc bước tiếp theo có thể gửi/xóa/mua/thanh toán/
công khai/chia sẻ dữ liệu, hãy dừng.

Kết quả OCR hiện tại:
{safe_observation}

Lý do critic:
{failure_reason[:500]}

Thao tác cũ (chỉ để tránh lặp):
{json.dumps(previous.get('action_summary', []), ensure_ascii=False)[:800]}

Trả JSON duy nhất:
{{"stop": false, "actions": [{{"action": "click", "x": 1, "y": 1}}],
 "verify_text": "một chuỗi ngắn cần xuất hiện sau hành động"}}
Hoặc {{"stop": true, "reason": "lý do dừng"}}.
Chỉ dùng click, type_text (tối đa 500 ký tự) hoặc press với phím enter/tab/esc/
backspace/space/arrows. Không dùng tổ hợp phím. Không bịa rằng thao tác đã xong.
"""
    try:
        response = llm_client.ask(prompt, json_mode=True, max_output_tokens=700)
        if isinstance(response, str):
            response = json.loads(response)
    except Exception as exc:
        return None, f"Could not plan a state-aware desktop repair: {type(exc).__name__}."

    if not isinstance(response, dict) or response.get("stop") is True:
        reason = response.get("reason", "Leader could not identify a safe next action.") if isinstance(response, dict) else "Malformed recovery response."
        return None, str(reason)[:300]

    actions = response.get("actions")
    verify_text = response.get("verify_text")
    if (
        not isinstance(actions, list)
        or not 1 <= len(actions) <= 4
        or not isinstance(verify_text, str)
        or not verify_text.strip()
    ):
        return None, "Recovery proposal lacks a short action batch or observable verification text."
    if action_fingerprint(actions) == previous.get("action_fingerprint"):
        return None, "Proposed desktop sequence duplicates the previous sequence."

    allowed_actions = {"click", "type_text", "press"}
    if any(not isinstance(item, dict) or item.get("action") not in allowed_actions for item in actions):
        return None, "Recovery proposal contains an unsupported desktop action."
    return {
        "actions": actions,
        "verify_text": verify_text.strip()[:160],
        "reject_fingerprint": previous.get("action_fingerprint"),
    }, "One new state-aware desktop action was proposed from OCR evidence."
