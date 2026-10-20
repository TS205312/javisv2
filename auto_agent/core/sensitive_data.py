"""Redaction helpers for prompts, logs, and tool output."""
import re


_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z_-]{16,}\b"),
    re.compile(r"\bAQ\.[0-9A-Za-z_-]{16,}\b"),
    re.compile(r"\bsk-(?:or-v1-)?[0-9A-Za-z_-]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{48,}\b"),
)
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|passwd|secret|credential|cookie|set-cookie|authorization)\b"
    r"\s*[:=]\s*(?:bearer\s+)?(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_SENSITIVE_FIELD_PARTS = {
    "token", "tokens", "cookie", "cookies", "password", "passwd",
    "secret", "secrets", "credential", "credentials", "authorization",
    "auth", "key",
}


def _is_sensitive_field(name):
    parts = set(filter(None, re.split(r"[^a-z0-9]+", str(name).casefold())))
    compact = "".join(parts)
    return bool(parts & _SENSITIVE_FIELD_PARTS) or compact in {
        "apikey", "accesstoken", "refreshtoken", "setcookie", "privatekey",
    }


def redact_text(value):
    if not isinstance(value, str):
        return value
    redacted = _ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        value,
    )
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
    for pattern in _PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def contains_sensitive_data(value):
    return isinstance(value, str) and redact_text(value) != value


def redact_data(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            redact_text(str(key)): (
                "[REDACTED]" if _is_sensitive_field(key) else redact_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_data(item) for item in value)
    return value
