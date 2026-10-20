"""Thread-owned Playwright sessions for isolated swarm-agent browser control."""
from functools import wraps
from threading import local
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None
from core.sensitive_data import contains_sensitive_data, redact_text
from core.tool_broker import browser_resource, resource_lease
from tools.write_safety import prepare_write_path, write_bytes_safely


_browser_state = local()
_PAGE_TEXT_LIMIT = 12000
_CANVA_URL = "https://www.canva.com/"


class BrowserToolError(RuntimeError):
    def __init__(self, code, message, retryable=False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def set_session_logger(logger):
    _browser_state.logger = logger


def _log(event, details):
    logger = getattr(_browser_state, "logger", None)
    if logger is not None:
        try:
            logger.log(event, details)
        except Exception:
            pass


def _page_is_closed(page):
    try:
        return page.is_closed() is True
    except Exception:
        return False


def _close_resources(session):
    errors = []
    for key in ("context", "browser", "playwright"):
        resource = session.get(key)
        if resource is None:
            continue
        try:
            resource.close() if key != "playwright" else resource.stop()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def _session(agent_id=None):
    session = getattr(_browser_state, "session", None)
    if session is not None:
        wrong_owner = agent_id is not None and session["agent_id"] != agent_id
        browser_disconnected = False
        try:
            browser_disconnected = session["browser"].is_connected() is False
        except Exception:
            pass
        if wrong_owner or browser_disconnected or _page_is_closed(session["page"]):
            _browser_state.recovery_pending = not wrong_owner
            _log("browser_session_replaced", {
                "agent_id": session["agent_id"],
                "session_id": session["session_id"],
                "reason": (
                    "agent_owner_changed" if wrong_owner
                    else "browser_disconnected" if browser_disconnected
                    else "page_closed"
                ),
            })
            close_session()
        else:
            return session

    playwright = None
    browser = None
    try:
        if sync_playwright is None:
            raise BrowserToolError(
                "browser_launch_failed",
                "Playwright is not installed in the active Python environment.",
            )
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
    except Exception as exc:
        for resource, closer in ((browser, "close"), (playwright, "stop")):
            if resource is not None:
                try:
                    getattr(resource, closer)()
                except Exception:
                    pass
        if isinstance(exc, BrowserToolError):
            raise
        if "Executable doesn't exist" in str(exc):
            message = (
                "Playwright Chromium is not installed. Run "
                "`python -m playwright install chromium` from the active "
                "virtual environment."
            )
            raise BrowserToolError("browser_launch_failed", message) from exc
        raise BrowserToolError(
            "browser_launch_failed",
            f"Could not launch an isolated browser for agent {agent_id}: "
            f"{type(exc).__name__}: {exc}",
        ) from exc

    session = {
        "agent_id": agent_id,
        "session_id": uuid4().hex,
        "playwright": playwright,
        "browser": browser,
        "context": context,
        "page": page,
        "recovered": bool(getattr(_browser_state, "recovery_pending", False)),
        "last_http_status": None,
    }
    _browser_state.recovery_pending = False
    _browser_state.session = session
    _log("browser_session_opened", {
        "agent_id": agent_id,
        "session_id": session["session_id"],
        "isolated_context": True,
        "headless": True,
    })
    return session


def open_agent_session(agent_id):
    """Create or return the isolated page owned by this agent thread."""
    session = _session(agent_id)
    return session_summary(agent_id)


def session_summary(agent_id=None):
    session = getattr(_browser_state, "session", None)
    if session is None or (
        agent_id is not None and session["agent_id"] != agent_id
    ):
        return {
            "agent_id": agent_id,
            "session_id": None,
            "status": "not_started",
            "isolated_context": True,
        }
    page = session["page"]
    closed = _page_is_closed(page)
    return {
        "agent_id": session["agent_id"],
        "session_id": session["session_id"],
        "status": "page_closed" if closed else "ready",
        "url": _public_url(getattr(page, "url", "")) if not closed else "",
        "isolated_context": True,
        "recovered": session["recovered"],
    }


def close_session(agent_id=None):
    """Close the browser owned by the current agent worker thread."""
    session = getattr(_browser_state, "session", None)
    if session is None:
        return
    if agent_id is not None and session["agent_id"] != agent_id:
        return

    del _browser_state.session
    _log("browser_session_closed", {
        "agent_id": session["agent_id"],
        "session_id": session["session_id"],
        "isolated_context": True,
    })
    _close_resources(session)


def _page(params):
    return _session(params.get("agent_id"))["page"]


def _validated_url(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("browser_tool requires a non-empty URL.")
    parsed = urlparse(url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("browser_tool only supports absolute HTTP(S) URLs.")
    if contains_sensitive_data(url):
        raise ValueError("Credential-like URL data is blocked from browser navigation.")
    return url.strip()


def _public_url(url):
    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _page_text(page):
    try:
        return page.locator("body").inner_text(timeout=3000).casefold()
    except Exception:
        return ""


def _page_status(page, http_status=None):
    url = getattr(page, "url", "") or ""
    parsed = urlparse(url)
    host = parsed.netloc.casefold()
    is_canva = host == "canva.com" or host.endswith(".canva.com")
    path = parsed.path.casefold()
    body = _page_text(page)
    blocked_terms = (
        "unusual traffic", "automated access", "automated requests",
        "temporarily blocked", "access denied", "verify you are human",
        "robot check",
    )
    auth_terms = (
        "log in to continue", "sign in to continue", "log in or sign up",
        "sign in or sign up",
    )
    if is_canva and (
        http_status in {403, 429} or any(term in body for term in blocked_terms)
    ):
        return "automation_blocked"
    if is_canva and http_status == 401:
        return "login_required"
    if is_canva and (
        any(term in path for term in ("/login", "/signin", "/auth"))
        or any(term in body for term in auth_terms)
    ):
        return "login_required"
    if http_status is not None and http_status >= 400:
        return "http_error"
    return "accessible" if is_canva else "ready"


def _classify_browser_error(exc):
    message = str(exc).casefold()
    if "timeout" in message or "timed out" in message:
        return "browser_timeout", True
    if any(term in message for term in (
        "target page, context or browser has been closed",
        "page has been closed", "context has been closed",
    )):
        return "page_closed", True
    return "browser_operation_failed", False


def _assert_canva_operable(session):
    if not session.get("session_id"):
        return
    status = _page_status(
        session["page"], session.get("last_http_status")
    )
    if status == "login_required":
        raise BrowserToolError(
            status,
            "Canva session requires login or has expired; no automated action was performed.",
        )
    if status == "automation_blocked":
        raise BrowserToolError(
            status,
            "Canva blocked automated browser access; no further action was performed.",
        )
    if status == "http_error":
        raise BrowserToolError(
            status,
            "The browser page returned an HTTP error; no automated action was performed.",
        )


def _raise_operation_error(session, exc):
    status = _page_status(
        session["page"], session.get("last_http_status")
    )
    if status in {"login_required", "automation_blocked", "http_error"}:
        message = {
            "login_required": "Canva session requires login or has expired.",
            "automation_blocked": "Canva blocked automated browser access.",
            "http_error": "The browser page returned an HTTP error during interaction.",
        }[status]
        raise BrowserToolError(status, message) from exc
    code, retryable = _classify_browser_error(exc)
    raise BrowserToolError(code, str(exc), retryable=retryable) from exc


def goto(params):
    """Navigate this agent's isolated page to an HTTP(S) URL."""
    session = _session(params.get("agent_id"))
    page = session["page"]
    try:
        response = page.goto(_validated_url(params.get("url")), timeout=30000)
    except Exception as exc:
        code, retryable = _classify_browser_error(exc)
        _log("browser_navigation_failed", {
            "agent_id": session["agent_id"],
            "session_id": session["session_id"],
            "url": _public_url(params.get("url")),
            "code": code,
            "error": str(exc)[:300],
        })
        raise BrowserToolError(code, str(exc), retryable=retryable) from exc
    status = response.status if response is not None else None
    session["last_http_status"] = status
    page_status = _page_status(page, status)
    _log("browser_navigation", {
        "agent_id": session["agent_id"],
        "session_id": session["session_id"],
        "url": _public_url(page.url),
        "http_status": status,
        "page_status": page_status,
    })
    return (
        f"Agent {session['agent_id']} session {session['session_id']} "
        f"navigated to {_public_url(page.url)} (HTTP {status or 'unknown'}, "
        f"page_status={page_status})."
    )


def canva_preflight(params):
    """Check whether Canva is reachable and whether its page requires login."""
    agent_id = params.get("agent_id")
    try:
        session = _session(agent_id)
        response = session["page"].goto(
            _CANVA_URL,
            timeout=int(params.get("timeout_ms", 15000)),
            wait_until="domcontentloaded",
        )
        http_status = response.status if response is not None else None
        session["last_http_status"] = http_status
        status = _page_status(session["page"], http_status)
        result = {
            "agent_id": agent_id,
            "session_id": session["session_id"],
            "status": status,
            "url": _public_url(session["page"].url),
            "http_status": http_status,
            "isolated_context": True,
            "design_created": False,
        }
    except Exception as exc:
        code = getattr(exc, "code", None)
        retryable = getattr(exc, "retryable", False)
        if not code:
            code, retryable = _classify_browser_error(exc)
            if code == "browser_operation_failed":
                code = "canva_access_failed"
        result = {
            "agent_id": agent_id,
            "session_id": session_summary(agent_id).get("session_id"),
            "status": code,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "retryable": retryable,
            "isolated_context": True,
            "design_created": False,
        }
    _log("canva_preflight", result)
    return result


def scrape(params):
    """Return the current page title and visible text, keeping this session open."""
    page = _page(params)
    if params.get("url"):
        goto({"agent_id": params.get("agent_id"), "url": params["url"]})
    if not page.url or page.url == "about:blank":
        raise ValueError("Navigate to a page before scraping it.")
    session = _session(params.get("agent_id"))
    status = _page_status(page, session.get("last_http_status"))
    text = page.locator("body").inner_text(timeout=10000)
    return redact_text(
        f"Agent: {session['agent_id']}\nSession: {session['session_id']}\n"
        f"Title: {page.title()}\nURL: {_public_url(page.url)}\n"
        f"Page status: {status}\n{text[:_PAGE_TEXT_LIMIT]}"
    )


def screenshot(params):
    """Save a screenshot of this agent's page."""
    path = params.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("browser_tool.screenshot requires a path.")
    page = _page(params)
    if not page.url or page.url == "about:blank":
        raise ValueError("Navigate to a page before taking a screenshot.")
    session = _session(params.get("agent_id"))
    _assert_canva_operable(session)
    safe_path = prepare_write_path(path)
    try:
        image_bytes = page.screenshot(full_page=True)
        write_bytes_safely(safe_path, image_bytes)
    except Exception as exc:
        _raise_operation_error(session, exc)
    _log("browser_action", {
        "agent_id": session.get("agent_id", params.get("agent_id")),
        "session_id": session.get("session_id"),
        "action": "screenshot",
        "byte_count": len(image_bytes),
    })
    return f"Agent {session['agent_id']} session {session['session_id']} saved a workspace screenshot."


def click(params):
    """Click a CSS selector within this agent's page."""
    selector = params.get("selector")
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("browser_tool.click requires a CSS selector.")
    session = _session(params.get("agent_id"))
    page = session["page"]
    _assert_canva_operable(session)
    try:
        page.locator(selector).click(timeout=10000)
    except Exception as exc:
        _raise_operation_error(session, exc)
    _log("browser_action", {
        "agent_id": session["agent_id"],
        "session_id": session["session_id"],
        "action": "click",
        "selector": selector,
    })
    return f"Agent {session['agent_id']} session {session['session_id']} clicked selector {selector!r}."


def click_at(params):
    """Move this page's virtual mouse pointer and click at viewport coordinates."""
    x = params.get("x")
    y = params.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("browser_tool.click_at requires numeric x and y coordinates.")
    session = _session(params.get("agent_id"))
    page = session["page"]
    _assert_canva_operable(session)
    try:
        page.mouse.move(x, y)
        page.mouse.click(x, y)
    except Exception as exc:
        _raise_operation_error(session, exc)
    _log("browser_action", {
        "agent_id": session.get("agent_id", params.get("agent_id")),
        "session_id": session.get("session_id"),
        "action": "click_at",
        "x": x,
        "y": y,
    })
    return (
        f"Agent {session.get('agent_id', params.get('agent_id', '?'))} "
        f"session {session.get('session_id', 'unknown')} clicked page "
        f"coordinates ({x}, {y})."
    )


def type_text(params):
    """Type into an optional selector, or the focused element, on this agent's page."""
    text = params.get("text")
    if not isinstance(text, str):
        raise ValueError("browser_tool.type_text requires text.")
    if contains_sensitive_data(text):
        raise ValueError("Credential-like browser text is blocked.")
    session = _session(params.get("agent_id"))
    page = session["page"]
    _assert_canva_operable(session)
    selector = params.get("selector")
    try:
        if selector:
            page.locator(selector).fill(text, timeout=10000)
        else:
            page.keyboard.type(text)
    except Exception as exc:
        _raise_operation_error(session, exc)
    _log("browser_action", {
        "agent_id": session["agent_id"],
        "session_id": session["session_id"],
        "action": "type_text",
        "character_count": len(text),
        "selector": selector,
    })
    return f"Agent {session['agent_id']} session {session['session_id']} typed {len(text)} characters."


def _with_browser_lease(function):
    @wraps(function)
    def wrapped(params):
        params = params if isinstance(params, dict) else {}
        target = (
            _CANVA_URL
            if function.__name__ == "canva_preflight"
            else params.get("url")
        )
        if target:
            resource = browser_resource(_validated_url(target))
        else:
            session = _session(params.get("agent_id"))
            current_url = session["page"].url
            resource = (
                browser_resource(current_url)
                if isinstance(current_url, str)
                and current_url.startswith(("http://", "https://"))
                else f"browser-agent:{params.get('agent_id', 'single')}"
            )
        with resource_lease(resource, timeout_seconds=30):
            return function(params)

    return wrapped


for _browser_action in (
    goto, canva_preflight, scrape, screenshot, click, click_at, type_text,
):
    globals()[_browser_action.__name__] = _with_browser_lease(_browser_action)
