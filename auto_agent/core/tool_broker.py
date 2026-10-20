"""Small in-process and desktop-session resource lease broker."""
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit


class ResourceLeaseTimeout(TimeoutError):
    pass


SAFE_TOOL_ACTIONS = {
    "file_tool": frozenset({"read", "write"}),
    "shell_tool": frozenset({"run"}),
    "search_tool": frozenset({"search"}),
    "browser_tool": frozenset({
        "goto", "scrape", "screenshot", "click", "click_at", "type_text",
    }),
    "presentation_draft": frozenset({"generate"}),
}


def tool_actions_for_agent(agent_id=None, desktop_operator_id=None):
    actions = dict(SAFE_TOOL_ACTIONS)
    if (
        agent_id is not None
        and desktop_operator_id is not None
        and agent_id == desktop_operator_id
    ):
        actions["computer_tool"] = frozenset({"observe", "perform"})
    return actions


_registry_lock = threading.Lock()
_resource_locks = {}


def _lock_for(resource):
    with _registry_lock:
        return _resource_locks.setdefault(resource, threading.RLock())


def file_resource(path):
    canonical = Path(path).resolve(strict=False)
    return f"file:{os.path.normcase(str(canonical))}"


def browser_resource(url):
    parsed = urlsplit(str(url or ""))
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Browser resource must be an absolute HTTP(S) URL.")
    scheme = parsed.scheme.casefold()
    host = parsed.hostname.casefold()
    port = parsed.port
    if port and not (scheme == "https" and port == 443) and not (scheme == "http" and port == 80):
        host = f"{host}:{port}"
    return f"browser:{scheme}://{host}"


def _desktop_mutex(timeout_seconds):
    if os.name != "nt":
        return None

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateMutexW(None, False, "Local\\JavisDesktopResourceLease")
    if not handle:
        raise OSError(ctypes.get_last_error(), "Could not create desktop resource mutex.")
    timeout_ms = max(0, min(int(timeout_seconds * 1000), 0xFFFFFFFE))
    result = kernel32.WaitForSingleObject(handle, timeout_ms)
    if result not in {0x00000000, 0x00000080}:
        kernel32.CloseHandle(handle)
        if result == 0x00000102:
            raise ResourceLeaseTimeout("Desktop resource lease timed out.")
        raise OSError(ctypes.get_last_error(), "Could not acquire desktop resource mutex.")
    return kernel32, handle


@contextmanager
def resource_lease(resource, timeout_seconds=30.0):
    if not isinstance(resource, str) or not resource.strip():
        raise ValueError("A resource lease requires a non-empty resource key.")
    timeout_seconds = max(0.0, min(float(timeout_seconds), 300.0))
    key = resource.strip()
    lock = _lock_for(key)
    acquired = lock.acquire(timeout=timeout_seconds)
    if not acquired:
        label = "desktop" if key.startswith("desktop:") else "shared resource"
        raise ResourceLeaseTimeout(f"Timed out waiting for the {label} lease.")

    os_mutex = None
    try:
        if key == "desktop:physical":
            os_mutex = _desktop_mutex(timeout_seconds)
        yield
    finally:
        try:
            if os_mutex is not None:
                kernel32, handle = os_mutex
                try:
                    kernel32.ReleaseMutex(handle)
                finally:
                    kernel32.CloseHandle(handle)
        finally:
            lock.release()
