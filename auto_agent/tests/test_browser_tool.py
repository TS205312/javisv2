"""Tests for isolated Playwright sessions and page-local mouse input."""
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import browser_tool
from tools.browser_tool import BrowserToolError


class BrowserToolIsolationTests(unittest.TestCase):
    def test_each_worker_gets_its_own_browser_context(self):
        barrier = threading.Barrier(2)
        sessions = []
        failures = []

        def create_session(agent_id):
            try:
                session = browser_tool._session(agent_id)
                barrier.wait(timeout=5)
                sessions.append(session)
                browser_tool.close_session(agent_id)
            except Exception as exc:
                failures.append(exc)

        with mock.patch.object(browser_tool, "sync_playwright") as playwright_factory:
            playwright_factory.return_value.start.side_effect = (
                lambda: mock.Mock()
            )
            threads = [
                threading.Thread(target=create_session, args=(agent_id,))
                for agent_id in (1, 2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

        self.assertFalse(failures)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(sessions), 2)
        self.assertIsNot(sessions[0], sessions[1])
        self.assertEqual(
            {session["agent_id"] for session in sessions},
            {1, 2},
        )
        self.assertEqual(
            len({session["session_id"] for session in sessions}),
            2,
        )

    def test_closed_page_is_replaced_only_in_its_owner_thread(self):
        old_page = mock.Mock()
        old_page.is_closed.return_value = True
        old_context = mock.Mock()
        old_browser = mock.Mock()
        old_playwright = mock.Mock()
        old_browser.is_connected.return_value = True
        browser_tool._browser_state.session = {
            "agent_id": 7,
            "session_id": "old-session",
            "page": old_page,
            "context": old_context,
            "browser": old_browser,
            "playwright": old_playwright,
            "recovered": False,
        }
        replacement = mock.Mock()
        replacement_page = mock.Mock()
        replacement_page.is_closed.return_value = False
        replacement.chromium.launch.return_value.new_context.return_value.new_page.return_value = replacement_page

        with mock.patch.object(browser_tool, "sync_playwright") as factory:
            factory.return_value.start.return_value = replacement
            session = browser_tool._session(7)
            browser_tool.close_session(7)

        self.assertNotEqual(session["session_id"], "old-session")
        self.assertTrue(session["recovered"])
        self.assertIs(session["page"], replacement_page)
        old_context.close.assert_called_once_with()

    def test_canva_preflight_classifies_login_expiry(self):
        page = mock.Mock()
        page.url = "https://www.canva.com/login"
        page.locator.return_value.inner_text.return_value = "Log in to continue"
        response = mock.Mock(status=200)
        session = {
            "agent_id": 2,
            "session_id": "canva-session",
            "page": page,
            "last_http_status": None,
        }
        with mock.patch.object(browser_tool, "_session", return_value=session):
            with mock.patch.object(page, "goto", return_value=response):
                result = browser_tool.canva_preflight({"agent_id": 2})

        self.assertEqual(result["status"], "login_required")
        self.assertFalse(result["design_created"])
        self.assertEqual(result["session_id"], "canva-session")

    def test_canva_preflight_classifies_automation_block(self):
        page = mock.Mock()
        page.url = "https://www.canva.com/"
        page.locator.return_value.inner_text.return_value = "Unusual traffic detected"
        response = mock.Mock(status=429)
        session = {
            "agent_id": 4,
            "session_id": "blocked-session",
            "page": page,
            "last_http_status": None,
        }
        with mock.patch.object(browser_tool, "_session", return_value=session):
            with mock.patch.object(page, "goto", return_value=response):
                result = browser_tool.canva_preflight({"agent_id": 4})

        self.assertEqual(result["status"], "automation_blocked")

    def test_canva_actions_stop_without_retry_when_login_is_required(self):
        page = mock.Mock()
        page.url = "https://www.canva.com/login"
        page.locator.return_value.inner_text.return_value = "Sign in to continue"
        session = {
            "agent_id": 5,
            "session_id": "expired-session",
            "page": page,
            "last_http_status": 200,
        }
        with mock.patch.object(browser_tool, "_session", return_value=session):
            with self.assertRaises(BrowserToolError) as caught:
                browser_tool.click({"agent_id": 5, "selector": "button"})

        self.assertEqual(caught.exception.code, "login_required")
        self.assertFalse(caught.exception.retryable)

    def test_click_at_uses_only_the_agent_page_mouse(self):
        page = mock.Mock()
        session = {"page": page}
        with mock.patch.object(browser_tool, "_session", return_value=session):
            result = browser_tool.click_at({
                "agent_id": 3,
                "x": 80,
                "y": 120,
            })

        page.mouse.move.assert_called_once_with(80, 120)
        page.mouse.click.assert_called_once_with(80, 120)
        self.assertIn("Agent 3", result)

    def test_navigation_rejects_non_http_urls(self):
        with self.assertRaises(ValueError):
            browser_tool._validated_url("file:///C:/Users/example.txt")

    def test_missing_chromium_has_install_instructions_and_cleans_up(self):
        playwright = mock.Mock()
        playwright.chromium.launch.side_effect = Exception(
            "Executable doesn't exist at expected path"
        )
        with mock.patch.object(
            browser_tool, "sync_playwright"
        ) as playwright_factory:
            playwright_factory.return_value.start.return_value = playwright
            with self.assertRaisesRegex(
                RuntimeError, "python -m playwright install chromium"
            ):
                browser_tool._session(agent_id=1)

        playwright.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
