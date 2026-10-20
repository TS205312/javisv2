"""Tests for serialized, bounded desktop action sequences."""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import computer_tool, write_safety
from core.recovery import action_fingerprint


class FakePyAutoGUI:
    FAILSAFE = True

    def __init__(self, on_click=None):
        self.on_click = on_click
        self.calls = []

    @staticmethod
    def size():
        return 1024, 768

    def click(self, x, y):
        self.calls.append(("click", x, y))
        if self.on_click:
            self.on_click()

    def write(self, text, interval):
        self.calls.append(("type_text", text, interval))

    def press(self, key):
        self.calls.append(("press", key))


class FakeCapture:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    @staticmethod
    def shot(output):
        Path(output).write_bytes(b"fake png")


class FakeMss:
    @staticmethod
    def mss():
        return FakeCapture()


class ComputerToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.root_patch = mock.patch.object(computer_tool, "_WORKSPACE_ROOT", self.root)
        self.write_root_patch = mock.patch.object(write_safety, "WORKSPACE_ROOT", self.root)
        self.root_patch.start()
        self.write_root_patch.start()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.write_root_patch.stop)

    def test_perform_captures_before_after_and_runs_allowed_actions(self):
        desktop = FakePyAutoGUI()
        with mock.patch.object(
            computer_tool, "_desktop_modules", return_value=(desktop, FakeMss)
        ), mock.patch.object(computer_tool, "_read_screen_text", return_value="Ready"):
            result = json.loads(computer_tool.perform({
                "agent_id": 2,
                "actions": [
                    {"action": "click", "x": 20, "y": 40},
                    {"action": "type_text", "text": "hello"},
                    {"action": "press", "key": "enter"},
                ],
            }))

        self.assertEqual(result["status"], "executed_unverified")
        self.assertEqual(result["agent_id"], 2)
        self.assertEqual(result["actions_completed"], 3)
        self.assertEqual(result["observed_text"], "Ready")
        self.assertTrue(Path(result["before_screenshot"]).is_file())
        self.assertTrue(Path(result["after_screenshot"]).is_file())
        self.assertEqual([call[0] for call in desktop.calls], ["click", "type_text", "press"])

    def test_observe_captures_screen_and_returns_bounded_ocr_coordinates(self):
        desktop = FakePyAutoGUI()
        words = ([{
            "text": "Open",
            "confidence": 96.0,
            "bounds": {"x": 10, "y": 20, "width": 30, "height": 12},
        }], None)
        with (
            mock.patch.object(
                computer_tool, "_desktop_modules", return_value=(desktop, FakeMss)
            ),
            mock.patch.object(computer_tool, "_read_screen_words", return_value=words),
        ):
            result = json.loads(computer_tool.observe({}))

        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["screen"], {"width": 1024, "height": 768})
        self.assertEqual(result["ocr_words"], words[0])
        self.assertTrue(Path(result["screenshot"]).is_file())

    def test_observe_reports_ocr_unavailable_without_faking_screen_text(self):
        desktop = FakePyAutoGUI()
        with (
            mock.patch.object(
                computer_tool, "_desktop_modules", return_value=(desktop, FakeMss)
            ),
            mock.patch.object(
                computer_tool,
                "_read_screen_words",
                return_value=([], "RuntimeError: OCR unavailable"),
            ),
        ):
            result = json.loads(computer_tool.observe({}))

        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["ocr_words"], [])
        self.assertIn("OCR unavailable", result["ocr_error"])

    def test_perform_accepts_more_than_eight_actions(self):
        desktop = FakePyAutoGUI()
        actions = [{"action": "press", "key": "tab"} for _ in range(12)]
        with (
            mock.patch.object(
                computer_tool, "_desktop_modules", return_value=(desktop, FakeMss)
            ),
            mock.patch.object(computer_tool, "_read_screen_text", return_value=""),
        ):
            result = json.loads(computer_tool.perform({"actions": actions}))

        self.assertEqual(result["actions_completed"], 12)
        self.assertEqual(len(desktop.calls), 12)

    def test_invalid_batch_is_rejected_before_any_action(self):
        desktop = FakePyAutoGUI()
        with mock.patch.object(
            computer_tool, "_desktop_modules", return_value=(desktop, FakeMss)
        ):
            with self.assertRaisesRegex(ValueError, "not allowed"):
                computer_tool.perform({
                    "actions": [
                        {"action": "click", "x": 10, "y": 10},
                        {"action": "press", "key": "ctrl+alt+delete"},
                    ],
                })

        self.assertEqual(desktop.calls, [])

    def test_partial_batch_is_not_verified_even_if_expected_text_is_visible(self):
        desktop = FakePyAutoGUI()
        with (
            mock.patch.object(
                computer_tool, "_desktop_modules", return_value=(desktop, FakeMss)
            ),
            mock.patch.object(
                computer_tool, "_run_action", side_effect=RuntimeError("input failed")
            ),
            mock.patch.object(
                computer_tool, "_verify_text", return_value={"status": "matched"}
            ),
        ):
            result = json.loads(computer_tool.perform({
                "actions": [{"action": "click", "x": 10, "y": 10}],
                "verify_text": "Ready",
            }))

        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["actions_completed"], 0)

    def test_duplicate_recovery_batch_is_rejected_before_desktop_input(self):
        desktop = FakePyAutoGUI()
        actions = [{"action": "press", "key": "enter"}]
        with mock.patch.object(
            computer_tool, "_desktop_modules", return_value=(desktop, FakeMss)
        ):
            with self.assertRaisesRegex(ValueError, "duplicates the previous"):
                computer_tool.perform({
                    "actions": actions,
                    "reject_fingerprint": action_fingerprint(actions),
                })

        self.assertEqual(desktop.calls, [])

    def test_concurrent_batches_are_serialized(self):
        state_lock = threading.Lock()
        state = {"active": 0, "max_active": 0}

        def slow_click():
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.03)
            with state_lock:
                state["active"] -= 1

        desktop = FakePyAutoGUI(on_click=slow_click)
        with (
            mock.patch.object(
                computer_tool, "_desktop_modules", return_value=(desktop, FakeMss)
            ),
            mock.patch.object(computer_tool, "_read_screen_text", return_value=""),
        ):
            threads = [
                threading.Thread(
                    target=computer_tool.perform,
                    args=({"agent_id": agent_id, "actions": [{
                        "action": "click", "x": 10, "y": 10,
                    }]},),
                )
                for agent_id in (1, 2)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(state["max_active"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
