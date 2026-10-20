"""Tests for bounded retries that avoid replaying side-effecting actions."""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.recovery import (
    action_fingerprint,
    propose_desktop_recovery,
    retry_decision,
)
from core.critic import evaluate_task


class RetryPolicyTests(unittest.TestCase):
    def test_read_only_action_can_retry_within_limit(self):
        allowed, reason = retry_decision("browser_tool", "scrape", 1)

        self.assertTrue(allowed)
        self.assertIn("lặp an toàn", reason)

    def test_desktop_and_shell_actions_are_not_replayed(self):
        for tool_name, action in (
            ("computer_tool", "perform"),
            ("shell_tool", "run"),
            ("browser_tool", "click"),
        ):
            with self.subTest(tool=tool_name, action=action):
                allowed, reason = retry_decision(tool_name, action, 1)
                self.assertFalse(allowed)
                self.assertIn("tác dụng phụ", reason)

    def test_unknown_action_does_not_retry(self):
        allowed, reason = retry_decision(None, None, 0)

        self.assertFalse(allowed)
        self.assertIn("không lặp lại mù", reason)

    def test_attempt_limit_is_enforced(self):
        allowed, reason = retry_decision("file_tool", "read", 3)

        self.assertFalse(allowed)
        self.assertIn("giới hạn 3", reason)

    def test_unverified_desktop_action_cannot_pass_critic(self):
        result = evaluate_task(
            {
                "_selected_tool": "computer_tool",
                "_selected_action": "perform",
                "desc": "Open the requested application",
            },
            '{"status":"executed_unverified","verification":"not checked"}',
        )

        self.assertFalse(result["passed"])

    def test_desktop_action_passes_only_with_matching_ocr(self):
        result = evaluate_task(
            {
                "_selected_tool": "computer_tool",
                "_selected_action": "perform",
                "desc": "Open the requested application",
            },
            '{"status":"verified","error":null,"actions_completed":1,'
            '"actions_requested":1,"verification":{"status":"matched"}}',
        )

        self.assertTrue(result["passed"])

    def test_desktop_recovery_requires_ocr_and_proposes_a_different_batch(self):
        previous_actions = [{"action": "click", "x": 10, "y": 20}]
        previous_output = {
            "status": "verification_failed",
            "verification": {
                "status": "not_found",
                "observed_text": "Calculator window is open",
            },
            "action_fingerprint": action_fingerprint(previous_actions),
            "action_summary": [{"action": "click", "x": 10, "y": 20}],
        }
        llm = mock.Mock()
        llm.ask.return_value = {
            "stop": False,
            "actions": [{"action": "press", "key": "tab"}],
            "verify_text": "Ready",
        }

        proposal, reason = propose_desktop_recovery(
            "Open calculator", json.dumps(previous_output), "Expected text was absent", llm
        )

        self.assertEqual(proposal["actions"], [{"action": "press", "key": "tab"}])
        self.assertEqual(proposal["reject_fingerprint"], previous_output["action_fingerprint"])
        self.assertIn("new state-aware", reason)

    def test_desktop_recovery_does_not_guess_without_ocr(self):
        llm = mock.Mock()
        proposal, reason = propose_desktop_recovery(
            "Open calculator",
            '{"status":"executed_unverified","verification":"not checked"}',
            "No confirmation",
            llm,
        )

        self.assertIsNone(proposal)
        self.assertIn("OCR observation is unavailable", reason)
        llm.ask.assert_not_called()

    def test_desktop_recovery_rejects_exact_duplicate(self):
        actions = [{"action": "press", "key": "tab"}]
        previous_output = {
            "status": "verification_failed",
            "verification": {"status": "not_found", "observed_text": "Current state"},
            "action_fingerprint": action_fingerprint(actions),
        }
        llm = mock.Mock()
        llm.ask.return_value = {
            "stop": False,
            "actions": actions,
            "verify_text": "Ready",
        }

        proposal, reason = propose_desktop_recovery(
            "Open calculator", json.dumps(previous_output), "Not found", llm
        )

        self.assertIsNone(proposal)
        self.assertIn("duplicates", reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
