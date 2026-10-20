"""Tests for single-agent desktop assignment and verified history selection."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.desktop_operator import select_desktop_operator
from core.tool_broker import tool_actions_for_agent


class DesktopOperatorTests(unittest.TestCase):
    def test_selects_agent_with_best_verified_record(self):
        selected, reason = select_desktop_operator({
            1: {"verified_successes": 2, "attempts": 4},
            2: {"verified_successes": 3, "attempts": 3},
            3: {"verified_successes": 0, "attempts": 2},
            4: {"verified_successes": 1, "attempts": 1},
        }, [1, 2, 3, 4])

        self.assertEqual(selected, 2)
        self.assertIn("3/3", reason)

    def test_cold_start_uses_completion_check_agent_provisionally(self):
        selected, reason = select_desktop_operator({}, [1, 2, 3, 4])

        self.assertEqual(selected, 4)
        self.assertIn("Cold start", reason)
        self.assertIn("provisionally", reason)

    def test_only_selected_agent_gets_computer_tool_in_allowlist(self):
        self.assertNotIn("computer_tool", tool_actions_for_agent(2, 4))
        self.assertEqual(
            tool_actions_for_agent(4, 4)["computer_tool"],
            frozenset({"perform"}),
        )

    def test_no_desktop_capability_without_operator_assignment(self):
        self.assertNotIn("computer_tool", tool_actions_for_agent(4, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
