"""Desktop actions stay scoped to the selected operator and explicit goal."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.plan_policy import _task_permissions
from core.tool_broker import tool_actions_for_agent


class DesktopPolicyTests(unittest.TestCase):
    def test_only_selected_operator_receives_observe_and_perform(self):
        operator_actions = tool_actions_for_agent(4, 4)
        other_actions = tool_actions_for_agent(3, 4)

        self.assertEqual(operator_actions["computer_tool"], {"observe", "perform"})
        self.assertNotIn("computer_tool", other_actions)

    def test_explicit_desktop_goal_authorizes_observe_task_for_operator(self):
        permissions = _task_permissions(
            "Observe the desktop screen before proceeding",
            {"desc": "Observe the desktop screen before operating the application"},
            agent_id=4,
            desktop_operator_id=4,
        )

        self.assertIn("computer_tool.observe", permissions)
        self.assertNotIn("computer_tool.perform", permissions)

    def test_explicit_desktop_goal_authorizes_action_task_for_operator(self):
        permissions = _task_permissions(
            "Use the desktop screen to inspect and operate an application",
            {"desc": "Click the requested control on the desktop screen"},
            agent_id=4,
            desktop_operator_id=4,
        )

        self.assertIn("computer_tool.perform", permissions)
        self.assertNotIn("computer_tool.observe", permissions)

    def test_non_operator_or_goal_without_desktop_scope_gets_no_desktop_access(self):
        task = {"desc": "Observe the desktop and click a control"}
        self.assertFalse(_task_permissions(
            "Use the desktop screen to inspect and operate an application",
            task,
            agent_id=3,
            desktop_operator_id=4,
        ))
        self.assertFalse(_task_permissions(
            "Summarize a research paper",
            task,
            agent_id=4,
            desktop_operator_id=4,
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
