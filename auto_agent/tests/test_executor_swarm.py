"""Tests ensuring swarm GUI tasks use browser sessions instead of desktop input."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import executor


class ExecutorSwarmTests(unittest.TestCase):
    def test_presentation_draft_uses_llm_drafting_not_search_router(self):
        task = {
            "id": "create_slides",
            "type": "presentation_draft",
            "desc": (
                "Soạn nội dung tiếng Việt 6–8 slide về AI trong giáo dục; "
                "mục tiêu gốc yêu cầu dùng công cụ tìm kiếm và ghi nguồn."
            ),
        }
        memory = mock.Mock()
        memory.get_short_term_context.return_value = "verified source results"
        logger = mock.Mock()
        llm = mock.Mock()
        llm.ask.return_value = "## Slide 1: Mở bài"
        with (
            mock.patch.object(executor, "route_task") as route_task,
            mock.patch.object(executor, "SkillResolver") as resolver,
        ):
            resolver.return_value.discover_best.return_value = {
                "status": "no_candidate",
                "reason": "No suitable public repository.",
            }
            result = executor.execute_task(
                task,
                memory=memory,
                logger=logger,
                llm_client=llm,
                agent_id=1,
            )

        self.assertEqual(result, "## Slide 1: Mở bài")
        self.assertEqual(task["_selected_tool"], "presentation_draft")
        llm.ask.assert_called_once_with(mock.ANY, max_output_tokens=4096)
        route_task.assert_not_called()
        resolver.return_value.discover_best.assert_called_once_with(task["desc"])
        self.assertIn(
            "GitHub skill reference metadata",
            llm.ask.call_args.args[0],
        )

    def test_desktop_mouse_tool_is_rejected_in_swarm(self):
        task = {"id": "ui", "desc": "Click a button in a browser"}
        memory = mock.Mock()
        logger = mock.Mock()
        skill = {"skill": "browser_tool"}
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch.object(
                executor,
                "route_task",
                return_value={
                    "tool": "computer_tool",
                    "action": "click",
                    "params": {"x": 1, "y": 2},
                },
            ) as route_task,
        ):
            resolver.return_value.resolve.return_value = skill
            result = executor.execute_task(
                task,
                memory=memory,
                logger=logger,
                llm_client=mock.Mock(),
                agent_id=2,
            )

        resolver.return_value.resolve.assert_called_once_with(
            task["desc"],
            allow_github=False,
        )
        route_task.assert_called_once()
        self.assertEqual(route_task.call_args.kwargs["swarm_agent_id"], 2)
        self.assertIn("Physical desktop control is disabled", result)

    def test_desktop_batch_is_blocked_for_agents(self):
        task = {"id": "desktop", "desc": "Open the calculator application"}
        memory = mock.Mock()
        logger = mock.Mock()
        computer = mock.Mock()
        computer.perform.return_value = '{"status":"executed_unverified"}'
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch.object(
                executor,
                "route_task",
                return_value={
                    "tool": "computer_tool",
                    "action": "perform",
                    "params": {
                        "actions": [{"action": "type_text", "text": "private screen text"}],
                        "verify_text": "private verification phrase",
                    },
                },
            ),
            mock.patch.dict(executor.TOOLS, {"computer_tool": computer}),
        ):
            resolver.return_value.resolve.return_value = {"skill": "computer_tool"}
            result = executor.execute_task(
                task,
                memory=memory,
                logger=logger,
                llm_client=mock.Mock(),
                agent_id=3,
            )

        computer.perform.assert_not_called()
        self.assertEqual(task["_selected_action"], "perform")
        self.assertIn("Physical desktop control is disabled", result)
        self.assertNotIn("private screen text", str(logger.log.call_args_list))
        self.assertNotIn("private verification phrase", str(logger.log.call_args_list))

    def test_assigned_operator_can_run_only_verified_desktop_batch(self):
        task = {
            "id": "desktop",
            "desc": "Use desktop to enter the requested text",
            "_authorized_actions": ["computer_tool.perform"],
        }
        desktop = mock.Mock()
        desktop.perform.return_value = '{"status":"verified"}'
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch.object(
                executor,
                "route_task",
                return_value={
                    "tool": "computer_tool",
                    "action": "perform",
                    "params": {
                        "actions": [{"action": "type_text", "text": "Hello"}],
                        "verify_text": "Hello",
                    },
                },
            ) as route_task,
            mock.patch.dict(executor.TOOLS, {"computer_tool": desktop}),
        ):
            resolver.return_value.resolve.return_value = {"skill": "computer_tool"}
            result = executor.execute_task(
                task,
                memory=mock.Mock(),
                logger=mock.Mock(),
                llm_client=mock.Mock(),
                agent_id=4,
                desktop_operator_id=4,
            )

        self.assertEqual(result, '{"status":"verified"}')
        self.assertEqual(route_task.call_args.kwargs["desktop_operator_id"], 4)
        desktop.perform.assert_called_once()
        resolver.return_value.discover_best.assert_called_once_with(task["desc"])

    def test_assigned_operator_can_observe_desktop_when_plan_authorizes_it(self):
        task = {
            "id": "observe_desktop",
            "desc": "Observe the desktop screen",
            "_authorized_actions": ["computer_tool.observe"],
        }
        desktop = mock.Mock()
        desktop.observe.return_value = '{"status":"observed","ocr_words":[]}'
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch.object(executor, "route_task", return_value={
                "tool": "computer_tool",
                "action": "observe",
                "params": {},
            }),
            mock.patch.dict(executor.TOOLS, {"computer_tool": desktop}),
        ):
            resolver.return_value.resolve.return_value = {"skill": "computer_tool"}
            result = executor.execute_task(
                task,
                memory=mock.Mock(),
                logger=mock.Mock(),
                llm_client=mock.Mock(),
                agent_id=4,
                desktop_operator_id=4,
            )

        self.assertIn('"status":"observed"', result)
        desktop.observe.assert_called_once_with({"agent_id": 4})

    def test_non_operator_cannot_observe_desktop(self):
        task = {"id": "observe_desktop", "desc": "Observe the screen"}
        desktop = mock.Mock()
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch.object(executor, "route_task", return_value={
                "tool": "computer_tool", "action": "observe", "params": {},
            }),
            mock.patch.dict(executor.TOOLS, {"computer_tool": desktop}),
        ):
            resolver.return_value.resolve.return_value = {"skill": "computer_tool"}
            result = executor.execute_task(
                task,
                memory=mock.Mock(),
                logger=mock.Mock(),
                llm_client=mock.Mock(),
                agent_id=3,
                desktop_operator_id=4,
            )

        self.assertIn("Physical desktop control is disabled", result)
        desktop.observe.assert_not_called()

    def test_assigned_operator_cannot_run_unverified_desktop_batch(self):
        task = {"id": "desktop", "desc": "Use desktop", "_authorized_actions": [
            "computer_tool.perform",
        ]}
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch.object(executor, "route_task", return_value={
                "tool": "computer_tool",
                "action": "perform",
                "params": {"actions": [{"action": "press", "key": "enter"}]},
            }),
            mock.patch.dict(executor.TOOLS, {"computer_tool": mock.Mock()}),
        ):
            resolver.return_value.resolve.return_value = {"skill": "computer_tool"}
            result = executor.execute_task(
                task,
                memory=mock.Mock(),
                logger=mock.Mock(),
                llm_client=mock.Mock(),
                agent_id=4,
                desktop_operator_id=4,
            )

        self.assertIn("require non-empty verify_text", result)

    def test_browser_route_receives_the_agent_id(self):
        task = {"id": "ui", "desc": "Inspect a page"}
        memory = mock.Mock()
        logger = mock.Mock()
        browser = mock.Mock()
        browser.goto.return_value = "Agent 2 navigated."
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch.object(
                executor,
                "route_task",
                return_value={
                    "tool": "browser_tool",
                    "action": "goto",
                    "params": {"url": "https://example.com"},
                },
            ),
            mock.patch.dict(executor.TOOLS, {"browser_tool": browser}),
        ):
            resolver.return_value.resolve.return_value = {
                "skill": "browser_tool"
            }
            result = executor.execute_task(
                task,
                memory=memory,
                logger=logger,
                llm_client=mock.Mock(),
                agent_id=2,
            )

        browser.goto.assert_called_once_with({
            "url": "https://example.com",
            "agent_id": 2,
        })
        self.assertEqual(result, "Agent 2 navigated.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
