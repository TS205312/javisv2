"""Regression coverage for scoped agent tools, plans, and shared leases."""
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import executor, orchestrator, plan_policy, swarm
from core.recovery import action_fingerprint
from core import tool_broker
from core.tool_broker import ResourceLeaseTimeout, browser_resource, resource_lease
from tools.search_tool import SearchResults


class AgentSafetyTests(unittest.TestCase):
    def test_plan_requires_relevance_success_criteria_and_goal_scope(self):
        valid_plan = {
            "tasks": [{
                "id": "research",
                "desc": "Find verified sources about AI in education",
                "allowed_tools": ["search_tool.search"],
                "success": "At least three relevant sources with URLs",
                "deps": [],
            }],
        }

        self.assertEqual(
            plan_policy.validate_plan("Research AI in education", valid_plan),
            [],
        )
        self.assertEqual(
            valid_plan["tasks"][0]["_authorized_actions"],
            ["search_tool.search"],
        )

        unrelated = {
            "tasks": [{
                "id": "delete",
                "desc": "Delete unrelated files",
                "success": "Files are deleted safely",
            }],
        }
        issues = plan_policy.validate_plan("Research AI in education", unrelated)
        self.assertTrue(any("not demonstrably relevant" in issue for issue in issues))
        self.assertTrue(any("delete not authorized" in issue for issue in issues))

    def test_plan_cannot_escalate_declared_tool_permissions(self):
        plan = {
            "tasks": [{
                "id": "research",
                "desc": "Find verified sources about AI in education",
                "allowed_tools": ["coder_tool.write_code"],
                "success": "At least three relevant sources with URLs",
            }],
        }

        issues = plan_policy.validate_plan("Research AI in education", plan)

        self.assertTrue(any("no tool permission within the goal scope" in issue for issue in issues))
        self.assertTrue(any("declares tools outside the goal scope" in issue for issue in issues))

    def test_plan_cannot_authorize_physical_desktop_control(self):
        plan = {
            "tasks": [{
                "id": "desktop",
                "desc": "Click the calculator application",
                "allowed_tools": ["computer_tool.perform"],
                "success": "The calculator displays the requested result",
            }],
        }

        issues = plan_policy.validate_plan("Use desktop to calculate a sum", plan)

        self.assertTrue(any("no tool permission within the goal scope" in issue for issue in issues))

    def test_only_assigned_operator_can_receive_desktop_plan_permission(self):
        plan = {"tasks": [{
            "id": "desktop",
            "desc": "Use computer to click the requested control",
            "allowed_tools": ["computer_tool.perform"],
            "success": "Requested screen text is visible after the action",
        }]}

        operator_plan = {"tasks": [dict(plan["tasks"][0])]}
        operator_issues = plan_policy.validate_plan(
            "Use the computer to click the requested control",
            operator_plan,
            agent_id=4,
            desktop_operator_id=4,
        )
        other_agent_issues = plan_policy.validate_plan(
            "Use the computer to click the requested control",
            {"tasks": [dict(plan["tasks"][0])]},
            agent_id=2,
            desktop_operator_id=4,
        )

        self.assertEqual(operator_issues, [])
        self.assertEqual(
            operator_plan["tasks"][0]["_authorized_actions"], ["computer_tool.perform"]
        )
        self.assertTrue(any("no tool permission" in issue for issue in other_agent_issues))

        unrelated_goal_issues = plan_policy.validate_plan(
            "Research computer graphics",
            {"tasks": [dict(plan["tasks"][0])]},
            agent_id=4,
            desktop_operator_id=4,
        )
        self.assertTrue(any("no tool permission" in issue for issue in unrelated_goal_issues))

    def test_plan_repair_is_limited_to_one_replan(self):
        invalid_plan = {
            "tasks": [{
                "id": "off-topic",
                "desc": "Delete unrelated files",
                "success": "Files are deleted",
            }],
        }
        agent = self._agent()
        valid_plan = {
            "tasks": [{
                "id": "research",
                "desc": "Find verified sources about AI in education",
                "success": "At least three relevant sources with URLs",
            }],
        }
        rows = [
            {"title": f"Source {index}", "url": f"https://example.org/{index}",
             "snippet": "AI education evidence", "source": "example.org"}
            for index in range(3)
        ]
        with (
            mock.patch("core.swarm.generate_plan", side_effect=[invalid_plan, valid_plan]) as generate,
            mock.patch("core.swarm.execute_task", return_value=SearchResults(rows, provider="test")),
            mock.patch("core.swarm.evaluate_task", return_value={"passed": True, "reason": "Evidence found", "fix": ""}),
        ):
            result = agent.run("Research AI in education")

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result["status"], "completed")

    def test_invalid_plan_fails_after_exactly_one_repair(self):
        invalid_plan = {
            "tasks": [{
                "id": "off-topic",
                "desc": "Delete unrelated files",
                "success": "Files are deleted",
            }],
        }
        agent = self._agent()
        with (
            mock.patch("core.swarm.generate_plan", return_value=invalid_plan) as generate,
            mock.patch("core.swarm.execute_task") as execute,
        ):
            result = agent.run("Research AI in education")

        self.assertEqual(generate.call_count, 2)
        execute.assert_not_called()
        self.assertEqual(result["status"], "plan_failed")
        self.assertIn("một lần sửa", result["error"])

    def test_task_github_recommendation_is_saved_in_long_term_memory(self):
        agent = self._agent()
        plan = {"tasks": [{
            "id": "open_page",
            "desc": "Open the target webpage in browser",
            "allowed_tools": ["browser_tool.goto"],
            "success": "The target page is reached and verified",
            "deps": [],
        }]}
        recommendation = {
            "status": "recommended_not_installed",
            "skill": "github:sample/browser-helper",
            "repo_url": "https://github.com/sample/browser-helper",
            "stars": 500,
            "license": "MIT",
        }

        def execute(task, **kwargs):
            task["_selected_tool"] = "browser_tool"
            task["_selected_action"] = "goto"
            task["_github_skill_recommendation"] = recommendation
            return "Page opened"

        with (
            mock.patch("core.swarm.generate_plan", return_value=plan),
            mock.patch("core.swarm.execute_task", side_effect=execute),
            mock.patch("core.swarm.evaluate_task", return_value={
                "passed": True,
                "reason": "Page is ready",
                "fix": "",
            }),
        ):
            result = agent.run("Open a webpage in browser")

        self.assertEqual(result["status"], "completed")
        saved_experience = agent.memory.save_long_term.call_args.args[2]
        self.assertEqual(
            saved_experience["github_skill_recommendation"]["skill"],
            "github:sample/browser-helper",
        )

    def test_executor_denies_unsupported_router_tool_even_if_yolo(self):
        task = {
            "id": "code",
            "desc": "Write and execute Python code",
            "_authorized_actions": ["shell_tool.run"],
        }
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch("core.executor.route_task", return_value={
                "tool": "coder_tool", "action": "write_code", "params": {},
            }),
        ):
            resolver.return_value.resolve.return_value = {"skill": "coder_tool"}
            result = executor.execute_task(
                task,
                yolo=True,
                memory=mock.Mock(),
                logger=mock.Mock(),
                agent_id=1,
            )

        self.assertTrue(result.startswith("ERROR [SAFETY]:"))

    def test_identical_failed_action_is_not_called_again(self):
        task = {
            "id": "research",
            "desc": "Find a source",
            "_authorized_actions": ["search_tool.search"],
        }
        route_params = {"query": "AI education"}
        task["_failed_action_fingerprints"] = [action_fingerprint({
            "tool": "search_tool",
            "action": "search",
            "params": route_params,
        })]
        with (
            mock.patch("core.executor.SkillResolver") as resolver,
            mock.patch("core.executor.route_task", return_value={
                "tool": "search_tool", "action": "search", "params": route_params,
            }),
            mock.patch.object(executor.search_tool, "search") as search,
        ):
            resolver.return_value.resolve.return_value = {"skill": "search_tool"}
            result = executor.execute_task(
                task, memory=mock.Mock(), logger=mock.Mock(), agent_id=1
            )

        self.assertIn("identical failed action", result)
        search.assert_not_called()

    def test_resource_lease_contends_and_releases_after_exception(self):
        entered = threading.Event()
        release = threading.Event()

        def holder():
            with resource_lease("test:shared-resource"):
                entered.set()
                release.wait(timeout=2)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(entered.wait(timeout=1))
        with self.assertRaises(ResourceLeaseTimeout):
            with resource_lease("test:shared-resource", timeout_seconds=0.01):
                pass
        release.set()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())

        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with resource_lease("test:release-on-error"):
                raise RuntimeError("test failure")
        with resource_lease("test:release-on-error", timeout_seconds=0.01):
            pass

    def test_desktop_process_lock_releases_even_if_os_mutex_cleanup_fails(self):
        kernel32 = mock.Mock()
        kernel32.ReleaseMutex.side_effect = RuntimeError("release failed")
        with mock.patch.object(
            tool_broker,
            "_desktop_mutex",
            return_value=(kernel32, object()),
        ):
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                with resource_lease("desktop:physical"):
                    pass

        acquired = threading.Event()

        def acquire_from_another_thread():
            with resource_lease("desktop:physical", timeout_seconds=0.1):
                acquired.set()

        with mock.patch.object(tool_broker, "_desktop_mutex", return_value=None):
            thread = threading.Thread(target=acquire_from_another_thread)
            thread.start()
            thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired.is_set())

    def test_browser_lease_key_ignores_path_and_query(self):
        self.assertEqual(
            browser_resource("https://example.org/a?token=ignored"),
            browser_resource("https://EXAMPLE.org/b"),
        )

    def test_partial_status_includes_missing_agents_and_failed_checks(self):
        status, evidence = orchestrator._run_status_evidence(
            [{"agent_id": 1, "status": "completed", "result": "usable result"}],
            expected_count=4,
            success=False,
            quality_checks={"peer_review_coverage": False},
        )

        self.assertEqual(status, "partial")
        self.assertEqual(evidence["missing_agent_ids"], [2, 3, 4])
        self.assertEqual(evidence["failed_quality_checks"], ["peer_review_coverage"])

    def _agent(self):
        patches = [
            mock.patch("core.swarm.LLMClient"),
            mock.patch("core.swarm.Memory"),
            mock.patch("core.swarm.SessionLogger"),
            mock.patch("core.swarm.close_session"),
            mock.patch("core.swarm.SkillResolver.record_outcome"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        swarm.Memory.return_value.search_similar.return_value = []
        return swarm.SwarmAgent(agent_id=1, max_steps=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
