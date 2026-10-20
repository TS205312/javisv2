"""Regression tests for failed-task recovery and truthful swarm status."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import swarm


class SwarmRecoveryTests(unittest.TestCase):
    def _agent(self, plan, max_steps=10):
        patches = [
            mock.patch("core.swarm.LLMClient"),
            mock.patch("core.swarm.Memory"),
            mock.patch("core.swarm.SessionLogger"),
            mock.patch("core.swarm.generate_plan", return_value=plan),
            mock.patch("core.swarm.close_session"),
            mock.patch("core.swarm.SkillResolver.record_outcome"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        swarm.Memory.return_value.search_similar.return_value = []
        return swarm.SwarmAgent(agent_id=1, max_steps=max_steps)

    def test_failed_search_is_not_repeated_after_provider_fallbacks(self):
        agent = self._agent({
            "tasks": [{
                "id": "research",
                "desc": "Find verified sources",
                "success": "At least one relevant source",
            }],
        })
        with (
            mock.patch("core.swarm.execute_task", return_value="irrelevant results") as execute,
            mock.patch(
                "core.swarm.evaluate_task",
                return_value={
                    "passed": False,
                    "reason": "The results are unrelated.",
                    "fix": "Use a more specific query.",
                },
            ),
        ):
            result = agent.run("Research AI in education")

        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["failed_tasks"]), 1)
        self.assertEqual(result["failed_tasks"][0]["attempts"], 1)
        self.assertIn("FAIL:", result["result"])

    def test_empty_plan_is_reported_as_plan_failure(self):
        agent = self._agent({"tasks": []})

        result = agent.run("Research AI in education")

        self.assertEqual(result["status"], "plan_failed")
        self.assertIn("không có task", result["error"])

    def test_unknown_tool_exception_is_not_replayed(self):
        agent = self._agent({
            "tasks": [{
                "id": "browse",
                "desc": "Open a source",
                "success": "Source is opened",
            }],
        })
        with (
            mock.patch(
                "core.swarm.execute_task",
                side_effect=RuntimeError("Chromium is missing"),
            ) as execute,
            mock.patch("core.swarm.evaluate_task") as evaluate,
        ):
            result = agent.run("Research AI in education")

        self.assertEqual(execute.call_count, 1)
        evaluate.assert_not_called()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["failed_tasks"]), 1)
        self.assertIn("Chromium is missing", result["failed_tasks"][0]["reason"])
        self.assertTrue(
            all("Chromium is missing" in action["output"] for action in result["actions"])
        )

    def test_retryable_navigation_exception_is_retried(self):
        agent = self._agent({
            "tasks": [{
                "id": "browse",
                "desc": "Open a source",
                "success": "Source is opened",
            }],
        })
        attempts = {"count": 0}

        def flaky_navigation(task, **kwargs):
            task["_selected_tool"] = "browser_tool"
            task["_selected_action"] = "goto"
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("Temporary navigation timeout")
            return "Page opened"

        with (
            mock.patch("core.swarm.execute_task", side_effect=flaky_navigation) as execute,
            mock.patch(
                "core.swarm.evaluate_task",
                return_value={"passed": True, "reason": "Page is accessible", "fix": ""},
            ),
        ):
            result = agent.run("Open a source")

        self.assertEqual(execute.call_count, 2)
        self.assertEqual(result["status"], "completed")
        agent.logger.log.assert_any_call("trial_recovery_decision", mock.ANY)
        agent.memory.search_similar.assert_called_once_with(
            "Open a source", success_only=True
        )
        agent.memory.save_long_term.assert_called_once()
        saved_experience = agent.memory.save_long_term.call_args
        self.assertIs(saved_experience.kwargs["success"], True)
        self.assertEqual(saved_experience.args[2]["attempts"], 2)
        self.assertIn("duration_seconds", saved_experience.args[2])

    def test_physical_desktop_plan_is_rejected_before_execution(self):
        agent = self._agent({
            "tasks": [{
                "id": "desktop",
                "desc": "Open Calculator and confirm its Ready label",
                "success": "Ready label is visible",
            }],
        })
        with mock.patch("core.swarm.execute_task") as execute:
            result = agent.run("Open Calculator and confirm its Ready label")

        execute.assert_not_called()
        self.assertEqual(result["status"], "plan_failed")
        self.assertIn("no tool permission within the goal scope", result["error"])

    def test_retry_loop_respects_agent_step_limit(self):
        agent = self._agent({
            "tasks": [{
                "id": "research",
                "desc": "Find verified sources",
                "success": "At least one relevant source",
            }],
        }, max_steps=1)
        with (
            mock.patch("core.swarm.execute_task", return_value="irrelevant") as execute,
            mock.patch(
                "core.swarm.evaluate_task",
                return_value={
                    "passed": False,
                    "reason": "Not relevant",
                    "fix": "Narrow query",
                },
            ),
        ):
            result = agent.run("Research")

        execute.assert_called_once()
        self.assertEqual(result["failed_tasks"][0]["attempts"], 1)

    def test_slide_draft_gets_one_targeted_repair_attempt(self):
        agent = self._agent({
            "tasks": [{
                "id": "create_slides",
                "type": "presentation_draft",
                "desc": "Soạn 6–8 slide có nguồn",
                "success": "Deck passes structure and evidence checks",
            }],
        })
        execute_outputs = iter(["incomplete draft", "repaired draft"])

        def draft_attempt(task, **kwargs):
            task["_selected_tool"] = "presentation_draft"
            task["_selected_action"] = "generate"
            return next(execute_outputs)

        with (
            mock.patch(
                "core.swarm.execute_task",
                side_effect=draft_attempt,
            ) as execute,
            mock.patch(
                "core.swarm.evaluate_task",
                side_effect=[
                    {"passed": False, "reason": "Missing image suggestions", "fix": "Add suggestions"},
                    {"passed": True, "reason": "Structure passes", "fix": ""},
                ],
            ) as evaluate,
        ):
            result = agent.run("Create a presentation")

        self.assertEqual(execute.call_count, 2)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(result["status"], "completed")
        self.assertIn("SỬA CÓ MỤC TIÊU", execute.call_args_list[1].args[0]["desc"])


    def test_slide_draft_appends_only_verified_references_before_critic(self):
        evidence = [
            {
                "title": f"Education study {index}",
                "url": f"https://example.org/{index}",
                "snippet": "AI education evidence",
                "source": "Example journal",
            }
            for index in range(3)
        ]
        agent = self._agent({
            "tasks": [{
                "id": "create_slides",
                "type": "presentation_draft",
                "desc": "Draft slides from evidence",
                "success": "References are present",
            }],
        }, max_steps=1)
        agent.shared_evidence = evidence

        with (
            mock.patch("core.swarm.execute_task", return_value="Draft without references"),
            mock.patch(
                "core.swarm.evaluate_task",
                return_value={"passed": True, "reason": "Pass", "fix": ""},
            ) as evaluate,
        ):
            result = agent.run("Create a presentation")

        self.assertEqual(result["status"], "completed")
        evaluated_draft = evaluate.call_args.args[1]
        self.assertIn("## Tài liệu tham khảo", evaluated_draft)
        self.assertTrue(all(source["url"] in evaluated_draft for source in evidence))
        agent.logger.log.assert_any_call(
            "presentation_references_repaired", mock.ANY
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
