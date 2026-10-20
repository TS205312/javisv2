"""Tests that the default swarm actually starts four independent agents."""
import asyncio
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import orchestrator


class OrchestratorSwarmTests(unittest.TestCase):
    @staticmethod
    def _sources():
        return [{
            "title": f"AI education source {index}",
            "url": f"https://example.org/{index}",
            "snippet": "AI in education source summary.",
            "source": "example.org",
        } for index in range(3)]

    def test_default_swarm_runs_four_agents_and_delegates_debate(self):
        agents = []
        barrier = threading.Barrier(4)

        class FakeAgent:
            def __init__(self, agent_id, model_name, **kwargs):
                self.agent_id = agent_id
                self.model_name = model_name
                self.role = kwargs["role"]
                self.logger = mock.Mock()
                agents.append(self)

            def run(self, goal):
                barrier.wait(timeout=2)
                return {
                    "agent_id": self.agent_id,
                    "model_used": self.model_name,
                    "status": "completed",
                    "result": f"Result for {goal}",
                    "elapsed": 0,
                }

        aggregator = mock.Mock()
        aggregator.aggregate.return_value = {
            "winner": "merged",
            "final_result": "Leader result",
        }
        with (
            mock.patch.object(orchestrator, "SwarmAgent", FakeAgent),
            mock.patch.object(orchestrator, "Aggregator", return_value=aggregator),
            mock.patch.object(orchestrator, "orch_logger"),
            mock.patch.object(orchestrator, "call_llm", return_value="OK") as api_call,
            mock.patch.object(orchestrator, "SWARM_AGENT_MODELS", ()),
            mock.patch.object(orchestrator, "GEMINI_API_KEY", "configured"),
            mock.patch.object(orchestrator, "GEMINI_MODEL", "gemini-test"),
            mock.patch.object(orchestrator, "GROQ_API_KEY", ""),
            mock.patch.object(orchestrator, "OLLAMA_MODEL", ""),
            mock.patch.object(orchestrator, "ollama_installed_models", return_value=[]),
            mock.patch.object(orchestrator, "KILO_API_KEY", "configured"),
            mock.patch.object(orchestrator, "KILO_MODEL", "kilo-auto/free"),
            mock.patch.object(orchestrator, "OPENROUTER_API_KEY", "configured"),
            mock.patch.object(orchestrator, "OPENROUTER_PRIMARY_MODEL", "or-a:free"),
            mock.patch.object(orchestrator, "OPENROUTER_FALLBACK_MODEL", "or-b:free"),
            mock.patch.object(
                orchestrator, "OPENROUTER_FREE_POOL", ("or-a:free", "or-b:free")
            ),
        ):
            result = asyncio.run(orchestrator.run_swarm_async(
                "test goal",
                max_steps=1,
            ))

        self.assertEqual(len(agents), 4)
        self.assertEqual(
            {agent.agent_id for agent in agents},
            {1, 2, 3, 4},
        )
        self.assertEqual(
            [agent.model_name for agent in agents],
            [
                "kilo:kilo-auto/free",
                "gemini:gemini-test",
                "openrouter:or-a:free",
                "openrouter:or-b:free",
            ],
        )
        self.assertEqual(
            len({agent.model_name for agent in agents}),
            4,
        )
        self.assertEqual(aggregator.aggregate.call_args.args[0], result["agent_results"])
        self.assertEqual(aggregator.aggregate.call_args.args[1], "test goal")
        self.assertEqual(result["aggregator"]["final_result"], "Leader result")
        self.assertEqual(
            len({agent.role for agent in agents}),
            4,
        )
        self.assertEqual(api_call.call_count, 4)
        self.assertTrue(all(
            call.kwargs["allow_fallback"] is False
            and call.kwargs["max_output_tokens"] == 8
            for call in api_call.call_args_list
        ))
        self.assertTrue(result["aggregator"]["quality_checks"]["model_apis_healthy"])

    def test_single_configured_model_is_not_repeated_for_four_agents(self):
        agents = []

        class FakeAgent:
            def __init__(self, agent_id, model_name, **kwargs):
                self.agent_id = agent_id
                self.model_name = model_name
                self.logger = mock.Mock()
                agents.append(self)

            def run(self, goal):
                return {
                    "agent_id": self.agent_id,
                    "model_used": self.model_name,
                    "status": "completed",
                    "result": f"Result for {goal}",
                    "elapsed": 0,
                }

        aggregator = mock.Mock()
        aggregator.aggregate.return_value = {
            "winner": "merged",
            "final_result": "Leader result",
            "peer_review_count": 4,
        }
        with (
            mock.patch.object(orchestrator, "SwarmAgent", FakeAgent),
            mock.patch.object(orchestrator, "Aggregator", return_value=aggregator),
            mock.patch.object(orchestrator, "orch_logger"),
            mock.patch.object(orchestrator, "SWARM_AGENT_MODELS", ()),
            mock.patch.object(orchestrator, "GEMINI_API_KEY", "configured"),
            mock.patch.object(orchestrator, "GEMINI_MODEL", "gemini-test"),
            mock.patch.object(orchestrator, "GROQ_API_KEY", ""),
            mock.patch.object(orchestrator, "OLLAMA_MODEL", ""),
            mock.patch.object(orchestrator, "ollama_installed_models", return_value=[]),
            mock.patch.object(orchestrator, "KILO_API_KEY", ""),
            mock.patch.object(orchestrator, "OPENROUTER_API_KEY", ""),
        ):
            result = asyncio.run(orchestrator.run_swarm_async("test goal"))

        self.assertEqual(agents, [])
        self.assertEqual(result["agent_count_completed"], 0)
        self.assertFalse(result["aggregator"]["success"])
        self.assertFalse(
            result["aggregator"]["quality_checks"]["distinct_models_assigned"]
        )
        self.assertIn("chỉ có 1 model", result["aggregator"]["reason"])

    def test_one_successful_agent_does_not_mark_swarm_successful(self):
        class FakeAgent:
            def __init__(self, agent_id, model_name, **kwargs):
                self.agent_id = agent_id
                self.logger = mock.Mock()

            def run(self, goal):
                return {
                    "agent_id": self.agent_id,
                    "status": "completed" if self.agent_id == 1 else "plan_failed",
                    "result": "Only one result" if self.agent_id == 1 else None,
                    "failed_tasks": [],
                }

        aggregator = mock.Mock()
        aggregator.aggregate.return_value = {
            "winner": 1,
            "final_result": "Leader result",
            "peer_review_count": 1,
        }
        with (
            mock.patch.object(orchestrator, "SwarmAgent", FakeAgent),
            mock.patch.object(orchestrator, "Aggregator", return_value=aggregator),
            mock.patch.object(orchestrator, "orch_logger"),
            mock.patch.object(orchestrator, "call_llm", return_value="OK"),
            mock.patch.object(orchestrator, "GEMINI_API_KEY", ""),
            mock.patch.object(
                orchestrator,
                "SWARM_AGENT_MODELS",
                ("kilo:a/free", "gemini:b", "openrouter:c:free", "openrouter:d:free"),
            ),
        ):
            result = asyncio.run(orchestrator.run_swarm_async("test goal"))

        self.assertEqual(result["agent_count_completed"], 1)
        self.assertFalse(result["aggregator"]["success"])
        self.assertFalse(result["aggregator"]["quality_checks"]["all_agents_completed"])

    def test_presentation_preflight_runs_once_and_shares_evidence(self):
        agents = []

        class FakeAgent:
            def __init__(self, agent_id, model_name, **kwargs):
                self.agent_id = agent_id
                self.model_name = model_name
                self.role = kwargs["role"]
                self.evidence = kwargs["shared_evidence"]
                self.logger = mock.Mock()
                agents.append(self)

            def run(self, goal):
                return {
                    "agent_id": self.agent_id,
                    "model_used": self.model_name,
                    "status": "completed",
                    "result": f"Result for {goal}",
                    "elapsed": 0,
                }

        search_results = orchestrator.search_tool.SearchResults(
            self._sources(),
            status="ok",
            provider="mock",
            query="AI education",
        )
        aggregator = mock.Mock()
        aggregator.aggregate.return_value = {
            "winner": "merged",
            "final_result": "Not a full slide deck",
            "peer_review_count": 4,
        }
        with (
            mock.patch.object(orchestrator, "search_tool") as search_tool,
            mock.patch.object(orchestrator, "SwarmAgent", FakeAgent),
            mock.patch.object(orchestrator, "Aggregator", return_value=aggregator),
            mock.patch.object(orchestrator, "orch_logger"),
            mock.patch.object(orchestrator, "call_llm", return_value="OK"),
            mock.patch.object(
                orchestrator,
                "SWARM_AGENT_MODELS",
                ("kilo:a/free", "gemini:b", "openrouter:c:free", "openrouter:d:free"),
            ),
            mock.patch.object(
                orchestrator,
                "_save_presentation_draft",
                return_value={"saved": False, "reason": "test"},
            ),
        ):
            search_tool.search.return_value = search_results
            result = asyncio.run(orchestrator.run_swarm_async(
                "Nghiên cứu ‘AI trong giáo dục’ và tạo bài Canva",
                max_steps=1,
            ))

        search_tool.search.assert_called_once_with({
            "query": "AI trong giáo dục",
            "min_results": 3,
        })
        self.assertEqual(len(agents), 4)
        self.assertTrue(all(agent.evidence == self._sources() for agent in agents))
        self.assertEqual(result["agent_count_expected"], 4)

    def test_leader_selected_agent_owns_final_canva_preflight(self):
        agents = []

        class FakeAgent:
            def __init__(self, agent_id, model_name, **kwargs):
                self.agent_id = agent_id
                self.model_name = model_name
                self.logger = mock.Mock()
                agents.append(self)

            def run(self, goal):
                return {
                    "agent_id": self.agent_id,
                    "model_used": self.model_name,
                    "status": "completed",
                    "result": f"Result for {goal}",
                    "failed_tasks": [],
                    "research_evidence": self._sources,
                    "browser_session": {
                        "agent_id": self.agent_id,
                        "session_id": None,
                        "status": "not_started",
                        "isolated_context": True,
                    },
                }

        FakeAgent._sources = self._sources()
        search_results = orchestrator.search_tool.SearchResults(
            self._sources(), status="ok", provider="mock", query="AI education"
        )
        aggregator = mock.Mock()
        aggregator.aggregate.return_value = {
            "winner": "merged",
            "final_executor_agent_id": 3,
            "final_result": "Leader draft",
            "peer_review_count": 4,
        }
        with (
            mock.patch.object(orchestrator, "search_tool") as search_tool,
            mock.patch.object(orchestrator, "SwarmAgent", FakeAgent),
            mock.patch.object(orchestrator, "Aggregator", return_value=aggregator),
            mock.patch.object(orchestrator, "orch_logger"),
            mock.patch.object(orchestrator, "call_llm", return_value="OK"),
            mock.patch.object(
                orchestrator,
                "SWARM_AGENT_MODELS",
                ("kilo:a/free", "gemini:b", "openrouter:c:free", "openrouter:d:free"),
            ),
            mock.patch.object(
                orchestrator,
                "_preflight_canva_for_agent",
                return_value={
                    "agent_id": 3,
                    "session_id": "agent-session-3",
                    "status": "login_required",
                    "isolated_context": True,
                    "design_created": False,
                },
            ) as canva_preflight,
            mock.patch.object(
                orchestrator,
                "_close_agent_browser_session",
            ),
            mock.patch.object(
                orchestrator,
                "_save_presentation_draft",
                return_value={"saved": False, "reason": "invalid draft"},
            ),
        ):
            search_tool.search.return_value = search_results
            result = asyncio.run(orchestrator.run_swarm_async(
                "Nghiên cứu AI trong giáo dục và tạo bài Canva",
                max_steps=1,
            ))

        canva_preflight.assert_called_once()
        self.assertEqual(canva_preflight.call_args.args[0], 3)
        self.assertIsNotNone(canva_preflight.call_args.kwargs["executor"])
        self.assertIs(canva_preflight.call_args.kwargs["logger"], agents[2].logger)
        self.assertTrue(result["aggregator"]["browser_session_coverage"]["complete"])
        self.assertEqual(result["aggregator"]["browser_session_coverage"]["expected"], 1)
        self.assertEqual(result["aggregator"]["browser_session_coverage"]["ready"], 1)
        self.assertEqual(
            result["aggregator"]["browser_session_coverage"]["final_executor_agent_id"],
            3,
        )

    def test_failed_presentation_preflight_skips_model_swarm(self):
        search_results = orchestrator.search_tool.SearchResults(
            [],
            status="irrelevant_results",
            message="off topic",
            provider="mock",
            query="AI education",
        )
        with (
            mock.patch.object(orchestrator, "search_tool") as search_tool,
            mock.patch.object(orchestrator, "SwarmAgent") as agent_factory,
            mock.patch.object(orchestrator, "orch_logger"),
            mock.patch.object(orchestrator, "call_llm", return_value="OK"),
            mock.patch.object(
                orchestrator,
                "SWARM_AGENT_MODELS",
                ("kilo:a/free", "gemini:b", "openrouter:c:free", "openrouter:d:free"),
            ),
        ):
            search_tool.search.return_value = search_results
            result = asyncio.run(orchestrator.run_swarm_async(
                "Nghiên cứu ‘AI trong giáo dục’ và tạo bài Canva",
                max_steps=1,
            ))

        agent_factory.assert_not_called()
        self.assertEqual(result["agent_count_completed"], 0)
        self.assertFalse(result["aggregator"]["success"])
        self.assertFalse(result["aggregator"]["presentation_output"]["saved"])
        self.assertIn("không ghi đè", result["aggregator"]["presentation_output"]["reason"])


    def test_api_preflight_uses_a_distinct_healthy_backup_when_primary_fails(self):
        models = [
            "kilo:a/free",
            "gemini:b",
            "openrouter:c:free",
            "openrouter:d:free",
        ]
        configured_models = (*models, "openrouter:e:free")

        def api_call(prompt, model, **kwargs):
            if model == "kilo:a/free":
                raise RuntimeError("429 rate limit")
            return "OK"

        with (
            mock.patch.object(orchestrator, "SWARM_AGENT_MODELS", configured_models),
            mock.patch.object(orchestrator, "KILO_API_KEY", ""),
            mock.patch.object(orchestrator, "GEMINI_API_KEY", ""),
            mock.patch.object(orchestrator, "OPENROUTER_API_KEY", ""),
            mock.patch.object(orchestrator, "call_llm", side_effect=api_call) as llm_call,
            mock.patch.object(orchestrator, "orch_logger"),
        ):
            assignments, checks, error = orchestrator._preflight_swarm_models(models)

        self.assertIsNone(error)
        self.assertEqual(assignments[0], "openrouter:e:free")
        self.assertEqual(len({model.casefold() for model in assignments}), 4)
        self.assertEqual(len(checks), 5)
        self.assertEqual(llm_call.call_count, 5)

    def test_api_preflight_uses_configured_groq_models_as_free_backups(self):
        models = [
            "kilo:kilo-a/free",
            "gemini:gemini-b",
            "openrouter:or-c:free",
            "openrouter:or-d:free",
        ]

        def api_call(prompt, model, **kwargs):
            if model in {
                "kilo:kilo-a/free",
                "openrouter:or-c:free",
                "openrouter:or-d:free",
            }:
                raise RuntimeError("provider unavailable")
            return "OK"

        with (
            mock.patch.object(orchestrator, "SWARM_AGENT_MODELS", tuple(models)),
            mock.patch.object(orchestrator, "KILO_API_KEY", "configured"),
            mock.patch.object(orchestrator, "KILO_MODEL", "kilo-a/free"),
            mock.patch.object(orchestrator, "GEMINI_API_KEY", "configured"),
            mock.patch.object(orchestrator, "GEMINI_MODEL", "gemini-b"),
            mock.patch.object(orchestrator, "GROQ_API_KEY", "configured"),
            mock.patch.object(orchestrator, "GROQ_MODEL", "qwen/qwen3.8-27b"),
            mock.patch.object(orchestrator, "GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"),
            mock.patch.object(orchestrator, "OLLAMA_MODEL", ""),
            mock.patch.object(orchestrator, "ollama_installed_models", return_value=[]),
            mock.patch.object(orchestrator, "OPENROUTER_API_KEY", "configured"),
            mock.patch.object(orchestrator, "OPENROUTER_PRIMARY_MODEL", "or-c:free"),
            mock.patch.object(orchestrator, "OPENROUTER_FALLBACK_MODEL", "or-d:free"),
            mock.patch.object(orchestrator, "OPENROUTER_FREE_POOL", ("poolside:free",)),
            mock.patch.object(orchestrator, "call_llm", side_effect=api_call),
            mock.patch.object(orchestrator, "orch_logger"),
        ):
            assignments, checks, error = orchestrator._preflight_swarm_models(models)

        self.assertIsNone(error)
        self.assertEqual(
            assignments,
            [
                "groq:qwen/qwen3.8-27b",
                "gemini:gemini-b",
                "groq:openai/gpt-oss-20b",
                "openrouter:poolside:free",
            ],
        )
        self.assertEqual(len({model.casefold() for model in assignments}), 4)
        self.assertEqual(len(checks), 7)

    def test_api_preflight_failure_prevents_agents_from_starting(self):
        models = (
            "kilo:a/free",
            "gemini:b",
            "openrouter:c:free",
            "openrouter:d:free",
        )
        with (
            mock.patch.object(orchestrator, "SWARM_AGENT_MODELS", models),
            mock.patch.object(orchestrator, "KILO_API_KEY", ""),
            mock.patch.object(orchestrator, "GEMINI_API_KEY", ""),
            mock.patch.object(orchestrator, "OPENROUTER_API_KEY", ""),
            mock.patch.object(
                orchestrator,
                "_probe_model_api",
                side_effect=lambda model: {
                    "model": model,
                    "ok": False,
                    "error": "connection unavailable",
                },
            ),
            mock.patch.object(orchestrator, "SwarmAgent") as agent_factory,
            mock.patch.object(orchestrator, "orch_logger"),
        ):
            result = asyncio.run(orchestrator.run_swarm_async("test goal"))

        agent_factory.assert_not_called()
        self.assertFalse(result["aggregator"]["success"])
        self.assertFalse(result["aggregator"]["quality_checks"]["model_apis_healthy"])
        self.assertEqual(result["agent_count_completed"], 0)

    def test_preflight_sources_are_available_to_final_source_gate(self):
        evidence = self._sources()

        verified = orchestrator._verified_search_urls([], evidence)

        self.assertEqual(verified, {row["url"] for row in evidence})


if __name__ == "__main__":
    unittest.main(verbosity=2)
