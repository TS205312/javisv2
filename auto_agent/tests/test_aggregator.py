"""Regression tests for partial and failed swarm results."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.aggregator import Aggregator


class AggregatorPartialResultTests(unittest.TestCase):
    def setUp(self):
        logger_patcher = mock.patch("core.aggregator.SessionLogger")
        self.addCleanup(logger_patcher.stop)
        self.logger_class = logger_patcher.start()
        llm_patcher = mock.patch("core.aggregator.LLMClient")
        self.addCleanup(llm_patcher.stop)
        self.llm_class = llm_patcher.start()
        self.aggregator = Aggregator(expected_count=4)

    def test_final_executor_uses_leader_selected_valid_agent_id(self):
        selected, reason = Aggregator._select_final_executor(
            {"winner": "merged", "final_executor_agent_id": 3},
            [
                {"agent_id": 1, "status": "completed", "result": "A"},
                {"agent_id": 3, "status": "partial", "result": "B"},
            ],
        )

        self.assertEqual(selected, 3)
        self.assertIn("leader", reason)

    def test_final_executor_falls_back_to_best_evidence_and_completion(self):
        selected, reason = Aggregator._select_final_executor(
            {"winner": "unknown"},
            [
                {
                    "agent_id": 1,
                    "status": "partial",
                    "result": "A",
                    "failed_tasks": [{"id": "x"}],
                    "research_evidence": [],
                },
                {
                    "agent_id": 2,
                    "status": "completed",
                    "result": "B",
                    "failed_tasks": [],
                    "research_evidence": [{"url": "https://example.org"}],
                },
            ],
        )

        self.assertEqual(selected, 2)
        self.assertIn("evidence coverage", reason)

    def test_two_partial_results_are_aggregated(self):
        self.aggregator.llm.ask.return_value = {
            "winner": "merged",
            "final_result": "Result from the two surviving agents.",
            "reason": "Partial swarm",
            "confidence": 0.6,
            "best_parts": [],
        }
        results = [
            {"status": "partial", "result": "Agent one output"},
            {"status": "crashed", "result": None},
            {"status": "partial", "result": "Agent three output"},
            None,
        ]

        final = self.aggregator.aggregate(results, "test goal")

        self.assertEqual(final["final_result"], "Result from the two surviving agents.")
        self.aggregator.llm.ask.assert_called_once()

    def test_all_agents_failed_returns_empty_winner(self):
        results = [
            {"status": "crashed", "result": None},
            {"status": "failed", "result": ""},
            None,
            {"status": "partial", "result": "FAIL: task failed"},
        ]

        final = self.aggregator.aggregate(results, "test goal")

        self.assertIsNone(final["winner"])
        self.assertEqual(final["reason"], "Tất cả agent fail")
        self.assertEqual(final["confidence"], 0)
        self.aggregator.llm.ask.assert_not_called()

    def test_single_valid_agent_is_preserved_but_not_counted_as_swarm_success(self):
        self.aggregator.llm.ask.return_value = {
            "winner": 1,
            "final_result": "Leader-reviewed useful result.",
            "reason": "Reviewed one result.",
            "confidence": 0.9,
            "best_parts": [],
        }
        final = self.aggregator.aggregate(
            [{"agent_id": 1, "status": "completed", "result": "Useful result"}],
            "test goal",
        )

        self.assertEqual(final["final_result"], "Useful result")
        self.assertEqual(final["confidence"], 0.25)
        self.assertEqual(final["expected_peer_review_count"], 4)
        self.assertTrue(final["needs_review"])
        self.assertTrue(final["leader_skipped"])
        self.assertIn("swarm is incomplete", final["reason"])
        self.aggregator.llm.ask.assert_not_called()

    def test_debate_collects_peer_reviews_before_leader_conclusion(self):
        leader = mock.Mock()
        leader.ask.side_effect = [
            {
                "winner": "merged",
                "final_result": "Leader result without sources.",
                "reason": "Initial draft.",
                "confidence": 0.85,
                "best_parts": [],
            },
            {
                "winner": "merged",
                "final_result": (
                    "Leader-reviewed result. Sources: "
                    "https://example.org/a https://example.org/b "
                    "https://example.org/c"
                ),
                "reason": "Compared evidence and resolved disagreements.",
                "confidence": 0.85,
                "best_parts": [],
            },
        ]
        reviewer_one = mock.Mock()
        reviewer_one.ask.return_value = "Agent 2: unsupported claim; verify the source."
        reviewer_two = mock.Mock()
        reviewer_two.ask.return_value = "Agent 1: source-backed and relevant."
        with mock.patch(
            "core.aggregator.LLMClient",
            side_effect=[leader, reviewer_one, reviewer_two],
        ) as clients:
            aggregator = Aggregator(strategy="debate", expected_count=2)
            final = aggregator.aggregate(
                [
                    {
                        "agent_id": 1,
                        "model_used": "groq:model-a",
                        "status": "completed",
                        "result": (
                            "Source-backed result https://example.org/a "
                            "https://example.org/b https://example.org/c"
                        ),
                    },
                    {
                        "agent_id": 2,
                        "model_used": "kilo:model-b",
                        "status": "partial",
                        "result": "Unsupported claim https://example.org/a",
                        "failed_tasks": [{"task_id": "t2"}],
                    },
                ],
                "Research a topic",
            )

        self.assertEqual(clients.call_count, 3)
        self.assertEqual(reviewer_one.ask.call_count, 1)
        self.assertEqual(reviewer_two.ask.call_count, 1)
        leader_prompt = leader.ask.call_args_list[0].args[0]
        self.assertEqual(
            leader.ask.call_args_list[0].kwargs["max_output_tokens"], 6144
        )
        self.assertIn("Phản biện chéo thực tế", leader_prompt)
        self.assertIn("Unsupported claim", leader_prompt)
        self.assertEqual(final["peer_review_count"], 2)
        self.assertEqual(len(final["peer_reviews"]), 2)
        self.assertEqual(leader.ask.call_count, 2)
        self.assertIn("https://example.org/c", final["final_result"])

    def test_vote_strategy_is_routed_through_evidence_debate(self):
        aggregator = Aggregator(strategy="vote", expected_count=2)
        with (
            mock.patch.object(
                aggregator,
                "_debate",
                return_value={
                    "winner": "merged",
                    "final_result": "Evidence-reviewed result",
                    "confidence": 0.7,
                },
            ) as debate,
            mock.patch.object(aggregator, "_vote") as vote,
        ):
            final = aggregator.aggregate(
                [
                    {"agent_id": 1, "status": "completed", "result": "A"},
                    {"agent_id": 2, "status": "completed", "result": "B"},
                ],
                "Compare results",
            )

        debate.assert_called_once()
        vote.assert_not_called()
        self.assertEqual(final["final_result"], "Evidence-reviewed result")

    def test_source_free_research_results_are_excluded(self):
        final = self.aggregator.aggregate(
            [
                {
                    "agent_id": 1,
                    "status": "completed",
                    "result": "Unsupported claim with no URL",
                },
            ],
            "Research AI in schools",
        )

        self.assertIsNone(final["winner"])
        self.assertEqual(final["valid_agent_count"], 0)
        self.aggregator.llm.ask.assert_not_called()

    def test_failed_leader_does_not_look_like_a_verified_conclusion(self):
        leader = mock.Mock()
        leader.ask.return_value = {"error": "quota exhausted"}
        reviewers = [mock.Mock(), mock.Mock()]
        for reviewer in reviewers:
            reviewer.ask.return_value = "Review completed."

        with mock.patch(
            "core.aggregator.LLMClient",
            side_effect=[leader, *reviewers],
        ):
            aggregator = Aggregator(strategy="debate")
            final = aggregator.aggregate(
                [
                    {"agent_id": 1, "status": "completed", "result": "Result A"},
                    {"agent_id": 2, "status": "completed", "result": "Result B"},
                ],
                "Compare results",
            )

        self.assertTrue(final["leader_failed"])
        self.assertTrue(final["needs_review"])
        self.assertIn("chưa qua đối chiếu", final["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
