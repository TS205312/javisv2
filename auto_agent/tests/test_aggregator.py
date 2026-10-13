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

    def test_single_result_uses_low_confidence_fallback(self):
        final = self.aggregator.aggregate(
            [{"status": "completed", "result": "Useful result"}],
            "test goal",
        )

        self.assertEqual(final["final_result"], "Useful result")
        self.assertEqual(final["confidence"], 0.3)
        self.aggregator.llm.ask.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
