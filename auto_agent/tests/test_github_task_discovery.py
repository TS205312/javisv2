"""Tests for safe, cached, discovery-only GitHub skill lookup per task."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import skill_resolver
from core.skill_resolver import SkillResolver


class GitHubTaskDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.resolver = SkillResolver(memory=mock.Mock())
        with skill_resolver._DISCOVERY_CACHE_LOCK:
            skill_resolver._DISCOVERY_CACHE.clear()

    def test_selects_best_filtered_repo_without_installing_or_importing(self):
        candidates = [
            {
                "full_name": "sample/solid-tool",
                "url": "https://github.com/sample/solid-tool",
                "stars": 900,
                "license": "MIT",
                "last_updated": "2026-01-01T00:00:00+00:00",
                "description": "Automation helper",
            },
            {
                "full_name": "sample/other-tool",
                "url": "https://github.com/sample/other-tool",
                "stars": 400,
                "license": "Apache-2.0",
                "last_updated": "2026-03-01T00:00:00+00:00",
                "description": "Related helper",
            },
        ]
        with (
            mock.patch.object(
                self.resolver,
                "_github_candidates",
                return_value=('"web automation" language:python stars:>100', candidates),
            ) as search,
            mock.patch.object(
                self.resolver,
                "_evaluate_candidates",
                return_value=candidates,
            ),
            mock.patch.object(self.resolver, "_install_candidate") as install,
        ):
            result = self.resolver.discover_best("Create a web automation task")

        self.assertEqual(result["status"], "recommended_not_installed")
        self.assertEqual(result["skill"], "github:sample/solid-tool")
        self.assertEqual(result["stars"], 900)
        search.assert_called_once()
        install.assert_not_called()

    def test_reuses_same_safe_topic_cache_for_parallel_agent_tasks(self):
        candidate = {
            "full_name": "sample/browser-helper",
            "url": "https://github.com/sample/browser-helper",
            "stars": 500,
            "license": "MIT",
            "last_updated": "2026-04-01T00:00:00+00:00",
            "description": "Browser automation",
        }
        with (
            mock.patch.object(
                self.resolver,
                "_github_candidates",
                return_value=("query", [candidate]),
            ) as search,
            mock.patch.object(self.resolver, "_evaluate_candidates", return_value=[candidate]),
        ):
            first = self.resolver.discover_best("Create browser automation")
            second = self.resolver.discover_best("CREATE browser automation")

        self.assertEqual(first, second)
        search.assert_called_once()

    def test_skips_search_when_task_contains_credential_like_data(self):
        with mock.patch.object(self.resolver, "_github_candidates") as search:
            result = self.resolver.discover_best(
                "Use api_key=private-secret to build a browser automation task"
            )

        self.assertEqual(result["status"], "skipped_sensitive")
        search.assert_not_called()

    def test_sensitive_task_keywords_are_not_sent_to_keyword_model(self):
        with mock.patch.object(skill_resolver, "ask_llm") as ask_llm:
            keywords = skill_resolver._extract_keywords(
                "Use api_key=private-secret to build a browser automation task"
            )

        self.assertEqual(keywords, [])
        ask_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
