"""Regression checks for malformed GitHub search results and URLs."""
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from github import GithubException

from core.skill_resolver import SkillResolver


class GitHubHunterBadInputTests(unittest.TestCase):
    def setUp(self):
        self.resolver = SkillResolver(memory=mock.Mock())
        self.resolver.last_keywords = ["video", "image"]

    def test_github_hunter_bad_input(self):
        with self.assertLogs("core.skill_resolver", level=logging.WARNING):
            self.assertIsNone(
                self.resolver._install_candidate(
                    {"url": "", "clone_url": "", "full_name": ""}
                )
            )

        with mock.patch("core.skill_resolver.Github") as github:
            github.return_value.search_repositories.return_value = {}
            with self.assertLogs("core.skill_resolver", level=logging.WARNING):
                self.assertIsNone(self.resolver.hunt_best("tạo video từ ảnh"))

        for status in (403, 429):
            with self.subTest(status=status):
                error = GithubException(status, {"message": "API error"}, {})
                with (
                    mock.patch("core.skill_resolver.Github") as github,
                    self.assertLogs("core.skill_resolver", level=logging.WARNING),
                ):
                    github.return_value.search_repositories.side_effect = error
                    self.assertIsNone(self.resolver.hunt_best("tạo video từ ảnh"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
