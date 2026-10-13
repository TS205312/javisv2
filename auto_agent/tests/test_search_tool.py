"""Regression tests for DuckDuckGo and Bing search fallback behavior."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import search_tool


class SearchToolTests(unittest.TestCase):
    def test_bing_fallback_runs_when_duckduckgo_returns_no_results(self):
        expected = [{"title": "AI paper", "url": "https://example.com"}]
        with (
            mock.patch.object(search_tool, "DDGS") as ddgs,
            mock.patch.object(search_tool, "_bing_search", return_value=expected) as bing,
        ):
            ddgs.return_value.text.return_value = []
            self.assertEqual(search_tool.search({"query": "AI trends"}), expected)
        bing.assert_called_once_with("AI trends")

    def test_search_failure_returns_empty_list(self):
        with (
            mock.patch.object(search_tool, "DDGS", side_effect=RuntimeError("blocked")),
            mock.patch.object(
                search_tool, "_bing_search", side_effect=search_tool.requests.Timeout
            ),
        ):
            self.assertEqual(search_tool.search({"query": "AI trends"}), [])

    def test_bing_parser_extracts_search_result(self):
        parser = search_tool._BingResultsParser()
        parser.feed(
            '<li class="b_algo"><h2><a href="https://example.com">AI Paper</a>'
            '</h2><p>Paper summary.</p></li>'
        )
        self.assertEqual(
            parser.results,
            [{
                "url": "https://example.com",
                "title": "AI Paper",
                "snippet": "Paper summary.",
            }],
        )

    def test_academic_query_uses_arxiv_results(self):
        expected = [{
            "title": "AI paper",
            "url": "https://arxiv.org/abs/1234.5678",
            "snippet": "Paper summary",
            "published": "2026-01-01T00:00:00Z",
            "source": "arXiv",
        }]
        with (
            mock.patch.object(search_tool, "_arxiv_search", return_value=expected),
            mock.patch.object(search_tool, "DDGS") as ddgs,
        ):
            self.assertEqual(
                search_tool.search({"query": "recent AI research papers"}),
                expected,
            )
        ddgs.assert_not_called()

    def test_artificial_intelligence_query_uses_arxiv(self):
        expected = [{
            "title": "AI paper",
            "url": "https://arxiv.org/abs/1234.5678",
            "snippet": "Paper summary",
            "published": "2026-01-01T00:00:00Z",
            "source": "arXiv",
        }]
        with (
            mock.patch.object(search_tool, "_arxiv_search", return_value=expected),
            mock.patch.object(search_tool, "DDGS") as ddgs,
        ):
            self.assertEqual(
                search_tool.search({"query": "Artificial Intelligence papers"}),
                expected,
            )
        ddgs.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
