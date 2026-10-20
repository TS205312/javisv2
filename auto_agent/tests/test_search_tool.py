"""Regression tests for DuckDuckGo and Bing search fallback behavior."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import search_tool


class SearchToolTests(unittest.TestCase):
    def test_bing_fallback_runs_when_duckduckgo_returns_no_results(self):
        expected = [{
            "title": "AI paper",
            "url": "https://example.com",
            "snippet": "An overview of AI trends.",
            "source": "example.com",
        }]
        with (
            mock.patch.object(search_tool, "DDGS") as ddgs,
            mock.patch.object(search_tool, "_bing_search", return_value=expected) as bing,
        ):
            ddgs.return_value.text.return_value = []
            self.assertEqual(search_tool.search({"query": "AI trends"}), expected)
        bing.assert_called_once_with("AI trends")

    def test_search_failure_reports_access_error(self):
        with (
            mock.patch.object(search_tool, "DDGS", side_effect=RuntimeError("blocked")),
            mock.patch.object(
                search_tool, "_bing_search", side_effect=search_tool.requests.Timeout
            ),
            mock.patch.object(
                search_tool, "_openalex_search", side_effect=search_tool.requests.Timeout
            ),
        ):
            result = search_tool.search({"query": "AI trends"})
        self.assertEqual(result.status, "access_error")
        self.assertFalse(result)

    def test_openalex_is_used_first_for_ai_education_queries(self):
        irrelevant = [{
            "title": "Lottery results",
            "url": "https://example.org/lottery",
            "snippet": "Latest lottery numbers.",
        }]
        expected = [{
            "title": f"Artificial intelligence education study {index}",
            "url": f"https://doi.org/10.0000/example.{index}",
            "snippet": "Artificial intelligence can support education and learning.",
            "source": "Journal of Education",
        } for index in range(3)]
        with (
            mock.patch.object(search_tool, "DDGS") as ddgs,
            mock.patch.object(search_tool, "_bing_search", return_value=irrelevant),
            mock.patch.object(search_tool, "_openalex_search", return_value=expected) as openalex,
        ):
            ddgs.return_value.text.return_value = []
            results = search_tool.search({"query": "AI in education", "min_results": 3})

        self.assertEqual(results.status, "ok")
        self.assertEqual(results.provider, "openalex")
        self.assertEqual(len(results), 3)
        openalex.assert_called_once()

    def test_openalex_reconstructs_abstract_and_metadata(self):
        response = mock.Mock()
        response.json.return_value = {
            "results": [{
                "title": "AI education study",
                "doi": "https://doi.org/10.0000/test",
                "publication_date": "2024-01-01",
                "primary_location": {
                    "source": {"display_name": "Education Journal"},
                },
                "abstract_inverted_index": {
                    "Artificial": [0],
                    "intelligence": [1],
                    "supports": [2],
                    "education.": [3],
                },
            }],
        }
        response.raise_for_status.return_value = None
        with mock.patch.object(search_tool.requests, "get", return_value=response):
            results = search_tool._openalex_search("AI in education")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["snippet"], "Artificial intelligence supports education.")
        self.assertEqual(results[0]["source"], "Education Journal")
        self.assertEqual(results[0]["published"], "2024-01-01")

    def test_openalex_rejects_cooccurring_but_off_topic_education_result(self):
        result = {
            "title": "Artificial Intelligence, Education, and Entrepreneurship",
            "url": "https://doi.org/10.0000/off-topic",
            "snippet": (
                "We document an unprecedented brain drain of AI professors "
                "from universities. Students from affected universities "
                "establish fewer AI startups and raise less funding."
            ),
        }

        self.assertFalse(
            search_tool._is_relevant_to_query("AI in education", result)
        )

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

    def test_irrelevant_education_results_are_rejected_before_bing_fallback(self):
        irrelevant = [{
            "title": "Lottery results",
            "href": "https://example.com/lottery",
            "body": "Latest lottery numbers.",
        }]
        expected = [{
            "title": "AI tutors improve personalized learning",
            "url": "https://unesco.org/ai-education",
            "snippet": "Artificial intelligence supports student learning.",
            "source": "unesco.org",
        }]
        with (
            mock.patch.object(search_tool, "DDGS") as ddgs,
            mock.patch.object(search_tool, "_openalex_search", return_value=irrelevant),
            mock.patch.object(
                search_tool, "_bing_search",
                return_value=[{
                    "title": expected[0]["title"],
                    "url": expected[0]["url"],
                    "snippet": expected[0]["snippet"],
                }],
            ) as bing,
        ):
            ddgs.return_value.text.return_value = irrelevant
            self.assertEqual(
                search_tool.search({
                    "query": "benefits of AI in education and student learning"
                }),
                expected,
            )
        bing.assert_called_once()

    def test_unrelated_web_results_do_not_count_as_search_success(self):
        irrelevant = [{
            "title": "Lottery results",
            "href": "https://example.com/lottery",
            "body": "Latest lottery numbers.",
        }]
        with (
            mock.patch.object(search_tool, "DDGS") as ddgs,
            mock.patch.object(search_tool, "_bing_search", return_value=irrelevant),
            mock.patch.object(search_tool, "_openalex_search", return_value=irrelevant),
        ):
            ddgs.return_value.text.return_value = irrelevant
            result = search_tool.search({
                "query": "benefits of AI in education and student learning"
            })
        self.assertEqual(result.status, "irrelevant_results")
        self.assertFalse(result)

    def test_mixed_results_keep_only_relevant_sources(self):
        mixed = [
            {
                "title": "AI tutors support personalized learning",
                "url": "https://unesco.org/ai-education",
                "snippet": "Artificial intelligence helps students learn.",
            },
            {
                "title": "Lottery results today",
                "url": "https://example.org/lottery",
                "snippet": "Numbers and prizes.",
            },
        ]
        remaining = [
            {
                "title": f"AI education research report {index}",
                "url": f"https://example.org/report-{index}",
                "snippet": "Artificial intelligence can support student learning.",
            }
            for index in range(2, 4)
        ]
        with (
            mock.patch.object(search_tool, "DDGS") as ddgs,
            mock.patch.object(search_tool, "_openalex_search", return_value=[]),
            mock.patch.object(search_tool, "_bing_search", return_value=remaining),
        ):
            ddgs.return_value.text.return_value = mixed
            results = search_tool.search({
                "query": "AI in education",
                "min_results": 3,
            })
        self.assertEqual(results.status, "ok")
        self.assertEqual(len(results), 3)
        self.assertFalse(any("lottery" in result["url"] for result in results))
        self.assertTrue(all(result["title"] and result["snippet"] for result in results))

    def test_normalizes_education_query_without_losing_topic(self):
        normalized = search_tool._normalize_query("AI trong giáo dục")
        self.assertIn("AI trong giáo dục", normalized)
        self.assertIn("artificial intelligence in education", normalized)
        self.assertIn("AI in learning", normalized)

    def test_results_missing_snippet_are_rejected(self):
        results = search_tool._standardize_results([{
            "title": "AI education article",
            "url": "https://example.org/article",
        }])
        self.assertEqual(results, [])

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
