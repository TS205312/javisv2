"""Tests for deterministic research planning, routing, and evidence gates."""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import critic, planner, router
from core.research_policy import (
    append_references_from_evidence,
    extract_research_topic,
    is_presentation_draft_task,
    is_search_task,
    presentation_issues,
)


class ResearchPolicyTests(unittest.TestCase):
    def test_missing_bibliography_is_built_only_from_validated_evidence(self):
        evidence = [
            {
                "title": f"Education study {index}",
                "url": f"https://example.org/{index}",
            }
            for index in range(3)
        ]

        result = append_references_from_evidence("### Slide 1: Opening", evidence)

        self.assertIn("## Tài liệu tham khảo", result)
        self.assertTrue(all(source["url"] in result for source in evidence))

    def test_missing_bibliography_is_not_invented_with_fewer_than_three_sources(self):
        result = append_references_from_evidence(
            "### Slide 1: Opening",
            [{"title": "One source", "url": "https://example.org/one"}],
        )

        self.assertEqual(result, "### Slide 1: Opening")

    def test_evidence_bibliography_completes_a_valid_cited_deck(self):
        evidence = [
            {
                "title": f"Education study {index}",
                "url": f"https://example.org/{index}",
            }
            for index in range(3)
        ]
        slides = []
        for number in range(1, 7):
            title = "Conclusion" if number == 6 else "Education topic"
            citation = (
                f"\n- Source: {evidence[(number - 2) % 3]['url']}"
                if number > 1
                else ""
            )
            slides.append(
                f"### Slide {number}: {title}\n- Evidence{citation}\n"
                "- Image suggestion: classroom"
            )

        completed = append_references_from_evidence("\n\n".join(slides), evidence)

        self.assertEqual(
            presentation_issues(
                completed,
                verified_source_urls={item["url"] for item in evidence},
            ),
            [],
        )

    def test_presentation_plan_starts_with_research_sources(self):
        client = mock.Mock()
        client.ask.return_value = {
            "tasks": [{
                "id": "slides",
                "desc": "Write 6 slides about AI in education",
                "deps": [],
                "success": "Six slides written",
            }],
        }

        result = planner.generate_plan(
            "Nghiên cứu AI trong giáo dục và tạo bài tiếng Việt 6–8 slide",
            llm_client=client,
        )

        self.assertEqual(result["tasks"][0]["id"], "research_sources")
        self.assertEqual(result["tasks"][0]["type"], "research")
        self.assertEqual(result["tasks"][0]["search_query"], "AI trong giáo dục")
        self.assertIn("3 kết quả liên quan", result["tasks"][0]["success"])
        self.assertIn("research_sources", result["tasks"][1]["deps"])
        client.ask.assert_not_called()

    def test_long_original_goal_does_not_turn_slide_task_into_search(self):
        original_goal = (
            "Dùng lại mục tiêu thử nghiệm: nghiên cứu “Trí tuệ nhân tạo trong giáo dục”, "
            "tạo 6–8 slide. Dùng công cụ tìm kiếm, ghi nguồn và không cài repository."
        )
        plan = planner.generate_plan(original_goal)
        draft_task = plan["tasks"][1]

        self.assertTrue(is_presentation_draft_task(draft_task))
        self.assertFalse(is_search_task(draft_task))
        self.assertEqual(draft_task["type"], "presentation_draft")
        self.assertTrue(is_search_task(plan["tasks"][0]))

    def test_shared_evidence_plan_skips_duplicate_search(self):
        evidence = [{
            "title": f"AI education source {index}",
            "url": f"https://example.org/{index}",
            "snippet": "AI in education source summary.",
            "source": "example.org",
        } for index in range(3)]
        client = mock.Mock()
        result = planner.generate_plan(
            "Nghiên cứu “AI trong giáo dục” và tạo slide",
            llm_client=client,
            shared_evidence=evidence,
        )

        self.assertEqual([task["type"] for task in result["tasks"]], ["presentation_draft"])
        self.assertEqual(result["tasks"][0]["research_evidence"], evidence)
        client.ask.assert_not_called()

    def test_extracts_quoted_topic_from_long_task(self):
        self.assertEqual(
            extract_research_topic(
                "Dùng lại mục tiêu: nghiên cứu “Trí tuệ nhân tạo trong giáo dục”, tạo slides"
            ),
            "Trí tuệ nhân tạo trong giáo dục",
        )

    def test_research_task_routes_to_search_without_llm(self):
        llm = mock.Mock()
        route = router.route_task(
            {"desc": "Tìm kiếm web, thu thập 3 nguồn về AI trong giáo dục"},
            "",
            llm_client=llm,
        )

        self.assertEqual(route["tool"], "search_tool")
        self.assertEqual(route["action"], "search")
        self.assertEqual(route["params"]["min_results"], 3)
        llm.ask.assert_not_called()

    def test_slide_drafting_is_not_misclassified_as_search(self):
        description = (
            "Soạn nội dung tiếng Việt 6–8 slide, dùng kết quả nghiên cứu "
            "và nguồn tham khảo đã thu thập."
        )
        self.assertFalse(is_search_task(description))
        self.assertTrue(is_presentation_draft_task(description))

    def test_tagged_slide_task_ignores_search_words_in_embedded_goal(self):
        task = {
            "id": "create_slides",
            "type": "presentation_draft",
            "desc": (
                "Soạn nội dung 6–8 slide cho mục tiêu: nghiên cứu AI trong giáo dục; "
                "dùng công cụ tìm kiếm, thêm nguồn và tài liệu tham khảo."
            ),
        }
        self.assertFalse(is_search_task(task))
        self.assertTrue(is_presentation_draft_task(task))

    def test_search_critic_rejects_empty_and_unverified_results(self):
        llm = mock.Mock()
        task = {"desc": "Tìm kiếm web 3 nguồn AI education research"}

        result = critic.evaluate_task(task, "SEARCH_NO_RESULTS: none", llm_client=llm)

        self.assertFalse(result["passed"])
        llm.ask.assert_not_called()

    def test_search_critic_requires_three_structured_sources(self):
        task = {"desc": "Tìm kiếm web 3 nguồn AI education research"}
        rows = [{
            "title": f"AI education source {index}",
            "url": f"https://example.org/{index}",
            "snippet": "AI supports learning.",
            "source": "example.org",
        } for index in range(3)]

        result = critic.evaluate_task(
            task,
            "SEARCH_OK provider=bing query='AI education'\n" + json.dumps(rows),
        )

        self.assertTrue(result["passed"])

    def test_slide_critic_rejects_slide_without_inline_source(self):
        urls = [f"https://example.org/{index}" for index in range(3)]
        slides = []
        for number in range(1, 7):
            title = "Kết luận" if number == 6 else "Nội dung"
            citation = "" if number == 3 else f"\n- Nguồn: {urls[number % 3]}"
            slides.append(
                f"## Slide {number}: {title}\n- Ý chính{citation}\n"
                "- Gợi ý hình ảnh: minh họa"
            )
        output = "\n\n".join(slides) + "\n\n## Tài liệu tham khảo\n" + "\n".join(urls)
        llm = mock.Mock()
        result = critic.evaluate_task(
            {
                "id": "create_slides",
                "type": "presentation_draft",
                "desc": "Soạn slide có trích dẫn",
                "verified_source_urls": urls,
            },
            output,
            llm_client=llm,
        )

        self.assertFalse(result["passed"])
        self.assertIn("slide 3 thiếu trích dẫn", result["reason"])
        llm.ask.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
