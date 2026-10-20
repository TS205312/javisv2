"""Tests for safe, explicit Canva-ready Markdown fallback output."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import orchestrator
from core.orchestrator import _save_presentation_draft
from tools import write_safety


class PresentationOutputTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        workspace_patch = mock.patch.object(
            orchestrator,
            "WORKSPACE_ROOT",
            Path(workspace.name),
        )
        workspace_patch.start()
        self.addCleanup(workspace_patch.stop)
        write_root_patch = mock.patch.object(
            write_safety,
            "WORKSPACE_ROOT",
            Path(workspace.name),
        )
        write_root_patch.start()
        self.addCleanup(write_root_patch.stop)

    @staticmethod
    def _deck(urls):
        slides = []
        for number in range(1, 7):
            title = "Kết luận" if number == 6 else "Nội dung"
            source = f"\n- Nguồn: {urls[(number - 2) % len(urls)]}" if number > 1 else ""
            slides.append(
                f"## Slide {number}: {title}\n- Ý chính{source}\n"
                "- Gợi ý hình ảnh: minh họa"
            )
        return "\n\n".join(slides) + "\n\n## Tài liệu tham khảo\n" + "\n".join(urls)

    def test_saves_six_slide_draft_without_claiming_canva_was_created(self):
        original_directory = Path.cwd()
        urls = [f"https://example.org/{letter}" for letter in "abc"]
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                slide_text = self._deck(urls)
                result = _save_presentation_draft(
                    "Tạo bài thuyết trình Canva",
                    {
                        "final_result": (
                            slide_text
                        )
                    },
                    "test-session",
                    verified_source_urls=urls,
                )
                self.assertTrue(result["saved"])
                self.assertFalse(result["canva_created"])
                self.assertEqual(result["slide_count"], 6)
                self.assertTrue(result["source_check_passed"])
                self.assertEqual(result["source_count"], 3)
                content = Path(result["path"]).read_text(encoding="utf-8")
                self.assertIn("chưa phải thiết kế đã lưu trên Canva", content)
            finally:
                os.chdir(original_directory)

    def test_builds_six_slide_evidence_fallback_when_leader_output_is_empty(self):
        original_directory = Path.cwd()
        evidence = [{
            "title": f"AI education study {index}",
            "url": f"https://example.org/{index}",
            "snippet": f"Validated summary for evidence source {index}.",
            "source": "Example research provider",
        } for index in range(1, 4)]
        urls = {row["url"] for row in evidence}
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                aggregate = {"final_result": ""}
                result = _save_presentation_draft(
                    "Research AI in education and create a Canva presentation",
                    aggregate,
                    "evidence-fallback",
                    verified_source_urls=urls,
                    research_evidence=evidence,
                    canva_preflight={
                        "status": "automation_blocked",
                        "reason": "HTTP 403",
                        "design_created": False,
                    },
                )
                content = Path(result["path"]).read_text(encoding="utf-8")
            finally:
                os.chdir(original_directory)

        self.assertTrue(result["saved"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["draft_type"], "source_evidence_fallback")
        self.assertEqual(result["slide_count"], 6)
        self.assertEqual(result["canva_status"], "automation_blocked")
        self.assertFalse(result["canva_created"])
        self.assertTrue(result["canva_fallback_saved"])
        self.assertTrue(aggregate["final_result"].startswith("## Slide 1:"))
        self.assertIn("HTTP 403", content)
        self.assertIn("Fallback:", content)

    def test_does_not_save_when_slide_count_is_outside_requested_range(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                result = _save_presentation_draft(
                    "Tạo bài Canva",
                    {"final_result": "## Slide 1: Intro"},
                    "test-session",
                )
                self.assertFalse(result["saved"])
                self.assertFalse(Path("output").exists())
            finally:
                os.chdir(original_directory)

    def test_valid_new_draft_does_not_overwrite_existing_education_draft(self):
        original_directory = Path.cwd()
        urls = [f"https://example.org/{letter}" for letter in "abc"]
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                stable_path = Path("output/canva_ai_giao_duc.md")
                stable_path.parent.mkdir(parents=True)
                stable_path.write_text("previous valid draft", encoding="utf-8")
                result = _save_presentation_draft(
                    "Tạo bài thuyết trình về giáo dục",
                    {"final_result": self._deck(urls)},
                    "new-session",
                    verified_source_urls=urls,
                )
                self.assertTrue(result["saved"])
                self.assertNotEqual(Path(result["path"]), stable_path)
                self.assertEqual(
                    stable_path.read_text(encoding="utf-8"),
                    "previous valid draft",
                )
            finally:
                os.chdir(original_directory)

    def test_duplicate_slide_numbers_do_not_pass_validation(self):
        result = _save_presentation_draft(
            "Create a Canva presentation",
            {"final_result": "\n".join("## Slide 1" for _ in range(6))},
            "duplicate-slides",
        )

        self.assertFalse(result["saved"])

    def test_file_write_error_is_reported_without_raising(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with mock.patch.object(
                    orchestrator,
                    "write_bytes_safely",
                    side_effect=PermissionError("read-only output"),
                ) as safe_write:
                    urls = [f"https://example.org/{letter}" for letter in "abc"]
                    final_result = self._deck(urls).replace(
                        "## Tài liệu tham khảo", "## Sources"
                    )
                    result = _save_presentation_draft(
                        "Create a Canva presentation",
                        {"final_result": final_result},
                        "write-error",
                        verified_source_urls=urls,
                    )
                    self.assertEqual(safe_write.call_count, 2)
            finally:
                os.chdir(original_directory)

        self.assertFalse(result["saved"])
        self.assertEqual(result["exception_type"], "PermissionError")

    def test_write_fallback_uses_a_distinct_workspace_path(self):
        urls = [f"https://example.org/{letter}" for letter in "abc"]
        fallback_path = Path(orchestrator.WORKSPACE_ROOT) / "fallback.md"
        with mock.patch.object(
            orchestrator,
            "write_bytes_safely",
            side_effect=[PermissionError("output unavailable"), fallback_path],
        ) as safe_write:
            result = _save_presentation_draft(
                "Create a Canva presentation",
                {"final_result": self._deck(urls)},
                "fallback-path",
                verified_source_urls=urls,
            )

        first_path = safe_write.call_args_list[0].args[0]
        second_path = safe_write.call_args_list[1].args[0]
        self.assertTrue(result["saved"])
        self.assertTrue(result["fallback_used"])
        self.assertNotEqual(Path(first_path), Path(second_path))
        self.assertTrue(Path(second_path).is_relative_to(orchestrator.WORKSPACE_ROOT))

    def test_does_not_save_unverified_sources(self):
        urls = [f"https://example.org/{letter}" for letter in "abc"]
        final_result = self._deck(urls)
        result = _save_presentation_draft(
            "Create a Canva presentation",
            {"final_result": final_result},
            "unverified",
            verified_source_urls={"https://other.org/source"},
        )
        self.assertFalse(result["saved"])
        self.assertIn("verified=False", result["reason"])

    def test_requires_each_content_slide_to_cite_a_source(self):
        urls = [f"https://example.org/{letter}" for letter in "abc"]
        deck = self._deck(urls).replace(
            f"- Nguồn: {urls[0]}",
            "- Ý chính chưa trích dẫn",
            1,
        )
        result = _save_presentation_draft(
            "Tạo bài Canva",
            {"final_result": deck},
            "missing-slide-citation",
            verified_source_urls=urls,
        )
        self.assertFalse(result["saved"])
        self.assertIn("thiếu trích dẫn nguồn", result["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
