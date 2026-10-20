"""Runtime verification tests for generated Python code."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import coder_tool


class CoderToolRuntimeTests(unittest.TestCase):
    def test_repairs_runtime_error_and_only_passes_after_success(self):
        workspace_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace_root) as temp_dir:
            script = Path(temp_dir) / "factorial.py"
            failed = SimpleNamespace(
                returncode=1, stdout="", stderr="NameError: name 'factorial' is not defined"
            )
            succeeded = SimpleNamespace(returncode=0, stdout="3628800\n", stderr="")
            with (
                mock.patch.object(coder_tool.subprocess, "run", side_effect=[failed, succeeded]),
                mock.patch.object(
                    coder_tool, "ask_llm",
                    return_value="def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\nprint(factorial(10))",
                ) as ask,
            ):
                result = coder_tool.write_code({
                    "file_path": str(script),
                    "content": "print(factorial(10))",
                })

            self.assertTrue(result.startswith("PASS:"))
            self.assertIn("3628800", result)
            self.assertIn("def factorial", script.read_text(encoding="utf-8"))
            ask.assert_called_once()

    def test_returns_stderr_after_three_failed_runs(self):
        workspace_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace_root) as temp_dir:
            script = Path(temp_dir) / "broken.py"
            failure = SimpleNamespace(returncode=1, stdout="", stderr="SyntaxError")
            with (
                mock.patch.object(
                    coder_tool.subprocess, "run", return_value=failure
                ) as run_mock,
                mock.patch.object(
                    coder_tool, "ask_llm", return_value="print('still broken')"
                ),
            ):
                result = coder_tool.write_code({
                    "path": str(script),
                    "code": "not valid python",
                })

            self.assertTrue(result.startswith("FAIL:"))
            self.assertIn("SyntaxError", result)
            self.assertEqual(run_mock.call_count, 3)

    def test_accepts_filepath_alias(self):
        workspace_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace_root) as temp_dir:
            script = Path(temp_dir) / "plot.py"
            with mock.patch.object(coder_tool.subprocess, "run") as run_mock:
                run_mock.return_value = SimpleNamespace(
                    returncode=0, stdout="chart created", stderr=""
                )
                result = coder_tool.write_code({
                    "filepath": str(script),
                    "content": "print('chart created')",
                })

            self.assertTrue(result.startswith("PASS:"))
            self.assertTrue(script.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
