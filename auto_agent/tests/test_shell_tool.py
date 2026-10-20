"""Regression tests for the structured, bounded shell policy."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import shell_tool, write_safety


class ShellToolTests(unittest.TestCase):
    def test_echo_alias_is_handled_without_a_shell_process(self):
        with mock.patch.object(shell_tool.subprocess, "run") as run:
            result = shell_tool.run({"command": "echo ok"})

        self.assertIn("STDOUT:\nok", result)
        run.assert_not_called()

    def test_structured_python_version_uses_argv_and_is_bounded(self):
        completed = SimpleNamespace(returncode=0, stdout="Python 3.x", stderr="")
        with mock.patch.object(shell_tool.subprocess, "run", return_value=completed) as run:
            result = shell_tool.run({
                "executable": "python",
                "args": ["--version"],
                "timeout": 900,
            })

        self.assertIn("Python 3.x", result)
        self.assertEqual(run.call_args.args[0], [sys.executable, "--version"])
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertLessEqual(run.call_args.kwargs["timeout"], 30)
        self.assertTrue(Path(run.call_args.kwargs["cwd"]).is_dir())

    def test_yolo_does_not_bypass_terminal_allowlist(self):
        with mock.patch.object(shell_tool.subprocess, "run") as run:
            result = shell_tool.run(
                {"command": "python -c import os"},
                yolo=True,
            )

        self.assertTrue(result.startswith("ERROR [SAFETY]:"))
        run.assert_not_called()

    def test_install_and_shell_interpreter_commands_are_denied(self):
        for command in ("pip install package", "cmd.exe /c whoami", "python script.py"):
            with self.subTest(command=command), mock.patch.object(
                shell_tool.subprocess, "run"
            ) as run:
                result = shell_tool.run({"command": command})
                self.assertTrue(result.startswith("ERROR [SAFETY]:"))
                run.assert_not_called()

    def test_process_working_directory_cannot_escape_workspace(self):
        workspace_parent = Path(shell_tool.__file__).resolve().parents[2]
        with mock.patch.object(shell_tool.subprocess, "run") as run:
            result = shell_tool.run({
                "executable": "python",
                "args": ["--version"],
                "cwd": str(workspace_parent),
            })

        self.assertTrue(result.startswith("ERROR [SAFETY]:"))
        run.assert_not_called()

    def test_directory_listing_hides_protected_paths(self):
        with tempfile.TemporaryDirectory(
            dir=write_safety.WORKSPACE_ROOT
        ) as directory:
            root = Path(directory)
            for name in ("token.json", "cookie.txt", "api_key.txt", "visible.txt"):
                (root / name).write_text("unit-test-only", encoding="utf-8")

            result = shell_tool._run_internal({"mode": "list", "cwd": root})

        self.assertIn("visible.txt", result)
        self.assertNotIn("token.json", result)
        self.assertNotIn("cookie.txt", result)
        self.assertNotIn("api_key.txt", result)

    def test_output_is_capped_and_redacted(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout="password=hunter2 " + ("safe " * 1100),
            stderr="",
        )
        with mock.patch.object(shell_tool.subprocess, "run", return_value=completed):
            result = shell_tool.run({
                "executable": "python",
                "args": ["--version"],
            })

        self.assertNotIn("hunter2", result)
        self.assertIn("[REDACTED]", result)
        self.assertIn("[OUTPUT TRUNCATED]", result)

    def test_rejects_non_mapping_parameters(self):
        self.assertTrue(shell_tool.run(None).startswith("ERROR [SAFETY]:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
