"""Tests for shell command parameter compatibility."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import shell_tool


class ShellToolTests(unittest.TestCase):
    def test_accepts_router_command_alias(self):
        completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        with (
            mock.patch.object(shell_tool, "is_safe", return_value=(True, "")),
            mock.patch.object(shell_tool.subprocess, "run", return_value=completed) as run,
        ):
            result = shell_tool.run({"command": "echo ok"})

        self.assertIn("STDOUT:\nok", result)
        self.assertEqual(run.call_args.args[0], "echo ok")

    def test_rejects_non_mapping_parameters(self):
        self.assertEqual(
            shell_tool.run(None),
            "Thiếu hoặc không hợp lệ param cmd",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
