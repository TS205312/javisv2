"""Tests for the default four-agent peer-review workflow."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


class SwarmDefaultsTests(unittest.TestCase):
    def test_cli_defaults_to_four_agents_and_debate(self):
        with (
            mock.patch.object(sys, "argv", ["main.py", "--goal", "test"]),
            mock.patch.object(main, "run_swarm_mode") as run_swarm,
            mock.patch.object(main, "run_single_agent") as run_single,
        ):
            main.main()

        self.assertEqual(run_swarm.call_args.args[1], 4)
        self.assertEqual(run_swarm.call_args.args[2], "debate")
        run_single.assert_not_called()

    def test_user_can_still_select_single_agent(self):
        with (
            mock.patch.object(
                sys, "argv", ["main.py", "--goal", "test", "--agents", "1"]
            ),
            mock.patch.object(main, "run_swarm_mode") as run_swarm,
            mock.patch.object(main, "run_single_agent") as run_single,
        ):
            main.main()

        run_single.assert_called_once()
        run_swarm.assert_not_called()

    def test_failed_swarm_returns_nonzero_cli_status(self):
        with (
            mock.patch.object(sys, "argv", ["main.py", "--goal", "test"]),
            mock.patch.object(main, "run_swarm_mode", return_value=False),
            mock.patch.object(main, "run_single_agent"),
        ):
            self.assertEqual(main.main(), 1)

    def test_failed_single_agent_returns_nonzero_cli_status(self):
        with (
            mock.patch.object(
                sys, "argv", ["main.py", "--goal", "test", "--agents", "1"]
            ),
            mock.patch.object(main, "run_swarm_mode"),
            mock.patch.object(main, "run_single_agent", return_value=False),
        ):
            self.assertEqual(main.main(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
