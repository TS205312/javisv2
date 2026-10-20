"""Command-line entry point for the autonomous agent."""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse

from rich.console import Console
from rich.panel import Panel

from core.orchestrator import run_single_agent, run_swarm_mode


console = Console()


def main():
    parser = argparse.ArgumentParser(description="Autonomous Agent CLI")
    parser.add_argument("--goal", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument(
        "--strategy",
        default="debate",
        choices=["debate"],
    )
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--cwd", type=str, default=".")
    args = parser.parse_args()

    console.print(Panel(f"🎯 {args.goal}", title="Agent"))
    if args.agents <= 1:
        result = run_single_agent(args.goal, args.max_steps, args.yolo, args.cwd)
        return 1 if result is False else 0
    else:
        succeeded = run_swarm_mode(
            args.goal,
            args.agents,
            args.strategy,
            args.models,
            args.max_steps,
            args.yolo,
            args.cwd,
        )
        return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
