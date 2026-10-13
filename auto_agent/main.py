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
    parser.add_argument("--agents", type=int, default=1)
    parser.add_argument(
        "--strategy",
        default="ensemble",
        choices=["ensemble", "debate", "vote"],
    )
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--yolo", action="store_true")
    parser.add_argument("--cwd", type=str, default=".")
    args = parser.parse_args()

    console.print(Panel(f"🎯 {args.goal}", title="Agent"))
    if args.agents <= 1:
        run_single_agent(args.goal, args.max_steps, args.yolo, args.cwd)
    else:
        run_swarm_mode(
            args.goal,
            args.agents,
            args.strategy,
            args.models,
            args.max_steps,
            args.yolo,
            args.cwd,
        )


if __name__ == "__main__":
    main()
