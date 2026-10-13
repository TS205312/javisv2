"""Command-line interface for local skill lookup and GitHub skill discovery."""
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from rich.console import Console
from rich.table import Table

from core.skill_resolver import SkillResolver, quality_score
from registry import get_all_skills

console = Console(width=160)


def _print_matches(matches, goal):
    if not matches:
        console.print(f"[yellow]Không tìm thấy skill local phù hợp với: {goal}[/yellow]")
        return
    table = Table(title=f"Local skills — {goal}")
    table.add_column("Skill")
    table.add_column("Match", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Decision")
    for item in matches:
        quality, _ = quality_score(item.get("meta", {}))
        table.add_row(
            item["skill"],
            f"{item['score']:.3f}",
            f"{quality}/10",
            "đủ tốt" if quality >= 7 else "cần hunt",
        )
    console.print(table)


def _list_skills():
    table = Table(title="Skill registry")
    table.add_column("Skill")
    table.add_column("Source")
    table.add_column("Repository")
    table.add_column("Stars", justify="right")
    table.add_column("License")
    table.add_column("Updated")
    table.add_column("Success", justify="right")
    table.add_column("Pass rate", justify="right")
    table.add_column("Rating", justify="right")
    for name, skill in sorted(get_all_skills().items()):
        meta = skill.get("meta", {})
        table.add_row(
            name,
            str(meta.get("source", "")),
            str(meta.get("repo_url", "")),
            str(meta.get("stars", 0)),
            str(meta.get("license", "")),
            str(meta.get("last_updated", ""))[:10],
            str(meta.get("success_count", 0)),
            f"{float(meta.get('test_pass_rate', 0) or 0):.2f}",
            str(meta.get("user_rating") if meta.get("user_rating") is not None else ""),
        )
    console.print(table)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resolve and manage autonomous-agent skills.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="Check local skills without GitHub hunting.")
    check.add_argument("goal", nargs="+")
    resolve = subparsers.add_parser("resolve", help="Resolve locally or hunt/install from GitHub.")
    resolve.add_argument("goal", nargs="+")
    subparsers.add_parser("list", help="List skills and metadata.")
    upgrade = subparsers.add_parser("upgrade", help="Hunt and install an updated skill.")
    upgrade.add_argument("skill", nargs="+")
    rate = subparsers.add_parser("rate", help="Set a user rating from 0 to 5.")
    rate.add_argument("skill")
    rate.add_argument("rating", type=int, choices=range(0, 6))
    args = parser.parse_args(argv)
    resolver = SkillResolver()

    if args.command == "list":
        _list_skills()
        return 0
    if args.command == "check":
        goal = " ".join(args.goal)
        matches = resolver.find_local_skills(goal)
        _print_matches(matches, goal)
        if not any(resolver.is_skill_good_enough(item, goal) for item in matches):
            console.print("[yellow]Cần hunt GitHub nếu resolve được gọi.[/yellow]")
        return 0
    if args.command == "resolve":
        goal = " ".join(args.goal)
        skill = resolver.resolve(goal)
        if not skill:
            console.print("[red]Không tìm được skill local hoặc GitHub.[/red]")
            return 1
        console.print(
            f"[green]Resolved:[/green] {skill['skill']} "
            f"(match={skill['score']:.3f}, source={skill.get('meta', {}).get('source', 'local')})"
        )
        console.print(f"[dim]{skill.get('desc', '')}[/dim]")
        return 0
    if args.command == "upgrade":
        goal = " ".join(args.skill)
        result = resolver.hunt_best(goal)
        if not result:
            console.print("[red]Không tìm được skill phù hợp để nâng cấp.[/red]")
            return 1
        console.print(f"[green]Installed:[/green] {result['skill']}")
        return 0
    if args.command == "rate":
        try:
            resolver.rate_skill(args.skill, args.rating)
        except (KeyError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        console.print(f"[green]Đã đánh giá {args.skill}: {args.rating}/5[/green]")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
