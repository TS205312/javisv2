"""
Orchestrator — điều phối N agent chạy song song với asyncio.

Nhận goal → spawn N SwarmAgent → chạy song song (asyncio.gather) →
thu kết quả → gửi cho Aggregator → trả final_result.

Mỗi agent có timeout riêng (mặc định 300s).
Agent crash không kill cả swarm.
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from core.swarm import SwarmAgent
from core.aggregator import Aggregator
from core.config import (
    AGENT_TIMEOUT,
    GROQ_FALLBACK_MODEL,
    GROQ_MODEL,
    KILO_MODEL,
    OPENROUTER_PRIMARY_MODEL,
)
from core.logger import session_logger as default_logger, SessionLogger

# Logger cho orchestrator
orch_logger = SessionLogger(agent_id="orchestrator")


def _run_agent_sync(agent, goal):
    """Chạy 1 SwarmAgent đồng bộ (dùng trong thread pool)."""
    return agent.run(goal)


async def _run_agent_with_timeout(executor, agent, goal, timeout):
    """Run the synchronous agent body in a worker thread with an async timeout."""
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(executor, _run_agent_sync, agent, goal),
        timeout=timeout,
    )


async def run_swarm_async(goal, agent_count=4, models=None, strategy="ensemble",
                          max_steps=50, yolo=False, timeout=None):
    """
    Chạy swarm gồm agent_count agent song song.
    """
    if timeout is None:
        timeout = AGENT_TIMEOUT

    if models is None and agent_count == 4:
        models = [
            f"groq:{GROQ_MODEL}",
            f"kilo:{KILO_MODEL}",
            f"openrouter:{OPENROUTER_PRIMARY_MODEL}",
            f"groq:{GROQ_FALLBACK_MODEL}",
        ]
    elif models is None:
        models = ["auto"] * agent_count
    while len(models) < agent_count:
        models.append("auto")
    if len(models) > agent_count:
        models = models[:agent_count]

    session_id = str(int(time.time() * 1000))
    orch_logger.print("SWARM", f"{agent_count} agent khởi động song song", "bold yellow")
    orch_logger.log("swarm_start", {
        "goal": goal,
        "agent_count": agent_count,
        "models": models,
        "strategy": strategy,
        "max_steps": max_steps,
        "timeout": timeout,
    })

    start = time.time()

    agents = [
        SwarmAgent(
            agent_id=i + 1,
            model_name=models[i],
            max_steps=max_steps,
            yolo=yolo,
            session_id=session_id,
        )
        for i in range(agent_count)
    ]

    executor = ThreadPoolExecutor(max_workers=agent_count)
    tasks = [
        asyncio.create_task(
            _run_agent_with_timeout(executor, agent, goal, timeout)
        )
        for agent in agents
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    executor.shutdown(wait=False)

    results = []
    for agent, result in zip(agents, raw_results):
        if isinstance(result, BaseException):
            orch_logger.print(
                "FAIL",
                f"A{agent.agent_id} FAIL ({result})",
                "bold red",
            )
            agent.logger.print("FAIL", str(result), "bold red")
            agent.logger.log("agent_fail", {
                "timestamp": time.time(),
                "status": "crashed",
                "error": str(result),
                "exception_type": type(result).__name__,
            })
            orch_logger.log("agent_failed", {
                "agent_id": agent.agent_id,
                "error": str(result),
                "exception_type": type(result).__name__,
            })
            continue
        if result is None:
            orch_logger.print(
                "FAIL", f"A{agent.agent_id} FAIL (không có kết quả)", "bold red"
            )
            orch_logger.log("agent_failed", {
                "agent_id": agent.agent_id,
                "error": "Agent returned None",
            })
            continue
        results.append(result)

    elapsed = time.time() - start
    orch_logger.print("SWARM", f"Tất cả agent hoàn tất trong {elapsed:.1f}s", "bold yellow")

    for r in results:
        aid = r.get("agent_id", "?")
        status = r.get("status", "?")
        t = r.get("elapsed", 0)
        orch_logger.print(f"A{aid}", f"status={status}, time={t:.1f}s", "cyan")

    aggregator = Aggregator(
        strategy=strategy,
        session_id=session_id,
        expected_count=agent_count,
    )
    final = aggregator.aggregate(results, goal)

    orch_logger.log("swarm_done", {
        "elapsed": elapsed,
        "final": final,
    })

    return {
        "agent_results": results,
        "aggregator": final,
        "elapsed": elapsed,
    }


def run_swarm(goal, cwd=".", **kwargs):
    """
    Entry point đồng bộ — tiện gọi từ main.py.
    """
    return asyncio.run(run_swarm_async(goal, **kwargs))

# --- Thêm run_single_agent và run_swarm_mode vào đây để main.py import ---

def run_single_agent(goal, max_steps, yolo, cwd="."):
    """Chạy chế độ đơn agent."""
    import os
    if cwd and cwd != ".":
        os.chdir(cwd)

    from core.planner import generate_plan
    from core.executor import execute_task
    from core.critic import evaluate_task
    from core.reflection import reflect
    from core.memory import memory
    from core.skill_resolver import SkillResolver

    default_logger.print("GOAL", goal, "bold green")
    default_logger.log("start_goal", goal)

    similars = memory.search_similar(goal)
    if similars:
        default_logger.print("MEMORY", f"Tìm thấy {len(similars)} kinh nghiệm cũ", "blue")

    default_logger.print("PLANNING", "Đang lập kế hoạch...", "cyan")
    plan = generate_plan(goal)

    if "error" in plan or not isinstance(plan, dict) or "tasks" not in plan:
        default_logger.print("ERROR", f"Lỗi sinh plan: {plan}", "red")
        return False

    default_logger.print("PLAN", f"Hiểu mục tiêu: {plan.get('understanding')}", "green")
    if plan.get("assumptions"):
        default_logger.print("ASSUMPTIONS", plan.get("assumptions"), "yellow")
        default_logger.log("assumptions", plan.get("assumptions"))

    tasks = plan.get("tasks", [])
    total = len(tasks)
    step = 0
    task_idx = 0

    while task_idx < total and step < max_steps:
        task = tasks[task_idx]
        default_logger.print(f"TASK {task_idx+1}/{total}", task["desc"], "bold magenta")

        retry_count = 0
        passed = False

        while retry_count < 3 and not passed:
            step += 1
            result = execute_task(task, yolo)

            critic_res = evaluate_task(task, str(result))
            if isinstance(critic_res, dict):
                passed = critic_res.get("passed", False)
                reason = critic_res.get("reason", "")
                fix = critic_res.get("fix", "")
            else:
                passed = True
                reason = "OK"
                fix = ""

            if passed:
                default_logger.print("CRITIC", f"Pass: {reason}", "green")
                memory.save_long_term(task["id"], task["desc"], "Thành công")
                SkillResolver.record_outcome(
                    [
                        (task.get("_resolved_skill") or {}).get("skill"),
                        task.get("_selected_tool"),
                    ],
                    True,
                    memory=memory,
                    logger=default_logger,
                )
                break
            else:
                retry_count += 1
                SkillResolver.record_outcome(
                    [
                        (task.get("_resolved_skill") or {}).get("skill"),
                        task.get("_selected_tool"),
                    ],
                    False,
                    memory=memory,
                    logger=default_logger,
                )
                default_logger.print("CRITIC", f"Fail ({retry_count}/3): {reason}. Gợi ý sửa: {fix}", "red")
                task["desc"] += f" [SỬA LỖI: {fix}]"

        if not passed:
            default_logger.print("REPLAN", "Task fail 3 lần, chuyển task tiếp", "yellow")

        task_idx += 1

        if step % 5 == 0 and step > 0:
            history = memory.get_short_term_context()
            ref = reflect(history)
            default_logger.print("REFLECT", str(ref), "cyan")
            default_logger.log("reflection", ref)

    default_logger.print("DONE", "HOÀN THÀNH – Autonomous Agent sẵn sàng", "bold green")
    default_logger.log("end", "Done")


def run_swarm_mode(goal, agent_count, strategy, models, max_steps, yolo, cwd="."):
    """Chạy chế độ swarm qua orchestrator. Gọi từ main.py."""
    import os
    if cwd and cwd != ".":
        os.chdir(cwd)

    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()

    console.print(Panel(
        Text(f"SWARM MODE: {agent_count} agents | strategy={strategy}", style="bold yellow"),
        title="🐝 Autonomous Agent Swarm",
        border_style="yellow",
    ))

    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(",")]

    result = run_swarm(
        goal=goal,
        agent_count=agent_count,
        models=model_list,
        strategy=strategy,
        max_steps=max_steps,
        yolo=yolo,
    )

    agg = result.get("aggregator", {})
    elapsed = result.get("elapsed", 0)

    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold]Winner:[/bold] {agg.get('winner', '?')}\n"
            f"[bold]Confidence:[/bold] {agg.get('confidence', 0):.2f}\n"
            f"[bold]Reason:[/bold] {agg.get('reason', '')}\n"
            f"[bold]Final Result:[/bold]\n{str(agg.get('final_result', ''))[:500]}"
        ),
        title="📊 Aggregator Result",
        border_style="green",
    ))

    best_parts = agg.get("best_parts", [])
    if best_parts:
        console.print("\n[bold cyan]Best Parts:[/bold cyan]")
        for part in best_parts:
            console.print(f"  • {part}")

    console.print(f"\n[bold green][DONE] Tổng thời gian: {elapsed:.1f}s[/bold green]")
    console.print(f"[dim]⚠️  Chi phí token: ~{agent_count}x so với chạy đơn agent[/dim]")
