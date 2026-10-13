"""
SwarmAgent — wrap toàn bộ pipeline (planner → executor → critic → reflection)
thành 1 agent độc lập có memory riêng, log riêng, LLM riêng.

Mỗi instance chạy đầy đủ vòng lặp plan → execute → evaluate → reflect
mà hoàn toàn không share state với các agent khác.
"""
import time
import copy
import traceback

from core.llm_helper import LLMClient
from core.memory import Memory
from core.logger import SessionLogger
from core.planner import generate_plan
from core.executor import execute_task
from core.critic import evaluate_task
from core.reflection import reflect
from core.skill_resolver import SkillResolver


class SwarmAgent:
    """
    Một agent độc lập trong swarm.
    agent_id: số thứ tự 0..N-1
    model_name: "auto" hoặc provider:model override, ví dụ "ollama:qwen2.5:7b".
    max_steps: giới hạn bước cho agent này.
    yolo: bỏ qua safety.
    """

    def __init__(
        self, agent_id, model_name="auto", max_steps=50, yolo=False,
        session_id=None,
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.max_steps = max_steps
        self.yolo = yolo

        # Mỗi agent có LLM client riêng
        self.llm = LLMClient(model_name)

        # Mỗi agent có memory namespace riêng (ChromaDB collection riêng)
        self.memory = Memory(namespace=f"agent{agent_id}")

        # Mỗi agent có log file riêng
        self.logger = SessionLogger(agent_id=agent_id, session_id=session_id)

    def run(self, goal):
        """
        Chạy toàn bộ pipeline cho goal.
        Trả về dict kết quả để aggregator tổng hợp.
        """
        start = time.time()
        start_monotonic = time.perf_counter()
        started_at = time.time()
        started_clock = time.strftime("%H:%M:%S", time.localtime(started_at))
        self.logger.print_agent_timing("START", started_at)
        self.logger.log("agent_start", {
            "timestamp": started_at,
            "display_time": started_clock,
        })
        result = {
            "agent_id": self.agent_id,
            "model_used": self.model_name,
            "plan": None,
            "actions": [],
            "result": None,
            "status": "unknown",
            "elapsed": 0,
            "error": None,
        }

        try:
            self.logger.print("START", f"Agent {self.agent_id} bắt đầu với model={self.model_name}", "bold cyan")
            self.logger.log("start", {"goal": goal, "model": self.model_name})

            # Tìm kinh nghiệm cũ
            similars = self.memory.search_similar(goal)
            if similars:
                self.logger.print("MEMORY", f"Tìm thấy {len(similars)} kinh nghiệm cũ", "blue")

            # Lập kế hoạch
            self.logger.print("PLANNING", "Đang lập kế hoạch...", "cyan")
            plan = generate_plan(goal, llm_client=self.llm)

            if "error" in plan or not isinstance(plan, dict) or "tasks" not in plan:
                result["status"] = "plan_failed"
                result["error"] = str(plan)
                result["elapsed"] = time.time() - start
                self.logger.print("ERROR", f"Lỗi sinh plan: {plan}", "red")
                self.logger.log("plan_failed", str(plan))
                return result

            result["plan"] = plan
            tasks = plan.get("tasks", [])
            total = len(tasks)
            self.logger.print("PLAN", f"Plan xong → {total} task", "green")
            self.logger.log("plan", plan)

            # Thực thi từng task
            step = 0
            task_idx = 0
            completed_tasks = 0
            all_outputs = []

            while task_idx < total and step < self.max_steps:
                task = copy.deepcopy(tasks[task_idx])
                self.logger.print(f"TASK {task_idx+1}/{total}", task["desc"], "bold magenta")

                retry_count = 0
                passed = False

                while retry_count < 3 and not passed:
                    step += 1
                    output = execute_task(
                        task,
                        yolo=self.yolo,
                        memory=self.memory,
                        logger=self.logger,
                        llm_client=self.llm,
                    )

                    result["actions"].append({
                        "task_id": task["id"],
                        "desc": task["desc"],
                        "output": str(output)[:500],
                        "retry": retry_count,
                    })

                    # Critic đánh giá
                    critic_res = evaluate_task(task, str(output), llm_client=self.llm)
                    if isinstance(critic_res, dict):
                        passed = critic_res.get("passed", False)
                        reason = critic_res.get("reason", "")
                        fix = critic_res.get("fix", "")
                    else:
                        passed = True
                        reason = "OK"
                        fix = ""

                    if passed:
                        self.logger.print("CRITIC", f"Pass: {reason}", "green")
                        self.memory.save_long_term(task["id"], task["desc"], "Thành công")
                        SkillResolver.record_outcome(
                            [
                                (task.get("_resolved_skill") or {}).get("skill"),
                                task.get("_selected_tool"),
                            ],
                            True,
                            memory=self.memory,
                            logger=self.logger,
                        )
                        completed_tasks += 1
                        all_outputs.append(str(output)[:500])
                        break
                    else:
                        retry_count += 1
                        SkillResolver.record_outcome(
                            [
                                (task.get("_resolved_skill") or {}).get("skill"),
                                task.get("_selected_tool"),
                            ],
                            False,
                            memory=self.memory,
                            logger=self.logger,
                        )
                        self.logger.print(
                            "CRITIC",
                            f"Fail ({retry_count}/3): {reason}. Gợi ý sửa: {fix}",
                            "red",
                        )
                        task["desc"] += f" [SỬA LỖI: {fix}]"

                if not passed:
                    self.logger.print("REPLAN", "Task fail 3 lần, chuyển task tiếp", "yellow")
                    all_outputs.append(f"FAIL: {task['desc']}")

                task_idx += 1

                # Reflection mỗi 5 bước
                if step % 5 == 0 and step > 0:
                    history = self.memory.get_short_term_context()
                    ref = reflect(history, llm_client=self.llm)
                    self.logger.print("REFLECT", str(ref), "cyan")
                    self.logger.log("reflection", ref)

            # Tổng kết
            result["result"] = "\n".join(all_outputs) if all_outputs else "Không có kết quả"
            result["status"] = "completed" if completed_tasks == total else "partial"
            result["elapsed"] = time.time() - start
            self.logger.print("DONE", f"Xong trong {result['elapsed']:.1f}s ({completed_tasks}/{total} task)", "bold green")
            self.logger.log("done", {
                "elapsed": result["elapsed"],
                "completed": completed_tasks,
                "total": total,
            })

        except Exception as e:
            result["status"] = "crashed"
            result["error"] = traceback.format_exc()
            result["elapsed"] = time.time() - start
            self.logger.print("CRASH", str(e), "bold red")
            self.logger.log("crash", {"error": str(e), "traceback": traceback.format_exc()})

        finally:
            ended_at = time.time()
            elapsed = time.perf_counter() - start_monotonic
            status = result["status"]
            event = "DONE" if status in {"completed", "partial"} else "FAIL"
            self.logger.print_agent_timing(event, ended_at, elapsed)
            self.logger.log(
                "agent_done" if event == "DONE" else "agent_fail",
                {
                    "timestamp": ended_at,
                    "display_time": time.strftime(
                        "%H:%M:%S", time.localtime(ended_at)
                    ),
                    "elapsed": elapsed,
                    "status": status,
                    "error": result.get("error"),
                },
            )

        return result
