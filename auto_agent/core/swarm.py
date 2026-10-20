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
from core.plan_policy import validate_plan
from core.executor import execute_task
from core.critic import evaluate_task
from core.reflection import reflect
from core.recovery import (
    desktop_action_summary,
    propose_desktop_recovery,
    retry_decision,
    sanitize_desktop_output,
)
from core.skill_resolver import SkillResolver
from tools.browser_tool import (
    BrowserToolError,
    close_session,
    session_summary,
    set_session_logger,
)
from core.research_policy import (
    append_references_from_evidence,
    is_presentation_draft_task,
    is_presentation_goal,
    is_search_task,
)
from tools.write_safety import reset_active_checkpoint, set_active_checkpoint


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
        session_id=None, role="", shared_evidence=None,
        desktop_operator_id=None,
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.role = role
        self.desktop_operator_id = desktop_operator_id
        self.shared_evidence = list(shared_evidence) if shared_evidence is not None else None
        self.max_steps = max_steps
        self.yolo = yolo

        # Mỗi agent có LLM client riêng
        self.llm = LLMClient(model_name)

        # All agents share persistent experience so later runs can reuse it.
        self.memory = Memory()

        # Mỗi agent có log file riêng
        self.logger = SessionLogger(agent_id=agent_id, session_id=session_id)

    def run(self, goal):
        """
        Chạy toàn bộ pipeline cho goal.
        Trả về dict kết quả để aggregator tổng hợp.
        """
        start = time.time()
        checkpoint_token = set_active_checkpoint(
            getattr(self, "file_checkpoint", None)
        )
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
            "role": self.role,
            "plan": None,
            "actions": [],
            "result": None,
            "status": "unknown",
            "elapsed": 0,
            "error": None,
            "failed_tasks": [],
            "research_evidence": list(self.shared_evidence or []),
            "github_skill_recommendations": [],
            "browser_session": {
                "agent_id": self.agent_id,
                "session_id": None,
                "status": "not_started",
                "isolated_context": True,
            },
        }

        try:
            set_session_logger(self.logger)
            self.logger.print("START", f"Agent {self.agent_id} bắt đầu với model={self.model_name}", "bold cyan")
            self.logger.log("start", {
                "goal": goal,
                "model": self.model_name,
                "role": self.role,
            })

            # Tìm kinh nghiệm cũ
            similars = self.memory.search_similar(goal, success_only=True)
            if similars:
                self.logger.print("MEMORY", f"Tìm thấy {len(similars)} kinh nghiệm cũ", "blue")
                experience_context = "\n".join(
                    f"- {str(item)[:1000]}" for item in similars
                )
                self.memory.add_short_term(
                    "system",
                    "Kinh nghiệm liên quan từ các lần chạy trước:\n"
                    f"{experience_context}",
                )
                self.logger.log("experience_reused", {
                    "count": len(similars),
                    "successful_only": True,
                    "context_characters": len(experience_context),
                })
            else:
                self.logger.log("experience_reused", {
                    "count": 0,
                    "successful_only": True,
                })

            # Lập kế hoạch
            self.logger.print("PLANNING", "Đang lập kế hoạch...", "cyan")
            planning_goal = goal
            if similars and not is_presentation_goal(goal):
                planning_goal += (
                    "\n\nKinh nghiệm liên quan từ các lần chạy trước "
                    "(chỉ áp dụng nếu phù hợp, không thay thế mục tiêu):\n"
                    + experience_context
                )
            if self.role:
                planning_goal += (
                    f"\n\nGóc độ riêng của Agent {self.agent_id}: {self.role} "
                    "Hãy tự lập kế hoạch và kết quả độc lập; vẫn phải đáp ứng "
                    "đầy đủ mục tiêu gốc."
                )
            if self.shared_evidence is not None:
                evidence_text = str(self.shared_evidence)
                self.logger.log("shared_research_received", {
                    "source_count": len(self.shared_evidence),
                    "urls": [
                        row.get("url") for row in self.shared_evidence
                        if isinstance(row, dict) and row.get("url")
                    ],
                })
                self.memory.add_short_term(
                    "system",
                    f"Shared validated research evidence: {evidence_text[:16000]}",
                )
                result["actions"].append({
                    "task_id": "shared_research",
                    "desc": "Preflight evidence pack shared by orchestrator",
                    "output": evidence_text[:16000],
                    "retry": 0,
                })
            plan = generate_plan(
                planning_goal,
                llm_client=self.llm,
                shared_evidence=self.shared_evidence,
                allow_desktop=self.agent_id == self.desktop_operator_id,
            )

            plan_issues = validate_plan(
                goal,
                plan,
                agent_id=self.agent_id,
                desktop_operator_id=self.desktop_operator_id,
            )
            if plan_issues:
                self.logger.log("plan_validation_failed", {
                    "attempt": 1,
                    "issues": plan_issues,
                    "next_step": "One targeted plan repair is allowed.",
                })
                repair_goal = (
                    f"{planning_goal}\n\nPlan policy rejected the previous plan. "
                    "Rebuild it once with tasks directly relevant to the original goal, "
                    "explicit success criteria, valid dependencies, and only permitted tools. "
                    f"Issues: {plan_issues}"
                )
                plan = generate_plan(
                    repair_goal,
                    llm_client=self.llm,
                    shared_evidence=self.shared_evidence,
                    allow_desktop=self.agent_id == self.desktop_operator_id,
                )
                plan_issues = validate_plan(
                    goal,
                    plan,
                    agent_id=self.agent_id,
                    desktop_operator_id=self.desktop_operator_id,
                )
                if plan_issues:
                    result["status"] = "plan_failed"
                    result["error"] = (
                        "Plan không có task hợp lệ hoặc không đạt policy sau một lần sửa: "
                        + "; ".join(plan_issues)
                    )
                    result["elapsed"] = time.time() - start
                    self.logger.print("ERROR", result["error"], "red")
                    self.logger.log("plan_rejected", {
                        "attempts": 2,
                        "issues": plan_issues,
                    })
                    return result

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
            if not tasks:
                result["status"] = "plan_failed"
                result["error"] = "Plan không có task nào để thực hiện."
                result["elapsed"] = time.time() - start
                self.logger.print("ERROR", result["error"], "red")
                self.logger.log("plan_failed", result["error"])
                return result
            self.logger.print("PLAN", f"Plan xong → {total} task", "green")
            self.logger.log("plan", plan)

            # Thực thi từng task
            step = 0
            task_idx = 0
            completed_tasks = 0
            all_outputs = []

            while task_idx < total and step < self.max_steps:
                task = copy.deepcopy(tasks[task_idx])
                if is_presentation_draft_task(task) and self.shared_evidence is not None:
                    task["research_evidence"] = self.shared_evidence
                    task["verified_source_urls"] = [
                        row.get("url") for row in self.shared_evidence
                        if isinstance(row, dict) and row.get("url")
                    ]
                self.logger.print(f"TASK {task_idx+1}/{total}", task["desc"], "bold magenta")

                retry_count = 0
                passed = False
                last_reason = ""
                attempts_made = 0
                task_started = time.perf_counter()

                while retry_count < 3 and not passed and step < self.max_steps:
                    step += 1
                    attempt_number = retry_count + 1
                    attempts_made += 1
                    self.logger.log("trial_attempt_started", {
                        "task_id": task.get("id"),
                        "attempt": attempt_number,
                        "max_attempts": 3,
                        "observation": "Use the latest tool output and prior failure as evidence.",
                        "hypothesis": task.get("desc"),
                        "strategy": "perform one bounded action, then evaluate its result",
                    })
                    try:
                        output = execute_task(
                            task,
                            yolo=self.yolo,
                            memory=self.memory,
                            logger=self.logger,
                            llm_client=self.llm,
                            agent_id=self.agent_id,
                            desktop_operator_id=self.desktop_operator_id,
                        )
                        recommendation = task.get("_github_skill_recommendation")
                        if isinstance(recommendation, dict):
                            summary = {
                                "task_id": task.get("id"),
                                **{
                                    key: recommendation.get(key)
                                    for key in (
                                        "status", "skill", "repo_url", "stars",
                                        "license", "reason",
                                    )
                                },
                            }
                            if summary not in result["github_skill_recommendations"]:
                                result["github_skill_recommendations"].append(summary)
                    except Exception as exc:
                        failed_fingerprint = task.get("_last_action_fingerprint")
                        if failed_fingerprint:
                            task.setdefault("_failed_action_fingerprints", []).append(
                                failed_fingerprint
                            )
                        last_reason = (
                            f"Tool execution error ({type(exc).__name__}): "
                            f"{str(exc)[:300]}"
                        )
                        result["actions"].append({
                            "task_id": task["id"],
                            "desc": task["desc"],
                            "output": last_reason,
                            "retry": retry_count,
                        })
                        retry_count += 1
                        can_retry, retry_reason = retry_decision(
                            task.get("_selected_tool"),
                            task.get("_selected_action"),
                            retry_count,
                            max_attempts=2 if is_presentation_draft_task(task) else 3,
                        )
                        self.logger.print(
                            "TOOL ERROR",
                            f"{last_reason} ({'sẽ thử lại' if can_retry else 'dừng'}; {retry_reason})",
                            "red",
                        )
                        self.logger.log("tool_execution_failed", {
                            "task_id": task["id"],
                            "reason": last_reason,
                            "retry": retry_count,
                            "tool": task.get("_selected_tool"),
                            "action": task.get("_selected_action"),
                            "retry_allowed": can_retry,
                            "retry_reason": retry_reason,
                        })
                        self.logger.log("trial_recovery_decision", {
                            "task_id": task.get("id"),
                            "tool": task.get("_selected_tool"),
                            "action": task.get("_selected_action"),
                            "attempt": retry_count,
                            "retry_allowed": can_retry,
                            "reason": retry_reason,
                        })
                        SkillResolver.record_outcome(
                            [
                                (task.get("_resolved_skill") or {}).get("skill"),
                                task.get("_selected_tool"),
                            ],
                            False,
                            memory=self.memory,
                            logger=self.logger,
                        )
                        if is_search_task(task):
                            self.logger.log("search_retry_stopped", {
                                "task_id": task.get("id"),
                                "reason": "Search tool owns provider fallbacks; identical task is not repeated.",
                            })
                            break
                        if isinstance(exc, BrowserToolError) and not exc.retryable:
                            self.logger.log("browser_retry_stopped", {
                                "task_id": task.get("id"),
                                "code": exc.code,
                                "reason": last_reason,
                            })
                            break
                        if not can_retry:
                            self.logger.log("trial_recovery_stopped", {
                                "task_id": task.get("id"),
                                "tool": task.get("_selected_tool"),
                                "action": task.get("_selected_action"),
                                "attempt": retry_count,
                                "reason": retry_reason,
                                "next_step": "Report failure; do not replay an action with possible side effects.",
                            })
                            break
                        task["desc"] += f" [TOOL ERROR: {last_reason}]"
                        continue

                    if is_presentation_draft_task(task) and isinstance(output, str):
                        repaired_output = append_references_from_evidence(
                            output,
                            task.get("research_evidence", []),
                        )
                        if repaired_output != output:
                            output = repaired_output
                            self.logger.log("presentation_references_repaired", {
                                "task_id": task["id"],
                                "source_count": len(task.get("research_evidence", [])),
                                "method": "append exact references from validated evidence",
                            })

                    action_output_limit = (
                        16000
                        if task.get("_selected_tool") in {"search_tool", "presentation_draft"}
                        else 5000
                    )
                    stored_output = (
                        sanitize_desktop_output(output)
                        if task.get("_selected_tool") == "computer_tool"
                        else str(output)
                    )
                    result["actions"].append({
                        "task_id": task["id"],
                        "desc": task["desc"],
                        "output": stored_output[:action_output_limit],
                        "retry": retry_count,
                    })

                    # Critic đánh giá; malformed responses never imply success.
                    if isinstance(output, str) and output.startswith("ERROR ["):
                        critic_res = {
                            "passed": False,
                            "reason": output[:500],
                            "fix": "Chọn một thao tác khác trong phạm vi được cấp; không lặp thao tác lỗi.",
                        }
                    elif is_search_task(task):
                        search_status = getattr(output, "status", "unknown")
                        source_rows_valid = (
                            isinstance(output, list)
                            and len(output) >= 3
                            and all(isinstance(row, dict) for row in output)
                            and all(
                                all(row.get(field) for field in (
                                    "title", "url", "snippet", "source",
                                ))
                                for row in output
                            )
                        )
                        if search_status != "ok" or not source_rows_valid:
                            critic_res = None
                            passed = False
                            reason = (
                                getattr(output, "message", "")
                                or f"Tìm kiếm chưa đạt: status={search_status}, "
                                   f"valid_sources={len(output) if isinstance(output, list) else 0}/3."
                            )
                            fix = (
                                "Thử đúng một truy vấn hẹp hơn, giữ chủ đề AI trong giáo dục; "
                                "nếu vẫn thiếu nguồn phù hợp thì dừng."
                            )
                        else:
                            critic_res = evaluate_task(task, str(output), llm_client=self.llm)
                    else:
                        critic_res = evaluate_task(task, str(output), llm_client=self.llm)
                    if critic_res is not None:
                        if isinstance(critic_res, dict) and isinstance(
                            critic_res.get("passed"), bool
                        ):
                            passed = critic_res["passed"]
                            reason = str(critic_res.get("reason", ""))
                            fix = str(critic_res.get("fix", ""))
                        else:
                            passed = False
                            reason = "Critic không trả JSON hợp lệ; không thể xác nhận thành công."
                            fix = "Thử lại tối đa một lần với đầu ra rõ hơn; nếu vẫn lỗi thì dừng."
                    last_reason = reason
                    if not passed and task.get("_last_action_fingerprint"):
                        task.setdefault("_failed_action_fingerprints", []).append(
                            task["_last_action_fingerprint"]
                        )
                    self.logger.log("trial_attempt_evaluated", {
                        "task_id": task.get("id"),
                        "attempt": attempt_number,
                        "tool": task.get("_selected_tool"),
                        "action": task.get("_selected_action"),
                        "passed": passed,
                        "observation": (
                            sanitize_desktop_output(output)[:1200]
                            if task.get("_selected_tool") == "computer_tool"
                            else str(output)[:1200]
                        ),
                        "reason": reason,
                        "targeted_fix": fix,
                    })

                    if passed:
                        self.logger.print("CRITIC", f"Pass: {reason}", "green")
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
                        if is_search_task(task) and isinstance(output, list):
                            self.shared_evidence = list(output)
                            result["research_evidence"] = list(output)
                        task["research_evidence"] = list(self.shared_evidence or [])
                        task["verified_source_urls"] = [
                            row.get("url") for row in (self.shared_evidence or [])
                            if isinstance(row, dict) and row.get("url")
                        ]
                        all_outputs.append(
                            sanitize_desktop_output(output)[:16000]
                            if task.get("_selected_tool") == "computer_tool"
                            else str(output)[:16000]
                        )
                        break
                    else:
                        if is_search_task(task):
                            retry_count += 1
                            self.logger.log("search_retry_stopped", {
                                "task_id": task.get("id"),
                                "reason": "Search tool exhausted provider fallbacks; no identical retry.",
                            })
                            SkillResolver.record_outcome(
                                [
                                    (task.get("_resolved_skill") or {}).get("skill"),
                                    task.get("_selected_tool"),
                                ],
                                False,
                                memory=self.memory,
                                logger=self.logger,
                            )
                            break
                        if (
                            task.get("_selected_tool") == "computer_tool"
                            and task.get("_selected_action") == "perform"
                        ):
                            retry_count += 1
                            proposal = None
                            recovery_reason = "Desktop recovery attempt limit reached."
                            if retry_count < 2 and step < self.max_steps:
                                proposal, recovery_reason = propose_desktop_recovery(
                                    task.get("desc", goal),
                                    output,
                                    reason,
                                    self.llm,
                                )
                            self.logger.log("desktop_recovery", {
                                "task_id": task.get("id"),
                                "attempt": retry_count,
                                "planned": proposal is not None,
                                "reason": recovery_reason,
                                "action_summary": desktop_action_summary(
                                    proposal.get("actions") if proposal else []
                                ),
                            })
                            SkillResolver.record_outcome(
                                [
                                    (task.get("_resolved_skill") or {}).get("skill"),
                                    task.get("_selected_tool"),
                                ],
                                False,
                                memory=self.memory,
                                logger=self.logger,
                            )
                            if proposal is not None:
                                task["_desktop_recovery_actions"] = proposal
                                task["desc"] += (
                                    " [DESKTOP RECOVERY: perform one new action based on "
                                    "the latest OCR state; do not replay the previous batch.]"
                                )
                                continue
                            break
                        if is_presentation_draft_task(task):
                            retry_count += 1
                            can_retry, retry_reason = retry_decision(
                                task.get("_selected_tool"),
                                task.get("_selected_action"),
                                retry_count,
                                max_attempts=2,
                            )
                            self.logger.log("presentation_repair", {
                                "task_id": task.get("id"),
                                "attempt": retry_count,
                                "reason": reason,
                                "fix": fix,
                                "retry_allowed": can_retry,
                                "retry_reason": retry_reason,
                            })
                            if can_retry and step < self.max_steps:
                                task["desc"] += f" [SỬA CÓ MỤC TIÊU: {fix}]"
                                continue
                            break
                        retry_count += 1
                        can_retry, retry_reason = retry_decision(
                            task.get("_selected_tool"),
                            task.get("_selected_action"),
                            retry_count,
                        )
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
                            f"Fail ({retry_count}/3): {reason}. Gợi ý sửa: {fix}. {retry_reason}",
                            "red",
                        )
                        self.logger.log("trial_recovery_decision", {
                            "task_id": task.get("id"),
                            "tool": task.get("_selected_tool"),
                            "action": task.get("_selected_action"),
                            "attempt": retry_count,
                            "retry_allowed": can_retry,
                            "reason": retry_reason,
                        })
                        if not can_retry:
                            break
                        task["desc"] += f" [SỬA LỖI: {fix}]"

                experience = {
                    "status": "success" if passed else "failed",
                    "tool": task.get("_selected_tool"),
                    "action": task.get("_selected_action"),
                    "attempts": attempts_made,
                    "duration_seconds": round(time.perf_counter() - task_started, 3),
                    "lesson": last_reason,
                    "github_skill_recommendation": {
                        key: task["_github_skill_recommendation"].get(key)
                        for key in ("status", "skill", "repo_url", "stars", "license")
                    } if isinstance(task.get("_github_skill_recommendation"), dict) else None,
                }
                saved = self.memory.save_long_term(
                    task.get("id", "unknown"),
                    task.get("desc", goal),
                    experience,
                    success=passed,
                    tool=task.get("_selected_tool"),
                    agent_id=self.agent_id,
                )
                self.logger.log("experience_learned", {
                    "task_id": task.get("id", "unknown"),
                    "success": passed,
                    "tool": task.get("_selected_tool"),
                    "attempts": attempts_made,
                    "duration_seconds": experience["duration_seconds"],
                    "persisted": bool(saved),
                })

                if not passed:
                    failure = {
                        "task_id": task.get("id"),
                        "description": task["desc"],
                        "reason": last_reason,
                        "attempts": retry_count,
                    }
                    result["failed_tasks"].append(failure)
                    reason_text = last_reason or "Vượt giới hạn số bước."
                    self.logger.print(
                        "TASK FAILED",
                        f"{task.get('id', '?')} thất bại sau {retry_count} lần: {reason_text}",
                        "red",
                    )
                    self.logger.log("task_failed", failure)
                    all_outputs.append(
                        f"FAIL: {task['desc']} | reason: {reason_text}"
                    )

                task_idx += 1

                # Reflection mỗi 5 bước
                if step % 5 == 0 and step > 0:
                    history = self.memory.get_short_term_context()
                    ref = reflect(history, llm_client=self.llm)
                    self.logger.print("REFLECT", str(ref), "cyan")
                    self.logger.log("reflection", ref)

            # Tổng kết
            result["result"] = (
                "\n".join(all_outputs)[:20000]
                if all_outputs
                else "Không có kết quả"
            )
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
            current_browser_session = session_summary(self.agent_id)
            retain_browser_session = getattr(
                self, "retain_browser_session", False
            )
            if current_browser_session.get("status") != "not_started":
                current_browser_session["status_at_agent_exit"] = current_browser_session[
                    "status"
                ]
                current_browser_session["status"] = (
                    "retained_for_leader"
                    if retain_browser_session
                    else "closed_after_agent"
                )
                result["browser_session"] = current_browser_session
                self.logger.log("browser_session_summary", current_browser_session)
            if not retain_browser_session:
                try:
                    close_session(self.agent_id)
                except Exception as exc:
                    self.logger.print(
                        "BROWSER CLEANUP",
                        f"Không đóng được phiên Playwright: {type(exc).__name__}: {exc}",
                        "red",
                    )
                    self.logger.log("browser_cleanup_failed", {
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                    })
                set_session_logger(None)
            reset_active_checkpoint(checkpoint_token)
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
