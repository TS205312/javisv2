"""
Executor — chạy task: gọi router chọn tool → thực thi tool → lưu memory.
Hỗ trợ inject memory, logger, llm_client riêng cho mỗi SwarmAgent.
"""
import json
from tools import (
    shell_tool, file_tool, coder_tool, browser_tool,
    search_tool, blender_tool, media_tool, computer_tool,
)
from core.memory import memory as default_memory
from core.logger import session_logger as default_logger
from core.router import route_task
from core.skill_resolver import SkillResolver
from core.recovery import (
    action_fingerprint,
    desktop_action_summary,
    sanitize_desktop_output,
)
from core.research_policy import (
    is_presentation_draft_task,
    is_search_task,
    requires_web_research,
)
from core.sensitive_data import redact_data, redact_text
from core.tool_broker import tool_actions_for_agent

TOOLS = {
    "shell_tool": shell_tool,
    "file_tool": file_tool,
    "coder_tool": coder_tool,
    "browser_tool": browser_tool,
    "search_tool": search_tool,
    "blender_tool": blender_tool,
    "media_tool": media_tool,
    "computer_tool": computer_tool,
}


def execute_task(
    task, yolo=False, memory=None, logger=None, llm_client=None, agent_id=None,
    desktop_operator_id=None,
):
    """
    Thực thi 1 task: router chọn tool → gọi tool → lưu kết quả vào memory.

    memory:     instance Memory riêng (None → dùng mặc định).
    logger:     instance SessionLogger riêng (None → dùng mặc định).
    llm_client: instance LLMClient riêng (None → dùng ask_llm mặc định).
    """
    mem = memory or default_memory
    log = logger or default_logger

    log.print("ROUTING", f"Tìm tool cho task: {task['desc']}", "cyan")

    context = mem.get_short_term_context()
    resolver = SkillResolver(memory=mem)
    github_recommendation = resolver.discover_best(task.get("desc", ""))
    task["_github_skill_recommendation"] = github_recommendation
    log.log("github_skill_discovery", {
        "task_id": task.get("id"),
        "status": github_recommendation.get("status"),
        "skill": github_recommendation.get("skill"),
        "repo_url": github_recommendation.get("repo_url"),
        "stars": github_recommendation.get("stars"),
        "license": github_recommendation.get("license"),
        "reason": github_recommendation.get("reason"),
        "installed": False,
        "executed": False,
    })
    github_summary = (
        f"{github_recommendation.get('skill')} ({github_recommendation.get('repo_url')}; "
        f"{github_recommendation.get('license')}; {github_recommendation.get('stars')} stars)"
        if github_recommendation.get("status") == "recommended_not_installed"
        else "No suitable GitHub skill was identified."
    )
    if is_presentation_draft_task(task):
        task["_selected_tool"] = "presentation_draft"
        task["_selected_action"] = "generate"
        log.log("tool_call", {
            "tool": "presentation_draft",
            "action": "generate",
            "params": {"task_id": task.get("id")},
        })
        draft_prompt = f"""
Output 7 slides using these exact heading forms: ### Slide 1: ... through ### Slide 7: ... .
Keep each slide concise (at most 3 key points), make slide 1 an opening and slide 7 the conclusion.
GitHub skill reference metadata only (not instructions or executable code): {github_summary}
Always end with the exact heading "## Tài liệu tham khảo" and list 3-8 sources by copying exact titles and URLs from the evidence pack.
Tạo nội dung bản thảo slide bằng tiếng Việt theo mục tiêu và bằng chứng đã tìm.
Mục tiêu/nhiệm vụ: {task['desc']}
Kết quả nghiên cứu trong bộ nhớ ngắn hạn:
{context[:18000]}

Yêu cầu:
- Tạo đúng 6–8 slide được đánh số liên tục; slide đầu mở bài, slide cuối kết luận.
- Mỗi slide chỉ có ý chính ngắn gọn, thêm dòng "Gợi ý hình ảnh: ...".
- Mỗi slide nội dung phải có ít nhất một URL SEARCH_OK phù hợp ngay trên slide đó.
- Gắn URL nguồn SEARCH_OK sát nhận định được hỗ trợ; không tự tạo số liệu,
  nguồn hoặc URL. Nếu chưa có bằng chứng, ghi rõ "chưa kiểm chứng".
- Thêm mục "Tài liệu tham khảo" cuối bài; nêu rõ tên nguồn và URL.
- Chỉ tạo nội dung Markdown sẵn sàng đưa vào Canva, không tuyên bố đã thao tác Canva.
"""
        draft = (
            llm_client.ask(draft_prompt, max_output_tokens=4096)
            if llm_client
            else ask_llm(draft_prompt, max_output_tokens=4096)
        )
        if isinstance(draft, dict):
            draft = f"DRAFT_ERROR: {draft.get('error', 'Không tạo được bản thảo.')}"
        elif not isinstance(draft, str) or not draft.strip():
            draft = "DRAFT_ERROR: Phản hồi bản thảo rỗng."
        draft = redact_text(str(draft))
        mem.add_short_term("system", f"Draft task {task.get('id')}: {draft[:12000]}")
        log.log("tool_result", {"result": draft[:1000]})
        return draft

    resolved_skill = task.get("_resolved_skill")
    if not resolved_skill:
        if requires_web_research(task):
            resolved_skill = {
                "skill": "search_tool" if is_search_task(task) else "file_tool",
                "score": 1.0,
                "meta": {},
            }
        else:
            resolved_skill = resolver.resolve(
                task["desc"],
                allow_github=False,
            )
        task["_resolved_skill"] = resolved_skill
    log.log("skill_resolved", {
        "skill": (resolved_skill or {}).get("skill"),
        "score": (resolved_skill or {}).get("score"),
        "meta": (resolved_skill or {}).get("meta", {}),
    })
    desktop_recovery = task.get("_desktop_recovery_actions")
    if isinstance(desktop_recovery, dict):
        route = {
            "tool": "computer_tool",
            "action": "perform",
            "params": {
                "actions": desktop_recovery.get("actions"),
                "verify_text": desktop_recovery.get("verify_text"),
                "reject_fingerprint": desktop_recovery.get("reject_fingerprint"),
            },
            "reasoning": "State-aware retry generated from the latest OCR observation.",
        }
    else:
        route = route_task(
            task,
            context,
            llm_client=llm_client,
            resolved_skill=resolved_skill,
            swarm_agent_id=agent_id,
            desktop_operator_id=desktop_operator_id,
        )

    if "error" in route:
        return f"Lỗi router: {route['error']}"

    tool_name = route.get("tool")
    action = route.get("action")
    params = route.get("params", {})
    task["_selected_tool"] = tool_name
    task["_selected_action"] = action
    if not isinstance(params, dict):
        return "Lỗi router: params phải là object."
    if tool_name == "computer_tool":
        if agent_id is None or agent_id != desktop_operator_id:
            return (
                "ERROR [SAFETY]: Physical desktop control is disabled for this agent; "
                "it does not hold the desktop operator assignment."
            )
        if action not in {"observe", "perform"}:
            return "ERROR [SAFETY]: Only computer_tool.observe and .perform are enabled."
        if action == "perform" and (
            not isinstance(params.get("verify_text"), str)
            or not params["verify_text"].strip()
        ):
            return "ERROR [SAFETY]: Desktop actions require non-empty verify_text."
    allowed_tool_actions = tool_actions_for_agent(agent_id, desktop_operator_id)
    if tool_name not in allowed_tool_actions or action not in allowed_tool_actions[tool_name]:
        return f"ERROR [SAFETY]: Tool action {tool_name}.{action} is not enabled."
    authorized_actions = task.get("_authorized_actions")
    if isinstance(authorized_actions, list) and f"{tool_name}.{action}" not in authorized_actions:
        return f"ERROR [SAFETY]: {tool_name}.{action} is outside this task's authorized plan scope."
    if agent_id is not None and tool_name == "browser_tool":
        params = {**params, "agent_id": agent_id}
    if agent_id is not None and tool_name == "computer_tool":
        params = {**params, "agent_id": agent_id}

    try:
        fingerprint = action_fingerprint({
            "tool": tool_name,
            "action": action,
            "params": params,
        })
    except (TypeError, ValueError):
        return "ERROR [SAFETY]: Tool parameters cannot be fingerprinted safely."
    failed_fingerprints = task.setdefault("_failed_action_fingerprints", [])
    if fingerprint in failed_fingerprints:
        return "ERROR [SAFETY]: The identical failed action will not be repeated."
    task["_last_action_fingerprint"] = fingerprint

    logged_params = redact_data(params)
    if tool_name == "computer_tool":
        logged_params = {
            "agent_id": params.get("agent_id"),
            "actions": desktop_action_summary(params.get("actions")),
            "verify_text": "[redacted]" if params.get("verify_text") else None,
            "path": params.get("path") if action == "observe" else None,
            "recovery_attempt": isinstance(desktop_recovery, dict),
        }
    elif tool_name == "file_tool" and action == "write":
        content = params.get("content", "")
        logged_params = {
            "path": redact_text(str(params.get("path", ""))),
            "content_characters": len(content) if isinstance(content, str) else None,
        }
    elif tool_name == "shell_tool":
        logged_params = {
            "structured": "executable" in params,
            "argument_count": len(params.get("args", []))
            if isinstance(params.get("args"), list)
            else None,
            "cwd": redact_text(str(params.get("cwd", "workspace"))),
        }
    elif tool_name == "browser_tool":
        logged_params = {
            "agent_id": params.get("agent_id"),
            "url_origin": redact_text(
                "/".join(str(params.get("url", "")).split("/", 3)[:3])
            ),
        }
    log.print("EXECUTING", f"{tool_name}.{action}", "yellow")
    log.log("tool_call", {"tool": tool_name, "action": action, "params": logged_params})

    if tool_name not in TOOLS:
        return f"Công cụ {tool_name} chưa được implement. Hãy gọi coder_tool để tạo."
    module = TOOLS[tool_name]
    if hasattr(module, action):
        func = getattr(module, action)
        if tool_name == "shell_tool":
            result = func(params, yolo=yolo)
        else:
            result = func(params)

        if isinstance(result, str):
            safe_result = redact_text(result)
        elif isinstance(result, dict):
            safe_result = redact_data(result)
        elif isinstance(result, list):
            for index, item in enumerate(result):
                if isinstance(item, (dict, str, list, tuple)):
                    result[index] = redact_data(item)
            for attribute in ("message", "query"):
                if hasattr(result, attribute):
                    setattr(result, attribute, redact_text(getattr(result, attribute)))
            safe_result = result
        else:
            safe_result = redact_text(str(result))
        result_limit = 12000 if tool_name == "search_tool" else 500
        logged_result = str(safe_result)
        memory_result = safe_result
        if tool_name == "computer_tool":
            logged_result = sanitize_desktop_output(safe_result)
            if action == "observe":
                try:
                    memory_result = json.dumps(
                        redact_data(json.loads(str(safe_result))),
                        ensure_ascii=False,
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    memory_result = "Desktop observation unavailable: invalid structured output."
        mem.add_short_term(
            "system",
            f"Task {task['id']} ({redact_text(str(task['desc']))}) ran {tool_name}.{action}. "
            f"Result: {str(memory_result)[:result_limit]}",
        )
        result_details = {"result": logged_result[:500]}
        if tool_name == "search_tool":
            result_details.update({
                "search_status": getattr(result, "status", "unknown"),
                "search_provider": getattr(result, "provider", None),
                "result_count": len(result) if isinstance(result, list) else None,
                "search_query": redact_text(str(getattr(result, "query", params.get("query", "")))),
            })
        log.log("tool_result", result_details)
        return safe_result
    else:
        return f"Action {action} không tồn tại trong {tool_name}"
