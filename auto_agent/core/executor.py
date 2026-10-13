"""
Executor — chạy task: gọi router chọn tool → thực thi tool → lưu memory.
Hỗ trợ inject memory, logger, llm_client riêng cho mỗi SwarmAgent.
"""
from tools import (
    shell_tool, file_tool, coder_tool, browser_tool,
    search_tool, blender_tool, media_tool, computer_tool, vision_tool,
)
from core.memory import memory as default_memory
from core.logger import session_logger as default_logger
from core.router import route_task
from core.skill_resolver import SkillResolver

TOOLS = {
    "shell_tool": shell_tool,
    "file_tool": file_tool,
    "coder_tool": coder_tool,
    "browser_tool": browser_tool,
    "search_tool": search_tool,
    "blender_tool": blender_tool,
    "media_tool": media_tool,
    "computer_tool": computer_tool,
    "vision_tool": vision_tool,
}


def execute_task(task, yolo=False, memory=None, logger=None, llm_client=None):
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
    resolved_skill = task.get("_resolved_skill")
    if not resolved_skill:
        resolved_skill = SkillResolver(memory=mem).resolve(task["desc"])
        task["_resolved_skill"] = resolved_skill
    log.log("skill_resolved", {
        "skill": (resolved_skill or {}).get("skill"),
        "score": (resolved_skill or {}).get("score"),
        "meta": (resolved_skill or {}).get("meta", {}),
    })
    route = route_task(
        task,
        context,
        llm_client=llm_client,
        resolved_skill=resolved_skill,
    )

    if "error" in route:
        return f"Lỗi router: {route['error']}"

    tool_name = route.get("tool")
    action = route.get("action")
    params = route.get("params", {})

    log.print("EXECUTING", f"{tool_name}.{action} with {params}", "yellow")
    log.log("tool_call", {"tool": tool_name, "action": action, "params": params})

    if tool_name not in TOOLS:
        return f"Công cụ {tool_name} chưa được implement. Hãy gọi coder_tool để tạo."
    task["_selected_tool"] = tool_name

    module = TOOLS[tool_name]
    if hasattr(module, action):
        func = getattr(module, action)
        if tool_name == "shell_tool":
            result = func(params, yolo=yolo)
        else:
            result = func(params)

        mem.add_short_term(
            "system",
            f"Task {task['id']} ({task['desc']}) ran {tool_name}.{action}. "
            f"Result: {str(result)[:200]}",
        )
        log.log("tool_result", {"result": str(result)[:500]})
        return result
    else:
        return f"Action {action} không tồn tại trong {tool_name}"
