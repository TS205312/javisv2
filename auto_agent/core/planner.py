"""
Planner — LLM sinh kế hoạch JSON từ goal tiếng Việt.
Hỗ trợ inject llm_client riêng cho mỗi SwarmAgent.
"""
from core.llm_helper import ask_llm
from registry import TOOL_REGISTRY
from core.research_policy import (
    extract_research_topic,
    is_presentation_goal,
    is_search_task,
    requires_web_research,
)
from core.tool_broker import tool_actions_for_agent
import json
import re


def _generate_plan(goal, llm_client=None, shared_evidence=None, allow_desktop=False):
    if is_presentation_goal(goal):
        topic = extract_research_topic(goal)
        role_match = re.search(
            r"Góc độ riêng của Agent \d+:\s*(.*?)\s+Hãy tự lập kế hoạch",
            str(goal),
            flags=re.IGNORECASE | re.DOTALL,
        )
        role_focus = role_match.group(1).strip() if role_match else ""
        tasks = []
        research_id = "research_sources"
        if shared_evidence is None:
            tasks.append({
                "id": research_id,
                "type": "research",
                "desc": f"Tìm kiếm ít nhất 3 nguồn đáng tin cậy về: {topic}",
                "search_query": topic,
                "deps": [],
                "allowed_tools": ["search_tool.search"],
                "success": (
                    "Có ít nhất 3 kết quả liên quan với tiêu đề, URL HTTP(S), tên nguồn "
                    "và đoạn trích/tóm tắt đủ đánh giá."
                ),
            })

        draft_desc = (
            f'Soạn nội dung 6–8 slide tiếng Việt về "{topic}"; '
            "dùng evidence pack đã tìm, trích dẫn nguồn liên quan trong slide, "
            "thêm gợi ý hình ảnh và phần tài liệu tham khảo."
        )
        if role_focus:
            draft_desc += f" Góc độ phản biện cần ưu tiên: {role_focus}"
        draft_task = {
            "id": "create_slides",
            "type": "presentation_draft",
            "desc": draft_desc,
            "topic": topic,
            "deps": [research_id] if shared_evidence is None else [],
            "allowed_tools": ["presentation_draft.generate"],
            "success": (
                "Có 6–8 slide đánh số liên tục, mở bài và kết luận; mỗi slide có "
                "gợi ý hình ảnh và nguồn được trích từ evidence pack; có tài liệu tham khảo."
            ),
        }
        if shared_evidence is not None:
            draft_task["research_evidence"] = list(shared_evidence)
        tasks.append(draft_task)
        return {
            "understanding": f"Nghiên cứu và trình bày chủ đề: {topic}",
            "assumptions": "Dùng nguồn tìm kiếm đã cấu trúc; không tự tạo dữ kiện hoặc URL.",
            "tasks": tasks,
            "risks": "Nếu không đủ nguồn liên quan hoặc cấu trúc slide không đạt, báo thất bại rõ ràng.",
        }

    _ask = llm_client.ask if llm_client else ask_llm
    tools_info = json.dumps(
        {
            **{
                name: {
                    **TOOL_REGISTRY[name],
                    "actions": sorted(actions),
                }
            for name, actions in tool_actions_for_agent(
                1 if allow_desktop else None,
                1 if allow_desktop else None,
            ).items()
                if name in TOOL_REGISTRY
            },
            "presentation_draft": {
                "desc": "Create Markdown slide content from validated research evidence.",
                "actions": ["generate"],
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt = f"""
Mục tiêu của người dùng: "{goal}"
Nhiệm vụ của bạn: Tạo một kế hoạch chi tiết để hoàn thành mục tiêu này.
Nếu mục tiêu mơ hồ, hãy tự đặt ra giả định hợp lý.

Công cụ có sẵn:
{tools_info}

Trả về ĐÚNG định dạng JSON sau (không chứa gì thêm):
{{
  "understanding": "Hiểu biết về mục tiêu",
  "assumptions": "Các giả định (nếu mục tiêu mơ hồ)",
  "tasks": [
    {{
      "id": "t1",
      "desc": "Mô tả bước làm",
      "deps": [],
      "allowed_tools": ["tool.action"],
      "success": "Tiêu chí thành công của bước này"
    }}
  ],
  "risks": "Rủi ro có thể gặp"
}}
"""
    plan = _ask(prompt, json_mode=True)
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        return plan

    tasks = [task for task in plan["tasks"] if isinstance(task, dict)]
    plan["tasks"] = tasks
    for task in tasks:
        if is_search_task(task):
            task["type"] = "research"
    if requires_web_research(goal):
        search_index = next((
            index for index, task in enumerate(tasks)
            if is_search_task(task)
        ), None)
        if search_index is None:
            research_task = {
                "id": "research_sources",
                "type": "research",
                "desc": (
                    "Tìm kiếm web và thu thập ít nhất 3 nguồn liên quan, có tiêu đề, "
                    "URL và đoạn tóm tắt; loại nguồn lạc đề hoặc thiếu thông tin."
                ),
                "deps": [],
                "success": (
                    "search_tool trả ít nhất 3 kết quả liên quan, mỗi kết quả có tiêu đề, "
                    "URL HTTP(S), nguồn và đoạn trích/tóm tắt đủ đánh giá."
                ),
            }
            tasks.insert(0, research_task)
            search_id = research_task["id"]
        else:
            research_task = tasks.pop(search_index)
            research_task["id"] = research_task.get("id") or "research_sources"
            research_task["type"] = "research"
            research_task["deps"] = []
            research_task["success"] = (
                "Có ít nhất 3 kết quả tìm kiếm liên quan, tiêu đề, URL HTTP(S), "
                "nguồn và đoạn trích/tóm tắt đủ đánh giá."
            )
            tasks.insert(0, research_task)
            search_id = research_task["id"]

        for task in tasks[1:]:
            deps = task.get("deps")
            if isinstance(deps, list) and search_id not in deps:
                deps.insert(0, search_id)

    return plan


def generate_plan(goal, llm_client=None, shared_evidence=None, allow_desktop=False):
    """Generate a plan, using a deterministic presentation pipeline when applicable."""
    return _generate_plan(
        goal,
        llm_client=llm_client,
        shared_evidence=shared_evidence,
        allow_desktop=allow_desktop,
    )
