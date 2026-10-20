"""
Router — LLM chọn tool phù hợp từ registry cho mỗi task.
Hỗ trợ inject llm_client riêng cho mỗi SwarmAgent.
"""
from core.llm_helper import ask_llm
from core.skill_resolver import SkillResolver
from registry import get_all_skills
from core.research_policy import is_search_task
from core.research_policy import requires_web_research
from core.tool_broker import tool_actions_for_agent
import json


def route_task(
    task, memory_context, llm_client=None, resolved_skill=None,
    swarm_agent_id=None, desktop_operator_id=None,
):
    """
    Resolve a skill first, then ask the LLM for the built-in tool action.
    llm_client: instance LLMClient (nếu None → dùng hàm mặc định ask_llm).
    """
    if is_search_task(task):
        query = task.get("search_query") or task.get("desc", "")
        return {
            "tool": "search_tool",
            "action": "search",
            "params": {"query": query, "min_results": 3},
            "reasoning": "Nhiệm vụ yêu cầu tìm nguồn web; dùng search_tool tích hợp, không cài skill/package.",
        }

    _ask = llm_client.ask if llm_client else ask_llm
    skill = resolved_skill or SkillResolver().resolve(task["desc"])
    available_tools = get_all_skills()
    tools_info = json.dumps(
        {
            name: {
                **available_tools[name],
                "actions": sorted(actions),
            }
            for name, actions in tool_actions_for_agent(
                swarm_agent_id, desktop_operator_id
            ).items()
            if name in available_tools
        },
        ensure_ascii=False,
        indent=2,
    )
    selected_skill = json.dumps(skill, ensure_ascii=False, indent=2)
    github_reference = task.get("_github_skill_recommendation")
    if isinstance(github_reference, dict) and github_reference.get("status") == "recommended_not_installed":
        selected_skill = json.dumps({
            "resolved_local_skill": skill,
            "github_reference_metadata_only": {
                key: github_reference.get(key)
                for key in ("skill", "repo_url", "stars", "license")
            },
        }, ensure_ascii=False, indent=2)
    swarm_policy = (
        "Bạn đang chạy trong swarm, agent ID "
        f"{swarm_agent_id}. Dùng browser_tool cho web để giữ phiên riêng. "
        "Không dùng computer_tool hoặc điều khiển desktop vật lý: tính năng này "
        "đang bị vô hiệu hóa cho agent. Chỉ chọn công cụ xuất hiện trong danh sách "
        "Công cụ và được policy cho phép."
        if swarm_agent_id is not None
        else ""
    )
    if swarm_agent_id is not None and swarm_agent_id == desktop_operator_id:
        swarm_policy = (
            f"Agent {swarm_agent_id} is the only selected desktop operator. "
            "Use computer_tool.observe / computer_tool.perform only when the plan authorizes them. "
            "Observe the screen before acting; use its OCR and coordinates as evidence, "
            "then perform one small bounded batch and include verify_text. Never replay a failed batch."
        )
    research_policy = (
        "Đây là tác vụ nghiên cứu/trình bày: chỉ dùng search_tool cho tìm nguồn; "
        "không chạy lệnh cài package, không tìm/cài repository hoặc skill ngoài."
        if requires_web_research(task.get("desc", ""))
        else ""
    )
    prompt = f"""
Resolver đã chọn skill liên quan dưới đây. Dùng skill/package này khi phù hợp,
nhưng chỉ gọi một tool tích hợp sẵn trong registry để thực thi hành động.
Nhiệm vụ: {task['desc']}
Bối cảnh trước đó: {memory_context}
Skill đã resolve: {selected_skill}
{swarm_policy}
GitHub reference metadata is not executable; never clone, install, import, or run that repo.
{research_policy}

Công cụ: {tools_info}

Trả về JSON định dạng:
{{
  "tool": "tên tool (vd: shell_tool)",
  "action": "action của tool",
  "params": {{
    "key": "value"
  }},
  "reasoning": "Lý do chọn"
}}
Lưu ý: Nếu cần chạy script Python, chỉ dùng shell_tool với executable và args "
được policy cho phép. Không gọi công cụ ngoài danh sách. Trả về JSON thuần.
"""
    return _ask(prompt, json_mode=True)
