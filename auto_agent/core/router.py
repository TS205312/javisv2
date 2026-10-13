"""
Router — LLM chọn tool phù hợp từ registry cho mỗi task.
Hỗ trợ inject llm_client riêng cho mỗi SwarmAgent.
"""
from core.llm_helper import ask_llm
from core.skill_resolver import SkillResolver
from registry import get_all_skills
import json


def route_task(task, memory_context, llm_client=None, resolved_skill=None):
    """
    Resolve a skill first, then ask the LLM for the built-in tool action.
    llm_client: instance LLMClient (nếu None → dùng hàm mặc định ask_llm).
    """
    _ask = llm_client.ask if llm_client else ask_llm
    skill = resolved_skill or SkillResolver().resolve(task["desc"])
    tools_info = json.dumps(get_all_skills(), ensure_ascii=False, indent=2)
    selected_skill = json.dumps(skill, ensure_ascii=False, indent=2)
    prompt = f"""
Resolver đã chọn skill liên quan dưới đây. Dùng skill/package này khi phù hợp,
nhưng chỉ gọi một tool tích hợp sẵn trong registry để thực thi hành động.
Nhiệm vụ: {task['desc']}
Bối cảnh trước đó: {memory_context}
Skill đã resolve: {selected_skill}

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
Lưu ý: Nếu cần chạy script python, dùng shell_tool hoặc coder_tool. Trả về JSON thuần.
"""
    return _ask(prompt, json_mode=True)
