"""
Planner — LLM sinh kế hoạch JSON từ goal tiếng Việt.
Hỗ trợ inject llm_client riêng cho mỗi SwarmAgent.
"""
from core.llm_helper import ask_llm
from registry import TOOL_REGISTRY
import json


def generate_plan(goal, llm_client=None):
    """
    Sinh kế hoạch thực thi cho goal.
    llm_client: instance LLMClient (nếu None → dùng hàm mặc định ask_llm).
    """
    _ask = llm_client.ask if llm_client else ask_llm
    tools_info = json.dumps(TOOL_REGISTRY, ensure_ascii=False, indent=2)
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
      "success": "Tiêu chí thành công của bước này"
    }}
  ],
  "risks": "Rủi ro có thể gặp"
}}
"""
    return _ask(prompt, json_mode=True)
