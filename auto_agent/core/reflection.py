"""
Reflection — LLM tự hỏi bản thân "đang đi đúng hướng không?" mỗi 5 bước.
Hỗ trợ inject llm_client riêng cho mỗi SwarmAgent.
"""
from core.llm_helper import ask_llm


def reflect(history, llm_client=None):
    """
    Phản chiếu dựa trên lịch sử gần đây.
    llm_client: instance LLMClient (nếu None → dùng hàm mặc định ask_llm).
    """
    _ask = llm_client.ask if llm_client else ask_llm
    prompt = f"""
Lịch sử 5 bước gần đây:
{history}

Bạn có đang đi đúng hướng để đạt mục tiêu không?
Trả về JSON định dạng:
{{
  "on_track": true/false,
  "advice": "Lời khuyên tự điều chỉnh nếu cần"
}}
"""
    return _ask(prompt, json_mode=True)
