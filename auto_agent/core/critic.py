"""
Critic — LLM đánh giá kết quả task: pass/fail + đề xuất sửa.
Hỗ trợ inject llm_client riêng cho mỗi SwarmAgent.
"""
from core.llm_helper import ask_llm


def evaluate_task(task, output, llm_client=None):
    """
    Đánh giá kết quả thực thi task.
    llm_client: instance LLMClient (nếu None → dùng hàm mặc định ask_llm).
    """
    _ask = llm_client.ask if llm_client else ask_llm
    prompt = f"""
Nhiệm vụ: {task['desc']}
Tiêu chí thành công: {task.get('success', 'Hoàn thành không có lỗi')}
Kết quả thực thi:
{output}

Đánh giá xem nhiệm vụ đã thành công chưa?
Trả về JSON định dạng:
{{
  "passed": true hoặc false,
  "reason": "Lý do",
  "fix": "Nếu fail, đề xuất cách sửa (để trống nếu passed)"
}}
"""
    return _ask(prompt, json_mode=True)
