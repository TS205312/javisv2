"""
Critic — LLM đánh giá kết quả task: pass/fail + đề xuất sửa.
Hỗ trợ inject llm_client riêng cho mỗi SwarmAgent.
"""
from core.llm_helper import ask_llm
from core.research_policy import (
    is_presentation_draft_task,
    is_search_task,
    presentation_issues,
)
import json


def evaluate_task(task, output, llm_client=None):
    """
    Đánh giá kết quả thực thi task.
    llm_client: instance LLMClient (nếu None → dùng hàm mặc định ask_llm).
    """
    if is_search_task(task):
        header, separator, payload = str(output).partition("\n")
        if not separator or not header.startswith("SEARCH_OK "):
            return {
                "passed": False,
                "reason": header or "Không có trạng thái tìm kiếm thành công.",
                "fix": "Dùng nguồn tìm kiếm dự phòng một lần hoặc dừng với lỗi rõ ràng.",
            }
        try:
            rows = json.loads(payload)
        except json.JSONDecodeError:
            rows = []
        valid_rows = (
            isinstance(rows, list)
            and len(rows) >= 3
            and all(
                isinstance(row, dict)
                and all(row.get(field) for field in ("title", "url", "snippet", "source"))
                and str(row["url"]).startswith(("http://", "https://"))
                for row in rows
            )
        )
        return {
            "passed": valid_rows,
            "reason": (
                f"Đạt {len(rows)} kết quả có tiêu đề, URL, nguồn và tóm tắt."
                if valid_rows
                else "Không đủ 3 kết quả có cấu trúc/nguồn để bắt đầu soạn nội dung."
            ),
            "fix": "Thử truy vấn hẹp hơn qua nguồn dự phòng; nếu vẫn thiếu thì báo thất bại.",
        }

    if (
        task.get("_selected_tool") == "computer_tool"
        and task.get("_selected_action") == "perform"
    ):
        try:
            desktop_result = json.loads(str(output))
        except (TypeError, json.JSONDecodeError):
            desktop_result = {}
        verification = desktop_result.get("verification", {})
        verified = (
            desktop_result.get("status") == "verified"
            and desktop_result.get("error") is None
            and desktop_result.get("actions_completed") == desktop_result.get("actions_requested")
            and isinstance(verification, dict)
            and verification.get("status") == "matched"
        )
        return {
            "passed": verified,
            "reason": (
                "OCR xác nhận trạng thái màn hình khớp điều kiện yêu cầu."
                if verified
                else "Desktop action chưa được xác minh bằng trạng thái sau thao tác."
            ),
            "fix": (
                "Dùng ảnh sau thao tác và OCR để xác định trạng thái hiện tại; "
                "lập một bước quan sát/sửa mới thay vì phát lại chuỗi GUI."
            ),
        }

    if is_presentation_draft_task(task):
        issues = presentation_issues(
            output,
            verified_source_urls=task.get("verified_source_urls"),
        )
        if issues:
            return {
                "passed": False,
                "reason": "Bản thảo chưa đạt cổng nội dung: " + "; ".join(issues),
                "fix": "Sửa đúng các mục còn thiếu: " + "; ".join(issues),
            }

    _ask = llm_client.ask if llm_client else ask_llm
    prompt = f"""
Nhiệm vụ: {task['desc']}
Tiêu chí thành công: {task.get('success', 'Hoàn thành không có lỗi')}
Evidence pack đã xác thực:
{json.dumps(task.get('research_evidence', []), ensure_ascii=False)[:12000]}
Kết quả thực thi:
{output}

Đánh giá xem nhiệm vụ đã thành công chưa; kiểm tra từng nhận định với evidence pack.
Trả về JSON định dạng:
{{
  "passed": true hoặc false,
  "reason": "Lý do",
  "fix": "Nếu fail, đề xuất cách sửa (để trống nếu passed)"
}}
"""
    return _ask(prompt, json_mode=True)
