"""
Aggregator — nhận kết quả từ N agent, dùng LLM tổng hợp thành 1 kết quả cuối.

Chiến lược:
  - ensemble: LLM đọc tất cả kết quả, chọn hoặc merge phần tốt nhất.
  - vote:     chọn kết quả xuất hiện nhiều nhất (majority).
  - debate:   cho các agent "tranh luận" qua LLM rồi ra kết luận.
"""
import json
import time
import os

from core.llm_helper import LLMClient
from core.logger import SessionLogger


class Aggregator:
    """
    Tổng hợp kết quả từ nhiều SwarmAgent.
    strategy: "ensemble" | "vote" | "debate"
    """

    def __init__(
        self, strategy="ensemble", model_name="auto", session_id=None,
        expected_count=None,
    ):
        self.strategy = strategy
        self.llm = LLMClient(model_name)
        self.expected_count = expected_count
        self.logger = SessionLogger(
            agent_id="aggregator", session_id=session_id
        )

    def aggregate(self, results, goal):
        """
        results: list[dict] — mỗi phần tử là output của SwarmAgent.run()
        goal:    mục tiêu ban đầu.
        Trả về dict kết quả cuối cùng.
        """
        results = results if isinstance(results, list) else []
        valid = [
            result
            for result in results
            if self._is_valid_result(result)
        ]
        expected = self.expected_count or len(results)
        self.logger.print(
            "AGGREGATOR",
            f"Nhận {len(valid)}/{expected} valid (strategy={self.strategy})",
            "bold blue",
        )
        self.logger.log("aggregate_start", {
            "strategy": self.strategy,
            "agent_count": len(results),
            "valid_count": len(valid),
            "expected_count": expected,
            "goal": goal,
        })

        if not valid:
            final = {
                "winner": None,
                "final_result": "",
                "reason": "Tất cả agent fail",
                "confidence": 0,
                "best_parts": [],
            }
        elif len(valid) == 1:
            final = self._fallback_result(valid, confidence=0.3)
            final["reason"] = "Chỉ có 1 kết quả agent hợp lệ; dùng kết quả đó."
        else:
            if len(valid) < expected:
                self.logger.print(
                    "WARNING",
                    f"Chỉ có {len(valid)}/{expected} agent hợp lệ; tổng hợp kết quả còn lại.",
                    "yellow",
                )
            try:
                if self.strategy == "vote":
                    final = self._vote(valid, goal)
                elif self.strategy == "debate":
                    final = self._debate(valid, goal)
                else:
                    final = self._ensemble(valid, goal)
                if not self._is_valid_aggregate(final):
                    raise ValueError("Aggregator LLM returned no usable result")
            except Exception as exc:
                self.logger.print(
                    "WARNING",
                    f"LLM aggregation failed ({type(exc).__name__}); using a valid agent result.",
                    "yellow",
                )
                final = self._fallback_result(valid, confidence=0.5)

        self.logger.print(
            "AGGREGATOR",
            f"Chọn: {final.get('winner', '?')} | Confidence: {final.get('confidence', 0):.2f} | {final.get('reason', '')}",
            "bold green",
        )
        self.logger.log("aggregate_result", final)
        return final

    @staticmethod
    def _is_valid_result(result):
        if not isinstance(result, dict):
            return False
        if result.get("status") not in {
            "success", "completed", "partial",
        }:
            return False
        output = result.get("result")
        return (
            isinstance(output, str)
            and bool(output.strip())
            and not output.strip().casefold().startswith("fail:")
            and output.strip().casefold() != "không có kết quả"
        )

    @staticmethod
    def _is_valid_aggregate(result):
        return (
            isinstance(result, dict)
            and not result.get("error")
            and isinstance(result.get("final_result"), str)
            and bool(result["final_result"].strip())
        )

    @staticmethod
    def _fallback_result(results, confidence):
        indexed = list(enumerate(results))
        winner, selected = max(
            indexed,
            key=lambda item: (
                sum(
                    1 for action in item[1].get("actions", [])
                    if isinstance(action, dict)
                    and action.get("output")
                    and not str(action["output"]).strip().casefold().startswith("fail:")
                ),
                len(str(item[1].get("result", ""))),
            ),
        )
        result_text = selected["result"]
        return {
            "winner": winner,
            "final_result": result_text,
            "reason": "Dùng trực tiếp kết quả hợp lệ của agent.",
            "confidence": confidence,
            "best_parts": [f"Agent {winner}: {result_text[:200]}"],
        }

    def _ensemble(self, results, goal):
        """LLM đọc tất cả kết quả, chọn hoặc merge phần tốt nhất."""
        summaries = self._build_summaries(results)
        prompt = f"""
Mục tiêu ban đầu: "{goal}"

Dưới đây là kết quả từ {len(results)} agent độc lập:
{summaries}

Hãy phân tích và tổng hợp:
- Nếu các kết quả giống nhau → chọn 1, confidence cao.
- Nếu khác nhau → merge phần tốt nhất từ mỗi agent.
- Bỏ qua agent bị crash hoặc failed.

Trả về JSON:
{{
  "winner": <index 0..{len(results)-1}> hoặc "merged",
  "final_result": "Kết quả tổng hợp cuối cùng",
  "reason": "Lý do chọn / merge",
  "confidence": <0.0 đến 1.0>,
  "best_parts": ["trích từ agent X: ...", "trích từ agent Y: ..."]
}}
"""
        return self.llm.ask(prompt, json_mode=True)

    def _vote(self, results, goal):
        """Chọn kết quả theo số phiếu (agent nào completed nhiều task nhất)."""
        completed = [
            (i, result)
            for i, result in enumerate(results)
            if self._is_valid_result(result)
        ]

        if not completed:
            # Không agent nào completed → fallback ensemble
            return self._ensemble(results, goal)

        # Chọn agent có nhiều action thành công nhất
        idx, best = max(
            completed,
            key=lambda item: len(item[1].get("actions", [])),
        )
        return {
            "winner": idx,
            "final_result": best.get("result", ""),
            "reason": f"Agent {idx} hoàn thành nhiều task nhất ({len(best.get('actions', []))} actions)",
            "confidence": 0.8,
            "best_parts": [f"Agent {idx}: {best.get('result', '')[:200]}"],
        }

    def _debate(self, results, goal):
        """Cho LLM đóng vai các agent tranh luận, rồi ra kết luận."""
        summaries = self._build_summaries(results)
        prompt = f"""
Mục tiêu: "{goal}"

{len(results)} agent đã thực hiện cùng mục tiêu. Kết quả:
{summaries}

Hãy mô phỏng cuộc tranh luận giữa các agent về kết quả tốt nhất,
rồi đưa ra kết luận cuối cùng.

Trả về JSON:
{{
  "winner": <index 0..{len(results)-1}> hoặc "merged",
  "final_result": "Kết quả tổng hợp cuối cùng",
  "reason": "Tóm tắt cuộc tranh luận + lý do chọn",
  "confidence": <0.0 đến 1.0>,
  "best_parts": ["trích từ agent X: ...", "trích từ agent Y: ..."]
}}
"""
        return self.llm.ask(prompt, json_mode=True)

    def _build_summaries(self, results):
        """Tạo bản tóm tắt kết quả của mỗi agent cho LLM đọc."""
        parts = []
        for i, r in enumerate(results):
            status = r.get("status", "unknown")
            elapsed = r.get("elapsed", 0)
            model = r.get("model_used", "unknown")
            plan_summary = ""
            if r.get("plan") and isinstance(r["plan"], dict):
                plan_summary = r["plan"].get("understanding", "")
            result_text = str(r.get("result", ""))[:400]
            error_text = str(r.get("error", ""))[:200] if r.get("error") else ""

            parts.append(
                f"--- Agent {i} (model={model}, status={status}, time={elapsed:.1f}s) ---\n"
                f"Plan: {plan_summary}\n"
                f"Result: {result_text}\n"
                f"{'Error: ' + error_text if error_text else ''}"
            )
        return "\n\n".join(parts)
