"""
Aggregator — nhận kết quả từ N agent, dùng LLM tổng hợp thành 1 kết quả cuối.

Chiến lược:
  - ensemble: LLM đọc tất cả kết quả, chọn hoặc merge phần tốt nhất.
  - vote:     chọn kết quả xuất hiện nhiều nhất (majority).
  - debate:   cho các agent "tranh luận" qua LLM rồi ra kết luận.
"""
import json
import re
import time
import os

from core.llm_helper import LLMClient
from core.logger import SessionLogger
from core.config import GEMINI_API_KEY, GEMINI_MODEL
from core.research_policy import is_presentation_goal, presentation_issues


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
        if model_name == "auto" and GEMINI_API_KEY:
            model_name = f"gemini:{GEMINI_MODEL}"
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
        source_required = self._requires_sources(goal)
        candidates = [
            result
            for result in results
            if self._is_valid_result(result)
        ]
        valid = [
            result
            for result in candidates
            if not source_required or self._source_urls(result.get("result", ""))
        ]
        evidence_rejected = [
            result
            for result in candidates
            if source_required and not self._source_urls(result.get("result", ""))
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
            "evidence_rejected_agent_ids": [
                result.get("agent_id") for result in evidence_rejected
            ],
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
            try:
                final = self._fallback_result(valid, confidence=0.25)
                if not self._is_valid_aggregate(final):
                    raise ValueError("Leader returned no usable result")
                final["reason"] = (
                    "Leader đã xem xét kết quả duy nhất còn hợp lệ; không có "
                    "đối chiếu chéo độc lập giữa các agent."
                )
                final["confidence"] = min(
                    float(final.get("confidence", 0.3)),
                    0.4,
                )
                final["peer_reviews"] = []
                final["peer_review_count"] = 0
                final["expected_peer_review_count"] = expected
                final["reason"] = (
                    f"Only 1/{expected} agents produced a valid result. "
                    "Preserved that draft without cross-agent debate; swarm is incomplete."
                )
                final["needs_review"] = True
                final["leader_skipped"] = True
            except Exception as exc:
                self.logger.print(
                    "WARNING",
                    f"Leader review failed ({type(exc).__name__}); using an explicitly unverified fallback.",
                    "yellow",
                )
                final = self._fallback_result(valid, confidence=0.3)
                final["reason"] = (
                    "Leader không xem xét được kết quả duy nhất; bản dự phòng "
                    "chưa qua đối chiếu đầy đủ: {0}".format(type(exc).__name__)
                )
                final["leader_failed"] = True
                final["needs_review"] = True
                final["peer_reviews"] = []
                final["peer_review_count"] = 0
                final["expected_peer_review_count"] = expected
        else:
            if len(valid) < expected:
                self.logger.print(
                    "WARNING",
                    f"Chỉ có {len(valid)}/{expected} agent hợp lệ; tổng hợp kết quả còn lại.",
                    "yellow",
                )
            try:
                if self.strategy == "vote":
                    final = self._debate(valid, goal)
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
                final["reason"] = (
                    "Leader không tổng hợp được kết quả; đang trả bản dự phòng "
                    "chưa qua đối chiếu đầy đủ: {0}".format(type(exc).__name__)
                )
                final["leader_failed"] = True
                final["needs_review"] = True

        self.logger.print(
            "AGGREGATOR",
            f"Chọn: {final.get('winner', '?')} | Confidence: {final.get('confidence', 0):.2f} | {final.get('reason', '')}",
            "bold green",
        )
        selected_executor, selection_reason = self._select_final_executor(
            final, valid
        )
        final["final_executor_agent_id"] = selected_executor
        final["final_executor_reason"] = selection_reason
        self.logger.log("final_executor_selected", {
            "agent_id": selected_executor,
            "reason": selection_reason,
            "browser_session": next((
                result.get("browser_session")
                for result in valid
                if result.get("agent_id") == selected_executor
            ), None),
        })
        final["valid_agent_count"] = len(valid)
        final["expected_agent_count"] = expected
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
    def _source_urls(text):
        return {
            url.rstrip(".,;:")
            for url in re.findall(r"https?://[^\s<>\])}]+", str(text))
        }

    @staticmethod
    def _requires_sources(goal):
        return any(
            term in goal.casefold()
            for term in (
                "research", "source", "citation", "reference", "presentation",
                "slide", "canva", "nghiên cứu", "nguồn", "trích dẫn",
                "thuyết trình", "thuyet trinh",
            )
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
            "winner": selected.get("agent_id", winner),
            "final_result": result_text,
            "reason": "Dùng trực tiếp kết quả hợp lệ của agent.",
            "confidence": confidence,
            "best_parts": [f"Agent {winner}: {result_text[:200]}"],
        }

    @staticmethod
    def _select_final_executor(final, results):
        if not results:
            return None, "No valid agent result is available."

        agent_ids = {result.get("agent_id") for result in results}
        requested = final.get("final_executor_agent_id")
        if requested in agent_ids:
            return requested, "Selected by leader from a valid agent result."

        winner = final.get("winner")
        if winner in agent_ids:
            return winner, "Leader winner selected as the final executor."

        selected = max(
            results,
            key=lambda result: (
                result.get("status") == "completed",
                not bool(result.get("failed_tasks")),
                len(result.get("research_evidence", [])),
                len(Aggregator._source_urls(result.get("result", ""))),
                sum(
                    1 for action in result.get("actions", [])
                    if isinstance(action, dict) and action.get("output")
                ),
                -int(result.get("agent_id") or 0),
            ),
        )
        return (
            selected.get("agent_id"),
            "Leader winner did not identify a valid agent; selected by completion, "
            "evidence coverage, and recorded work.",
        )

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
  "winner": <agent_id đúng như trong dữ liệu đầu vào> hoặc "merged",
  "final_result": "Kết quả tổng hợp cuối cùng",
  "reason": "Lý do chọn / merge",
  "confidence": <0.0 đến 1.0>,
  "best_parts": ["trích từ agent X: ...", "trích từ agent Y: ..."],
  "final_executor_agent_id": <agent_id>
}}
"""
        return self.llm.ask(prompt, json_mode=True, max_output_tokens=6144)

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
        """Have each agent review its peers, then ask the leader to decide."""
        summaries = self._build_summaries(results)
        peer_reviews = []
        for index, result in enumerate(results):
            agent_id = result.get("agent_id", index + 1)
            peers = [
                candidate
                for peer_index, candidate in enumerate(results)
                if peer_index != index
            ]
            prompt = f"""
Bạn là Agent {agent_id}, đang phản biện kết quả của các agent khác.
Mục tiêu: "{goal}"

Kết quả của bạn:
{str(result.get('result', ''))[:3000]}

Kết quả của các agent cần phản biện:
{self._build_summaries(peers)}

Nêu cụ thể:
- Điểm nào của từng kết quả có bằng chứng và phù hợp mục tiêu.
- Điểm nào thiếu nguồn, sai chủ đề, mâu thuẫn hoặc chưa hoàn thành.
- Đề xuất phần tốt nhất để leader dùng và điều cần xác minh.
Không bịa bằng chứng hoặc tuyên bố đã kiểm tra nguồn nếu chưa có URL/nội dung.
"""
            try:
                reviewer = LLMClient(result.get("model_used", "auto"))
                review = reviewer.ask(prompt)
                if not isinstance(review, str) or not review.strip():
                    raise ValueError("Agent returned an empty peer review")
                if review.strip().casefold().startswith("error:"):
                    raise RuntimeError(review.strip())
                peer_reviews.append({
                    "agent_id": agent_id,
                    "review": review,
                })
                self.logger.log("peer_review", peer_reviews[-1])
            except Exception as exc:
                failure = {
                    "agent_id": agent_id,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
                self.logger.print(
                    "PEER REVIEW FAILED",
                    f"Agent {agent_id}: {type(exc).__name__}: {exc}",
                    "yellow",
                )
                self.logger.log("peer_review_failed", failure)

        reviews_text = "\n\n".join(
            f"--- Phản biện từ Agent {review['agent_id']} ---\n{review['review']}"
            for review in peer_reviews
        ) or "Không có agent nào gửi được phản biện."
        expected_reviews = self.expected_count or len(results)
        review_coverage = (
            f"Phản biện thành công: {len(peer_reviews)}/{expected_reviews} agent. "
            "Nêu rõ trong kết luận nếu thiếu lượt phản biện."
        )
        prompt = f"""
Mục tiêu: "{goal}"

Kết quả ban đầu của các agent:
{summaries}

{review_coverage}

Phản biện chéo thực tế do từng agent gửi:
{reviews_text}

Bạn là leader. Đối chiếu kết quả ban đầu với phản biện, giải quyết bất đồng
bằng bằng chứng hiện có, nêu rõ thiếu sót và chọn/ghép phần tốt nhất. Không
coi nội dung là đã xác minh nếu kết quả không kèm nguồn kiểm tra được. Nếu còn
task lỗi hoặc agent chỉ hoàn thành một phần, phải nêu rõ trong kết luận; không
được tuyên bố toàn bộ nhiệm vụ hoàn tất.
Nếu mục tiêu yêu cầu bài thuyết trình/slide, hãy định dạng final_result thành
6–8 slide có tiêu đề rõ, ý chính ngắn gọn, gợi ý hình ảnh và nguồn tham khảo.
Mỗi slide phải có bullet "Gợi ý hình ảnh"; slide cuối là kết luận. Mỗi slide
nội dung (trừ trang tiêu đề) phải có URL nguồn phù hợp ngay trong slide. Gắn URL
nguồn đã xuất hiện trong evidence pack vào nhận định/số liệu liên quan; thêm mục
"Tài liệu tham khảo" cuối bài. Chỉ dùng nguồn được trả về dưới trạng thái
SEARCH_OK. Nếu không có ít nhất 3 nguồn liên quan đã kiểm tra, không bịa slide
hay kết luận; nêu rõ chưa đủ bằng chứng.
Không tuyên bố đã tạo hoặc lưu thiết kế Canva nếu không có bằng chứng thao tác
thành công trên Canva.

Trả về JSON:
{{
  "winner": <agent_id> hoặc "merged",
  "final_result": "Kết quả tổng hợp cuối cùng",
  "reason": "Nhận xét của leader, điểm bất đồng và lý do chọn",
  "confidence": <0.0 đến 1.0>,
  "best_parts": ["trích từ agent X: ...", "trích từ agent Y: ..."]
}}
"""
        final = self.llm.ask(prompt, json_mode=True, max_output_tokens=6144)
        if self._is_valid_aggregate(final):
            issues = self._output_issues(goal, final["final_result"])
            if issues:
                repair_prompt = f"""
Sửa kết luận để đáp ứng các điều kiện kiểm tra còn thiếu: {issues}

Kết luận hiện tại:
{final['final_result']}

Chỉ dùng URL có trong kết quả ban đầu hoặc phản biện. Không bịa nguồn. Giữ lại
điểm đúng, loại nhận định thiếu bằng chứng, và trả JSON cùng cấu trúc trước.
"""
                repaired = self.llm.ask(
                    repair_prompt,
                    json_mode=True,
                    max_output_tokens=6144,
                )
                if self._is_valid_aggregate(repaired):
                    final = repaired
                    remaining_issues = self._output_issues(
                        goal,
                        final["final_result"],
                    )
                    if remaining_issues:
                        final["needs_review"] = True
                        final["quality_issues"] = remaining_issues
                else:
                    final["needs_review"] = True
                    final["quality_issues"] = issues
        if isinstance(final, dict):
            final["peer_reviews"] = peer_reviews
            final["peer_review_count"] = len(peer_reviews)
            final["expected_peer_review_count"] = expected_reviews
        return final

    @staticmethod
    def _output_issues(goal, result_text):
        issues = []
        lowered_goal = goal.casefold()
        source_required = any(
            term in lowered_goal
            for term in (
                "research", "source", "citation", "reference", "presentation",
                "slide", "canva", "nghiên cứu", "nguồn", "trích dẫn",
                "thuyết trình", "thuyet trinh",
            )
        )
        urls = Aggregator._source_urls(result_text)
        if source_required and len(urls) < 3:
            issues.append(f"chỉ có {len(urls)} URL nguồn hợp lệ; cần ít nhất 3")

        if is_presentation_goal(goal):
            issues.extend(presentation_issues(result_text))
        return issues

    def _build_summaries(self, results):
        """Tạo bản tóm tắt kết quả của mỗi agent cho LLM đọc."""
        parts = []
        for i, r in enumerate(results):
            status = r.get("status", "unknown")
            elapsed = r.get("elapsed", 0)
            model = r.get("model_used", "unknown")
            agent_id = r.get("agent_id", i + 1)
            role = r.get("role", "unspecified")
            plan_summary = ""
            if r.get("plan") and isinstance(r["plan"], dict):
                plan_summary = r["plan"].get("understanding", "")
            result_text = str(r.get("result", ""))[:3000]
            error_text = str(r.get("error", ""))[:200] if r.get("error") else ""
            failed_tasks = r.get("failed_tasks", [])
            failure_summary = (
                json.dumps(failed_tasks, ensure_ascii=False)[:400]
                if failed_tasks
                else "none"
            )
            evidence_summary = json.dumps(
                r.get("research_evidence", []),
                ensure_ascii=False,
            )[:2000]
            browser_summary = json.dumps(
                r.get("browser_session", {}),
                ensure_ascii=False,
            )[:500]

            parts.append(
                f"--- Agent {agent_id} (role={role}, model={model}, status={status}, time={elapsed:.1f}s) ---\n"
                f"Plan: {plan_summary}\n"
                f"Validated research evidence: {evidence_summary}\n"
                f"Isolated browser session: {browser_summary}\n"
                f"Result: {result_text}\n"
                f"Failed tasks: {failure_summary}\n"
                f"{'Error: ' + error_text if error_text else ''}"
            )
        return "\n\n".join(parts)
