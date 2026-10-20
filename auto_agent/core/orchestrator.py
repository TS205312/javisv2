"""
Orchestrator — điều phối N agent chạy song song với asyncio.

Nhận goal → spawn N SwarmAgent → chạy song song (asyncio.gather) →
thu kết quả → gửi cho Aggregator → trả final_result.

Mỗi agent có timeout riêng (mặc định 300s).
Agent crash không kill cả swarm.
"""
import asyncio
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from core.swarm import SwarmAgent
from core.aggregator import Aggregator
from core.desktop_operator import select_desktop_operator
from core.memory import Memory
from core.config import (
    AGENT_TIMEOUT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_FALLBACK_MODEL,
    GROQ_MODEL,
    KILO_API_KEY,
    KILO_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_FREE_POOL,
    OPENROUTER_PRIMARY_MODEL,
    OLLAMA_MODEL,
    SWARM_AGENT_MODELS,
)
from core.logger import session_logger as default_logger, SessionLogger
from core.llm_helper import call_llm, ollama_installed_models
from core.research_policy import (
    extract_research_topic,
    is_presentation_goal,
    presentation_issues,
    slide_count,
)
from tools import browser_tool
from tools import search_tool
from tools.write_safety import FileCheckpoint, write_bytes_safely
from core.sensitive_data import redact_data, redact_text

# Logger cho orchestrator
orch_logger = SessionLogger(agent_id="orchestrator")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROLES = (
    "Tạo lời giải độc lập, làm đủ mục tiêu và ghi nguồn nếu phù hợp.",
    "Kiểm tra bằng chứng, nguồn và tính đúng đắn của từng nhận định.",
    "Tìm phản chứng, quan điểm thay thế và các rủi ro bị bỏ sót.",
    "Rà tiêu chí hoàn thành, phát hiện thiếu sót và đề xuất cấu trúc tốt nhất.",
)


def _model_key(model_name):
    return (model_name or "").strip().casefold()


def _free_model_candidates():
    candidates = list(SWARM_AGENT_MODELS)
    if KILO_API_KEY and "free" in KILO_MODEL.casefold():
        candidates.append(f"kilo:{KILO_MODEL}")
    if GEMINI_API_KEY:
        candidates.append(f"gemini:{GEMINI_MODEL}")
    if GROQ_API_KEY:
        candidates.append(f"groq:{GROQ_MODEL}")
    if OLLAMA_MODEL:
        candidates.append(f"ollama:{OLLAMA_MODEL}")
    candidates.extend(f"ollama:{model}" for model in ollama_installed_models())
    if GROQ_API_KEY:
        candidates.append(f"groq:{GROQ_FALLBACK_MODEL}")
    if OPENROUTER_API_KEY:
        openrouter_models = (
            OPENROUTER_PRIMARY_MODEL,
            OPENROUTER_FALLBACK_MODEL,
            *OPENROUTER_FREE_POOL,
        )
        candidates.extend(
            f"openrouter:{model}"
            for model in openrouter_models
            if model and model.casefold().endswith(":free")
        )

    unique = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        key = _model_key(candidate)
        if candidate and key != "auto" and key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _resolve_swarm_models(agent_count, requested_models=None):
    candidates = _free_model_candidates()
    requested = list(requested_models or [])
    if len(requested) > agent_count:
        requested = requested[:agent_count]
    requested.extend(["auto"] * (agent_count - len(requested)))

    resolved = []
    used = set()
    for model_name in requested:
        model_name = (model_name or "auto").strip()
        if model_name.casefold() == "auto":
            model_name = next(
                (candidate for candidate in candidates if _model_key(candidate) not in used),
                "",
            )
            if not model_name and agent_count == 1:
                model_name = "auto"
            if not model_name:
                return None, (
                    f"Cần {agent_count} model khác nhau để chạy swarm, nhưng chỉ có "
                    f"{len(candidates)} model miễn phí riêng biệt đã cấu hình. "
                    "Cấu hình thêm Kilo/Gemini/OpenRouter, hoặc đặt đủ "
                    "SWARM_AGENT_MODELS; không chạy agent với model bị lặp."
                )
        key = _model_key(model_name)
        if key in used:
            return None, (
                f"Model bị gán trùng ({model_name}). Mỗi agent cần một model riêng; "
                "hãy sửa --models hoặc SWARM_AGENT_MODELS."
            )
        resolved.append(model_name)
        used.add(key)
    return resolved, None


def _probe_model_api(model_name):
    try:
        response = call_llm(
            "Reply only with OK. This is a short API connectivity check.",
            model=model_name,
            max_output_tokens=8,
            allow_fallback=False,
        )
        if not isinstance(response, str) or not response.strip():
            raise ValueError("empty API health-check response")
        return {"model": model_name, "ok": True, "error": None}
    except Exception as exc:
        return {
            "model": model_name,
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def _preflight_swarm_models(models):
    assignments = list(models)
    if assignments:
        with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            checks = list(executor.map(_probe_model_api, assignments))
    else:
        checks = []

    for check in checks:
        orch_logger.log("model_api_preflight", check)
        orch_logger.print(
            "API PREFLIGHT",
            f"{check['model']} → {'OK' if check['ok'] else 'FAIL'}"
            + (f" ({check['error']})" if check["error"] else ""),
            "green" if check["ok"] else "yellow",
        )

    used = {_model_key(model) for model in assignments}
    checked = {_model_key(check["model"]) for check in checks}
    failed_agents = []
    for index, initial_check in enumerate(list(checks)):
        if initial_check["ok"]:
            continue

        replacement = None
        for candidate in _free_model_candidates():
            candidate_key = _model_key(candidate)
            if candidate_key in used or candidate_key in checked:
                continue
            checked.add(candidate_key)
            backup_check = _probe_model_api(candidate)
            checks.append(backup_check)
            orch_logger.log("model_api_preflight", backup_check)
            orch_logger.print(
                "API PREFLIGHT",
                f"Dự phòng {candidate} → "
                f"{'OK' if backup_check['ok'] else 'FAIL'}"
                + (f" ({backup_check['error']})" if backup_check["error"] else ""),
                "green" if backup_check["ok"] else "yellow",
            )
            if backup_check["ok"]:
                replacement = candidate
                assignments[index] = candidate
                used.add(candidate_key)
                orch_logger.log("model_api_reassigned", {
                    "agent_id": index + 1,
                    "failed_model": initial_check["model"],
                    "replacement_model": candidate,
                })
                break

        if replacement is None:
            failed_agents.append(
                f"Agent {index + 1}: {initial_check['model']} ({initial_check['error']}); "
                "không tìm được model dự phòng khỏe và khác biệt"
            )

    if failed_agents:
        return assignments, checks, "Kiểm tra API thất bại: " + "; ".join(failed_agents)
    return assignments, checks, None


def _run_agent_sync(agent, goal):
    """Chạy 1 SwarmAgent đồng bộ (dùng trong thread pool)."""
    return agent.run(goal)


async def _run_agent_with_timeout(executor, agent, goal, timeout):
    """Run the synchronous agent body in a worker thread with an async timeout."""
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(executor, _run_agent_sync, agent, goal),
        timeout=timeout,
    )


def _verified_search_urls(agent_results, shared_evidence=None):
    verified = set()
    for agent_result in agent_results:
        for action in agent_result.get("actions", []):
            output = str(action.get("output", ""))
            header, separator, payload = output.partition("\n")
            if not separator or not header.startswith("SEARCH_OK "):
                continue
            try:
                rows = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(rows, list) or len(rows) < 3:
                continue
            for row in rows:
                if not isinstance(row, dict) or not all(
                    row.get(field) for field in ("title", "url", "snippet", "source")
                ):
                    continue
                url = str(row["url"]).rstrip(".,;:")
                parsed = urlparse(url)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    verified.add(url)
    for row in shared_evidence or []:
        if not isinstance(row, dict) or not all(
            row.get(field) for field in ("title", "url", "snippet", "source")
        ):
            continue
        url = str(row["url"]).rstrip(".,;:")
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            verified.add(url)
    return verified


def _build_source_fallback(goal, research_evidence, verified_source_urls):
    verified = {str(url).rstrip(".,;:") for url in verified_source_urls or ()}
    sources = []
    seen_urls = set()
    for row in research_evidence or []:
        if not isinstance(row, dict):
            continue
        title = " ".join(str(row.get("title", "")).split())
        snippet = " ".join(str(row.get("snippet", "")).split())
        url = str(row.get("url", "")).rstrip(".,;:")
        parsed = urlparse(url)
        if (
            not title
            or not snippet
            or url not in verified
            or url in seen_urls
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
        ):
            continue
        title = title.replace("#", "").replace("<", "").replace(">", "")[:180]
        snippet = snippet.replace("#", "").replace("<", "").replace(">", "")[:360]
        sources.append({"title": title, "snippet": snippet, "url": url})
        seen_urls.add(url)
        if len(sources) == 3:
            break
    if len(sources) < 3:
        return ""

    topic = " ".join(extract_research_topic(goal).split())[:160]
    first, second, third = sources
    slides = [
        f"## Slide 1: {topic}\n"
        "- Focus: evidence-led overview based only on validated search snippets.\n"
        "- Image suggestion: classroom with students using digital learning tools.",
    ]
    for number, source in enumerate(sources, start=2):
        slides.append(
            f"## Slide {number}: Evidence summary {number - 1}\n"
            f"- Source title: {source['title']}\n"
            f"- Retrieved snippet (summary only): {source['snippet']}\n"
            f"- Source: {source['url']}\n"
            "- Image suggestion: a simple visual matching the cited source topic."
        )
    slides.extend([
        "## Slide 5: What the sources support\n"
        "- Compare the three retrieved snippets; do not infer causation or rank outcomes.\n"
        f"- Sources: {first['url']} {second['url']} {third['url']}\n"
        "- Image suggestion: three evidence cards with source labels.",
        "## Slide 6: Conclusion\n"
        "- Conclusion is limited to the retrieved snippets; check full texts before presenting claims.\n"
        f"- Sources: {first['url']} {second['url']} {third['url']}\n"
        "- Image suggestion: a checklist for evidence and responsible interpretation.",
    ])
    references = "\n".join(
        f"{index}. {source['title']} — {source['url']}"
        for index, source in enumerate(sources, start=1)
    )
    return "\n\n".join(slides) + f"\n\n## Sources\n{references}"


def _save_presentation_draft(
    goal,
    aggregate,
    session_id,
    verified_source_urls=None,
    research_evidence=None,
    canva_preflight=None,
    final_executor_agent_id=None,
):
    if not is_presentation_goal(goal):
        return None

    canva_preflight = canva_preflight or {
        "status": "not_checked",
        "reason": "No isolated final browser session was selected.",
        "design_created": False,
    }

    def canva_details():
        status = canva_preflight.get("status", "unknown")
        reason = canva_preflight.get("reason") or canva_preflight.get("error")
        if not reason:
            if status == "accessible":
                reason = (
                    "Canva was reachable, but automatic design creation is not "
                    "implemented or confirmed."
                )
            elif status == "login_required":
                reason = "Canva requires an authenticated session; no design was created."
            elif status == "automation_blocked":
                reason = "Canva blocked automated browser access; no design was created."
            elif status in {"browser_launch_failed", "canva_access_failed", "browser_timeout"}:
                reason = f"Canva preflight failed: {status}. No design was created."
            else:
                reason = f"Canva was not confirmed available ({status}); no design was created."
        return {
            "canva_created": False,
            "canva_status": status,
            "canva_reason": reason,
            "final_executor_agent_id": final_executor_agent_id,
            "final_browser_session_id": canva_preflight.get("session_id"),
        }

    canva_fields = canva_details()

    final_result = aggregate.get("final_result", "")
    slide_numbers = slide_count(final_result)
    initial_issues = presentation_issues(final_result)
    if initial_issues:
        fallback = _build_source_fallback(
            goal,
            research_evidence,
            verified_source_urls,
        )
        fallback_issues = presentation_issues(
            fallback,
            verified_source_urls=verified_source_urls,
        ) if fallback else ["fewer than three verified source snippets"]
        if fallback and not fallback_issues:
            final_result = fallback
            slide_numbers = slide_count(final_result)
            aggregate["final_result"] = final_result
            aggregate["fallback_used"] = True
            aggregate["fallback_reason"] = (
                "Leader output failed slide validation; built a six-slide "
                "evidence-only fallback from three validated search results."
            )
            initial_issues = []
    if initial_issues:
        message = (
            "Chưa lưu bản thảo: leader output không đạt kiểm tra: "
            f"{'; '.join(initial_issues)}. Canva chưa được thao tác."
        )
        aggregate["presentation_output"] = {
            "saved": False,
            "reason": message,
            "canva_fallback_saved": False,
            **canva_fields,
        }
        return aggregate["presentation_output"]

    extracted_urls = re.findall(r"https?://[^\s<>\])}]+", final_result)
    source_urls = list(dict.fromkeys(
        url.rstrip(".,;:")
        for url in extracted_urls
        if urlparse(url.rstrip(".,;:")).scheme in {"http", "https"}
        and urlparse(url.rstrip(".,;:")).netloc
    ))
    verified_source_urls = set(verified_source_urls or ())
    source_urls_are_verified = bool(source_urls) and set(source_urls).issubset(
        verified_source_urls
    )
    references_present = re.search(
        r"(?im)^#{1,3}\s*(?:nguồn|tài liệu tham khảo|references|sources)\b",
        final_result,
    ) is not None
    content_issues = presentation_issues(
        final_result,
        verified_source_urls=verified_source_urls,
    )
    source_check_passed = not content_issues
    source_note = (
        "Nguồn có tiêu đề, URL, tóm tắt và đã qua lọc chủ đề; chưa xác minh nội dung từng trang."
        if source_check_passed
        else (
            "Bản thảo không đạt cổng nguồn/cấu trúc: "
            f"{'; '.join(content_issues)}. "
            f"Hiện có {len(source_urls)} URL; verified={source_urls_are_verified}; "
            f"references={references_present}."
        )
    )

    if not source_check_passed:
        result = {
            "saved": False,
            "reason": source_note,
            "canva_fallback_saved": False,
            **canva_fields,
        }
        aggregate["presentation_output"] = result
        return result

    workspace_root = WORKSPACE_ROOT
    output_path = workspace_root / "output" / (
        "canva_ai_giao_duc.md"
        if "giáo dục" in goal.casefold() or "giao duc" in goal.casefold()
        else f"presentation_draft_{session_id}.md"
    )
    if output_path.exists():
        output_path = output_path.with_name(
            f"{output_path.stem}_{session_id}{output_path.suffix}"
        )
    content = (
        "# Bản thảo bài thuyết trình\n\n"
        "> Nội dung soạn sẵn để đưa vào Canva; chưa phải thiết kế đã lưu trên Canva. "
        f"{canva_fields['canva_status']}: {canva_fields['canva_reason']} "
        f"{source_note}\n\n"
        + (
            f"> Fallback: {aggregate['fallback_reason']}\n\n"
            if aggregate.get("fallback_used")
            else ""
        )
        + f"## Mục tiêu\n\n{goal}\n\n"
        + f"{final_result.strip()}\n"
    )
    try:
        output_path = write_bytes_safely(output_path, content.encode("utf-8"))
    except (OSError, ValueError, TimeoutError) as exc:
        fallback_path = workspace_root / (
            f"presentation_draft_fallback_{session_id}_{time.time_ns()}.md"
        )
        try:
            fallback_path = write_bytes_safely(
                fallback_path,
                content.encode("utf-8"),
            )
        except (OSError, ValueError, TimeoutError) as fallback_exc:
            result = {
                "saved": False,
                "reason": (
                    "Không ghi được bản thảo ở output/ hoặc thư mục hiện tại: "
                    f"{type(exc).__name__}; {type(fallback_exc).__name__}: "
                    f"{fallback_exc}"
                ),
                "exception_type": type(fallback_exc).__name__,
                "canva_fallback_saved": False,
                **canva_fields,
            }
            aggregate["presentation_output"] = result
            return result
        output_path = fallback_path
        fallback_used = True
    else:
        fallback_used = False
    result = {
        "saved": True,
        "path": str(output_path),
        "slide_count": len(slide_numbers),
        "source_count": len(source_urls),
        "source_check_passed": source_check_passed,
        "source_check": source_note,
        "fallback_used": fallback_used or bool(aggregate.get("fallback_used")),
        "draft_type": (
            "source_evidence_fallback"
            if aggregate.get("fallback_used")
            else "leader_output"
        ),
        "canva_fallback_saved": not canva_fields["canva_created"],
        **canva_fields,
    }
    aggregate["presentation_output"] = result
    return result


def _preflight_canva_for_agent(agent_id, executor=None, logger=None, timeout=20):
    """Check Canva on the selected agent's retained page and close its session."""
    def inspect():
        browser_tool.set_session_logger(logger or orch_logger)
        try:
            return browser_tool.canva_preflight({
                "agent_id": agent_id,
                "timeout_ms": 12000,
            })
        except Exception as exc:
            return {
                "agent_id": agent_id,
                "status": "canva_access_failed",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "isolated_context": True,
                "design_created": False,
            }
        finally:
            try:
                browser_tool.close_session(agent_id)
            except Exception as exc:
                orch_logger.log("final_browser_cleanup_failed", {
                    "agent_id": agent_id,
                    "error": str(exc),
                })
            browser_tool.set_session_logger(None)

    owns_executor = executor is None
    executor = executor or ThreadPoolExecutor(max_workers=1)
    future = executor.submit(inspect)
    try:
        return future.result(timeout=timeout)
    except Exception as exc:
        return {
            "agent_id": agent_id,
            "status": "browser_timeout" if isinstance(exc, TimeoutError) else "canva_access_failed",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "isolated_context": True,
            "design_created": False,
        }
    finally:
        if owns_executor:
            executor.shutdown(wait=False, cancel_futures=True)


def _close_agent_browser_session(agent_id, logger):
    browser_tool.set_session_logger(logger)
    try:
        browser_tool.close_session(agent_id)
    except Exception as exc:
        logger.log("browser_cleanup_failed", {
            "agent_id": agent_id,
            "error": str(exc),
            "exception_type": type(exc).__name__,
        })
    finally:
        browser_tool.set_session_logger(None)


def _select_final_executor(final, results):
    valid_ids = {row.get("agent_id") for row in results if isinstance(row, dict)}
    selected = final.get("final_executor_agent_id")
    if selected in valid_ids:
        return selected, final.get(
            "final_executor_reason", "Selected by the leader."
        )
    winner = final.get("winner")
    if winner in valid_ids:
        return winner, "Leader winner selected for the final browser handoff."
    candidates = [row for row in results if isinstance(row, dict) and row.get("agent_id")]
    if not candidates:
        return None, "No agent returned a result that can own the final browser session."
    chosen = max(
        candidates,
        key=lambda row: (
            row.get("status") == "completed",
            not bool(row.get("failed_tasks")),
            len(row.get("research_evidence", [])),
            -int(row.get("agent_id") or 0),
        ),
    )
    return chosen["agent_id"], "Leader omitted a valid ID; selected by completion and evidence coverage."


def _run_status_evidence(results, expected_count, success, quality_checks):
    expected_ids = set(range(1, expected_count + 1))
    received_ids = {
        row.get("agent_id") for row in results if isinstance(row, dict)
    }
    missing_ids = sorted(expected_ids - received_ids)
    incomplete_agents = [
        {
            "agent_id": row.get("agent_id"),
            "status": row.get("status", "unknown"),
            "failed_tasks": row.get("failed_tasks", []),
            "error": row.get("error"),
        }
        for row in results
        if row.get("status") != "completed" or row.get("failed_tasks")
    ]
    has_usable_result = any(
        row.get("status") in {"success", "completed", "partial"}
        and isinstance(row.get("result"), str)
        and bool(row["result"].strip())
        and not row["result"].strip().casefold().startswith("fail:")
        for row in results
    )
    return (
        "completed" if success else "partial" if has_usable_result else "failed",
        {
            "expected_agent_ids": sorted(expected_ids),
            "received_agent_ids": sorted(
                agent_id for agent_id in received_ids
                if isinstance(agent_id, int)
            ),
            "missing_agent_ids": missing_ids,
            "completed_agent_count": sum(
                row.get("status") == "completed" and not row.get("failed_tasks")
                for row in results
            ),
            "incomplete_agents": incomplete_agents,
            "failed_quality_checks": sorted(
                name for name, passed in quality_checks.items() if not passed
            ),
        },
    )


async def run_swarm_async(goal, agent_count=4, models=None, strategy="debate",
                          max_steps=50, yolo=False, timeout=None):
    """
    Chạy swarm gồm agent_count agent song song.
    """
    goal = redact_text(str(goal))
    if timeout is None:
        timeout = AGENT_TIMEOUT
    start = time.time()
    if strategy != "debate":
        orch_logger.print(
            "STRATEGY",
            "Swarm aggregation uses evidence-based debate.",
            "yellow",
        )
        strategy = "debate"

    models, model_error = _resolve_swarm_models(agent_count, models)
    if model_error:
        orch_logger.print("MODEL ASSIGNMENT FAILED", model_error, "red")
        orch_logger.log("model_assignment_failed", {
            "agent_count": agent_count,
            "available_models": _free_model_candidates(),
            "requested_models": models,
            "reason": model_error,
        })
        aggregator_result = {
            "winner": None,
            "final_result": "",
            "reason": model_error,
            "confidence": 0.0,
            "success": False,
            "valid_agent_count": 0,
            "expected_agent_count": agent_count,
            "quality_checks": {"distinct_models_assigned": False},
        }
        if is_presentation_goal(goal):
            aggregator_result["presentation_output"] = {
                "saved": False,
                "canva_created": False,
                "reason": model_error,
            }
        return {
            "agent_results": [],
            "aggregator": aggregator_result,
            "agent_count_expected": agent_count,
            "agent_count_completed": 0,
            "model_assignments": [],
            "model_api_checks": [],
            "elapsed": time.time() - start,
        }

    models, model_api_checks, api_error = _preflight_swarm_models(models)
    if api_error:
        orch_logger.print("API PREFLIGHT FAILED", api_error, "red")
        orch_logger.log("model_api_preflight_failed", {
            "models": models,
            "checks": model_api_checks,
            "reason": api_error,
        })
        aggregator_result = {
            "winner": None,
            "final_result": "",
            "reason": api_error,
            "confidence": 0.0,
            "success": False,
            "valid_agent_count": 0,
            "expected_agent_count": agent_count,
            "quality_checks": {
                "distinct_models_assigned": len({_model_key(model) for model in models}) == agent_count,
                "model_apis_healthy": False,
            },
        }
        if is_presentation_goal(goal):
            aggregator_result["presentation_output"] = {
                "saved": False,
                "canva_created": False,
                "reason": api_error,
            }
        return {
            "agent_results": [],
            "aggregator": aggregator_result,
            "agent_count_expected": agent_count,
            "agent_count_completed": 0,
            "model_assignments": [],
            "model_api_checks": model_api_checks,
            "elapsed": time.time() - start,
        }

    session_id = str(int(time.time() * 1000))
    shared_evidence = None
    if is_presentation_goal(goal):
        topic = extract_research_topic(goal)
        orch_logger.print(
            "RESEARCH PREFLIGHT",
            f"Tìm một bộ nguồn chung cho chủ đề: {topic}",
            "cyan",
        )
        try:
            search_result = search_tool.search({
                "query": topic,
                "min_results": 3,
            })
            search_status = getattr(search_result, "status", "unknown")
            shared_evidence = list(search_result) if isinstance(search_result, list) else []
            valid_source_count = sum(
                1 for row in shared_evidence
                if isinstance(row, dict)
                and all(row.get(field) for field in ("title", "url", "snippet", "source"))
                and urlparse(str(row.get("url", ""))).scheme in {"http", "https"}
                and bool(urlparse(str(row.get("url", ""))).netloc)
            )
            if search_status == "ok" and valid_source_count < 3:
                search_status = "invalid_results"
            preflight_details = {
                "topic": topic,
                "status": search_status,
                "provider": getattr(search_result, "provider", None),
                "query": getattr(search_result, "query", topic),
                "source_count": len(shared_evidence),
                "valid_source_count": valid_source_count,
                "message": getattr(search_result, "message", ""),
            }
        except Exception as exc:
            search_status = "access_error"
            shared_evidence = []
            preflight_details = {
                "topic": topic,
                "status": search_status,
                "provider": None,
                "query": topic,
                "source_count": 0,
                "message": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
        orch_logger.log("research_preflight", preflight_details)
        if (
            search_status != "ok"
            or len(shared_evidence) < 3
            or preflight_details.get("valid_source_count", 0) < 3
        ):
            reason = (
                "Research preflight failed; swarm agents were not started to avoid "
                "repeating a failed search or spending model quota. "
                f"status={search_status}, "
                f"valid_sources={preflight_details.get('valid_source_count', 0)}/3, "
                f"provider={preflight_details.get('provider')}, "
                f"detail={preflight_details.get('message')}"
            )
            orch_logger.print("RESEARCH PREFLIGHT FAILED", reason, "red")
            orch_logger.log("research_preflight_failed", preflight_details)
            return {
                "agent_results": [],
                "aggregator": {
                    "winner": None,
                    "final_result": "",
                    "reason": reason,
                    "confidence": 0.0,
                    "success": False,
                    "valid_agent_count": 0,
                    "expected_agent_count": agent_count,
                    "quality_checks": {
                        "research_preflight": False,
                        "all_agents_completed": False,
                        "leader_succeeded": False,
                        "sources_present": False,
                        "presentation_saved": False,
                    },
                    "presentation_output": {
                        "saved": False,
                        "canva_created": False,
                        "reason": (
                            "Chưa đủ nguồn hợp lệ để tạo slide; nếu có bản dự phòng "
                            "trước đó thì giữ nguyên, không ghi đè."
                        ),
                    },
                },
                "agent_count_expected": agent_count,
                "agent_count_completed": 0,
                "elapsed": time.time() - start,
            }
        orch_logger.print(
            "RESEARCH PREFLIGHT",
            f"Đạt {len(shared_evidence)} nguồn; dùng chung cho mọi agent.",
            "green",
        )

    orch_logger.print("SWARM", f"{agent_count} agent khởi động song song", "bold yellow")
    orch_logger.print(
        "SWARM",
        "Mỗi agent có memory, model fallback và phiên Playwright riêng; "
        "leader sẽ thu phản biện chéo.",
        "cyan",
    )
    model_assignments = [
        {
            "agent_id": index + 1,
            "model": models[index],
            "role": _AGENT_ROLES[index % len(_AGENT_ROLES)],
        }
        for index in range(agent_count)
    ]
    orch_logger.log("swarm_start", {
        "goal": goal,
        "agent_count": agent_count,
        "models": models,
        "model_assignments": model_assignments,
        "strategy": strategy,
        "max_steps": max_steps,
        "timeout": timeout,
    })
    for assignment in model_assignments:
        orch_logger.print(
            "MODEL ASSIGNMENT",
            f"Agent {assignment['agent_id']} → {assignment['model']} | "
            f"{assignment['role']}",
            "cyan",
        )

    desktop_terms = ("desktop", "computer", "mouse", "keyboard", "calculator", "máy tính", "chuột", "bàn phím", "màn hình")
    if any(term in goal.casefold() for term in desktop_terms):
        desktop_candidate_ids = list(range(1, agent_count + 1))
        desktop_scores = Memory().desktop_operator_scores(desktop_candidate_ids)
        desktop_operator_id, desktop_operator_reason = select_desktop_operator(
            desktop_scores,
            desktop_candidate_ids,
        )
        orch_logger.log("desktop_operator_selected", {
            "agent_id": desktop_operator_id,
            "reason": desktop_operator_reason,
            "verified_history": desktop_scores,
        })
        orch_logger.print(
            "DESKTOP OPERATOR",
            f"Agent {desktop_operator_id}: {desktop_operator_reason}",
            "cyan",
        )
    else:
        desktop_operator_id = None
        desktop_operator_reason = "Desktop access was not requested by the goal."
    agents = [
        SwarmAgent(
            agent_id=i + 1,
            model_name=models[i],
            max_steps=max_steps,
            yolo=yolo,
            session_id=session_id,
            role=_AGENT_ROLES[i % len(_AGENT_ROLES)],
            shared_evidence=shared_evidence,
            desktop_operator_id=desktop_operator_id,
        )
        for i in range(agent_count)
    ]
    file_checkpoint = FileCheckpoint()
    for agent in agents:
        agent.retain_browser_session = True
        agent.file_checkpoint = file_checkpoint

    agent_executors = [
        ThreadPoolExecutor(max_workers=1)
        for _ in range(agent_count)
    ]
    tasks = [
        asyncio.create_task(
            _run_agent_with_timeout(agent_executors[index], agent, goal, timeout)
        )
        for index, agent in enumerate(agents)
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for agent, result in zip(agents, raw_results):
        if isinstance(result, BaseException):
            orch_logger.print(
                "FAIL",
                f"A{agent.agent_id} FAIL ({result})",
                "bold red",
            )
            agent.logger.print("FAIL", str(result), "bold red")
            agent.logger.log("agent_fail", {
                "timestamp": time.time(),
                "status": "crashed",
                "error": str(result),
                "exception_type": type(result).__name__,
            })
            orch_logger.log("agent_failed", {
                "agent_id": agent.agent_id,
                "error": str(result),
                "exception_type": type(result).__name__,
            })
            continue
        if result is None:
            orch_logger.print(
                "FAIL", f"A{agent.agent_id} FAIL (không có kết quả)", "bold red"
            )
            orch_logger.log("agent_failed", {
                "agent_id": agent.agent_id,
                "error": "Agent returned None",
            })
            continue
        results.append(result)

    elapsed = time.time() - start
    orch_logger.print("SWARM", f"Các lượt agent kết thúc trong {elapsed:.1f}s", "bold yellow")

    for r in results:
        aid = r.get("agent_id", "?")
        status = r.get("status", "?")
        t = r.get("elapsed", 0)
        orch_logger.print(f"A{aid}", f"status={status}, time={t:.1f}s", "cyan")

    aggregator = Aggregator(
        strategy=strategy,
        session_id=session_id,
        expected_count=agent_count,
    )
    final = aggregator.aggregate(results, goal)
    verified_source_urls = _verified_search_urls(results, shared_evidence)
    canva_preflight = None
    final_executor_agent_id = None
    if is_presentation_goal(goal):
        final_executor_agent_id, executor_reason = _select_final_executor(
            final, results
        )
        final["final_executor_agent_id"] = final_executor_agent_id
        final["final_executor_reason"] = executor_reason
        usable_sessions = [
            row.get("browser_session", {})
            for row in results
            if isinstance(row, dict)
            and row.get("browser_session", {}).get("isolated_context") is True
            and row.get("browser_session", {}).get("session_id")
            and row.get("browser_session", {}).get("status_at_agent_exit") == "ready"
        ]
        unique_session_ids = {
            row.get("session_id") for row in usable_sessions
        }
        sessions_ready = (
            len(unique_session_ids) == len(usable_sessions)
            and len(results) == agent_count
        )
        if sessions_ready and final_executor_agent_id is not None:
            selected_index = final_executor_agent_id - 1
            canva_preflight = _preflight_canva_for_agent(
                final_executor_agent_id,
                executor=agent_executors[selected_index],
                logger=agents[selected_index].logger,
            )
        else:
            if final_executor_agent_id is None:
                canva_preflight = {
                    "agent_id": None,
                    "status": "no_final_executor",
                    "reason": (
                        "The leader had no valid agent result to assign. "
                        "Canva was not opened."
                    ),
                    "isolated_context": False,
                    "design_created": False,
                }
            else:
                canva_preflight = {
                    "agent_id": final_executor_agent_id,
                    "status": "insufficient_agent_sessions",
                    "reason": "The agent result set was incomplete or browser sessions were not isolated.",
                    "isolated_context": False,
                    "design_created": False,
                }
        preflight_session_id = (canva_preflight or {}).get("session_id")
        browser_coverage_complete = bool(
            sessions_ready
            and final_executor_agent_id is not None
            and preflight_session_id
            and (canva_preflight or {}).get("isolated_context") is True
        )
        final["browser_session_coverage"] = {
            "expected": len(usable_sessions) + (
                0 if any(
                    row.get("agent_id") == final_executor_agent_id
                    and row.get("session_id") == preflight_session_id
                    for row in usable_sessions
                ) else 1
            ),
            "ready": len(unique_session_ids | ({preflight_session_id} if preflight_session_id else set())),
            "session_ids_unique": len(unique_session_ids) == len(usable_sessions),
            "complete": browser_coverage_complete,
            "final_executor_agent_id": final_executor_agent_id,
        }
        orch_logger.log("final_browser_handoff", {
            "final_executor_agent_id": final_executor_agent_id,
            "selection_reason": executor_reason,
            "browser_session_coverage": final["browser_session_coverage"],
            "canva_preflight": canva_preflight,
        })
    for index, (agent, worker_executor) in enumerate(zip(agents, agent_executors)):
        try:
            worker_executor.submit(
                _close_agent_browser_session,
                agent.agent_id,
                agent.logger,
            ).result(timeout=5)
        except Exception as exc:
            orch_logger.log("browser_session_cleanup_timeout", {
                "agent_id": agent.agent_id,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })
        worker_executor.shutdown(wait=False, cancel_futures=False)
        for result in results:
            if result.get("agent_id") == agent.agent_id:
                browser_session = result.get("browser_session", {})
                if browser_session.get("session_id"):
                    browser_session["status"] = "closed_after_handoff"
    presentation_output = _save_presentation_draft(
        goal,
        final,
        session_id,
        verified_source_urls=verified_source_urls,
        research_evidence=shared_evidence,
        canva_preflight=canva_preflight,
        final_executor_agent_id=final_executor_agent_id,
    )
    if presentation_output and not presentation_output["saved"]:
        orch_logger.print("PRESENTATION", presentation_output["reason"], "yellow")
        orch_logger.log("presentation_not_saved", presentation_output)
    elif presentation_output:
        orch_logger.print(
            "PRESENTATION",
            f"Đã lưu bản thảo {presentation_output['slide_count']} slide tại "
            f"{presentation_output['path']}; Canva status="
            f"{presentation_output.get('canva_status', 'unknown')}: "
            f"{presentation_output.get('canva_reason', 'not checked')}. "
            f"{presentation_output['source_check']}",
            "green" if presentation_output["source_check_passed"] else "yellow",
        )
        orch_logger.log("presentation_draft_saved", presentation_output)

    final_urls = re.findall(
        r"https?://[^\s<>\])}]+",
        str(final.get("final_result", "")),
    )
    source_count = len({
        url.rstrip(".,;:")
        for url in final_urls
        if urlparse(url.rstrip(".,;:")).scheme in {"http", "https"}
        and urlparse(url.rstrip(".,;:")).netloc
    })
    agent_source_urls = {
        url.rstrip(".,;:")
        for result in results
        for url in re.findall(
            r"https?://[^\s<>\])}]+",
            str(result.get("result", "")),
        )
        if urlparse(url.rstrip(".,;:")).scheme in {"http", "https"}
        and urlparse(url.rstrip(".,;:")).netloc
    }
    agent_source_urls.update(verified_source_urls)
    final_source_urls = {
        url.rstrip(".,;:")
        for url in final_urls
        if urlparse(url.rstrip(".,;:")).scheme in {"http", "https"}
        and urlparse(url.rstrip(".,;:")).netloc
    }
    completed_agents = sum(
        result.get("status") == "completed" and not result.get("failed_tasks")
        for result in results
    )
    source_required = any(
        term in goal.casefold()
        for term in (
            "research", "source", "citation", "reference", "presentation",
            "slide", "canva", "nghiên cứu", "nguồn", "trích dẫn",
            "thuyết trình", "thuyet trinh",
        )
    )
    healthy_model_keys = {
        _model_key(check["model"])
        for check in model_api_checks
        if check["ok"]
    }
    isolated_sessions = [
        result.get("browser_session", {}).get("session_id")
        for result in results
        if result.get("browser_session", {}).get("isolated_context") is True
        and result.get("browser_session", {}).get("session_id")
    ]
    browser_sessions_isolated = (
        len(results) == agent_count
        and len(isolated_sessions) == len(set(isolated_sessions))
        and all(
            result.get("browser_session", {}).get("status") == "closed_after_handoff"
            and result.get("browser_session", {}).get("status_at_agent_exit") == "ready"
            for result in results
            if result.get("browser_session", {}).get("session_id")
        )
        and (
            not is_presentation_goal(goal)
            or final.get("browser_session_coverage", {}).get("complete") is True
        )
    )
    checks = {
        "distinct_models_assigned": len({_model_key(model) for model in models}) == agent_count,
        "model_apis_healthy": all(
            _model_key(model) in healthy_model_keys for model in models
        ),
        "all_agents_completed": completed_agents == agent_count,
        "all_agent_ids_unique": (
            len({result.get("agent_id") for result in results}) == agent_count
        ),
        "browser_sessions_isolated": browser_sessions_isolated,
        "leader_succeeded": not final.get("leader_failed") and not final.get("needs_review"),
        "peer_review_coverage": (
            strategy != "debate"
            or final.get("peer_review_count", 0) == agent_count
        ),
        "sources_present": not source_required or source_count >= 3,
        "sources_from_validated_search": (
            not source_required or (
                len(verified_source_urls) >= 3
                and final_source_urls.issubset(verified_source_urls)
            )
        ),
        "sources_trace_to_agent_results": (
            not source_required or final_source_urls.issubset(agent_source_urls)
        ),
        "presentation_saved": (
            not presentation_output
            or (
                presentation_output.get("saved")
                and presentation_output.get("source_check_passed")
                and 6 <= presentation_output.get("slide_count", 0) <= 8
            )
        ),
        "canva_created": (
            "canva" not in goal.casefold()
            or bool(
                presentation_output
                and presentation_output.get("canva_created") is True
            )
        ),
    }
    if presentation_output and presentation_output.get("saved"):
        checks["presentation_sources"] = presentation_output.get(
            "source_check_passed", False
        )
    final["quality_checks"] = checks
    final["success"] = (
        final.get("winner") is not None
        and bool(str(final.get("final_result", "")).strip())
        and all(checks.values())
    )
    if final["success"]:
        checkpoint_result = file_checkpoint.commit()
    else:
        checkpoint_result = file_checkpoint.rollback()
    final["file_checkpoint"] = checkpoint_result

    final["run_status"], final["run_evidence"] = _run_status_evidence(
        results,
        agent_count,
        final["success"],
        checks,
    )
    orch_logger.log("quality_checks", checks)
    orch_logger.log("swarm_done", {
        "elapsed": elapsed,
        "final": final,
    })

    github_skill_recommendations = [
        {
            "agent_id": row.get("agent_id"),
            **recommendation,
        }
        for row in results
        for recommendation in row.get("github_skill_recommendations", [])
        if isinstance(recommendation, dict)
    ]
    return {
        "agent_results": redact_data(results),
        "aggregator": redact_data(final),
        "github_skill_recommendations": redact_data(github_skill_recommendations),
        "agent_count_expected": agent_count,
        "desktop_operator_agent_id": desktop_operator_id,
        "desktop_operator_selection_reason": desktop_operator_reason,
        "agent_count_completed": completed_agents,
        "model_assignments": redact_data(model_assignments),
        "model_api_checks": redact_data(model_api_checks),
        "elapsed": elapsed,
    }


def run_swarm(goal, cwd=".", **kwargs):
    """
    Entry point đồng bộ — tiện gọi từ main.py.
    """
    return asyncio.run(run_swarm_async(goal, **kwargs))

# --- Thêm run_single_agent và run_swarm_mode vào đây để main.py import ---

def run_single_agent(goal, max_steps, yolo, cwd="."):
    """Chạy chế độ đơn agent."""
    import os
    from core.sensitive_data import redact_text
    from tools.write_safety import prepare_workspace_path

    goal = redact_text(str(goal))
    safe_cwd = prepare_workspace_path(
        cwd or ".",
        must_exist=True,
        directory=True,
    )
    os.chdir(safe_cwd)

    from core.planner import generate_plan
    from core.executor import execute_task
    from core.critic import evaluate_task
    from core.reflection import reflect
    from core.memory import memory
    from core.skill_resolver import SkillResolver
    from core.plan_policy import validate_plan

    default_logger.print("GOAL", goal, "bold green")
    default_logger.log("start_goal", goal)

    similars = memory.search_similar(goal)
    if similars:
        default_logger.print("MEMORY", f"Tìm thấy {len(similars)} kinh nghiệm cũ", "blue")

    default_logger.print("PLANNING", "Đang lập kế hoạch...", "cyan")
    plan = generate_plan(goal)
    plan_issues = validate_plan(goal, plan)
    if plan_issues:
        default_logger.log("plan_validation_failed", {
            "attempt": 1,
            "issues": plan_issues,
        })
        plan = generate_plan(
            f"{goal}\n\nPlan policy rejected the prior plan. Repair it once with "
            f"goal-relevant tasks, explicit success criteria, and permitted tools. Issues: {plan_issues}"
        )
        plan_issues = validate_plan(goal, plan)
        if plan_issues:
            default_logger.print(
                "ERROR",
                "Plan bị từ chối sau một lần sửa: " + "; ".join(plan_issues),
                "red",
            )
            default_logger.log("plan_rejected", {"attempts": 2, "issues": plan_issues})
            return False

    default_logger.print("PLAN", f"Hiểu mục tiêu: {plan.get('understanding')}", "green")
    if plan.get("assumptions"):
        default_logger.print("ASSUMPTIONS", plan.get("assumptions"), "yellow")
        default_logger.log("assumptions", plan.get("assumptions"))

    tasks = plan.get("tasks", [])
    total = len(tasks)
    step = 0
    task_idx = 0

    while task_idx < total and step < max_steps:
        task = tasks[task_idx]
        default_logger.print(f"TASK {task_idx+1}/{total}", task["desc"], "bold magenta")

        retry_count = 0
        passed = False

        while retry_count < 3 and not passed:
            step += 1
            result = execute_task(task, yolo)

            if isinstance(result, str) and result.startswith("ERROR ["):
                critic_res = {"passed": False, "reason": result, "fix": "Use a different authorized action."}
            else:
                critic_res = evaluate_task(task, str(result))
            if isinstance(critic_res, dict) and isinstance(critic_res.get("passed"), bool):
                passed = critic_res.get("passed", False)
                reason = critic_res.get("reason", "")
                fix = critic_res.get("fix", "")
            else:
                passed = False
                reason = "Critic returned no valid pass/fail evidence."
                fix = "Re-evaluate the task with explicit evidence."

            if passed:
                default_logger.print("CRITIC", f"Pass: {reason}", "green")
                memory.save_long_term(task["id"], task["desc"], "Thành công")
                SkillResolver.record_outcome(
                    [
                        (task.get("_resolved_skill") or {}).get("skill"),
                        task.get("_selected_tool"),
                    ],
                    True,
                    memory=memory,
                    logger=default_logger,
                )
                break
            else:
                retry_count += 1
                SkillResolver.record_outcome(
                    [
                        (task.get("_resolved_skill") or {}).get("skill"),
                        task.get("_selected_tool"),
                    ],
                    False,
                    memory=memory,
                    logger=default_logger,
                )
                default_logger.print("CRITIC", f"Fail ({retry_count}/3): {reason}. Gợi ý sửa: {fix}", "red")
                if task.get("_last_action_fingerprint"):
                    task.setdefault("_failed_action_fingerprints", []).append(
                        task["_last_action_fingerprint"]
                    )
                from core.recovery import retry_decision

                can_retry, retry_reason = retry_decision(
                    task.get("_selected_tool"),
                    task.get("_selected_action"),
                    retry_count,
                )
                default_logger.log("trial_recovery_decision", {
                    "task_id": task.get("id"),
                    "retry_allowed": can_retry,
                    "reason": retry_reason,
                })
                if not can_retry:
                    break
                task["desc"] += f" [SỬA LỖI: {fix}]"

        if not passed:
            default_logger.print("REPLAN", "Task fail 3 lần, chuyển task tiếp", "yellow")

        task_idx += 1

        if step % 5 == 0 and step > 0:
            history = memory.get_short_term_context()
            ref = reflect(history)
            default_logger.print("REFLECT", str(ref), "cyan")
            default_logger.log("reflection", ref)

    default_logger.print("DONE", "HOÀN THÀNH – Autonomous Agent sẵn sàng", "bold green")
    default_logger.log("end", "Done")


def run_swarm_mode(goal, agent_count, strategy, models, max_steps, yolo, cwd="."):
    """Chạy chế độ swarm qua orchestrator. Gọi từ main.py."""
    import os
    if cwd and cwd != ".":
        os.chdir(cwd)
    if strategy != "debate":
        strategy = "debate"

    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()

    console.print(Panel(
        Text(f"SWARM MODE: {agent_count} agents | strategy={strategy}", style="bold yellow"),
        title="🐝 Autonomous Agent Swarm",
        border_style="yellow",
    ))

    model_list = None
    if models:
        model_list = [m.strip() for m in models.split(",")]

    result = run_swarm(
        goal=goal,
        agent_count=agent_count,
        models=model_list,
        strategy=strategy,
        max_steps=max_steps,
        yolo=yolo,
    )

    agg = result.get("aggregator", {})
    elapsed = result.get("elapsed", 0)
    presentation_output = agg.get("presentation_output")

    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold]Winner:[/bold] {agg.get('winner', '?')}\n"
            f"[bold]Confidence:[/bold] {agg.get('confidence', 0):.2f}\n"
            f"[bold]Reason:[/bold] {agg.get('reason', '')}\n"
            f"[bold]Final Result:[/bold]\n{str(agg.get('final_result', ''))[:500]}"
        ),
        title="📊 Aggregator Result",
        border_style="green" if agg.get("success") else "red",
    ))

    best_parts = agg.get("best_parts", [])
    if best_parts:
        console.print("\n[bold cyan]Best Parts:[/bold cyan]")
        for part in best_parts:
            console.print(f"  • {part}")
    if presentation_output:
        if presentation_output.get("saved"):
            console.print(
                "[bold cyan]Canva-ready draft:[/bold cyan] "
                f"{presentation_output['path']} "
                f"(Canva: {presentation_output.get('canva_status', 'unknown')}; "
                f"{presentation_output.get('canva_reason', 'design not created')})"
            )
        else:
            console.print(
                "[yellow]Presentation draft not saved:[/yellow] "
                f"{presentation_output.get('reason', 'Slide output is incomplete.')}"
            )

    succeeded = bool(agg.get("success"))
    if succeeded:
        console.print(f"\n[bold green][DONE] Tổng thời gian: {elapsed:.1f}s[/bold green]")
    else:
        console.print(f"\n[bold red][FAILED] Tổng thời gian: {elapsed:.1f}s[/bold red]")
        failed_checks = [
            name for name, passed in agg.get("quality_checks", {}).items()
            if not passed
        ]
        if failed_checks:
            console.print(
                "[bold red]Quality checks failed:[/bold red] "
                + ", ".join(failed_checks)
            )
    console.print(
        f"[dim]Chi phí token phụ thuộc số agent và lượt debate/leader "
        f"(agents={agent_count}).[/dim]"
    )
    return succeeded
