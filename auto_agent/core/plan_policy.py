"""Deterministic scope and permission checks for model-generated plans."""
import re
import unicodedata

from core.research_policy import (
    is_presentation_draft_task,
    is_search_task,
    requires_web_research,
)
from core.tool_broker import tool_actions_for_agent


_STOP_WORDS = {
    "a", "an", "and", "as", "at", "by", "do", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "the", "to", "with", "va", "và", "cua",
    "cho", "de", "để", "mot", "một", "cac", "các", "nhung", "những", "ve",
    "về", "cho", "hay", "hãy", "can", "cần", "phai", "phải", "duoc", "được",
    "này", "đó", "trong", "theo", "muc", "tiêu", "mục", "tieu", "task", "step",
}
_ACTION_WORDS = {
    "create", "make", "draft", "write", "prepare", "generate", "produce", "build",
    "tao", "soan", "viet", "lap", "thuc", "hien", "kiem", "tra", "analyze",
    "inspect", "verify", "find", "tim", "kiem", "search", "research", "nghien",
    "cuu", "read", "open", "use", "dung", "mo", "xem", "collect", "gather",
}
_GROUPS = (
    {"research", "researching", "nghien", "cuu", "search", "source", "sources", "evidence", "citation", "reference", "nguon", "timkiem"},
    {"presentation", "presentations", "slide", "slides", "deck", "canva", "thuyet", "trinh"},
    {"browser", "web", "website", "site", "page", "browser", "trang", "webpage"},
    {"desktop", "computer", "mouse", "keyboard", "screen", "calculator", "desktop", "chuot", "banphim", "manhinh"},
    {"file", "files", "document", "documents", "path", "folder", "report", "output", "tep", "hoso"},
    {"terminal", "shell", "command", "cmd", "powershell", "test", "build", "run", "console"},
    {"image", "photo", "screenshot", "ocr", "vision", "picture", "anh", "hinh"},
)
_DANGEROUS_INTENTS = {
    "install": re.compile(r"\b(?:install|pip|package|cài đặt|cai dat)\b", re.IGNORECASE),
    "delete": re.compile(r"\b(?:delete|remove|erase|rm|xóa|xoa|gỡ bỏ|go bo)\b", re.IGNORECASE),
    "system configuration": re.compile(
        r"\b(?:system config|registry|sudo|administrator|admin rights|cấu hình hệ thống|cau hinh he thong)\b",
        re.IGNORECASE,
    ),
    "external publication": re.compile(
        r"\b(?:publish|send|share publicly|purchase|pay|submit|công khai|gui di|mua hàng|thanh toán)\b",
        re.IGNORECASE,
    ),
}


def _normalize(value):
    decomposed = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _tokens(value):
    return set(re.findall(r"[a-z0-9]+", _normalize(value)))


def _group_hits(tokens):
    return {index for index, group in enumerate(_GROUPS) if tokens & group}


def _goal_scope(goal):
    text = _normalize(goal)
    scope = set()
    if requires_web_research(goal) or re.search(
        r"\b(research|search|source|citation|reference|evidence|nghien cuu|nguon|tim kiem)\b",
        text,
    ):
        scope.add("research")
    if re.search(r"\b(browser|web|website|canva|trang web|website)\b", text):
        scope.add("browser")
    if re.search(r"\b(desktop|computer|mouse|keyboard|calculator|may tinh|chuot|ban phim|man hinh)\b", text):
        scope.add("desktop")
    if re.search(r"\b(file|document|path|folder|save|export|output|report|tep|luu|ghi file|doc file)\b", text):
        scope.add("file")
    if re.search(r"\b(terminal|shell|command|cmd|powershell|test|build|run command)\b", text):
        scope.add("shell")
    if re.search(r"\b(image|photo|screenshot|ocr|vision|picture|anh|hinh anh)\b", text):
        scope.add("vision")
    return scope


def _task_permissions(goal, task, agent_id=None, desktop_operator_id=None):
    description = str(task.get("desc", ""))
    if is_presentation_draft_task(task):
        return {"presentation_draft.generate"}
    if is_search_task(task):
        return {"search_tool.search"} if "research" in _goal_scope(goal) else set()

    text = _normalize(description)
    scope = _goal_scope(goal)
    permissions = set()

    if "research" in scope and re.search(
        r"\b(search|research|source|citation|reference|evidence|find|open|read|nguon|tim kiem)\b",
        text,
    ):
        permissions.add("search_tool.search")
        if re.search(r"\b(open|read|page|website|browser|url|source|mo|trang)\b", text):
            permissions.update({
                "browser_tool.goto",
                "browser_tool.scrape",
            })
            if "screenshot" in _normalize(goal) and "screenshot" in text:
                permissions.add("browser_tool.screenshot")
    if "browser" in scope and re.search(
        r"\b(browser|web|website|page|canva|open|click|navigate|browse|trang)\b", text
    ):
        permissions.update({
            "browser_tool.goto",
            "browser_tool.scrape",
        })
        if "screenshot" in _normalize(goal) and "screenshot" in text:
            permissions.add("browser_tool.screenshot")
        goal_allows_interaction = bool(re.search(
            r"\b(canva|click|interact|edit|create|fill|type|thao tác|thao tac)\b",
            _normalize(goal),
        ))
        task_requests_interaction = bool(re.search(
            r"\b(click|interact|edit|fill|type|nhap|bam|sua)\b", text
        ))
        if goal_allows_interaction and task_requests_interaction:
            permissions.update({
                "browser_tool.click",
                "browser_tool.click_at",
                "browser_tool.type_text",
            })
    if "file" in scope and re.search(
        r"\b(file|document|path|folder|read|open|inspect|doc|tep|mo file)\b", text
    ):
        permissions.add("file_tool.read")
    if "file" in scope and re.search(
        r"\b(write|save|create|edit|export|output|report|ghi|luu|tao|viet)\b", text
    ):
        permissions.add("file_tool.write")
    if "shell" in scope and re.search(
        r"\b(terminal|shell|command|cmd|powershell|test|build|run|git|python)\b", text
    ):
        permissions.add("shell_tool.run")
    goal_requests_desktop_action = bool(re.search(
        r"\b(click|type|press|interact|control|operate|open|launch|enter|edit|save|observe|inspect|look|check|bam|nhap|mo|xem|kiem tra|dieu khien)\b",
        _normalize(goal),
    ))
    if (
        agent_id is not None
        and agent_id == desktop_operator_id
        and "desktop" in scope
        and goal_requests_desktop_action
        and re.search(
            r"\b(desktop|computer|mouse|keyboard|click|type|screen|chuot|ban phim|man hinh)\b",
            text,
        )
    ):
        if re.search(
            r"\b(observe|inspect|look|read|check|xem|quan sat|kiem tra)\b", text
        ):
            permissions.add("computer_tool.observe")
        if re.search(
            r"\b(click|type|press|interact|control|operate|open|launch|enter|edit|save|bam|nhap|mo|dieu khien)\b",
            text,
        ):
            permissions.add("computer_tool.perform")
    allowed_tools = tool_actions_for_agent(agent_id, desktop_operator_id)
    declared = task.get("allowed_tools")
    if declared is not None:
        if not isinstance(declared, list):
            return set()
        declared_permissions = set()
        for item in declared:
            if not isinstance(item, str):
                continue
            if "." in item:
                declared_permissions.add(item)
            elif item in allowed_tools:
                declared_permissions.update(
                    f"{item}.{action}" for action in allowed_tools[item]
                )
        permissions &= declared_permissions
    return permissions


def validate_plan(goal, plan, agent_id=None, desktop_operator_id=None):
    """Return policy violations and annotate accepted tasks with trusted permissions."""
    if not isinstance(plan, dict):
        return ["plan must be a JSON object"]
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return ["plan must contain at least one task"]
    if len(tasks) > 20:
        return ["plan exceeds the 20-task limit"]

    issues = []
    task_ids = []
    goal_tokens = _tokens(goal)
    goal_subject_tokens = goal_tokens - _STOP_WORDS - _ACTION_WORDS
    goal_groups = _group_hits(goal_tokens)

    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            issues.append(f"task {index} must be an object")
            continue
        task_id = task.get("id")
        description = task.get("desc")
        success = task.get("success")
        if not isinstance(task_id, str) or not task_id.strip():
            issues.append(f"task {index} needs a non-empty id")
        else:
            task_ids.append(task_id)
        if not isinstance(description, str) or not description.strip():
            issues.append(f"task {task_id or index} needs a description")
            continue
        if not isinstance(success, str) or len(success.strip()) < 8:
            issues.append(f"task {task_id or index} needs explicit success criteria")

        task_tokens = _tokens(description)
        task_subject_tokens = task_tokens - _STOP_WORDS - _ACTION_WORDS
        if not (goal_subject_tokens & task_subject_tokens) and not (
            goal_groups & _group_hits(task_tokens)
        ):
            issues.append(f"task {task_id or index} is not demonstrably relevant to the goal")

        for intent, pattern in _DANGEROUS_INTENTS.items():
            if pattern.search(description) and not pattern.search(str(goal)):
                issues.append(
                    f"task {task_id or index} requests {intent} not authorized by the goal"
                )

        permissions = _task_permissions(
            goal,
            {**task, "allowed_tools": None},
            agent_id=agent_id,
            desktop_operator_id=desktop_operator_id,
        )
        allowed_tools = tool_actions_for_agent(agent_id, desktop_operator_id)
        declared = task.get("allowed_tools")
        if declared is not None:
            if not isinstance(declared, list):
                issues.append(f"task {task_id or index} allowed_tools must be a list")
                permissions = set()
            else:
                declared_permissions = set()
                for item in declared:
                    if not isinstance(item, str):
                        continue
                    if "." in item:
                        declared_permissions.add(item)
                    elif item in allowed_tools:
                        declared_permissions.update(
                            f"{item}.{action}" for action in allowed_tools[item]
                        )
                unauthorized = declared_permissions - permissions
                if unauthorized:
                    issues.append(
                        f"task {task_id or index} declares tools outside the goal scope: "
                        + ", ".join(sorted(unauthorized))
                    )
                permissions &= declared_permissions
        if not permissions:
            issues.append(f"task {task_id or index} has no tool permission within the goal scope")
        task["_authorized_actions"] = sorted(permissions)

    if len(task_ids) != len(set(task_ids)):
        issues.append("task ids must be unique")
    known_ids = set(task_ids)
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue
        dependencies = task.get("deps", [])
        if not isinstance(dependencies, list):
            issues.append(f"task {task.get('id', index)} dependencies must be a list")
            continue
        for dependency in dependencies:
            if dependency not in known_ids or dependency == task.get("id"):
                issues.append(
                    f"task {task.get('id', index)} has an invalid dependency"
                )
    return issues
