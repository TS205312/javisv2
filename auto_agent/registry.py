"""Built-in tools and persistent metadata for discovered GitHub skills."""
import json
import os
import tempfile
from pathlib import Path


TOOL_REGISTRY = {
    "shell_tool": {
        "desc": "Chạy các lệnh terminal/cmd. Hỗ trợ shell commands.",
        "use_when": "Cần chạy python, pip, git, ls, dir, hoặc các công cụ CLI.",
        "actions": ["run"],
    },
    "file_tool": {
        "desc": "Đọc, ghi, và tìm kiếm file trên hệ thống.",
        "use_when": "Cần tạo file, đọc nội dung file, hoặc tìm file.",
        "actions": ["read", "write", "search"],
    },
    "coder_tool": {
        "desc": "Sinh code, sửa code trực tiếp vào file.",
        "use_when": "Cần viết script, sửa file python, tạo tool mới.",
        "actions": ["write_code", "fix_code"],
    },
    "browser_tool": {
        "desc": "Điều khiển một phiên trang Playwright riêng cho từng AI; mỗi phiên có chuột ảo độc lập.",
        "use_when": "Cần mở website, đọc nội dung, click phần tử hoặc tọa độ trang, nhập văn bản, chụp ảnh.",
        "actions": ["goto", "scrape", "screenshot", "click", "click_at", "type_text"],
    },
    "search_tool": {
        "desc": "Tìm kiếm thông tin trên internet qua DuckDuckGo.",
        "use_when": "Cần lấy thông tin mới nhất từ web, tìm kiếm tin tức, tài liệu.",
        "actions": ["search"],
    },
    "blender_tool": {
        "desc": "Chạy Blender script (headless) để render 3D.",
        "use_when": "Cần tạo 3D, render video/image qua blender --background.",
        "actions": ["run_script"],
    },
    "media_tool": {
        "desc": "Xử lý audio/video dùng ffmpeg.",
        "use_when": "Cần chuyển đổi định dạng, cắt ghép video/audio.",
        "actions": ["ffmpeg_run"],
    },
    "computer_tool": {
        "desc": "Điều khiển desktop dùng chung không giới hạn số thao tác mỗi chuỗi; thao tác được khóa tuần tự, chụp ảnh trước/sau và hỗ trợ kiểm chứng OCR.",
        "use_when": "Dùng cho ứng dụng trên máy tính; trong swarm chỉ dùng perform với verify_text ngắn, dễ quan sát. Không phát lại thao tác GUI đã chạy.",
        "actions": ["perform", "click", "type_text", "press", "screenshot"],
    },
    "vision_tool": {
        "desc": "Phân tích hình ảnh, đọc chữ (OCR), mô tả ảnh bằng AI.",
        "use_when": "Cần hiểu nội dung ảnh, lấy text từ ảnh chụp.",
        "actions": ["analyze", "ocr"],
    },
}

_DEFAULT_META = {
    "source": "builtin",
    "repo_url": "",
    "stars": 0,
    "license": "",
    "last_updated": "",
    "success_count": 0,
    "test_pass_rate": 0.0,
    "test_count": 0,
    "user_rating": None,
}
_REGISTRY_PATH = Path(__file__).resolve().parent / "skill_registry.json"
SKILL_REGISTRY = {}

for _skill in TOOL_REGISTRY.values():
    _skill["meta"] = dict(_DEFAULT_META)


def _write_registry():
    """Persist metadata atomically without modifying this Python source file."""
    payload = {
        name: {
            key: value
            for key, value in skill.items()
            if key != "actions" or isinstance(value, list)
        }
        for name, skill in get_all_skills().items()
    }
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".skill_registry_", suffix=".json", dir=_REGISTRY_PATH.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, _REGISTRY_PATH)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _load_registry():
    if not _REGISTRY_PATH.is_file():
        return
    try:
        with _REGISTRY_PATH.open(encoding="utf-8") as file:
            saved_skills = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[SKILL WARNING] Không đọc được {_REGISTRY_PATH.name}: {exc}")
        return

    if not isinstance(saved_skills, dict):
        print(f"[SKILL WARNING] Bỏ qua {_REGISTRY_PATH.name}: dữ liệu không hợp lệ.")
        return

    for name, saved in saved_skills.items():
        if not isinstance(saved, dict):
            continue
        meta = saved.get("meta")
        if not isinstance(meta, dict):
            continue
        if name in TOOL_REGISTRY:
            TOOL_REGISTRY[name]["meta"].update(meta)
        elif name.startswith("github:"):
            SKILL_REGISTRY[name] = saved


def get_skill(name):
    if name in TOOL_REGISTRY:
        return TOOL_REGISTRY[name]
    return SKILL_REGISTRY.get(name)


def get_all_skills():
    return {**TOOL_REGISTRY, **SKILL_REGISTRY}


def register_skill(name, skill):
    """Register a validated GitHub skill and persist its metadata."""
    if not name.startswith("github:") or not isinstance(skill, dict):
        raise ValueError("GitHub skill names must use the 'github:' prefix.")
    if not isinstance(skill.get("meta"), dict):
        raise ValueError("Registered skills must include a metadata object.")
    SKILL_REGISTRY[name] = skill
    _write_registry()


def update_skill_meta(name, updates):
    skill = get_skill(name)
    if skill is None:
        raise KeyError(f"Không tìm thấy skill: {name}")
    skill["meta"].update(updates)
    if name.startswith("github:"):
        SKILL_REGISTRY[name] = skill
    _write_registry()


_load_registry()
