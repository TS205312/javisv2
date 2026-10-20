"""Small deterministic rules for web-research and slide-generation tasks."""
import re
from urllib.parse import urlparse


_RESEARCH_TERMS = (
    "research", "literature", "evidence", "source", "citation", "reference",
    "nghiên cứu", "nguồn", "trích dẫn", "tài liệu tham khảo", "bằng chứng",
)
_PRESENTATION_TERMS = (
    "slide", "presentation", "canva", "thuyết trình", "thuyet trinh",
)

_TASK_TYPES = {
    "search": "research",
    "research": "research",
    "presentation_draft": "presentation_draft",
    "draft_slides": "presentation_draft",
    "slides": "presentation_draft",
}


def _description(task):
    if isinstance(task, dict):
        return str(task.get("desc", "") or "")
    return str(task or "")


def _task_type(task):
    if not isinstance(task, dict):
        return None
    raw_type = task.get("type") or task.get("kind")
    return _TASK_TYPES.get(str(raw_type or "").casefold())


def extract_research_topic(text):
    value = str(text or "").strip()
    quoted = re.search(r"[\u201c\u2018\"']([^\u201d\u2019\"']{3,160})[\u201d\u2019\"']", value)
    if quoted:
        return " ".join(quoted.group(1).split())

    topic_match = re.search(
        r"(?:nghiên cứu|research|study)\s+(.+?)(?=[,;.!?\n]|\s+(?:và tạo|and create|to create|and make)\b|$)",
        value,
        flags=re.IGNORECASE,
    )
    if topic_match:
        topic = " ".join(topic_match.group(1).split()).strip(" -*\t:–—")
        if topic:
            return topic[:160]

    first_line = value.splitlines()[0].strip(" -*\t") if value else ""
    first_sentence = re.split(r"[.;!?]", first_line, maxsplit=1)[0].strip()
    first_sentence = re.sub(
        r"^(?:dùng lại đúng mục tiêu thử nghiệm:\s*)?(?:nghiên cứu|research|study)\s+",
        "",
        first_sentence,
        flags=re.IGNORECASE,
    )
    return first_sentence[:160].strip(" ,:–—-") or value[:160]


def is_presentation_goal(text):
    value = _description(text).casefold()
    return any(term in value for term in _PRESENTATION_TERMS)


def requires_web_research(text):
    value = _description(text).casefold()
    return is_presentation_goal(value) or any(
        term in value for term in _RESEARCH_TERMS
    )


def is_search_task(text):
    tagged_type = _task_type(text)
    if tagged_type:
        return tagged_type == "research"

    value = _description(text).casefold().strip()
    if _starts_with_draft_intent(value):
        return False
    if not requires_web_research(value):
        return False
    has_search_action = re.match(
        r"^(?:tìm kiếm|tìm nguồn|thu thập nguồn|tra cứu|search|find|gather|collect|research)\b",
        value,
    ) is not None
    has_source_target = any(term in value for term in (
        "source", "sources", "citation", "reference", "nguồn", "tài liệu",
    ))
    return has_search_action and has_source_target


def _starts_with_draft_intent(value):
    return re.match(
        r"^(?:soạn|viết|tạo|draft|write|create|compose|prepare)\b",
        value.strip(),
    ) is not None


def is_presentation_draft_task(text):
    tagged_type = _task_type(text)
    if tagged_type:
        return tagged_type == "presentation_draft"

    value = _description(text).casefold().strip()
    return (
        is_presentation_goal(value)
        and _starts_with_draft_intent(value)
        and any(term in value for term in (
            "slide", "presentation", "thuyết trình", "thuyet trinh",
        ))
    )


def slide_count(text):
    matches = re.findall(
        r"(?im)^\s*(?:#{1,3}\s*)?(?:slide|trang)\s+(\d+)\b",
        str(text or ""),
    )
    return [int(number) for number in matches]


def append_references_from_evidence(text, evidence):
    value = str(text or "")
    if not value.strip() or re.search(
        r"(?im)^#{1,3}\s*(?:references|sources|tài liệu tham khảo|nguồn)\b",
        value,
    ):
        return value

    references = []
    seen_urls = set()
    for row in evidence or []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        url = str(row.get("url", "")).strip().rstrip(".,;:")
        parsed = urlparse(url)
        if (
            title
            and parsed.scheme in {"http", "https"}
            and parsed.netloc
            and url not in seen_urls
        ):
            references.append((title, url))
            seen_urls.add(url)
        if len(references) == 8:
            break

    if len(references) < 3:
        return value

    bibliography = "\n".join(
        f"{index}. {title}. {url}"
        for index, (title, url) in enumerate(references, start=1)
    )
    return f"{value.rstrip()}\n\n## Tài liệu tham khảo\n{bibliography}"


def presentation_issues(text, verified_source_urls=None):
    value = str(text or "")
    headings = list(re.finditer(
        r"(?im)^\s*(?:#{1,3}\s*)?(?:slide|trang)\s+(\d+)\b([^\n]*)",
        value,
    ))
    numbers = [int(match.group(1)) for match in headings]
    issues = []
    if not (
        6 <= len(numbers) <= 8
        and len(set(numbers)) == len(numbers)
        and set(numbers) == set(range(1, max(numbers) + 1))
    ):
        issues.append(f"cần 6–8 slide liên tục; hiện có {len(numbers)}")

    if headings and not re.search(
        r"(?:kết luận|conclusion)", headings[-1].group(2), re.IGNORECASE
    ):
        issues.append("slide cuối cần là kết luận")

    references_match = re.search(
        r"(?im)^#{1,3}\s*(?:nguồn|tài liệu tham khảo|references|sources)\b",
        value,
    )
    if not references_match:
        issues.append("cần mục tài liệu tham khảo")
    elif headings and references_match.start() < headings[-1].end():
        issues.append("mục tài liệu tham khảo phải nằm sau các slide")

    verified = {
        str(url).rstrip(".,;:")
        for url in (verified_source_urls or ())
    }
    all_urls = {
        url.rstrip(".,;:")
        for url in re.findall(r"https?://[^\s<>)\]}]+", value)
        if urlparse(url.rstrip(".,;:")).scheme in {"http", "https"}
        and urlparse(url.rstrip(".,;:")).netloc
    }
    if len(all_urls) < 3:
        issues.append(f"cần ít nhất 3 URL nguồn; hiện có {len(all_urls)}")
    if verified_source_urls is not None and not all_urls.issubset(verified):
        issues.append("có URL không thuộc kết quả tìm kiếm đã xác thực")

    slide_urls = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(value)
        section = value[heading.end():end]
        if references_match:
            section = section[:max(0, references_match.start() - heading.end())]
        if not re.search(
            r"(?i)gợi ý hình ảnh|hình ảnh đề xuất|image suggestion",
            section,
        ):
            issues.append(f"slide {numbers[index]} thiếu gợi ý hình ảnh")
        section_urls = {
            url.rstrip(".,;:")
            for url in re.findall(r"https?://[^\s<>)\]}]+", section)
            if urlparse(url.rstrip(".,;:")).scheme in {"http", "https"}
            and urlparse(url.rstrip(".,;:")).netloc
        }
        slide_urls.append(section_urls)

    for number, section_urls in zip(numbers, slide_urls):
        if number != 1 and not section_urls:
            issues.append(f"slide {number} thiếu trích dẫn nguồn trong nội dung slide")
        if verified_source_urls is not None and not section_urls.issubset(verified):
            issues.append(f"slide {number} trích dẫn URL chưa xác thực")
    return issues
