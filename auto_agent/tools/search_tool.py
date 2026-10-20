"""Web search with relevance filtering and academic fallback providers."""
import logging
import json
import re
import unicodedata
from html.parser import HTMLParser
from xml.etree import ElementTree
from urllib.parse import urlparse

import requests
from core.sensitive_data import contains_sensitive_data, redact_text
from duckduckgo_search import DDGS


logger = logging.getLogger(__name__)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
_ACADEMIC_TERMS = (
    "paper", "papers", "research", "arxiv", "study", "studies",
    "article", "articles", "bài báo", "bài viết", "nghiên cứu",
)
_SEARCH_STOP_WORDS = {
    "a", "an", "and", "article", "articles", "artificial", "bai", "bao",
    "bài", "báo", "find", "for", "intelligence", "latest", "new", "of",
    "paper", "papers", "recent",
    "research", "study", "studies", "the", "three", "top", "tìm", "và",
    "về",
}
_EDUCATION_TERMS = {
    "education", "educational", "learning", "teacher", "teachers", "student",
    "students", "school", "schools", "classroom", "classrooms", "tutor",
    "tutoring", "pedagogy", "curriculum", "teaching", "learner", "learners",
    "giaoduc", "giao duc", "hoctap", "hoc tap", "giaovien", "giao vien",
    "hocsinh", "hoc sinh", "truonghoc", "truong hoc", "dayhoc", "day hoc",
    "lop hoc",
}
_EDUCATION_PRACTICE_TERMS = _EDUCATION_TERMS - {"education", "educational"}
_AI_TERMS = {
    "ai", "artificial intelligence", "generative ai", "machine learning",
    "large language model", "llm", "tri tue nhan tao",
}
_SEARCH_PROVIDERS = ("arxiv", "duckduckgo", "bing", "openalex")


class SearchResults(list):
    def __init__(self, results=(), status="ok", message="", provider=None, query=""):
        super().__init__(results)
        self.status = status
        self.message = message
        self.provider = provider
        self.query = query

    def __str__(self):
        if not self:
            return f"SEARCH_{self.status.upper()}: {self.message} (query={self.query!r})"
        return (
            f"SEARCH_{self.status.upper()} provider={self.provider} "
            f"query={self.query!r}\n"
            + json.dumps(list(self), ensure_ascii=False)
        )


def _normalize_query(query):
    query = " ".join(query.strip().split())
    normalized = _search_text(query)
    has_education_context = _contains_any_term(normalized, _EDUCATION_TERMS)
    has_ai_context = _contains_any_term(normalized, _AI_TERMS)
    if has_education_context and has_ai_context:
        core_terms = "AI education artificial intelligence in education AI in learning"
        return f"{query} {core_terms}"
    return query


def _search_text(value):
    normalized = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _contains_any_term(text, terms):
    return any(
        term in text
        if " " in term
        else re.search(r"\b{0}\b".format(re.escape(term)), text) is not None
        for term in terms
    )


def _standardize_results(results):
    standardized = []
    for result in results:
        if not isinstance(result, dict):
            continue
        title = result.get("title")
        url = result.get("url") or result.get("href")
        snippet = result.get("snippet") or result.get("body") or result.get("description")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        parsed_url = urlparse(url.strip())
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            continue
        source = result.get("source") or parsed_url.netloc
        normalized = {
            "title": title.strip()[:300],
            "url": url.strip(),
            "snippet": snippet.strip()[:2000],
            "source": str(source).strip()[:200] or parsed_url.netloc,
        }
        published = result.get("published")
        if isinstance(published, str) and published.strip():
            normalized["published"] = published.strip()[:100]
        standardized.append(normalized)
    return standardized


def _is_relevant_to_query(query, result):
    query_text = _search_text(query)
    result_text = _search_text(
        f"{result.get('title', '')} {result.get('snippet', '')}"
    )
    query_is_education = _contains_any_term(query_text, _EDUCATION_TERMS)
    query_terms = {
        token for token in re.findall(r"[a-z0-9]+", query_text)
        if len(token) > 2 and token not in _SEARCH_STOP_WORDS
    }
    result_terms = set(re.findall(r"[a-z0-9]+", result_text))
    if not query_is_education and query_terms and not query_terms & result_terms:
        return False
    if query_is_education:
        has_education_context = _contains_any_term(
            result_text,
            _EDUCATION_TERMS,
        )
        has_ai_context = _contains_any_term(result_text, _AI_TERMS)
        title_text = _search_text(result.get("title", ""))
        title_is_about_ai_education = re.search(
            r"\b(?:artificial intelligence|ai)(?:\s+in)?\s+education\b",
            title_text,
        ) is not None
        has_education_practice_context = _contains_any_term(
            title_text,
            _EDUCATION_PRACTICE_TERMS,
        )
        return (
            has_education_context
            and has_ai_context
            and (title_is_about_ai_education or has_education_practice_context)
        )
    return True


def _filter_relevant_results(query, results):
    standardized = _standardize_results(results)
    relevant = [
        result for result in standardized
        if _is_relevant_to_query(query, result)
    ]
    rejected_count = len(standardized) - len(relevant)
    if rejected_count:
        logger.warning(
            "Rejected %d search result(s) unrelated to query topic.",
            rejected_count,
        )
    return relevant


class _BingResultsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._in_result = False
        self._field = None
        self._current = {}
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "li" and "b_algo" in attributes.get("class", "").split():
            self._in_result = True
            self._current = {}
        elif self._in_result and tag == "h2":
            self._field = "title"
            self._buffer = []
        elif self._in_result and tag == "p":
            self._field = "snippet"
            self._buffer = []
        elif self._in_result and tag == "a" and self._field == "title":
            self._current["url"] = attributes.get("href", "")

    def handle_endtag(self, tag):
        if tag == "h2" and self._field == "title":
            self._current["title"] = "".join(self._buffer).strip()
            self._field = None
        elif tag == "p" and self._field == "snippet":
            self._current["snippet"] = "".join(self._buffer).strip()
            self._field = None
        elif tag == "li" and self._in_result:
            if self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
            self._in_result = False

    def handle_data(self, data):
        if self._field:
            self._buffer.append(data)


def _bing_search(query):
    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "count": 10},
        headers={"User-Agent": _USER_AGENT},
        timeout=15,
    )
    response.raise_for_status()
    parser = _BingResultsParser()
    parser.feed(response.text)
    return parser.results[:5]


def _arxiv_search(query):
    lowered_query = query.casefold()
    terms = [
        term for term in re.findall(r"[A-Za-z0-9]+", query.casefold())
        if term not in _SEARCH_STOP_WORDS
    ]
    has_ai_topic = (
        re.search(r"\bai\b", lowered_query) is not None
        or "artificial intelligence" in lowered_query
    )
    if has_ai_topic:
        search_query = "cat:cs.AI"
    else:
        search_query = " AND ".join(
            f"all:{term}" for term in terms[:5]
        ) or "all:AI"
    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": search_query,
            "start": 0,
            "max_results": 5,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", namespace):
        title = " ".join(
            entry.findtext("atom:title", default="", namespaces=namespace).split()
        )
        url = entry.findtext("atom:id", default="", namespaces=namespace).strip()
        summary = " ".join(
            entry.findtext("atom:summary", default="", namespaces=namespace).split()
        )
        published = entry.findtext(
            "atom:published", default="", namespaces=namespace
        ).strip()
        if title and url:
            results.append({
                "title": title,
                "url": url,
                "snippet": summary,
                "published": published,
                "source": "arXiv",
            })
    return results


def _openalex_search(query):
    query_text = _search_text(query)
    if (
        _contains_any_term(query_text, _EDUCATION_TERMS)
        and _contains_any_term(query_text, _AI_TERMS)
    ):
        search_query = "artificial intelligence education"
    else:
        search_query = query
    response = requests.get(
        "https://api.openalex.org/works",
        params={
            "search": search_query,
            "per-page": 10,
            "select": (
                "title,doi,publication_date,primary_location,"
                "abstract_inverted_index,is_retracted"
            ),
        },
        headers={"User-Agent": "auto-agent-research/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("OpenAlex returned an invalid response shape")

    results = []
    for item in payload["results"]:
        if not isinstance(item, dict) or item.get("is_retracted"):
            continue
        abstract_index = item.get("abstract_inverted_index") or {}
        words = sorted(
            (position, word)
            for word, positions in abstract_index.items()
            if isinstance(positions, list)
            for position in positions
            if isinstance(position, int)
        )
        snippet = " ".join(word for _, word in words)
        if len(snippet) > 1000:
            snippet = snippet[:1000].rsplit(" ", 1)[0]
        location = item.get("primary_location") or {}
        source_info = location.get("source") or {}
        source = source_info.get("display_name")
        doi = item.get("doi")
        url = doi or location.get("landing_page_url") or item.get("id")
        if not url or not item.get("title") or not snippet:
            continue
        results.append({
            "title": item["title"],
            "url": url,
            "snippet": snippet,
            "published": item.get("publication_date"),
            "source": source or urlparse(url).netloc,
        })
    return results


def search(params):
    if not isinstance(params, dict):
        return SearchResults(status="invalid_query", message="Search parameters must be an object.")
    query = params.get("query")
    if not isinstance(query, str) or not query.strip():
        logger.warning("Web search skipped: query is empty or invalid.")
        return SearchResults(status="invalid_query", message="Query rỗng hoặc không hợp lệ.")
    if contains_sensitive_data(query):
        return SearchResults(
            status="blocked_sensitive_query",
            message="Credential-like search text was blocked before network access.",
        )

    normalized_query = redact_text(_normalize_query(query))
    min_results = max(0, int(params.get("min_results", 0) or 0))
    logger.info("Normalized search query: %s", normalized_query)
    failures = []
    attempted = []
    collected = []
    collected_urls = set()
    openalex_attempted = False

    def try_provider(name, search_call):
        attempted.append(name)
        try:
            raw_results = search_call()
        except (requests.RequestException, ElementTree.ParseError, ValueError) as exc:
            failures.append(f"{name}: access_error ({type(exc).__name__})")
            logger.warning("%s search failed (%s).", name, type(exc).__name__)
            return None
        except Exception as exc:
            failures.append(f"{name}: access_error ({type(exc).__name__})")
            logger.warning("%s search failed (%s).", name, type(exc).__name__)
            return None

        raw_count = len(raw_results) if isinstance(raw_results, (list, tuple)) else 0
        standardized = _standardize_results(raw_results or [])
        relevant = [
            item for item in standardized
            if _is_relevant_to_query(normalized_query, item)
        ]
        if relevant:
            for item in relevant:
                if item["url"] not in collected_urls:
                    collected.append(item)
                    collected_urls.add(item["url"])
            status = "ok" if len(collected) >= min_results else "insufficient_results"
            return SearchResults(
                collected, status=status,
                message=(f"Chỉ tìm được {len(collected)}/{min_results} nguồn đạt yêu cầu."
                         if status != "ok" else ""),
                provider=name, query=normalized_query,
            )
        status = (
            "irrelevant_results" if standardized
            else "insufficient_info" if raw_count
            else "no_results"
        )
        failures.append(f"{name}: {status}")
        logger.warning("%s search returned %s.", name, status)
        return None

    is_academic = any(term in normalized_query.casefold() for term in _ACADEMIC_TERMS)
    query_text = _search_text(normalized_query)
    is_ai_education = (
        _contains_any_term(query_text, _EDUCATION_TERMS)
        and _contains_any_term(query_text, _AI_TERMS)
    )
    if is_ai_education:
        openalex_attempted = True
        results = try_provider(
            "openalex",
            lambda: _openalex_search(normalized_query),
        )
        if results is not None and results.status == "ok":
            return results
        if results is not None:
            failures.append(f"openalex: {results.status}")

    if is_academic:
        results = try_provider("arxiv", lambda: _arxiv_search(normalized_query))
        if results is not None and results.status == "ok":
            return results
        if results is not None:
            failures.append(f"arxiv: {results.status}")

    results = try_provider(
        "duckduckgo",
        lambda: list(DDGS().text(normalized_query, max_results=5)),
    )
    if results is not None and results.status == "ok":
        return results
    if results is not None:
        failures.append(f"duckduckgo: {results.status}")

    results = try_provider("bing", lambda: _bing_search(normalized_query))
    if results is not None and results.status == "ok":
        return results
    if results is not None:
        failures.append(f"bing: {results.status}")

    if not openalex_attempted:
        results = try_provider(
            "openalex",
            lambda: _openalex_search(normalized_query),
        )
        if results is not None and results.status == "ok":
            return results
        if results is not None:
            failures.append(f"openalex: {results.status}")

    if any("irrelevant_results" in failure for failure in failures):
        status = "irrelevant_results"
        message = "Có kết quả nhưng không đủ dấu hiệu liên quan/chất lượng để dùng."
    elif any("insufficient_results" in failure for failure in failures):
        status = "insufficient_results"
        message = "Có một số nguồn phù hợp nhưng chưa đạt số lượng tối thiểu."
    elif any("insufficient_info" in failure for failure in failures):
        status = "insufficient_info"
        message = "Có phản hồi nhưng thiếu tiêu đề, URL hoặc đoạn tóm tắt để đánh giá."
    elif failures and all("access_error" in failure for failure in failures):
        status = "access_error"
        message = "Không truy cập được các nguồn tìm kiếm dự phòng."
    else:
        status = "no_results"
        message = "Các nguồn tìm kiếm không trả về kết quả phù hợp."
    if collected:
        status = "insufficient_results"
        message = f"Có {len(collected)} nguồn phù hợp, dưới mức tối thiểu {min_results}."
    return SearchResults(
        collected,
        status=status,
        message=f"{message} Providers: {'; '.join(failures)}",
        provider=",".join(attempted),
        query=normalized_query,
    )
