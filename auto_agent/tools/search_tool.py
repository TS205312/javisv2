"""Web search using DuckDuckGo with a Bing HTML fallback."""
import logging
import re
from html.parser import HTMLParser
from xml.etree import ElementTree

import requests
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


def search(params):
    query = params.get("query")
    if not isinstance(query, str) or not query.strip():
        logger.warning("Web search skipped: query is empty or invalid.")
        return []

    is_academic = any(term in query.casefold() for term in _ACADEMIC_TERMS)
    if is_academic:
        try:
            results = _arxiv_search(query.strip())
            if results:
                return results
            logger.warning("arXiv returned no results; trying general web search.")
        except (requests.RequestException, ElementTree.ParseError, ValueError) as exc:
            logger.warning(
                "arXiv search failed (%s); trying general web search.",
                type(exc).__name__,
            )

    try:
        results = list(DDGS().text(query.strip(), max_results=5))
        if results:
            return results
        logger.warning("DuckDuckGo returned no results; trying Bing fallback.")
    except Exception as exc:
        logger.warning(
            "DuckDuckGo search failed (%s); trying Bing fallback.",
            type(exc).__name__,
        )

    try:
        results = _bing_search(query.strip())
        if not results:
            logger.warning("Bing fallback returned no results for the query.")
        return results
    except requests.RequestException as exc:
        logger.warning("Bing fallback failed (%s).", type(exc).__name__)
    except (ValueError, TypeError) as exc:
        logger.warning("Bing response could not be parsed (%s).", type(exc).__name__)
    return []
