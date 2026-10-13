"""Resolve existing local skills or discover, install, and register GitHub skills."""
import importlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from github import Github, GithubException

from core.memory import memory as default_memory
from core.llm_helper import ask_llm, call_llm
from tools.write_safety import backup_existing_file, prepare_write_path
from registry import (
    get_all_skills,
    get_skill,
    register_skill,
    update_skill_meta,
)

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
ALLOWED_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-3",
    "BSD-3-Clause",
    "GPL-3",
    "GPL-3.0",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
}
BLOCKED_TOPICS = {"malware", "exploit", "miner"}
STOP_WORDS = {
    "a", "an", "and", "app", "bằng", "cho", "code", "của", "để", "hãy",
    "làm", "một", "the", "tạo", "to", "và", "viết", "xây",
}
logger = logging.getLogger(__name__)


def _log(message, *args):
    rendered = message % args if args else message
    print(f"[SKILL] {rendered}")
    logger.info(rendered)


def _tokens(text):
    return {
        token.casefold()
        for token in re.findall(r"[\w.+#-]{2,}", text, flags=re.UNICODE)
        if token.casefold() not in STOP_WORDS
    }


def _extract_keywords(goal):
    prompt = (
        "Trích xuất tối đa 5 từ khóa ngắn bằng tiếng Anh để tìm Python skill/repository "
        "phù hợp nhất với mục tiêu. Với 'code web' hoặc tạo website/API, ưu tiên "
        "'fastapi' và 'web framework'. Chỉ trả JSON {\"keywords\": [\"...\"]}.\n"
        f"Goal: {goal}"
    )
    try:
        result = ask_llm(prompt, json_mode=True)
        if isinstance(result, dict) and "error" in result:
            _log("Keyword LLM unavailable; extracting terms from the goal text.")
        values = result.get("keywords", []) if isinstance(result, dict) else []
        keywords = [str(value).strip().lower() for value in values if str(value).strip()]
    except Exception as exc:
        _log("Không trích được keyword bằng LLM; dùng keyword heuristic (%s).", exc)
        keywords = []

    goal_tokens = _tokens(goal)
    if goal_tokens & {"web", "website", "frontend", "backend", "api"}:
        keywords.extend(("fastapi", "web framework", "api"))
    if not keywords:
        keywords = sorted(goal_tokens)
    deduplicated = list(dict.fromkeys(keywords))[:8]
    _log("Goal keywords: %s", ", ".join(deduplicated) or "(none)")
    return deduplicated


def _updated_within(last_updated, days):
    if not last_updated:
        return False
    try:
        updated = datetime.fromisoformat(str(last_updated).replace("Z", "+00:00"))
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - updated.astimezone(timezone.utc)
    return timedelta(0) <= age < timedelta(days=days)


def quality_score(meta):
    stars = int(meta.get("stars", 0) or 0)
    recent = _updated_within(meta.get("last_updated"), 365)
    successes = int(meta.get("success_count", 0) or 0)
    pass_rate = float(meta.get("test_pass_rate", 0) or 0)
    parts = {
        "stars": 3 if stars >= 500 else 0,
        "updated<1y": 2 if recent else 0,
        "success>=3": 2 if successes >= 3 else 0,
        "test_pass>=0.8": 3 if pass_rate >= 0.8 else 0,
    }
    return sum(parts.values()), parts


class SkillResolver:
    """Three-stage skill lookup, quality assessment, and GitHub discovery."""

    def __init__(self, memory=None):
        self.memory = memory or default_memory
        self.last_keywords = []

    def find_local_skills(self, goal):
        """Find matching registered tools and relevant procedural-memory entries."""
        self.last_keywords = _extract_keywords(goal)
        goal_tokens = _tokens(" ".join(self.last_keywords) or goal)
        try:
            memory_matches = self.memory.search_similar(goal, n=5)
        except Exception as exc:
            _log("Procedural memory lookup failed: %s", exc)
            memory_matches = []

        found = []
        for name, skill in get_all_skills().items():
            searchable = " ".join((
                name,
                str(skill.get("desc", "")),
                str(skill.get("use_when", "")),
                str(skill.get("import_name", "")),
            ))
            skill_tokens = _tokens(searchable)
            overlaps = goal_tokens & skill_tokens
            memory_evidence = [
                str(item) for item in memory_matches
                if _tokens(str(item)) & (overlaps or goal_tokens)
            ]
            score = len(overlaps) / max(len(goal_tokens), 1)
            if memory_evidence:
                score = min(1.0, score + 0.15 * len(memory_evidence))
            if score > 0 or not found:
                found.append({
                    "skill": name,
                    "score": round(score, 3),
                    "desc": skill.get("desc", ""),
                    "meta": dict(skill.get("meta", {})),
                    "memory_matches": memory_evidence,
                })

        if not found:
            for name in ("coder_tool", "file_tool", "shell_tool"):
                skill = get_skill(name)
                found.append({
                    "skill": name,
                    "score": 0.0,
                    "desc": skill["desc"],
                    "meta": dict(skill["meta"]),
                    "memory_matches": [],
                })

        found.sort(key=lambda item: (-item["score"], item["skill"]))
        _log(
            "Local candidates: %s",
            ", ".join(f"{item['skill']}={item['score']:.3f}" for item in found[:10]),
        )
        return found

    def is_skill_good_enough(self, skill, goal):
        """Score a local skill using stars, recency, success, and test pass rate."""
        del goal
        if isinstance(skill, str):
            skill = {"skill": skill, "meta": (get_skill(skill) or {}).get("meta", {})}
        meta = skill.get("meta") or {}
        score, parts = quality_score(meta)
        _log(
            "Quality %s: stars %s (+%d), update<1y (+%d), success %s (+%d), "
            "test_pass %.2f (+%d) => %d/10; decision=%s",
            skill.get("skill", "unknown"),
            meta.get("stars", 0),
            parts["stars"],
            parts["updated<1y"],
            meta.get("success_count", 0),
            parts["success>=3"],
            float(meta.get("test_pass_rate", 0) or 0),
            parts["test_pass>=0.8"],
            score,
            "use" if score >= 7 else "hunt",
        )
        return score >= 7

    @staticmethod
    def _license_allowed(license_id):
        return license_id in ALLOWED_LICENSES

    @staticmethod
    def _repo_value(repo, name, default=None):
        if isinstance(repo, dict):
            return repo.get(name, default)
        return getattr(repo, name, default)

    @staticmethod
    def _valid_github_url(value):
        if not isinstance(value, str) or not value.strip():
            return False
        parsed = urlparse(value.strip())
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        return (
            parsed.scheme in {"http", "https"}
            and parsed.netloc.casefold() == "github.com"
            and len(parts) >= 2
            and all(part not in {".", ".."} for part in parts[:2])
        )

    def _github_candidates(self, keywords):
        search_term = next(
            (word for word in keywords if word in {"fastapi", "web framework"}),
            " ".join(keywords[:3]),
        )
        query = f'"{search_term}" language:python stars:>100'
        _log("GitHub query (sort=stars, order=desc): %s", query)
        github = Github(GITHUB_TOKEN or None, timeout=20)
        try:
            results = github.search_repositories(
                query=query,
                sort="stars",
                order="desc",
            )
        except GithubException as exc:
            logger.warning(
                "GitHub search failed with HTTP %s; skipping hunt.",
                getattr(exc, "status", "unknown"),
            )
            return query, []
        except (OSError, TimeoutError) as exc:
            logger.warning(
                "GitHub search network error (%s); skipping hunt.",
                type(exc).__name__,
            )
            return query, []

        if isinstance(results, dict):
            items = results.get("items")
            if not isinstance(items, list):
                logger.warning(
                    "GitHub search response has no valid items list; skipping hunt."
                )
                return query, []
            repositories = items[:10]
        else:
            try:
                repositories = list(results[:10])
            except (TypeError, KeyError, IndexError):
                logger.warning(
                    "GitHub search response is not a repository list; skipping hunt."
                )
                return query, []

        candidates = []
        for repo in repositories:
            try:
                license_info = self._repo_value(repo, "license")
                license_id = (
                    license_info.get("spdx_id")
                    if isinstance(license_info, dict)
                    else getattr(license_info, "spdx_id", None)
                )
                topics_method = getattr(repo, "get_topics", None)
                topics_value = topics_method() if callable(topics_method) else self._repo_value(repo, "topics", [])
                if not isinstance(topics_value, (list, tuple, set)):
                    raise TypeError("topics is not a list")
                topics = {str(topic).casefold() for topic in topics_value}
                updated_at = self._repo_value(repo, "updated_at")
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(
                        updated_at.replace("Z", "+00:00")
                    )
                if not isinstance(updated_at, datetime):
                    raise TypeError("updated_at is missing or invalid")
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)
                full_name = self._repo_value(repo, "full_name", "")
                name = self._repo_value(repo, "name", "")
                html_url = self._repo_value(repo, "html_url", "")
                clone_url = self._repo_value(repo, "clone_url", "")
                stars = self._repo_value(repo, "stargazers_count", 0)
                is_fork = self._repo_value(repo, "fork", True)
                if not self._valid_github_url(html_url) or not self._valid_github_url(clone_url):
                    logger.warning(
                        "Skipping GitHub repository with malformed URLs: %s",
                        full_name or "(unknown)",
                    )
                    continue
                accepted = (
                    self._license_allowed(license_id)
                    and timedelta(0) <= age < timedelta(days=730)
                    and not is_fork
                    and not (topics & BLOCKED_TOPICS)
                )
                candidates.append({
                    "full_name": full_name,
                    "name": name,
                    "description": self._repo_value(repo, "description", "") or "",
                    "url": html_url,
                    "clone_url": clone_url,
                    "stars": int(stars or 0),
                    "license": license_id,
                    "last_updated": updated_at.astimezone(timezone.utc).isoformat(),
                    "topics": sorted(topics),
                })
                _log(
                    "Candidate %s stars=%d license=%s updated=%s fork=%s blocked_topics=%s decision=%s",
                    full_name or "(unknown)",
                    int(stars or 0),
                    license_id or "unknown",
                    updated_at.date().isoformat(),
                    is_fork,
                    ",".join(sorted(topics & BLOCKED_TOPICS)) or "-",
                    "eligible" if accepted else "filtered",
                )
                if not accepted:
                    candidates.pop()
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed GitHub repository result (%s).",
                    type(exc).__name__,
                )

        candidates.sort(key=lambda item: item["stars"], reverse=True)
        return query, candidates

    def _evaluate_candidates(self, goal, keywords, candidates):
        top_candidates = candidates[:5]
        if not top_candidates:
            return []
        _log(
            "Evaluating top %d candidates: %s",
            len(top_candidates),
            ", ".join(
                f"{item['full_name']} ({item['stars']} stars)"
                for item in top_candidates
            ),
        )
        prompt = (
            "Evaluate which GitHub Python repositories are directly relevant to the "
            "user's goal. A request for web code or a web/API backend is relevant to "
            "FastAPI. Return JSON only: {\"relevant\": [\"owner/repo\", ...]}.\n"
            f"Goal: {goal}\nKeywords: {', '.join(keywords)}\nCandidates:\n"
            + json.dumps(top_candidates, ensure_ascii=False)
        )
        try:
            response = call_llm(prompt, json_mode=True)
            data = json.loads(response)
            relevant = data.get("relevant", [])
            names = {str(name).casefold() for name in relevant} if isinstance(relevant, list) else set()
            selected = [item for item in top_candidates if item["full_name"].casefold() in names]
            _log("LLM relevance: %s", ", ".join(names) or "(none)")
            return sorted(selected, key=lambda item: item["stars"], reverse=True)
        except Exception as exc:
            _log("LLM candidate evaluation failed; using keyword relevance fallback: %s", exc)
            goal_tokens = _tokens(" ".join(keywords) + " " + goal)
            selected = []
            for item in top_candidates:
                text = " ".join((
                    item["full_name"],
                    item["description"],
                    " ".join(item["topics"]),
                ))
                overlap = goal_tokens & _tokens(text)
                text_tokens = _tokens(text)
                if "fastapi" in keywords:
                    relevant = (
                        "fastapi" in text_tokens
                        or (
                            "web" in text_tokens
                            and bool(text_tokens & {"framework", "api"})
                        )
                    )
                else:
                    meaningful = overlap - {"python", "api", "code", "web"}
                    relevant = len(meaningful) >= 2 or bool(
                        meaningful and overlap & {"web", "api"}
                    )
                _log(
                    "Relevance %s score=%d decision=%s",
                    item["full_name"],
                    len(overlap),
                    "relevant" if relevant else "skip",
                )
                if relevant:
                    selected.append(item)
            return sorted(selected, key=lambda item: item["stars"], reverse=True)

    @staticmethod
    def _import_name(candidate):
        name = candidate["name"].replace("-", "_").replace(".", "_")
        if name.casefold() == "fastapi":
            return "fastapi"
        return name

    def _install_candidate(self, candidate):
        if not isinstance(candidate, dict):
            logger.warning("Skipping install: repository candidate is not an object.")
            return None
        repo_url = candidate.get("url")
        clone_url = candidate.get("clone_url")
        full_name = candidate.get("full_name")
        parts = full_name.split("/") if isinstance(full_name, str) else []
        if (
            len(parts) != 2
            or not all(parts)
            or not self._valid_github_url(repo_url)
            or not self._valid_github_url(clone_url)
        ):
            logger.warning("Skipping install: malformed GitHub repository URL or name.")
            return None

        from git import Repo

        project_root = Path(__file__).resolve().parents[1]
        skills_dir = project_root / ".skills"
        directory_name = "gh_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", full_name)
        checkout = prepare_write_path(skills_dir / directory_name)
        if checkout.parent != skills_dir.resolve():
            logger.warning("Skipping install: skill checkout escaped .skills directory.")
            return None
        skills_dir.mkdir(parents=True, exist_ok=True)
        _log("Installing %s from %s", full_name, repo_url)

        if checkout.exists():
            _log("Using existing isolated skill checkout: %s", directory_name)
        else:
            Repo.clone_from(
                candidate["clone_url"],
                checkout,
                depth=1,
                allow_unsafe_options=False,
            )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(checkout),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        import_name = self._import_name(candidate)
        importlib.import_module(import_name)
        _log("Install and import test passed: %s", import_name)
        return import_name

    def hunt_best(self, goal):
        """Search, filter, assess, install, and register the highest-star relevant repo."""
        keywords = self.last_keywords or _extract_keywords(goal)
        if not keywords:
            _log("Hunt aborted: no search keywords.")
            return None
        try:
            query, candidates = self._github_candidates(keywords)
            if not candidates:
                _log("Hunt found no repositories passing the license/update/fork/topic filters.")
                return None
            relevant = self._evaluate_candidates(goal, keywords, candidates)
            if not relevant:
                _log("Hunt found no relevant repository among the top 5 candidates.")
                return None

            best = max(relevant, key=lambda item: item["stars"])
            _log(
                "Selected highest-star relevant candidate: %s (%d stars)",
                best["full_name"],
                best["stars"],
            )
            import_name = self._install_candidate(best)
            if not import_name:
                _log("Install skipped because candidate URLs or metadata were invalid.")
                return None
            registry_name = f"github:{best['full_name']}"
            skill = {
                "desc": best["description"],
                "use_when": f"Use installed Python package {import_name} for: {goal}",
                "actions": ["use_as_dependency"],
                "import_name": import_name,
                "query": query,
                "meta": {
                    "source": "github",
                    "repo_url": best["url"],
                    "stars": best["stars"],
                    "license": best["license"],
                    "last_updated": best["last_updated"],
                    "success_count": 0,
                    "test_pass_rate": 1.0,
                    "test_count": 1,
                    "user_rating": None,
                },
            }
            backup_existing_file(
                    Path(__file__).resolve().parents[1] / "skill_registry.json"
            )
            register_skill(registry_name, skill)
            _log("Registered skill: %s", registry_name)
            return {
                "skill": registry_name,
                "score": 1.0,
                "desc": skill["desc"],
                "meta": dict(skill["meta"]),
                "import_name": import_name,
                "memory_matches": [],
            }
        except Exception as exc:
            logger.warning(
                "GitHub hunt/install failed (%s); falling back to local skills.",
                type(exc).__name__,
            )
            return None

    def resolve(self, goal):
        """Use an adequate local skill, otherwise hunt and finally fall back locally."""
        local = self.find_local_skills(goal)
        for skill in local:
            if self.is_skill_good_enough(skill, goal):
                _log("Decision: use local skill %s", skill["skill"])
                return skill

        discovered = self.hunt_best(goal)
        if discovered:
            _log("Decision: use discovered skill %s", discovered["skill"])
            return discovered

        fallback = local[0] if local else None
        if fallback:
            _log("Decision: hunt failed; fallback to local skill %s", fallback["skill"])
        return fallback

    @staticmethod
    def record_outcome(skill_names, passed, memory=None, logger=None):
        """Update quality metadata and record task outcomes in procedural memory."""
        names = list(dict.fromkeys(name for name in skill_names if name))
        memory_store = memory or default_memory
        for name in names:
            skill = get_skill(name)
            if skill is None:
                continue
            meta = skill.get("meta", {})
            old_count = int(meta.get("test_count", 0) or 0)
            old_rate = float(meta.get("test_pass_rate", 0) or 0)
            new_count = old_count + 1
            new_rate = ((old_rate * old_count) + (1 if passed else 0)) / new_count
            changes = {
                "test_count": new_count,
                "test_pass_rate": round(new_rate, 4),
            }
            if passed:
                changes["success_count"] = int(meta.get("success_count", 0) or 0) + 1
            update_skill_meta(name, changes)
            outcome = "passed" if passed else "failed"
            _log("Outcome %s: %s (success_count=%s, test_pass_rate=%.2f)",
                 name, outcome, changes.get("success_count", meta.get("success_count", 0)),
                 new_rate)
            memory_store.add_short_term(
                "skill_outcome",
                f"Skill {name} {outcome}; test_pass_rate={new_rate:.2f}",
            )
            memory_id = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
            memory_store.save_long_term(
                f"skill_{memory_id}_{time.time_ns()}",
                f"Skill {name} task outcome",
                f"{outcome}; test_pass_rate={new_rate:.2f}",
            )
            if logger:
                logger.log("skill_outcome", {
                    "skill": name,
                    "passed": passed,
                    "test_pass_rate": round(new_rate, 4),
                })

    @staticmethod
    def rate_skill(name, rating):
        if not 0 <= rating <= 5:
            raise ValueError("Rating must be between 0 and 5.")
        if get_skill(name) is None:
            raise KeyError(f"Không tìm thấy skill: {name}")
        update_skill_meta(name, {"user_rating": rating})
