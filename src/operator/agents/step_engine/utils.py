from __future__ import annotations

from dataclasses import dataclass, field
import ast
import base64
import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from pathlib import Path
from datetime import datetime, timezone

from pydantic import BaseModel, Field

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


FSM_MODES = {
    "BOOTSTRAP",
    "POPUP",
    "PLAN",
    "NAV",
    "EXTRACT",
    "SYNTH",
    "REPORT",
    "STUCK",
    "DONE",
}

OBS_META_TOOLS = {
    "META.EXTRACT_LINKS",
    "META.EXTRACT_FACTS",
    "META.SEARCH_TEXT",
    "META.FIND_ELEMENTS",
    "META.VISION_QA",
}

CONTROL_META_TOOLS = {
    "META.SOLVE_POPUPS",
    "META.REPLAN",
    "META.SELECT_NEXT_TARGET",
    "META.ESCALATE",
    "META.SET_MODE",
    "META.MARK_PROGRESS",
}

META_TOOLS = OBS_META_TOOLS | CONTROL_META_TOOLS

SUPPORTED_BROWSER_TOOL_NAMES = {
    "browser.search",
    "browser.navigate",
    "browser.go_back",
    "browser.click",
    "browser.dblclick",
    "browser.rightclick",
    "browser.middleclick",
    "browser.tripleclick",
    "browser.input",
    "browser.scroll",
    "browser.wait",
    "browser.done",
    "browser.select_dropdown",
    "browser.dropdown_options",
    "browser.hover",
    "browser.screenshot",
    "browser.send_keys",
    "browser.hold_key",
    "browser.extract",
}

UNAVAILABLE_BROWSER_TOOLS = (
    "switch_tab",
    "close_tab",
    "upload_file",
    "read_file",
    "write_file",
    "replace_file",
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
SITE_KNOWLEDGE_TASK_CACHE = _REPO_ROOT / "data" / "task_cache" / "tasks_5_projects.json"
SITE_KNOWLEDGE_STATIC_MAP_PATH = _REPO_ROOT / "data" / "site_maps.json"

MAX_INTERNAL_META_STEPS = 3
MAX_FACTS = 32
MAX_CHECKPOINTS = 20
MAX_PENDING_URLS = 30
MAX_PENDING_ELEMENTS = 50
MAX_VISITED_URLS = 80
MAX_PAGE_HASHES = 80
MAX_STR = 400
MAX_HISTORY_SUMMARY_CHARS = 2400
HISTORY_RECENT_LIMIT = 12
MAX_VISUAL_NOTES = 12
MAX_VISUAL_HINTS = 20

_TASK_TERM_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "then",
    "open",
    "click",
    "visit",
    "please",
    "show",
    "website",
    "page",
    "current",
    "value",
    "finish",
    "first",
    "your",
    "using",
    "account",
    "successfully",
    "continue",
    "task",
    "next",
    "value",
    "total",
    "price",
    "count",
    "number",
    "amount",
    "many",
    "much",
    "tell",
    "show",
    "find",
    "what",
    "when",
    "which",
    "balance",
}


def _utc_now() -> str:
    try:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _norm_ws(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _merge_usage_dicts(base: Dict[str, Any] | None, extra: Dict[str, Any] | None) -> Dict[str, int]:
    out = {
        "prompt_tokens": int(((base or {}).get("prompt_tokens") or 0)),
        "completion_tokens": int(((base or {}).get("completion_tokens") or 0)),
        "total_tokens": int(((base or {}).get("total_tokens") or 0)),
    }
    out["prompt_tokens"] += int(((extra or {}).get("prompt_tokens") or 0))
    out["completion_tokens"] += int(((extra or {}).get("completion_tokens") or 0))
    out["total_tokens"] += int(((extra or {}).get("total_tokens") or 0))
    return out


def _empty_usage_dict() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _empty_call_breakdown() -> Dict[str, int]:
    return {
        "policy_llm_calls": 0,
        "obs_extract_llm_calls": 0,
        "vision_llm_calls": 0,
    }


def _usage_breakdown_template() -> Dict[str, Dict[str, int]]:
    return {
        "policy": _empty_usage_dict(),
        "obs_extract": _empty_usage_dict(),
        "vision": _empty_usage_dict(),
    }


def _dom_digest(html: str) -> str:
    try:
        data = str(html or "").encode("utf-8", errors="ignore")
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return ""


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", str(text or "").lower())}


def _focus_terms(text: str, *, max_terms: int = 18) -> set[str]:
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", str(text or "").lower()) if t not in _TASK_TERM_STOPWORDS]
    freq: Dict[str, int] = {}
    for token in tokens:
        freq[token] = int(freq.get(token) or 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (item[1], len(item[0]), item[0]), reverse=True)
    return {token for token, _ in ranked[: max(1, int(max_terms))]}


def _looks_like_informational_task(prompt: str) -> bool:
    text = _norm_ws(prompt).lower()
    if not text:
        return False
    info_patterns = [
        r"\bwhat(?:'s| is)\b",
        r"\btell me\b",
        r"\bhow much\b",
        r"\bhow many\b",
        r"\bvalue\b",
        r"\bprice\b",
        r"\btotal\b",
        r"\bbalance\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\bwhich\b",
        r"\bwhen\b",
        r"\bfind\b",
        r"\bshow me\b",
    ]
    return any(re.search(pattern, text) for pattern in info_patterns)


def _valueish_text(text: str) -> bool:
    value = _norm_ws(text)
    if not value:
        return False
    if len(value) > 96:
        return False
    if len(re.findall(r"\d", value)) >= 1:
        return True
    if re.search(r"\b(?:yes|no|active|inactive|online|offline|open|closed)\b", value.lower()):
        return True
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", value.lower()):
        return True
    return False


def _labelish_text(text: str) -> bool:
    label = _norm_ws(text)
    if not label:
        return False
    if len(label) > 64:
        return False
    if len(label.split()) > 8:
        return False
    if len(re.findall(r"\d", label)) > 0:
        return False
    if re.search(r"^[^a-zA-Z]*$", label):
        return False
    return True


def _value_line_text(text: str) -> bool:
    line = _norm_ws(text)
    if not line:
        return False
    if len(line) > 140:
        return False
    if len(re.findall(r"\d", line)) >= 1:
        return True
    if re.search(r"\b(?:yes|no|active|inactive|online|offline|open|closed)\b", line.lower()):
        return True
    return False


def _is_generic_tool_placeholder(value: Any, *, kind: str) -> bool:
    text = _norm_ws(str(value or "")).strip().lower()
    if not text:
        return True
    generic_common = {
        "text",
        "value",
        "input",
        "field",
        "enter text",
        "type here",
        "your text",
        "<text>",
        "<value>",
        "<input>",
    }
    generic_select = {
        "option",
        "select option",
        "choose option",
        "<option>",
        "dropdown option",
    }
    valid_placeholders = {"<username>", "<password>", "<signup_email>", "<email>"}
    if text in valid_placeholders:
        return False
    if kind == "select":
        return text in (generic_common | generic_select)
    return text in generic_common


def _normalize_reasoning_trace(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    allowed = (
        "task_interpretation",
        "success_state",
        "current_subgoal",
        "next_expected_proof",
        "drift_risks",
        "where_am_i",
        "state_assessment",
        "plan",
    )
    out: Dict[str, str] = {}
    for key in allowed:
        text = _norm_ws(str(raw.get(key) or ""))
        if text:
            out[key] = text[:280]
    return out


def _normalize_working_state(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in (
        "current_page_kind",
        "active_region",
        "active_workflow",
        "next_milestone",
        "completion_state",
    ):
        text = _candidate_text(raw.get(key))
        if text:
            out[key] = text[:160]
    for key in (
        "completed_fields",
        "pending_fields",
        "completion_evidence_missing",
    ):
        values: List[str] = []
        for item in list(raw.get(key) or []):
            text = _candidate_text(item)
            if text:
                values.append(text[:120])
        if values:
            out[key] = _dedupe_keep_order(values, 8)
    return out


def _reasoning_trace_summary(trace: Dict[str, str]) -> str:
    if not isinstance(trace, dict) or not trace:
        return ""
    parts: List[str] = []
    if trace.get("task_interpretation"):
        parts.append(f"Task means: {trace['task_interpretation']}")
    if trace.get("success_state"):
        parts.append(f"Success: {trace['success_state']}")
    if trace.get("current_subgoal"):
        parts.append(f"Now: {trace['current_subgoal']}")
    if trace.get("next_expected_proof"):
        parts.append(f"Next proof: {trace['next_expected_proof']}")
    if trace.get("drift_risks"):
        parts.append(f"Avoid: {trace['drift_risks']}")
    if trace.get("where_am_i"):
        parts.append(f"Where: {trace['where_am_i']}")
    if trace.get("state_assessment"):
        parts.append(f"State: {trace['state_assessment']}")
    if trace.get("plan"):
        parts.append(f"Plan: {trace['plan']}")
    return " | ".join(parts)[:900]


def _working_state_summary(ws: Dict[str, Any]) -> str:
    if not isinstance(ws, dict) or not ws:
        return ""
    parts: List[str] = []
    if ws.get("current_page_kind"):
        parts.append(f"Page: {ws['current_page_kind']}")
    if ws.get("active_region"):
        parts.append(f"Region: {ws['active_region']}")
    if ws.get("active_workflow"):
        parts.append(f"Workflow: {ws['active_workflow']}")
    completed = ws.get("completed_fields") if isinstance(ws.get("completed_fields"), list) else []
    if completed:
        parts.append("Done: " + ", ".join(str(x) for x in completed[:4]))
    pending = ws.get("pending_fields") if isinstance(ws.get("pending_fields"), list) else []
    if pending:
        parts.append("Pending: " + ", ".join(str(x) for x in pending[:4]))
    missing = ws.get("completion_evidence_missing") if isinstance(ws.get("completion_evidence_missing"), list) else []
    if missing:
        parts.append("Missing proof: " + ", ".join(str(x) for x in missing[:3]))
    if ws.get("next_milestone"):
        parts.append(f"Next: {ws['next_milestone']}")
    if ws.get("completion_state"):
        parts.append(f"Status: {ws['completion_state']}")
    return " | ".join(parts)[:900]


def _normalize_use_case_info(raw: Any) -> Dict[str, str]:
    if isinstance(raw, dict):
        return {
            "name": _candidate_text(raw.get("name"))[:80],
            "description": _candidate_text(raw.get("description"))[:400],
        }
    name = _candidate_text(getattr(raw, "name", None), raw)
    description = _candidate_text(getattr(raw, "description", None))
    out = {"name": name[:80], "description": description[:400]}
    return {k: v for k, v in out.items() if v}


def _site_section_templates() -> Dict[str, Dict[str, str]]:
    return {
        "home": {
            "label": "home / landing",
            "when_useful": "starting navigation, broad discovery, finding main flows",
            "unlikely_for": "completing deep item-specific workflows by itself",
        },
        "auth": {
            "label": "auth / account access",
            "when_useful": "login, registration, logout, account access",
            "unlikely_for": "item detail tasks unless authentication is explicitly required",
        },
        "catalog": {
            "label": "search / browse / filters",
            "when_useful": "searching, filtering, browsing lists before opening an item",
            "unlikely_for": "submitting item-specific forms once the correct item is already open",
        },
        "detail": {
            "label": "item detail pages",
            "when_useful": "details, comments, share, trailer, item-specific actions",
            "unlikely_for": "global auth and generic site info",
        },
        "form": {
            "label": "mutation forms",
            "when_useful": "create, edit, delete, submit, contact, checkout-like flows",
            "unlikely_for": "broad exploration after the correct form is already visible",
        },
        "account": {
            "label": "profile / saved items",
            "when_useful": "profile edits, watchlists, wishlists, user-specific actions",
            "unlikely_for": "anonymous discovery tasks",
        },
        "info": {
            "label": "informational pages",
            "when_useful": "about, contact, static information, policies",
            "unlikely_for": "most transactional or item-specific tasks",
        },
    }


def _section_keys_for_use_case(name: str, description: str) -> List[str]:
    text = f"{name} {description}".lower()
    keys = ["home"]
    if any(tok in text for tok in ("login", "sign in", "sign up", "register", "logout", "auth")):
        keys.append("auth")
    if any(tok in text for tok in ("search", "filter", "browse", "find", "list")):
        keys.append("catalog")
    if any(tok in text for tok in ("detail", "movie", "book", "product", "profile", "comment", "review", "share", "trailer", "view")):
        keys.append("detail")
    if any(tok in text for tok in ("add", "create", "edit", "delete", "contact", "submit", "form", "message")):
        keys.append("form")
    if any(tok in text for tok in ("watchlist", "wishlist", "profile", "account", "saved")):
        keys.append("account")
    if any(tok in text for tok in ("about", "contact", "policy", "info", "help", "support")):
        keys.append("info")
    return _dedupe_keep_order(keys, 6)


@lru_cache(maxsize=1)
def _load_task_cache_site_index() -> Dict[str, Dict[str, Any]]:
    path = SITE_KNOWLEDGE_TASK_CACHE
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    raw_tasks = data.get("tasks") if isinstance(data, dict) else data
    out: Dict[str, Dict[str, Any]] = {}
    for item in raw_tasks if isinstance(raw_tasks, list) else []:
        if not isinstance(item, dict):
            continue
        project_id = _candidate_text(item.get("web_project_id"))
        if not project_id:
            continue
        project = out.setdefault(project_id, {"use_cases": {}, "examples": []})
        uc = _normalize_use_case_info(item.get("use_case"))
        uc_name = uc.get("name") or ""
        if uc_name and uc_name not in project["use_cases"]:
            project["use_cases"][uc_name] = uc
        prompt = _candidate_text(item.get("prompt"))
        if prompt:
            project["examples"] = _dedupe_keep_order(list(project["examples"]) + [prompt[:180]], 12)
    return out


@lru_cache(maxsize=1)
def _load_static_site_maps() -> Dict[str, Dict[str, Any]]:
    path = SITE_KNOWLEDGE_STATIC_MAP_PATH
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    projects = payload.get("projects") if isinstance(payload.get("projects"), dict) else payload
    out: Dict[str, Dict[str, Any]] = {}
    for project_id, project_payload in projects.items() if isinstance(projects, dict) else []:
        pid = _candidate_text(project_id)
        if not pid or not isinstance(project_payload, dict):
            continue
        out[pid] = dict(project_payload)
    return out


def _section_key_for_path(path: str) -> str:
    clean = str(path or "").strip().lower() or "/"
    if clean in {"", "/"}:
        return "home"
    if any(tok in clean for tok in ("/login", "/register", "/signup", "/signin", "/auth")):
        return "auth"
    if any(tok in clean for tok in ("/search", "/browse", "/catalog", "/books", "/movies", "/restaurants")):
        return "catalog"
    if any(tok in clean for tok in ("/detail", "/movie/", "/book/", "/restaurant/", "/item/", "/film/")):
        return "detail"
    if any(tok in clean for tok in ("/contact", "/create", "/add", "/edit", "/delete", "/checkout", "/reserve", "/booking")):
        return "form"
    if any(tok in clean for tok in ("/profile", "/account", "/watchlist", "/wishlist", "/saved", "/menu", "/calendar")):
        return "account"
    if any(tok in clean for tok in ("/about", "/help", "/support", "/faq", "/policy")):
        return "info"
    return "home"


def _normalize_route_entry(route: Dict[str, Any], *, base_url: str = "") -> Dict[str, Any] | None:
    if not isinstance(route, dict):
        return None
    raw_href = _candidate_text(route.get("href"), route.get("url"), route.get("path"))
    if not raw_href:
        return None
    safe_href = _safe_url(raw_href, base=base_url)
    parsed = urlsplit(str(safe_href or raw_href))
    path = str(parsed.path or "/")
    label = _candidate_text(route.get("label"), route.get("text"), path)
    section_id = _candidate_text(route.get("section_id")) or _section_key_for_path(path)
    return {
        "label": label[:120],
        "href": str(safe_href or raw_href)[:300],
        "path": path[:180],
        "section_id": section_id,
        "source": _candidate_text(route.get("source"), "static"),
    }


def _discover_page_routes(*, snapshot_html: str, current_url: str, candidates: List[Any]) -> List[Dict[str, Any]]:
    discovered: List[Dict[str, Any]] = []
    if snapshot_html and BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(snapshot_html, "lxml")
        except Exception:
            soup = None
        if soup is not None:
            try:
                for anchor in soup.find_all("a", href=True, limit=80):
                    href = _candidate_text(anchor.get("href"))
                    if not href or href.startswith(("javascript:", "mailto:", "tel:")):
                        continue
                    label = _candidate_text(anchor.get_text(" ", strip=True), anchor.get("aria-label"), href)
                    discovered.append(
                        {
                            "href": href,
                            "label": label,
                            "source": "page_anchor",
                        }
                    )
            except Exception:
                pass
    for cand in candidates:
        href = _candidate_text(getattr(cand, "href", ""))
        if not href:
            continue
        discovered.append(
            {
                "href": href,
                "label": _candidate_text(getattr(cand, "text", ""), getattr(cand, "context", ""), href),
                "source": "candidate_href",
            }
        )
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in discovered:
        route = _normalize_route_entry(item, base_url=current_url)
        if not isinstance(route, dict):
            continue
        dedupe_key = f"{route.get('path')}::{route.get('label')}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(route)
        if len(normalized) >= 20:
            break
    return normalized


def _canonical_crawl_path(url: str) -> str:
    parsed = urlsplit(str(url or ""))
    path = str(parsed.path or "/")
    return path or "/"


@lru_cache(maxsize=32)
def _crawl_site_routes(start_url: str, depth: int, max_pages: int, timeout_s: float) -> tuple[tuple[str, str, str, str], ...]:
    if BeautifulSoup is None:
        return ()
    start = str(start_url or "").strip()
    parsed_start = urlsplit(start)
    if not (parsed_start.scheme and parsed_start.netloc):
        return ()
    same_origin = f"{parsed_start.scheme}://{parsed_start.netloc}"
    queue: List[tuple[str, int]] = [(start, 0)]
    root = _root_url(start)
    if root and root != start:
        queue.append((root, 0))
    fetched_paths: set[str] = set()
    emitted: List[tuple[str, str, str, str]] = []
    emitted_paths: set[str] = set()
    while queue and len(fetched_paths) < max(1, int(max_pages)):
        page_url, current_depth = queue.pop(0)
        page_path = _canonical_crawl_path(page_url)
        if page_path in fetched_paths:
            continue
        fetched_paths.add(page_path)
        try:
            req = Request(page_url, headers={"User-Agent": "autoppia-operator-site-crawler/1.0"})
            with urlopen(req, timeout=max(0.5, float(timeout_s))) as resp:
                content_type = str(resp.headers.get("Content-Type") or "").lower()
                if "html" not in content_type:
                    continue
                raw_html = resp.read()
        except Exception:
            continue
        try:
            html = raw_html.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            continue
        try:
            anchors = soup.find_all("a", href=True, limit=120)
        except Exception:
            anchors = []
        for anchor in anchors:
            href = _candidate_text(anchor.get("href"))
            if not href or href.startswith(("javascript:", "mailto:", "tel:")):
                continue
            absolute = _safe_url(href, base=page_url)
            parsed = urlsplit(str(absolute or ""))
            if f"{parsed.scheme}://{parsed.netloc}" != same_origin:
                continue
            route_path = _canonical_crawl_path(absolute)
            if route_path not in emitted_paths:
                emitted_paths.add(route_path)
                emitted.append(
                    (
                        _candidate_text(anchor.get_text(" ", strip=True), anchor.get("aria-label"), route_path)[:120],
                        str(absolute or href)[:300],
                        route_path[:180],
                        _section_key_for_path(route_path),
                    )
                )
            if current_depth + 1 <= max(0, int(depth)) and route_path not in fetched_paths:
                queue.append((str(absolute), current_depth + 1))
    return tuple(emitted[:40])


def _merge_site_routes(static_routes: List[Dict[str, Any]], discovered_routes: List[Dict[str, Any]], *, base_url: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(static_routes or []) + list(discovered_routes or []):
        route = _normalize_route_entry(item, base_url=base_url)
        if not isinstance(route, dict):
            continue
        dedupe_key = str(route.get("path") or route.get("href") or "")
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(route)
        if len(out) >= 18:
            break
    return out


def _build_site_knowledge(
    project_id: str,
    use_case: Dict[str, str],
    prompt: str,
    *,
    current_url: str = "",
    snapshot_html: str = "",
    candidates: List[Any] | None = None,
) -> Dict[str, Any]:
    project_id = _candidate_text(project_id)
    uc = _normalize_use_case_info(use_case)
    cache_index = _load_task_cache_site_index()
    static_index = _load_static_site_maps()
    cached = cache_index.get(project_id, {}) if project_id else {}
    static_project = static_index.get(project_id, {}) if project_id else {}
    known_use_cases = list((cached.get("use_cases") or {}).values()) if isinstance(cached, dict) else []
    all_use_cases = list(known_use_cases)
    if uc.get("name") and not any(str(item.get("name") or "") == uc["name"] for item in all_use_cases if isinstance(item, dict)):
        all_use_cases.append(uc)
    templates = _site_section_templates()
    section_sources: Dict[str, List[str]] = {}
    for item in all_use_cases:
        if not isinstance(item, dict):
            continue
        name = _candidate_text(item.get("name"))
        desc = _candidate_text(item.get("description"))
        for key in _section_keys_for_use_case(name, desc):
            section_sources.setdefault(key, []).append(name or desc or "unknown")
    known_sections: List[Dict[str, Any]] = []
    for key in ("home", "auth", "catalog", "detail", "form", "account", "info"):
        if key not in section_sources:
            continue
        tpl = templates[key]
        known_sections.append(
            {
                "section_id": key,
                "label": tpl["label"],
                "when_useful": tpl["when_useful"],
                "unlikely_for": tpl["unlikely_for"],
                "supported_by_use_cases": _dedupe_keep_order(section_sources.get(key, []), 6),
            }
        )
    current_name = _candidate_text(uc.get("name"))
    current_desc = _candidate_text(uc.get("description"), prompt)
    current_keys = _section_keys_for_use_case(current_name, current_desc)
    best_key = next((key for key in current_keys if key != "home"), "home")
    unlikely_keys = [key for key in ("info", "auth", "home") if key != best_key and key in section_sources]
    static_routes = list(static_project.get("routes") or []) if isinstance(static_project, dict) else []
    discovered_routes = _discover_page_routes(
        snapshot_html=str(snapshot_html or ""),
        current_url=str(current_url or ""),
        candidates=list(candidates or []),
    )
    crawled_routes: List[Dict[str, Any]] = []
    if (not static_routes) and _env_bool("FSM_ENABLE_SITE_CRAWLER", True):
        crawl_depth = max(0, min(_env_int("FSM_SITE_CRAWL_DEPTH", 3), 3))
        crawl_max_pages = max(1, min(_env_int("FSM_SITE_CRAWL_MAX_PAGES", 12), 24))
        crawl_timeout_s = max(0.5, min(float(os.getenv("FSM_SITE_CRAWL_TIMEOUT_S", "1.5") or 1.5), 4.0))
        try:
            crawled_routes = [
                {
                    "label": label,
                    "href": href,
                    "path": path,
                    "section_id": section_id,
                    "source": "crawler",
                }
                for (label, href, path, section_id) in _crawl_site_routes(
                    str(current_url or ""),
                    int(crawl_depth),
                    int(crawl_max_pages),
                    float(crawl_timeout_s),
                )
            ]
        except Exception:
            crawled_routes = []
    merged_routes = _merge_site_routes(static_routes, list(discovered_routes) + list(crawled_routes), base_url=str(current_url or ""))
    route_sections: Dict[str, List[str]] = {}
    for route in merged_routes:
        if not isinstance(route, dict):
            continue
        key = _candidate_text(route.get("section_id"))
        label = _candidate_text(route.get("label"), route.get("path"))
        if key and label:
            route_sections.setdefault(key, []).append(label)
    return {
        "project_id": project_id,
        "available": bool(known_sections or merged_routes),
        "known_sections": known_sections[:6],
        "current_use_case": current_name,
        "current_task_routing": {
            "likely_best_section": best_key,
            "section_label": templates.get(best_key, {}).get("label", best_key),
            "why": templates.get(best_key, {}).get("when_useful", ""),
            "unlikely_sections_for_this_task": unlikely_keys[:3],
        },
        "routes": merged_routes[:12],
        "discovered_routes": discovered_routes[:8],
        "crawled_routes": crawled_routes[:8],
        "route_sections": {k: _dedupe_keep_order(v, 5) for k, v in route_sections.items()},
        "site_source": {
            "static_map": bool(static_routes),
            "page_discovery": bool(discovered_routes),
            "crawler": bool(crawled_routes),
        },
        "example_task_prompts": list((cached.get("examples") or [])[:5]) if isinstance(cached, dict) else [],
    }


def _fact_overlap_score(prompt: str, fact: str) -> int:
    p_terms = _focus_terms(prompt, max_terms=16)
    if not p_terms:
        return 0
    f_terms = _tokenize(fact)
    return len(p_terms.intersection(f_terms))


def _anchor_overlap_score(prompt: str, fact: str) -> int:
    anchors = _focus_terms(prompt, max_terms=12)
    if not anchors:
        return 0
    return len(anchors.intersection(_tokenize(fact)))


def _digit_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", str(text or "")))


def _best_page_evidence(prompt: str, text_ir: Dict[str, Any]) -> str:
    text_ir = text_ir if isinstance(text_ir, dict) else {}
    page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
    value_lines = text_ir.get("value_lines") if isinstance(text_ir.get("value_lines"), list) else []
    relevant_lines = text_ir.get("relevant_lines") if isinstance(text_ir.get("relevant_lines"), list) else []
    candidates = _dedupe_keep_order(
        [str(x) for x in page_facts[:12] + relevant_lines[:12] + value_lines[:16] if _norm_ws(x)],
        24,
    )
    if not candidates:
        return ""
    ranked: List[tuple[int, str]] = []
    for fact in candidates:
        clean = _norm_ws(fact)
        if not clean:
            continue
        raw_overlap = _fact_overlap_score(prompt, clean)
        anchor_overlap = _anchor_overlap_score(prompt, clean)
        score = raw_overlap + (anchor_overlap * 2)
        if _valueish_text(clean):
            score += 1
        if ":" in clean or "-" in clean:
            score += 1
        if clean in page_facts:
            score += 2
        if clean in relevant_lines:
            score += 1
        ranked.append((score, f"{raw_overlap}:{anchor_overlap}::{clean}"))
    ranked.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    if not ranked:
        return ""
    best_score, tagged_fact = ranked[0]
    overlap_blob, _, best_fact = tagged_fact.partition("::")
    try:
        raw_overlap_str, anchor_overlap_str = overlap_blob.split(":", 1)
        raw_overlap = int(raw_overlap_str)
        anchor_overlap = int(anchor_overlap_str)
    except Exception:
        raw_overlap = 0
        anchor_overlap = 0
    if raw_overlap <= 0 and anchor_overlap <= 0:
        return ""
    return best_fact[:220]


def _content_supported_by_page_evidence(prompt: str, content: str, text_ir: Dict[str, Any]) -> bool:
    answer = _norm_ws(content)
    if not answer:
        return False
    page_facts = text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else []
    likely = []
    if page_facts:
        likely = sorted(
            [_norm_ws(fact) for fact in page_facts if _norm_ws(fact)],
            key=lambda fact: (_fact_overlap_score(prompt, fact), len(fact)),
            reverse=True,
        )[:8]
    if not likely:
        return False
    answer_terms = _tokenize(answer)
    answer_digits = _digit_tokens(answer)
    for fact in likely:
        if answer.lower() in fact.lower() or fact.lower() in answer.lower():
            if _fact_overlap_score(prompt, fact) >= 1:
                return True
        fact_terms = _tokenize(fact)
        fact_digits = _digit_tokens(fact)
        if answer_digits and fact_digits and answer_digits.intersection(fact_digits):
            if answer_terms.intersection(fact_terms) and _fact_overlap_score(prompt, fact) >= 1:
                return True
        if _fact_overlap_score(answer, fact) >= 2:
            if _fact_overlap_score(prompt, fact) >= 1:
                return True
    return False


def _looks_like_vague_informational_answer(text: str) -> bool:
    answer = _norm_ws(text).lower()
    if not answer:
        return True
    vague_patterns = [
        r"\blikely\b",
        r"\bprobably\b",
        r"\bappears to\b",
        r"\bseems to\b",
        r"\bmay be\b",
        r"\bmight be\b",
        r"\bshould\b",
        r"\buser is on\b",
        r"\bpage .*display",
        r"\bpage .*show",
    ]
    return any(re.search(pattern, answer) for pattern in vague_patterns)


def _page_context_overlap(prompt: str, url: str, text_ir: Dict[str, Any]) -> int:
    terms = _focus_terms(prompt, max_terms=16)
    if not terms:
        return 0
    text_ir = text_ir if isinstance(text_ir, dict) else {}
    headings = text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else []
    try:
        parsed = urlsplit(str(url or ""))
        url_context = " ".join([str(parsed.path or ""), str(parsed.query or "")])
    except Exception:
        url_context = str(url or "")
    haystack = " ".join(
        [
            url_context,
            str(text_ir.get("title") or ""),
            " ".join(str(x) for x in headings[:8]),
        ]
    ).lower()
    hay_terms = _tokenize(haystack)
    return len(terms.intersection(hay_terms))


def _page_context_ready_for_informational_answer(prompt: str, url: str, text_ir: Dict[str, Any]) -> bool:
    overlap = _page_context_overlap(prompt, url, text_ir)
    try:
        path = str(urlsplit(str(url or "")).path or "/").strip() or "/"
    except Exception:
        path = "/"
    non_root_path = path not in {"", "/"}
    return bool(non_root_path or overlap >= 2)


def _runtime_page_evidence_ready(prompt: str, url: str, text_ir: Dict[str, Any], *, step_index: int) -> bool:
    if _page_context_ready_for_informational_answer(prompt, url, text_ir):
        return True
    if int(step_index) < 1:
        return False
    best_fact = _best_page_evidence(prompt, text_ir)
    if not best_fact:
        return False
    return bool(
        _fact_overlap_score(prompt, best_fact) >= 1
        and _anchor_overlap_score(prompt, best_fact) >= 1
    )


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    return str(raw).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _use_vision() -> bool:
    return _env_bool("USE_VISION", False) or _env_bool("AGENT_USE_VISION", False)


def _supported_browser_tool_names() -> set[str]:
    out = set(SUPPORTED_BROWSER_TOOL_NAMES)
    if not _use_vision():
        out.discard("browser.screenshot")
    return out


def _obs_meta_tools() -> set[str]:
    out = set(OBS_META_TOOLS)
    if not _use_vision():
        out.discard("META.VISION_QA")
    return out


def _meta_tools() -> set[str]:
    return _obs_meta_tools() | set(CONTROL_META_TOOLS)


def _unavailable_browser_tools() -> tuple[str, ...]:
    out = list(UNAVAILABLE_BROWSER_TOOLS)
    if not _use_vision():
        out.append("browser.screenshot")
    return tuple(out)


def _screenshot_available(value: Any) -> bool:
    if not _use_vision():
        return False
    return bool(_normalize_screenshot_data_url(value))


def _normalize_screenshot_data_url(value: Any) -> str:
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("data:image/"):
        return raw
    return f"data:image/png;base64,{raw}"


def _vision_signature(*, screenshot: Any, question: str, url: str) -> str:
    data_url = _normalize_screenshot_data_url(screenshot)
    if not data_url:
        return ""
    payload = json.dumps(
        {
            "url": str(url or "")[:300],
            "question": str(question or "")[:600],
            "image_hash": hashlib.sha1(data_url.encode("utf-8", errors="ignore")).hexdigest()[:16],
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _task_constraints(task: str) -> Dict[str, str]:
    text = str(task or "")
    out: Dict[str, str] = {}
    pattern = re.compile(
        r"\b([a-z][a-z0-9 _-]{1,40})\b\s*(?:equals|=|is|:)\s*(?:'([^']+)'|\"([^\"]+)\"|(<[^>]+>)|([0-9]+(?:\.[0-9]+)?)|([^\s,;]+))",
        flags=re.I,
    )
    for match in pattern.finditer(text):
        raw_key = _norm_ws(match.group(1) or "").lower()
        key = re.sub(r"^(?:the|a|an)\s+", "", raw_key).strip().replace(" ", "_")
        value = next((g for g in match.groups()[1:] if g), "")
        value = _norm_ws(value).strip(" \t\r\n'\"`.,;:!?")
        if not key or not value:
            continue
        if len(key) > 40:
            key = key[:40]
        out[key] = value[:120]
        if len(out) >= 16:
            break
    return out


def _constraint_key_tokens(key: str) -> set[str]:
    return _tokenize(str(key or "").replace("_", " "))


def _constraint_value_matches(expected: str, actual: str) -> bool:
    exp = _norm_ws(expected).lower()
    act = _norm_ws(actual).lower()
    if not exp or not act:
        return False
    if exp == act:
        return True
    if exp in act or act in exp:
        return True
    exp_tokens = _tokenize(exp)
    act_tokens = _tokenize(act)
    if exp_tokens and act_tokens and exp_tokens.issubset(act_tokens):
        return True
    return False


def _safe_url(raw: str, base: str = "") -> str:
    txt = str(raw or "").strip()
    if not txt:
        return ""
    if txt.startswith("http://") or txt.startswith("https://"):
        return txt
    if txt.startswith("/"):
        return urljoin(base or "", txt)
    # Relative path support (e.g. "search", "products/list") under current host.
    if base and " " not in txt and not re.match(r"^[a-z]+://", txt, flags=re.I):
        if txt.startswith("./"):
            txt = txt[2:]
        if txt:
            return urljoin(base or "", txt)
    if "." in txt and " " not in txt and "/" not in txt:
        return f"https://{txt}"
    return txt


def _query_map(url: str) -> Dict[str, str]:
    try:
        parsed = urlsplit(str(url or "").strip())
        out: Dict[str, str] = {}
        for k, v in parse_qsl(str(parsed.query or ""), keep_blank_values=True):
            key = str(k or "").strip()
            if not key:
                continue
            out[key[:80]] = str(v or "")[:120]
            if len(out) >= 16:
                break
        return out
    except Exception:
        return {}


def _with_query(url: str, query: Dict[str, str]) -> str:
    try:
        parsed = urlsplit(str(url or "").strip())
        if not parsed.scheme or not parsed.netloc:
            return str(url or "")
        q_items = list(query.items())[:16]
        q_encoded = urlencode(q_items, doseq=False)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", q_encoded, parsed.fragment or ""))
    except Exception:
        return str(url or "")


def _candidate_text(*parts: Any) -> str:
    for item in parts:
        if isinstance(item, str):
            cleaned = _norm_ws(item)
            if cleaned:
                return cleaned[:MAX_STR]
    return ""


def _sanitize_selector(selector: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(selector, dict):
        return None
    out = dict(selector)
    raw_type = str(out.get("type") or "").strip()
    sel_type = raw_type.lower()
    case_sensitive = bool(out.get("case_sensitive", False))

    def first_text(*keys: str) -> str:
        for key in keys:
            value = _candidate_text(out.get(key))
            if value:
                return value[:MAX_STR]
        return ""

    if sel_type in {"text", "textselector", "textcontains", "textcontainsselector", "linktext", "partiallinktext"}:
        value = first_text("value", "text", "label", "name", "query")
        if not value:
            return None
        return {"type": "tagContainsSelector", "value": value, "case_sensitive": case_sensitive}
    if sel_type in {"attribute", "attributevalueselector"}:
        attribute = first_text("attribute", "attr", "name")
        value = first_text("value", "text", "label")
        if not attribute or not value:
            return None
        return {
            "type": "attributeValueSelector",
            "attribute": attribute,
            "value": value,
            "case_sensitive": case_sensitive,
        }
    if sel_type in {"id", "class", "name", "href", "placeholder", "aria-label", "aria_label", "title", "role", "value", "type"}:
        value = first_text("value", "text", "label")
        if not value:
            return None
        attribute = "aria-label" if sel_type == "aria_label" else raw_type
        return {
            "type": "attributeValueSelector",
            "attribute": attribute,
            "value": value,
            "case_sensitive": case_sensitive,
        }
    if sel_type not in {"attributevalueselector", "tagcontainsselector", "xpathselector"}:
        attribute = first_text("attribute", "attr", "name")
        value = first_text("value", "text", "label", "query")
        if attribute and value:
            return {
                "type": "attributeValueSelector",
                "attribute": attribute,
                "value": value,
                "case_sensitive": case_sensitive,
            }
        if value:
            return {"type": "tagContainsSelector", "value": value, "case_sensitive": case_sensitive}
        return None
    if sel_type == "tagcontainsselector":
        value = first_text("value", "text", "label")
        if not value:
            return None
        return {"type": "tagContainsSelector", "value": value, "case_sensitive": case_sensitive}
    if sel_type == "xpathselector":
        value = str(out.get("value") or out.get("text") or out.get("xpath") or "").strip()
        if value.lower().startswith("xpath="):
            value = value[6:].strip()
        if value.startswith("///"):
            value = "//" + value.lstrip("/")
        elif value.startswith("/") and not value.startswith("//"):
            value = value.lstrip("/")
        value = re.sub(r"\s+", " ", value).strip()
        if not value:
            return None
        return {"type": "xpathSelector", "value": value, "case_sensitive": case_sensitive}
    return out


def _dedupe_keep_order(values: List[str], limit: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _norm_ws(raw)[:MAX_STR]
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _action_type_for_tool(tool_name: str) -> str | None:
    t = str(tool_name or "").strip().lower()
    mapping = {
        "browser.search": "SearchAction",
        "browser.navigate": "NavigateAction",
        "browser.go_back": "GoBackAction",
        "browser.click": "ClickAction",
        "browser.dblclick": "DoubleClickAction",
        "browser.rightclick": "RightClickAction",
        "browser.middleclick": "MiddleClickAction",
        "browser.tripleclick": "TripleClickAction",
        "browser.input": "TypeAction",
        "browser.scroll": "ScrollAction",
        "browser.wait": "WaitAction",
        "browser.select_dropdown": "SelectDropDownOptionAction",
        "browser.dropdown_options": "GetDropDownOptionsAction",
        "browser.hover": "HoverAction",
        "browser.screenshot": "ScreenshotAction",
        "browser.send_keys": "SendKeysIWAAction",
        "browser.hold_key": "HoldKeyAction",
        "browser.extract": "ExtractAction",
    }
    return mapping.get(t)


def _root_url(value: str) -> str:
    try:
        parsed = urlsplit(str(value or "").strip())
        if parsed.scheme and parsed.netloc:
            return urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))
    except Exception:
        pass
    return ""


def _canonical_allowed_tool_name(raw: str) -> str:
    name = str(raw or "").strip().lower()
    if not name:
        return ""
    if name.startswith("user."):
        return name
    if name.startswith("meta."):
        return name.upper()
    if name == "request_user_input":
        return "user.request_input"
    if name.startswith("browser."):
        name = name.split(".", 1)[1].strip()
    if name == "evaluate":
        return ""
    alias = {
        "search": "browser.search",
        "navigate": "browser.navigate",
        "go_back": "browser.go_back",
        "click": "browser.click",
        "dblclick": "browser.dblclick",
        "rightclick": "browser.rightclick",
        "middleclick": "browser.middleclick",
        "tripleclick": "browser.tripleclick",
        "hover": "browser.hover",
        "input": "browser.input",
        "select_dropdown": "browser.select_dropdown",
        "dropdown_options": "browser.dropdown_options",
        "screenshot": "browser.screenshot",
        "scroll": "browser.scroll",
        "wait": "browser.wait",
        "send_keys": "browser.send_keys",
        "hold_key": "browser.hold_key",
        "extract": "browser.extract",
        "done": "browser.done",
    }
    if name in alias:
        return alias[name]
    # Keep permissive for custom browser tools.
    return f"browser.{name}"


__all__ = [name for name in globals() if not name.startswith("__")]
