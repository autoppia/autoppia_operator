from __future__ import annotations

from importlib import import_module

from .utils import *


def _package_helper(name: str, default: Any) -> Any:
    try:
        package = import_module("src.operator.agents.fsm")
        helper = getattr(package, name, default)
        if callable(helper):
            return helper
    except Exception:
        pass
    return default


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
                    discovered.append({"href": href, "label": label, "source": "page_anchor"})
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
    cache_index = _package_helper("_load_task_cache_site_index", _load_task_cache_site_index)()
    static_index = _package_helper("_load_static_site_maps", _load_static_site_maps)()
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
                for (label, href, path, section_id) in _package_helper("_crawl_site_routes", _crawl_site_routes)(
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


__all__ = [name for name in globals() if not name.startswith("__")]
