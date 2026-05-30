from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
IWA_PROJECTS_ROOT = REPO_ROOT.parent / "autoppia_iwa" / "autoppia_iwa" / "src" / "demo_webs" / "projects"
WEBS_DEMO_ROOT = REPO_ROOT.parent / "autoppia_webs_demo"


@dataclass(frozen=True)
class DemoProjectContext:
    web_project_id: str
    project_dir: Path
    frontend_dir: Path | None
    project_slug: str
    project_index: int | None


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _project_id_from_main(path: Path) -> str | None:
    try:
        text = _load_text(path)
    except Exception:
        return None
    match = re.search(r'id\s*=\s*["\']([^"\']+)["\']', text)
    if match:
        return str(match.group(1)).strip()
    return None


def resolve_demo_project(web_project_id: str) -> DemoProjectContext:
    target = str(web_project_id).strip()
    if not target:
        raise ValueError("web_project_id is required")
    for project_dir in sorted(IWA_PROJECTS_ROOT.glob("*")):
        main_py = project_dir / "main.py"
        if not main_py.exists():
            continue
        project_id = _project_id_from_main(main_py)
        if project_id != target:
            continue
        match = re.match(r"p0*(\d+)_([a-z0-9_]+)$", project_dir.name)
        project_index = int(match.group(1)) if match else None
        project_slug = str(match.group(2) if match else project_dir.name).strip()
        frontend_dir = None
        if project_index is not None:
            candidate = WEBS_DEMO_ROOT / f"web_{project_index}_{project_slug}"
            if candidate.exists():
                frontend_dir = candidate
        return DemoProjectContext(
            web_project_id=target,
            project_dir=project_dir,
            frontend_dir=frontend_dir,
            project_slug=project_slug,
            project_index=project_index,
        )
    raise ValueError(f"Unable to resolve demo project for web_project_id={target}")


def load_task_row(*, web_project_id: str, use_case: str, task_cache_path: str | Path, seed: int | None = None) -> dict[str, Any]:
    task_cache_path = Path(task_cache_path).resolve()
    payload = json.loads(task_cache_path.read_text(encoding="utf-8"))
    rows = payload.get("tasks") if isinstance(payload, dict) and isinstance(payload.get("tasks"), list) else payload
    if not isinstance(rows, list):
        raise ValueError(f"unexpected task cache format: {task_cache_path}")
    normalized_use_case = str(use_case).strip().upper()
    matches: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("web_project_id") or "").strip() != str(web_project_id).strip():
            continue
        row_use_case = ((row.get("use_case") or {}).get("name") if isinstance(row.get("use_case"), dict) else None)
        if str(row_use_case or "").strip().upper() != normalized_use_case:
            continue
        matches.append(row)
    if not matches:
        raise ValueError(
            f"use case {normalized_use_case} for project {web_project_id} not found in {task_cache_path}"
        )
    if seed is not None:
        needle = f"seed={int(seed)}"
        for row in matches:
            url = str(row.get("url") or "")
            if needle in url:
                return row
        cloned = dict(matches[0])
        base_url = str(cloned.get("url") or "").split("?", 1)[0]
        cloned["url"] = f"{base_url}?seed={int(seed)}"
        return cloned
    return matches[0]


def _extract_named_block(text: str, *, anchor_pattern: str, radius: int = 40, max_chars: int = 2500) -> str:
    lines = text.splitlines()
    anchor_idx = None
    regex = re.compile(anchor_pattern)
    for idx, line in enumerate(lines):
        if regex.search(line):
            anchor_idx = idx
            break
    if anchor_idx is None:
        return text[:max_chars]
    start = max(0, anchor_idx - radius)
    end = min(len(lines), anchor_idx + radius)
    snippet = "\n".join(lines[start:end]).strip()
    return snippet[:max_chars]


def _candidate_frontend_files(frontend_dir: Path | None, use_case: str) -> list[Path]:
    if frontend_dir is None or not frontend_dir.exists():
        return []
    normalized = str(use_case).strip().upper()
    keyword_map = {
        "LOGIN": ["login", "auth", "user", "header"],
        "REGISTRATION": ["signup", "register", "auth", "user"],
        "SEARCH": ["search", "filter", "header", "grid", "card"],
        "CONTACT": ["contact", "footer"],
        "DETAIL": ["detail", "book", "film", "hero", "meta", "product"],
        "ADD_TO_CART": ["cart", "card", "product"],
    }
    keywords = []
    for token, values in keyword_map.items():
        if token in normalized:
            keywords.extend(values)
    if not keywords:
        keywords = [normalized.lower().replace("_book", "").replace("_film", "").replace("_product", "")]

    candidates: list[Path] = []
    for path in sorted((frontend_dir / "src").rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".tsx", ".ts", ".json", ".js"}:
            continue
        rel = str(path.relative_to(frontend_dir)).lower()
        if any(keyword in rel for keyword in keywords):
            candidates.append(path)
    preferred = [
        frontend_dir / "src" / "app" / "page.tsx",
        frontend_dir / "src" / "app" / "login" / "page.tsx",
        frontend_dir / "src" / "context" / "AuthContext.tsx",
        frontend_dir / "src" / "data" / "users.ts",
        frontend_dir / "tests" / "test-events.js",
    ]
    merged: list[Path] = []
    seen: set[str] = set()
    for path in preferred + candidates:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        merged.append(path)
        if len(merged) >= 12:
            break
    return merged




def _concretize_demo_placeholders(payload: Any, *, web_agent_id: int) -> Any:
    if isinstance(payload, str):
        value = payload
        suffix = str(int(web_agent_id))
        value = value.replace("newuser<web_agent_id>@gmail.com", f"newuser{suffix}@gmail.com")
        value = value.replace("newuser<web_agent_id>", f"newuser{suffix}")
        value = value.replace("user<web_agent_id>@site.com", f"user{suffix}@site.com")
        value = value.replace("user<web_agent_id>", f"user{suffix}")
        value = value.replace("<web_agent_id>", suffix)
        return value
    if isinstance(payload, list):
        return [_concretize_demo_placeholders(item, web_agent_id=web_agent_id) for item in payload]
    if isinstance(payload, dict):
        return {key: _concretize_demo_placeholders(value, web_agent_id=web_agent_id) for key, value in payload.items()}
    return payload


def _seed_specialize_auth_payload(payload: Any, *, use_case: str, seed: int) -> Any:
    suffix = str(int(seed))
    registration_like = {"REGISTRATION", "REGISTRATION_BOOK"}
    login_like = {"LOGIN", "LOGIN_BOOK", "LOGOUT", "LOGOUT_BOOK"}
    uc = str(use_case).strip().upper()
    if isinstance(payload, str):
        value = payload
        if uc in registration_like:
            value = value.replace("newuser1@gmail.com", f"newuser{suffix}@gmail.com")
            value = value.replace("newuser1", f"newuser{suffix}")
        if uc in login_like:
            value = value.replace("user1@site.com", f"user{suffix}@site.com")
            value = value.replace("user1", f"user{suffix}")
        return value
    if isinstance(payload, list):
        return [_seed_specialize_auth_payload(item, use_case=use_case, seed=seed) for item in payload]
    if isinstance(payload, dict):
        return {key: _seed_specialize_auth_payload(value, use_case=use_case, seed=seed) for key, value in payload.items()}
    return payload

def build_project_context(*, web_project_id: str, use_case: str, task_cache_path: str | Path, seed: int) -> dict[str, Any]:
    resolved = resolve_demo_project(web_project_id)
    task_row = _concretize_demo_placeholders(
        load_task_row(
            web_project_id=web_project_id,
            use_case=use_case,
            task_cache_path=task_cache_path,
            seed=seed,
        ),
        web_agent_id=int(seed),
    )
    task_row = _seed_specialize_auth_payload(task_row, use_case=use_case, seed=int(seed))
    use_cases_path = resolved.project_dir / "use_cases.py"
    events_path = resolved.project_dir / "events.py"
    main_path = resolved.project_dir / "main.py"
    use_cases_text = _load_text(use_cases_path)
    events_text = _load_text(events_path)
    normalized_use_case = str(use_case).strip().upper()
    use_case_block = _extract_named_block(
        use_cases_text,
        anchor_pattern=rf'{re.escape(normalized_use_case)}_USE_CASE\s*=\s*UseCase\(',
        radius=45,
        max_chars=2500,
    )
    event_name = ""
    tests = task_row.get("tests")
    if isinstance(tests, list) and tests:
        first = tests[0]
        if isinstance(first, dict):
            event_name = str(first.get("event_name") or "").strip()
    if not event_name:
        event_name = str(use_case).strip()
    event_class_name = ''
    if isinstance(task_row.get('use_case'), dict):
        event_class_name = str((task_row.get('use_case') or {}).get('event') or '').strip()
    event_anchor = rf'class\s+{re.escape(event_class_name)}\b' if event_class_name else rf'event_name\s*:\s*str\s*=\s*["\']{re.escape(event_name)}["\']'
    event_block = _extract_named_block(
        events_text,
        anchor_pattern=event_anchor,
        radius=35,
        max_chars=2200,
    )
    backend_event_contract = {
        "get_events_endpoint": "/get_events/",
        "reset_events_endpoint": "/reset_events/",
        "params": ["web_url", "web_agent_id", "validator_id"],
    }
    frontend_files = _candidate_frontend_files(resolved.frontend_dir, use_case)
    source_files = [
        main_path,
        use_cases_path,
        events_path,
        resolved.project_dir / "generation_functions.py",
        resolved.project_dir / "replace_functions.py",
        resolved.project_dir / "utils.py",
        resolved.project_dir / "data_utils.py",
        resolved.project_dir / "data.py",
    ]
    file_snippets: list[dict[str, str]] = []
    for path in source_files + frontend_files:
        if not path.exists():
            continue
        try:
            text = _load_text(path)
        except Exception:
            continue
        file_snippets.append(
            {
                "path": str(path),
                "content": text[:1800],
            }
        )
    return {
        "resolved_project": {
            "web_project_id": resolved.web_project_id,
            "project_dir": str(resolved.project_dir),
            "frontend_dir": str(resolved.frontend_dir) if resolved.frontend_dir else "",
            "project_slug": resolved.project_slug,
            "project_index": resolved.project_index,
        },
        "task_row": task_row,
        "event_name": event_name,
        "backend_event_contract": backend_event_contract,
        "use_case_block": use_case_block,
        "event_block": event_block,
        "file_snippets": file_snippets,
    }
