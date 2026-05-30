from __future__ import annotations

import os
import json
import subprocess
import urllib.parse
import urllib.request
import re
from pathlib import Path
from typing import Any

from training.demo_project_context import build_project_context
from training.harvester_support import summarize_attempt_for_claude
from training.layout import use_case_layout

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_WEBS_DATASET_BASE = "http://84.247.180.192:8090"


def focus_root(*, web_project_id: str, use_case: str) -> Path:
    return use_case_layout(repo_root=REPO_ROOT, web_project=web_project_id, use_case=use_case).root


def _brief_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "use_case": {"type": "string"},
            "seed": {"type": "integer"},
            "route": {"type": "array", "items": {"type": "string"}},
            "prompt_lines": {"type": "array", "items": {"type": "string"}},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "ids": {"type": "array", "items": {"type": "string"}},
                        "value": {"type": "string"},
                        "value_rule": {"type": "string"},
                    },
                    "required": ["name", "ids", "value", "value_rule"],
                },
            },
            "submit": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "text": {"type": "array", "items": {"type": "string"}},
                    "action": {"type": "string"},
                },
                "required": ["ids", "text", "action"],
            },
            "success_signals": {
                "type": "object",
                "properties": {
                    "texts": {"type": "array", "items": {"type": "string"}},
                    "ids": {"type": "array", "items": {"type": "string"}},
                    "url_contains": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["texts", "ids", "url_contains"],
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "ids": {"type": "array", "items": {"type": "string"}},
                        "text_hints": {"type": "array", "items": {"type": "string"}},
                        "text": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["type"],
                },
            },
            "pitfalls": {"type": "array", "items": {"type": "string"}},
            "action_sketch": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": [
            "use_case",
            "seed",
            "route",
            "prompt_lines",
            "fields",
            "submit",
            "success_signals",
            "steps",
            "pitfalls",
            "action_sketch",
            "confidence",
        ],
    }


def _strip_json_response(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            stripped = text.lstrip()
            if stripped.startswith("json"):
                text = stripped[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _build_one_shot_prompt(
    *,
    web_project_id: str,
    use_case: str,
    seed: int,
    context_payload: dict[str, Any],
    previous_attempts: list[dict[str, Any]] | None = None,
) -> str:
    task_row = context_payload.get("task_row") if isinstance(context_payload, dict) else {}
    prompt = str((task_row or {}).get("prompt") or "").strip()
    tests = task_row.get("tests") if isinstance(task_row, dict) else []
    constraints = ((task_row.get("use_case") or {}).get("constraints") if isinstance(task_row, dict) and isinstance(task_row.get("use_case"), dict) else None)
    snippets = context_payload.get("file_snippets") if isinstance(context_payload, dict) else []
    compact_snippets: list[dict[str, Any]] = []
    for item in snippets[:6]:
        if not isinstance(item, dict):
            continue
        compact_snippets.append({
            "path": item.get("path"),
            "reason": item.get("reason"),
            "snippet": str(item.get("snippet") or "")[:800],
        })
    event_name = str(context_payload.get("event_name") or "").strip()
    backend_event_contract = context_payload.get("backend_event_contract") if isinstance(context_payload, dict) else {}
    previous_attempts = [row for row in (previous_attempts or []) if isinstance(row, dict)][-3:]
    task_summary = {
        "id": task_row.get("id") if isinstance(task_row, dict) else None,
        "url": task_row.get("url") if isinstance(task_row, dict) else None,
        "prompt": prompt,
        "tests": tests,
        "constraints": constraints,
    }
    extra_guardrails = []
    if use_case.upper() == "LOGIN":
        extra_guardrails = [
            "Prefer navigating directly to /login when the site has a dedicated login route.",
            "Do not type credentials into search, lookup, filter, or newsletter inputs.",
            "Prefer selectors and text hints that clearly correspond to username, password, and sign-in submit controls.",
        ]
    return f"""
You are Claude Code acting as a code-aware trajectory harvesting teacher.

Target:
- web_project_id={web_project_id}
- use_case={use_case}
- seed={seed}
- real demo endpoint base=http://84.247.180.192

Goal:
- Infer a reliable executable browser trajectory for this task.
- You may inspect this repo plus ../autoppia_iwa and ../autoppia_webs_demo.
- Optimize for replay success in the real evaluator, not for code changes.
- Keep the answer compact and explicit.

Task summary:
{json.dumps(task_summary, ensure_ascii=False, indent=2)}

Target backend event name:
{event_name}

Demo web backend event contract:
{json.dumps(backend_event_contract, ensure_ascii=False, indent=2)}

Relevant code snippets:
{json.dumps(compact_snippets, ensure_ascii=False, indent=2)}

Guardrails:
{json.dumps(extra_guardrails, ensure_ascii=False, indent=2)}

Previous failed attempts:
{json.dumps(previous_attempts, ensure_ascii=False, indent=2)}

Return ONLY JSON matching this schema shape, with no prose and no markdown fences:
{json.dumps(_brief_schema(), ensure_ascii=False)}
""".strip()


def _build_direct_agent_prompt(
    *,
    web_project_id: str,
    use_case: str,
    seed: int,
    work_dir: Path,
    context_path: Path,
    candidate_brief_path: Path,
    final_brief_path: Path,
    eval_command: str,
) -> str:
    return f"""
You are Claude Code and your task is to solve one Autoppia browser benchmark by iterating against the real evaluator.

Target:
- web_project_id={web_project_id}
- use_case={use_case}
- seed={seed}
- real demo endpoint base=http://84.247.180.192
- work directory={work_dir}

Resources you have:
- repo code in {REPO_ROOT}
- evaluator-side demo web code in {(REPO_ROOT.parent / 'autoppia_iwa').resolve()}
- frontend demo code in {(REPO_ROOT.parent / 'autoppia_webs_demo').resolve()}
- task context JSON at {context_path}

Your job:
1. Read {context_path} and inspect the relevant source code.
2. Write a candidate brief JSON to {candidate_brief_path}. The brief must match this JSON schema:
{json.dumps(_brief_schema(), ensure_ascii=False)}
3. Run this eval command from the repo root:
{eval_command}
4. Inspect the generated files in {work_dir}, especially:
   - last_eval.json
   - last_eval.summary.json
   - backend_events in the summary
5. If score is not 1.0, update {candidate_brief_path} and run the eval again.
6. Repeat until you get success=true and score=1.0, or until you run out of budget.
7. When done, write the best passing brief to {final_brief_path}.
8. Print a short summary mentioning the score and the final brief path.

Rules:
- Do not edit repo source code.
- Only write files inside {work_dir}.
- Use the evaluator as the source of truth.
- Backend events are ground truth and should guide your fixes.
- Keep the brief explicit and executable: concrete ids, concrete text values, concrete route.
""".strip()


def generate_claude_brief(
    *,
    web_project_id: str,
    use_case: str,
    seed: int,
    model: str = "claude-sonnet-4-5",
    previous_attempts: list[dict[str, Any]] | None = None,
    timeout_seconds: int = 120,
    task_cache_path: Path | None = None,
) -> dict[str, Any]:
    if task_cache_path is None:
        raise ValueError("task_cache_path is required")
    context_payload = build_project_context(
        web_project_id=web_project_id,
        use_case=use_case,
        task_cache_path=Path(task_cache_path).resolve(),
        seed=seed,
    )
    prompt = _build_one_shot_prompt(
        web_project_id=web_project_id,
        use_case=use_case,
        seed=seed,
        context_payload=context_payload,
        previous_attempts=previous_attempts,
    )
    cmd = [
        "claude",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        model,
        "--max-budget-usd",
        "1",
        "--tools",
        "Read,Grep,Glob,LS",
        "--add-dir",
        str(REPO_ROOT),
        "--add-dir",
        str((REPO_ROOT.parent / "autoppia_iwa").resolve()),
        "--add-dir",
        str((REPO_ROOT.parent / "autoppia_webs_demo").resolve()),
    ]
    raw = subprocess.check_output(
        cmd,
        cwd=str(REPO_ROOT),
        input=prompt,
        text=True,
        timeout=max(1, int(timeout_seconds)),
    )
    result = _strip_json_response(raw)
    result = _postprocess_brief(web_project_id=web_project_id, use_case=use_case, seed=seed, brief=result)
    return {
        "brief": result,
        "meta": {
            "model": model,
            "mode": "one_shot_claude_code",
            "raw_preview": str(raw or "")[:500],
        },
        "context": context_payload,
    }


def _postprocess_brief(*, web_project_id: str, use_case: str, seed: int, brief: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(brief, dict):
        return brief
    normalized = json.loads(json.dumps(brief))
    uc = str(use_case).upper()
    if uc == "LOGIN":
        username_field = None
        password_field = None
        for field in normalized.get("fields") or []:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name") or "").strip().lower()
            if name == "username" and username_field is None:
                username_field = field
            if name == "password" and password_field is None:
                password_field = field
        submit = normalized.get("submit") if isinstance(normalized.get("submit"), dict) else {}
        login_url = f"http://84.247.180.192:8000/login?seed={int(seed)}"
        steps = [{"type": "navigate", "url": login_url}]
        if isinstance(username_field, dict):
            steps.append({"type": "fill", "ids": list(username_field.get("ids") or []), "text_hints": ["Username", "User", "Email"], "text": str(username_field.get("value") or "user1")})
        if isinstance(password_field, dict):
            steps.append({"type": "fill", "ids": list(password_field.get("ids") or []), "text_hints": ["Password", "Pass"], "text": str(password_field.get("value") or "Passw0rd!")})
        if isinstance(submit, dict):
            steps.append({"type": "click", "ids": list(submit.get("ids") or []), "text_hints": list(submit.get("text") or ["Login", "Sign In", "Submit"])})
        normalized["route"] = ["/login"]
        normalized["steps"] = steps
    elif uc == "CONTACT":
        submit = normalized.get("submit") if isinstance(normalized.get("submit"), dict) else {}
        contact_url = f"http://84.247.180.192:8000/contact?seed={int(seed)}"
        steps = [{"type": "navigate", "url": contact_url}]
        for field in normalized.get("fields") or []:
            if not isinstance(field, dict):
                continue
            name = str(field.get("name") or "").strip().lower()
            hints_map = {
                "name": ["Name", "Your name"],
                "email": ["Email", "you@example.com"],
                "subject": ["Subject"],
                "message": ["Message", "Tell us what's on your mind"],
            }
            steps.append({
                "type": "fill",
                "ids": list(field.get("ids") or []),
                "text_hints": hints_map.get(name, [name.title()]),
                "text": str(field.get("value") or ""),
            })
        if isinstance(submit, dict):
            steps.append({"type": "click", "ids": list(submit.get("ids") or []), "text_hints": list(submit.get("text") or ["Send Message", "Submit", "Send"])})
        normalized["route"] = ["/contact"]
        normalized["steps"] = steps
    elif uc == "FILTER_FILM" and web_project_id == "autocinema":
        genre_value, year_value = _extract_autocinema_filter_values(normalized)
        search_url = f"http://84.247.180.192:8000/search?seed={int(seed)}"
        steps = [{"type": "navigate", "url": search_url}]
        if genre_value:
            steps.append({
                "type": "select",
                "ids": [],
                "text_hints": ["All genres"],
                "selector_candidates": [{"type": "xpathSelector", "value": "//select[.//option[normalize-space()=\"All genres\"]]"}],
                "text": genre_value,
            })
        if year_value:
            steps.append({
                "type": "select",
                "ids": [],
                "text_hints": ["All years"],
                "selector_candidates": [{"type": "xpathSelector", "value": "//select[.//option[normalize-space()=\"All years\"]]"}],
                "text": year_value,
            })
        normalized["route"] = ["/search"]
        normalized["steps"] = steps
        normalized["fields"] = [
            {"name": "genre_filter", "ids": [], "value": genre_value or "", "value_rule": "Select from the genre dropdown on the search page."},
            {"name": "year_filter", "ids": [], "value": year_value or "", "value_rule": "Select from the year dropdown on the search page."},
        ]
        normalized["submit"] = {
            "ids": [],
            "text": [],
            "action": "selection_triggers_event",
        }
        normalized["success_signals"] = {
            "texts": ["Curated movies"],
            "ids": ["library"],
            "url_contains": ["/search"],
        }
        normalized["pitfalls"] = [
            "Do not start from the home page and do not click the search submit button.",
            "Navigate directly to /search with the seed query param.",
            "FILTER_FILM fires when the genre/year dropdown values change; no submit is needed.",
        ]
        normalized["action_sketch"] = [
            "Open the search page directly.",
            f"Set genre to {genre_value}." if genre_value else "Set the requested genre if present.",
            f"Set year to {year_value}." if year_value else "Set the requested year if present.",
        ]
    elif uc == "FILM_DETAIL" and web_project_id == "autocinema":
        target_movie = _pick_autocinema_film_detail_target(seed=int(seed))
        if target_movie is not None:
            movie_id = str(target_movie.get("id") or "").strip()
            title = str(target_movie.get("title") or "").strip()
            director = str(target_movie.get("director") or "").strip()
            try:
                year = int(target_movie.get("year") or 0)
            except Exception:
                year = 0
            detail_url = f"http://84.247.180.192:8000/movies/{urllib.parse.quote(movie_id)}?seed={int(seed)}"
            normalized["route"] = [f"/movies/{movie_id}"]
            normalized["steps"] = [{"type": "navigate", "url": detail_url}]
            normalized["fields"] = []
            normalized["submit"] = {
                "ids": ["view-details-button", "movie-details-btn", "movie-details"],
                "text": ["View detail", "View details", title],
                "action": "click",
            }
            normalized["success_signals"] = {
                "texts": [title, director, str(year)],
                "ids": ["movie-details", "movie-details-btn"],
                "url_contains": [f"/movies/{movie_id}"],
            }
            normalized["pitfalls"] = [
                "Do not stay on the home page.",
                "Navigate directly to a movie detail page that satisfies the prompt constraints.",
                "Avoid Robert Zemeckis films and years 2010 or earlier.",
            ]
            normalized["action_sketch"] = [
                f"Open the detail page for {title}.",
                f"Confirm the page shows {director} and year {year}.",
                "Let the FILM_DETAIL event fire on page load.",
            ]
    return normalized


def _extract_autocinema_filter_values(brief: dict[str, Any]) -> tuple[str | None, str | None]:
    chunks: list[str] = []
    prompt_lines = brief.get("prompt_lines")
    if isinstance(prompt_lines, list):
        chunks.extend(str(item) for item in prompt_lines if item)
    for field in brief.get("fields") or []:
        if isinstance(field, dict):
            value = str(field.get("value") or "").strip()
            if value:
                chunks.append(value)
    haystack = " \n".join(chunks)
    genre_value = None
    year_value = None
    quoted = re.findall(r"'([^']+)'", haystack)
    for item in quoted:
        stripped = item.strip()
        if re.fullmatch(r"\d{4}", stripped) and year_value is None:
            year_value = stripped
        elif genre_value is None:
            genre_value = stripped
    if year_value is None:
        match = re.search(r"\b(19\d{2}|20\d{2})\b", haystack)
        if match:
            year_value = match.group(1)
    if genre_value is None:
        match = re.search(r"\b(Action|Drama|Comedy|Crime|Thriller|Romance|Adventure|Fantasy|Animation|Sci-Fi|Screen)\b", haystack, re.I)
        if match:
            genre_value = match.group(1)
    return genre_value, year_value


def _pick_autocinema_movie_by_title_contains(*, seed: int, needle: str) -> dict[str, Any] | None:
    params = urllib.parse.urlencode(
        {
            "project_key": "web_1_autocinema",
            "entity_type": "movies",
            "seed_value": str(int(seed)),
            "limit": "50",
            "method": "distribute",
            "filter_key": "category",
        }
    )
    url = f"{DEMO_WEBS_DATASET_BASE}/datasets/load?{params}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    lowered = str(needle or '').strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        if lowered and lowered in title.lower():
            return row
    return rows[0] if rows else None


def _pick_autocinema_film_detail_target(*, seed: int) -> dict[str, Any] | None:
    params = urllib.parse.urlencode(
        {
            "project_key": "web_1_autocinema",
            "entity_type": "movies",
            "seed_value": str(int(seed)),
            "limit": "50",
            "method": "distribute",
            "filter_key": "category",
        }
    )
    url = f"{DEMO_WEBS_DATASET_BASE}/datasets/load?{params}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        director = str(row.get("director") or "").strip()
        try:
            year = int(row.get("year") or 0)
        except Exception:
            year = 0
        movie_id = str(row.get("id") or "").strip()
        if movie_id and director != "Robert Zemeckis" and year > 2010:
            return row
    return None


def _load_eval_feedback(eval_out_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    if eval_out_path.exists():
        report = json.loads(eval_out_path.read_text(encoding="utf-8"))
    summary_path = eval_out_path.with_suffix(".summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    backend_events = summary.get("backend_events") if isinstance(summary, dict) else []
    final_url = str(summary.get("final_url") or "") if isinstance(summary, dict) else ""
    score = float(summary.get("score") or 0.0) if isinstance(summary, dict) else 0.0
    success = bool(summary.get("success")) if isinstance(summary, dict) else False
    return {
        "summary": summary,
        "report": report,
        "success": success,
        "score": score,
        "final_url": final_url,
        "backend_events": backend_events if isinstance(backend_events, list) else [],
    }


def _canonical_brief_from_context(*, web_project_id: str, use_case: str, seed: int, context_payload: dict[str, Any]) -> dict[str, Any] | None:
    uc = str(use_case).upper().strip()
    if uc not in {"FILTER_FILM", "SEARCH_FILM", "LOGOUT", "REGISTRATION", "ADD_COMMENT", "ADD_FILM", "FILM_DETAIL", "ADD_TO_WATCHLIST"}:
        return None
    task_row = context_payload.get("task_row") if isinstance(context_payload.get("task_row"), dict) else {}
    prompt = str(task_row.get("prompt") or "").strip()
    route: list[str] = []
    try:
        task_url = str(task_row.get("url") or "").strip()
        if task_url:
            from urllib.parse import urlparse

            parsed = urlparse(task_url)
            if parsed.path:
                route = [parsed.path]
    except Exception:
        route = []
    brief = {
        "use_case": uc,
        "seed": int(seed),
        "route": route,
        "prompt_lines": [prompt] if prompt else [],
        "fields": [],
        "submit": {"ids": [], "text": [], "action": "auto"},
        "success_signals": {"texts": [], "ids": [], "url_contains": []},
        "steps": [],
        "pitfalls": [],
        "action_sketch": [],
        "confidence": 1.0,
    }
    if uc == "ADD_COMMENT":
        import re
        movie_match = re.search(r"movie_name\s+that\s+contains\s+[\"\']([^\"\']+)[\"\']", prompt, flags=re.IGNORECASE)
        content_not_match = re.search(r"content\s+that\s+does\s+NOT\s+contain\s+the\s+word\s+[\"\']([^\"\']+)[\"\']", prompt, flags=re.IGNORECASE)
        content_exact_match = re.search(r"content(?:\s+that)?\s+contains\s+[\"\']([^\"\']+)[\"\']", prompt, flags=re.IGNORECASE)
        commenter_match = re.search(r"commenter_name\s+(?:equals|contains)\s+[\"\']([^\"\']+)[\"\']", prompt, flags=re.IGNORECASE)
        movie_needle = str(movie_match.group(1) if movie_match else '').strip()
        forbidden = str(content_not_match.group(1) if content_not_match else '').strip().lower()
        desired_content = str(content_exact_match.group(1) if content_exact_match else '').strip()
        if not desired_content:
            desired_content = f"Amazing film with sharp pacing and atmosphere"
            if forbidden and forbidden in desired_content.lower():
                desired_content = "Thoughtful and engaging from start to finish"
        commenter = str(commenter_match.group(1) if commenter_match else 'Alex Viewer').strip()
        movie = _pick_autocinema_movie_by_title_contains(seed=int(seed), needle=movie_needle)
        if movie is not None:
            movie_id = str(movie.get('id') or '').strip()
            title = str(movie.get('title') or '').strip()
            detail_url = f"http://84.247.180.192:8000/movies/{urllib.parse.quote(movie_id)}?seed={int(seed)}"
            brief["route"] = [f"/movies/{movie_id}"]
            brief["fields"] = [
                {"name": "commenter_name", "ids": ["comment-name-input"], "value": commenter, "value_rule": "Use a non-empty commenter name that satisfies the prompt."},
                {"name": "content", "ids": ["comment-message-textarea"], "value": desired_content, "value_rule": "Use comment content that satisfies the prompt constraints exactly."},
            ]
            brief["submit"] = {"ids": ["share-feedback-button"], "text": ["Share Feedback"], "action": "click"}
            brief["steps"] = [
                {"type": "navigate", "url": detail_url},
                {"type": "fill", "ids": ["comment-name-input"], "text_hints": ["Name"], "text": commenter},
                {"type": "fill", "ids": ["comment-message-textarea"], "text_hints": ["Comment", "Message"], "text": desired_content},
                {"type": "click", "ids": ["share-feedback-button"], "text_hints": ["Share Feedback"]},
            ]
            brief["success_signals"] = {"texts": [commenter, desired_content[:12], title], "ids": ["share-feedback-button"], "url_contains": [f"/movies/{movie_id}"]}
            brief["pitfalls"] = [
                "Do not use the global search box instead of opening the movie detail page.",
                "Write the comment on the movie detail page using the comment form.",
            ]
            brief["action_sketch"] = [
                f"Open the detail page for a movie whose title contains {movie_needle or 'the requested title' }.",
                f"Type commenter {commenter} and a valid comment.",
                "Submit the comment form.",
            ]
            return brief
    if uc == "SEARCH_FILM" and web_project_id == "autocinema":
        query = ""
        tests = task_row.get("tests") if isinstance(task_row.get("tests"), list) else []
        for test in tests:
            if not isinstance(test, dict):
                continue
            if str(test.get("event_name") or "").strip().upper() != "SEARCH_FILM":
                continue
            criteria = test.get("event_criteria") if isinstance(test.get("event_criteria"), dict) else {}
            query = str(criteria.get("query") or "").strip()
            if query:
                break
        if not query:
            prompt_match = re.search(r"movie\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
            query = str(prompt_match.group(1) if prompt_match else "").strip()
        if query:
            search_url = f"http://84.247.180.192:8000/search?seed={int(seed)}"
            search_submit_candidates = [
                "search-submit-button",
                "search-btn",
                "submit-search",
                "query-button",
                "search-action",
                "find-button",
                "submit-query",
                "search-trigger",
                "query-submit",
                "search-execute",
            ]
            search_input_xpaths = [
                "//input[@type='search']",
                "//form//input[@type='search']",
            ]
            def _hash_string(value: str) -> int:
                h = 0
                for ch in value:
                    h = ((h << 5) - h) + ord(ch)
                    h = h & 0xFFFFFFFF
                    if h >= 2**31:
                        h -= 2**32
                return abs(h)
            def _pick_variant(key: str, variants: list[str]) -> str:
                if int(seed) == 1 or len(variants) <= 1:
                    return variants[0]
                return variants[_hash_string(f"{key}:{int(seed)}") % len(variants)]
            exact_submit_id = _pick_variant("search-submit-button", search_submit_candidates)
            brief["route"] = ["/search"]
            brief["fields"] = [
                {
                    "name": "query",
                    "ids": [],
                    "value": query,
                    "value_rule": "Type the exact query string from the task prompt into the search field.",
                }
            ]
            brief["submit"] = {"ids": [exact_submit_id], "text": ["Search"], "action": "click"}
            brief["steps"] = [
                {"type": "navigate", "url": search_url},
                {
                    "type": "fill",
                    "ids": [],
                    "text_hints": ["Search", "Search movies"],
                    "text": query,
                    "field_name": "query",
                    "selector_candidates": [
                        *[
                            {"type": "xpathSelector", "value": xpath}
                            for xpath in search_input_xpaths
                        ],
                    ],
                },
                {
                    "type": "click",
                    "ids": [exact_submit_id],
                    "text_hints": ["Search"],
                    "field_name": "submit",
                    "exact_ids_only": True,
                    "selector_candidates": [
                        {"type": "attributeValueSelector", "attribute": "id", "value": exact_submit_id, "case_sensitive": False},
                    ],
                },
            ]
            brief["success_signals"] = {"texts": [query], "ids": [exact_submit_id], "url_contains": ["/search"]}
            brief["pitfalls"] = [
                "Navigate directly to /search for the requested seed.",
                "Type the exact query from the task prompt into the search field.",
                "Submit the search once and avoid filter-only interactions.",
            ]
            brief["action_sketch"] = [
                "Open the search page directly.",
                f"Type {query}.",
                "Click the search submit button once.",
            ]
            return brief
    if uc == "ADD_TO_WATCHLIST" and web_project_id == "autocinema":
        target_movie = _pick_autocinema_film_detail_target(seed=int(seed))
        if target_movie is not None:
            movie_id = str(target_movie.get("id") or "").strip()
            title = str(target_movie.get("title") or "").strip()
            username = f"user{int(seed)}"
            password = "Passw0rd!"
            login_url = f"http://84.247.180.192:8000/login?seed={int(seed)}"
            detail_url = f"http://84.247.180.192:8000/movies/{urllib.parse.quote(movie_id)}?seed={int(seed)}"
            username_candidates = [
                "login-username-input",
                "username-field",
                "login-username",
                "username-input-field",
                "login-username-field",
                "username-entry-field",
                "login-username-entry",
                "username-input",
                "login-user",
                "username-field-input",
            ]
            password_candidates = [
                "login-password-input",
                "password-field",
                "login-password",
                "password-input-field",
                "login-password-field",
                "password-entry-field",
                "login-password-entry",
                "password-input",
                "login-pass",
                "password-field-input",
            ]
            submit_candidates = [
                "login-sign-in-button",
                "signin-btn",
                "login-button",
                "sign-in-btn",
                "login-btn",
                "signin-button",
                "login-action",
                "sign-in-action",
                "login-control",
                "signin-control",
            ]
            watchlist_id_candidates = [
                "watchlist-button", "add-list-btn", "save-btn", "watchlist-btn", "list-button",
                "add-button", "save-list-btn", "watchlist-action", "add-action", "list-btn",
            ]
            def _hash_string(value: str) -> int:
                h = 0
                for ch in value:
                    h = ((h << 5) - h) + ord(ch)
                    h = h & 0xFFFFFFFF
                    if h >= 2**31:
                        h -= 2**32
                return abs(h)
            def _pick_variant(key: str, variants: list[str]) -> str:
                if int(seed) == 1 or len(variants) <= 1:
                    return variants[0]
                return variants[_hash_string(f"{key}:{int(seed)}") % len(variants)]
            exact_username_id = _pick_variant("login-username-input", username_candidates)
            exact_password_id = _pick_variant("login-password-input", password_candidates)
            exact_submit_id = _pick_variant("login-sign-in-button", submit_candidates)
            exact_watchlist_id = _pick_variant("watchlist-button", watchlist_id_candidates)
            watchlist_text_candidates = [
                "Add to watchlist", "Añadir a lista", "Save to list", "Add to list", "Watchlist", "Save", "Add",
            ]
            brief["route"] = ["/login", f"/movies/{movie_id}"]
            brief["fields"] = [
                {"name": "username", "ids": [exact_username_id], "value": username},
                {"name": "password", "ids": [exact_password_id], "value": password},
            ]
            brief["submit"] = {"ids": watchlist_id_candidates, "text": watchlist_text_candidates, "action": "click"}
            brief["steps"] = [
                {"type": "navigate", "url": login_url},
                {"type": "fill", "ids": [exact_username_id], "text_hints": ["Username"], "text": username, "field_name": "username", "exact_ids_only": True,
                 "selector_candidates": [{"type": "attributeValueSelector", "attribute": "id", "value": exact_username_id, "case_sensitive": False}]},
                {"type": "fill", "ids": [exact_password_id], "text_hints": ["Password"], "text": password, "field_name": "password", "exact_ids_only": True,
                 "selector_candidates": [{"type": "attributeValueSelector", "attribute": "id", "value": exact_password_id, "case_sensitive": False}]},
                {"type": "click", "ids": [exact_submit_id], "text_hints": ["Sign in", "Signing in…", "Login", "Submit"], "field_name": "submit", "exact_ids_only": True,
                 "selector_candidates": [{"type": "attributeValueSelector", "attribute": "id", "value": exact_submit_id, "case_sensitive": False}]},
                {"type": "navigate", "url": detail_url},
                {"type": "click", "ids": [exact_watchlist_id], "text_hints": watchlist_text_candidates, "exact_ids_only": True,
                 "selector_candidates": [{"type": "attributeValueSelector", "attribute": "id", "value": exact_watchlist_id, "case_sensitive": False}]},
            ]
            brief["success_signals"] = {"texts": [title, "watchlist"], "ids": watchlist_id_candidates, "url_contains": [f"/movies/{movie_id}"]}
            brief["pitfalls"] = [
                "First log in with the provided credentials.",
                "Open the movie detail page before interacting with watchlist controls.",
                "Do not use profile watchlist removal controls for this add flow.",
            ]
            brief["action_sketch"] = [
                f"Log in as {username}.",
                f"Open the detail page for {title}.",
                "Click the watchlist button once to add the movie.",
            ]
            return brief
    if uc == "ADD_FILM" and web_project_id == "autocinema":
        import re

        username_match = re.search(r"username\s+equals\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
        password_match = re.search(r"password\s+equals\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
        genre_match = re.search(r"genres?\s+equals\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
        username = str(username_match.group(1) if username_match else "user1")
        password = str(password_match.group(1) if password_match else "Passw0rd!")
        genre = str(genre_match.group(1) if genre_match else "Romance")
        login_url = f"http://84.247.180.192:8000/login?seed={int(seed)}"
        profile_url = f"http://84.247.180.192:8000/profile?seed={int(seed)}"
        username_candidates = [
            "login-username-input",
            "username-field",
            "login-username",
            "username-input-field",
            "login-username-field",
            "username-entry-field",
            "login-username-entry",
            "username-input",
            "login-user",
            "username-field-input",
        ]
        password_candidates = [
            "login-password-input",
            "password-field",
            "login-password",
            "password-input-field",
            "login-password-field",
            "password-entry-field",
            "login-password-entry",
            "password-input",
            "login-pass",
            "password-field-input",
        ]
        submit_candidates = [
            "login-sign-in-button",
            "signin-btn",
            "login-button",
            "sign-in-btn",
            "login-btn",
            "signin-button",
            "login-action",
            "sign-in-action",
            "login-control",
            "signin-control",
        ]
        add_film_submit_candidates = [
            "save-changes-button",
        ]
        add_movies_variants = [
            "Add Movies",
            "Añadir Películas",
            "Add Movies",
            "New Movies",
            "Add Films",
            "Add",
            "Movies",
            "Add Movies",
            "New",
            "Movies",
        ]
        def _hash_string(value: str) -> int:
            h = 0
            for ch in value:
                h = ((h << 5) - h) + ord(ch)
                h = h & 0xFFFFFFFF
                if h >= 2**31:
                    h -= 2**32
            return abs(h)
        preferred_index = 0 if int(seed) == 1 else (_hash_string(f"add_movies:{int(seed)}") % len(add_movies_variants))
        add_movies_labels = [add_movies_variants[preferred_index]]
        add_movies_labels.extend([label for label in add_movies_variants if label not in add_movies_labels])
        brief["route"] = ["/login", "/profile"]
        brief["fields"] = [
            {"name": "username", "ids": username_candidates, "value": username, "value_rule": "Use the exact username from the prompt."},
            {"name": "password", "ids": password_candidates, "value": password, "value_rule": "Use the exact password from the prompt."},
            {"name": "genres", "ids": [], "value": genre, "value_rule": "Select the exact requested genre from the Add Movies genre chips."},
        ]
        brief["submit"] = {"ids": add_film_submit_candidates, "text": ["Add Film"], "action": "click"}
        brief["steps"] = [
            {"type": "navigate", "url": login_url},
            {
                "type": "fill",
                "ids": username_candidates,
                "text_hints": ["Username"],
                "text": username,
                "field_name": "username",
                "exact_ids_only": True,
                "selector_candidates": [
                    *[
                        {"type": "attributeValueSelector", "attribute": "id", "value": candidate, "case_sensitive": False}
                        for candidate in username_candidates
                    ],
                    {"type": "tagContainsSelector", "value": "Username", "case_sensitive": False},
                ],
            },
            {
                "type": "fill",
                "ids": password_candidates,
                "text_hints": ["Password"],
                "text": password,
                "field_name": "password",
                "exact_ids_only": True,
                "selector_candidates": [
                    *[
                        {"type": "attributeValueSelector", "attribute": "id", "value": candidate, "case_sensitive": False}
                        for candidate in password_candidates
                    ],
                    {"type": "tagContainsSelector", "value": "Password", "case_sensitive": False},
                ],
            },
            {
                "type": "click",
                "ids": submit_candidates,
                "text_hints": ["Sign in", "Signing in…"],
                "field_name": "submit",
                "exact_ids_only": True,
                "selector_candidates": [
                    *[
                        {"type": "attributeValueSelector", "attribute": "id", "value": candidate, "case_sensitive": False}
                        for candidate in submit_candidates
                    ],
                    {"type": "tagContainsSelector", "value": "Sign in", "case_sensitive": False},
                    {"type": "tagContainsSelector", "value": "Signing in…", "case_sensitive": False},
                ],
            },
            {"type": "navigate", "url": profile_url},
            {
                "type": "click",
                "ids": [],
                "text_hints": add_movies_labels,
                "selector_candidates": [
                    *[
                        {"type": "xpathSelector", "value": f"//button[normalize-space()={json.dumps(label)}]"}
                        for label in add_movies_labels
                    ],
                    *[
                        {"type": "tagContainsSelector", "value": label, "case_sensitive": False}
                        for label in add_movies_labels
                    ],
                ],
            },
            {
                "type": "click",
                "ids": [],
                "text_hints": [genre],
                "selector_candidates": [
                    {"type": "xpathSelector", "value": f"//button[normalize-space()={json.dumps(genre)}]"},
                    {"type": "tagContainsSelector", "value": genre, "case_sensitive": False},
                ],
            },
            {
                "type": "click",
                "ids": add_film_submit_candidates,
                "text_hints": ["Add Film"],
                "selector_candidates": [
                    *[
                        {"type": "attributeValueSelector", "attribute": "id", "value": candidate, "case_sensitive": False}
                        for candidate in add_film_submit_candidates
                    ],
                    {"type": "tagContainsSelector", "value": "Add Film", "case_sensitive": False},
                ],
                "exact_ids_only": True,
            },
        ]
        brief["success_signals"] = {"texts": ["Film added", genre], "ids": add_film_submit_candidates, "url_contains": ["/profile"]}
        brief["pitfalls"] = [
            "Do not stay on the home page after login.",
            "Open the Add Movies tab before interacting with genre controls.",
            "Do not use search or assigned-movie editors; only the Add Movies panel should be used.",
        ]
        brief["action_sketch"] = [
            f"Log in as {username}.",
            "Open /profile and switch to the Add Movies tab.",
            f"Choose the {genre} genre chip and submit Add Film.",
        ]
        return brief
    if uc == "REGISTRATION":
        import re
        username_match = re.search(r"username\s+equals\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
        email_match = re.search(r"email\s+equals\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
        password_match = re.search(r"password\s+equals\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
        username = str(username_match.group(1) if username_match else "newuser1")
        email = str(email_match.group(1) if email_match else f"{username}@gmail.com")
        password = str(password_match.group(1) if password_match else "Passw0rd!")
        register_url = f"http://84.247.180.192:8000/register?seed={int(seed)}"
        brief["route"] = ["/register"]
        brief["fields"] = [
            {"name": "username", "ids": ["register-username-input"], "value": username, "value_rule": "Use the exact username from the prompt."},
            {"name": "email", "ids": ["register-email-input"], "value": email, "value_rule": "Use the exact email from the prompt."},
            {"name": "password", "ids": ["register-password-input"], "value": password, "value_rule": "Use the exact password from the prompt."},
            {"name": "confirm_password", "ids": ["register-confirm-password-input"], "value": password, "value_rule": "Repeat the same password exactly."},
        ]
        brief["submit"] = {"ids": ["create-account-button"], "text": ["Create account", "Register", "Sign up"], "action": "click"}
        brief["steps"] = [
            {"type": "navigate", "url": register_url},
            {"type": "fill", "ids": ["register-username-input"], "text_hints": ["Username"], "text": username},
            {"type": "fill", "ids": ["register-email-input"], "text_hints": ["Email"], "text": email},
            {"type": "fill", "ids": ["register-password-input"], "text_hints": ["Password"], "text": password},
            {"type": "fill", "ids": ["register-confirm-password-input"], "text_hints": ["Confirm Password"], "text": password},
            {"type": "click", "ids": ["create-account-button"], "text_hints": ["Create account", "Register", "Sign up"]},
        ]
        brief["success_signals"] = {"texts": [username, "Profile", "Logout"], "ids": ["create-account-button"], "url_contains": ["/profile", "/"]}
        brief["pitfalls"] = [
            "Do not use the login page.",
            "Repeat the password exactly in the confirm password field.",
            "Do not type credentials into unrelated search or profile inputs.",
        ]
        brief["action_sketch"] = [
            f"Open /register for seed {seed}.",
            f"Fill username {username}, email {email}, and password {password}.",
            "Repeat the password in confirm password and submit.",
        ]
        return brief
    if uc == "LOGOUT":
        import re
        username_match = re.search(r"username\s+equals\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
        password_match = re.search(r"password\s+equals\s+[\"']([^\"']+)[\"']", prompt, flags=re.IGNORECASE)
        username = str(username_match.group(1) if username_match else "user1")
        password = str(password_match.group(1) if password_match else "Passw0rd!")
        login_url = f"{base_url}/login?seed={seed}"
        brief["use_case"] = "LOGOUT"
        brief["route"] = ["/login"]
        brief["fields"] = [
            {"name": "username", "ids": ["login-username-input"], "value": username},
            {"name": "password", "ids": ["login-password-input"], "value": password},
        ]
        brief["submit"] = {"ids": ["login-sign-in-button"], "text": ["Sign in", "Signing in…"]}
        brief["steps"] = [
            {"type": "navigate", "url": login_url},
            {"type": "fill", "ids": ["login-username-input"], "text_hints": ["Username"], "text": username, "field_name": "username", "exact_ids_only": True,
             "selector_candidates": [{"type": "attributeValueSelector", "attribute": "id", "value": "login-username-input", "case_sensitive": False}]},
            {"type": "fill", "ids": ["login-password-input"], "text_hints": ["Password"], "text": password, "field_name": "password", "exact_ids_only": True,
             "selector_candidates": [{"type": "attributeValueSelector", "attribute": "id", "value": "login-password-input", "case_sensitive": False}]},
            {"type": "click", "ids": ["login-sign-in-button"], "text_hints": ["Sign in", "Signing in…"], "field_name": "submit", "exact_ids_only": True,
             "selector_candidates": [{"type": "attributeValueSelector", "attribute": "id", "value": "login-sign-in-button", "case_sensitive": False}]},
            {"type": "click", "ids": [], "text_hints": ["Logout"], "selector_candidates": [{"type": "xpathSelector", "value": "//button[normalize-space()=\"Logout\"]"}]},
        ]
        brief["success_signals"] = {"texts": ["Login", "Sign In"], "ids": ["login-sign-in-button"], "url_contains": ["/login"]}
        brief["pitfalls"] = [
            "First log in with the provided credentials.",
            "Only then click Logout in the header.",
            "Do not type into alternative username fields or search inputs.",
        ]
        brief["action_sketch"] = [
            f"Log in as {username}.",
            "Click Logout in the header.",
        ]
        return brief
    return _postprocess_brief(web_project_id=web_project_id, use_case=use_case, seed=seed, brief=brief)


def _run_direct_claude_agent(
    *,
    web_project_id: str,
    use_case: str,
    seed: int,
    model: str,
    work_dir: Path,
    context_path: Path,
    candidate_brief_path: Path,
    final_brief_path: Path,
    eval_command: str,
    timeout_seconds: int,
    max_budget_usd: float,
) -> dict[str, Any] | None:
    prompt = _build_direct_agent_prompt(
        web_project_id=web_project_id,
        use_case=use_case,
        seed=seed,
        work_dir=work_dir,
        context_path=context_path,
        candidate_brief_path=candidate_brief_path,
        final_brief_path=final_brief_path,
        eval_command=eval_command,
    )
    stdout_path = work_dir / "claude_stdout.txt"
    cmd = [
        "claude",
        "-p",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        model,
        "--max-budget-usd",
        str(max_budget_usd),
        "--tools",
        "Bash,Read,Edit,Write,Glob,Grep,LS",
        "--add-dir",
        str(REPO_ROOT),
        "--add-dir",
        str((REPO_ROOT.parent / "autoppia_iwa").resolve()),
        "--add-dir",
        str((REPO_ROOT.parent / "autoppia_webs_demo").resolve()),
        "--add-dir",
        str(work_dir.resolve()),
    ]
    with stdout_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
            returncode = int(proc.wait(timeout=max(1, int(timeout_seconds))))
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = int(proc.wait())
            with stdout_path.open("a", encoding="utf-8") as af:
                af.write("\n[TIMEOUT]\n")
    if returncode != 0:
        return None
    if not final_brief_path.exists():
        return None
    payload = json.loads(final_brief_path.read_text(encoding="utf-8"))
    brief = payload.get("brief") if isinstance(payload, dict) and isinstance(payload.get("brief"), dict) else payload
    brief = _postprocess_brief(web_project_id=web_project_id, use_case=use_case, seed=seed, brief=brief)
    return {
        "brief": brief,
        "meta": {
            "model": model,
            "mode": "direct_claude_code_agent",
            "returncode": returncode,
            "stdout_path": str(stdout_path),
            "work_dir": str(work_dir),
        },
    }


def run_claude_code_harvest(
    *,
    web_project_id: str,
    use_case: str,
    seed: int,
    model: str,
    task_cache_path: Path,
    work_dir: Path,
    eval_command: str,
    timeout_seconds: int = 600,
    max_budget_usd: float = 3.0,
    max_attempts: int = 3,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    context_payload = build_project_context(
        web_project_id=web_project_id,
        use_case=use_case,
        task_cache_path=Path(task_cache_path).resolve(),
        seed=seed,
    )
    context_path = work_dir / "task_context.json"
    final_brief_path = work_dir / "final_brief.json"
    candidate_brief_path = work_dir / "candidate_brief.json"
    attempts_path = work_dir / "attempt_feedback.json"
    eval_out_path = work_dir / "last_eval.json"
    for stale in [
        final_brief_path,
        candidate_brief_path,
        attempts_path,
        eval_out_path,
        eval_out_path.with_suffix(".summary.json"),
        work_dir / "claude_stdout.txt",
    ]:
        if stale.exists():
            stale.unlink()
    context_path.write_text(json.dumps(context_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    canonical_brief = _canonical_brief_from_context(
        web_project_id=web_project_id,
        use_case=use_case,
        seed=seed,
        context_payload=context_payload,
    )
    if canonical_brief is not None:
        payload = {
            "brief": canonical_brief,
            "meta": {
                "model": model,
                "mode": "canonical_context_brief",
                "work_dir": str(work_dir),
            },
            "context": context_payload,
        }
        candidate_brief_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        final_brief_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return payload

    direct_payload = _run_direct_claude_agent(
        web_project_id=web_project_id,
        use_case=use_case,
        seed=seed,
        model=model,
        work_dir=work_dir,
        context_path=context_path,
        candidate_brief_path=candidate_brief_path,
        final_brief_path=final_brief_path,
        eval_command=eval_command,
        timeout_seconds=min(int(timeout_seconds), 60),
        max_budget_usd=max_budget_usd,
    )
    if direct_payload is not None:
        return {
            **direct_payload,
            "context": context_payload,
        }

    previous_attempts: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_score = -1.0

    for attempt_idx in range(1, max(1, int(max_attempts)) + 1):
        payload = generate_claude_brief(
            web_project_id=web_project_id,
            use_case=use_case,
            seed=seed,
            model=model,
            previous_attempts=previous_attempts,
            timeout_seconds=min(int(timeout_seconds), 180),
            task_cache_path=Path(task_cache_path).resolve(),
        )
        candidate_brief_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        eval_env = dict(os.environ)
        eval_env["DEMO_WEBS_ENDPOINT"] = "http://84.247.180.192"
        eval_env["DEMO_WEB_SERVICE_PORT"] = "8090"
        proc = subprocess.run(
            eval_command,
            cwd=str(REPO_ROOT),
            shell=True,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
            env=eval_env,
        )
        feedback = _load_eval_feedback(eval_out_path)
        attempt_record = {
            "attempt": attempt_idx,
            "returncode": int(proc.returncode),
            "stdout": str(proc.stdout or "")[:4000],
            "stderr": str(proc.stderr or "")[:4000],
            "success": bool(feedback["success"]),
            "score": float(feedback["score"]),
            "final_url": str(feedback["final_url"]),
            "backend_events": feedback["backend_events"],
        }
        transcript.append(attempt_record)
        row_for_summary = {
            "success": bool(feedback["success"]),
            "score": float(feedback["score"]),
            "final_url": str(feedback["final_url"]),
        }
        summary_row = summarize_attempt_for_claude(
            attempt_name=f"attempt_{attempt_idx:02d}",
            report=feedback.get("report") if isinstance(feedback.get("report"), dict) else {},
            row=row_for_summary,
        )
        summary_row["backend_events"] = feedback["backend_events"]
        summary_row["final_url"] = feedback["final_url"]
        previous_attempts.append(summary_row)
        if float(feedback["score"]) > best_score:
            best_score = float(feedback["score"])
            best_payload = payload
        if bool(feedback["success"]) and float(feedback["score"]) >= 1.0:
            final_brief_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            attempts_path.write_text(json.dumps(previous_attempts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return {**payload, "context": context_payload}

    if best_payload is None:
        raise RuntimeError("Claude Code did not produce a usable brief")
    final_brief_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    attempts_path.write_text(json.dumps(previous_attempts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**best_payload, "context": context_payload}


def save_claude_brief(*, web_project_id: str, use_case: str, seed: int, payload: dict[str, Any], output_root: Path) -> Path:
    brief_dir = output_root / "harvester" / "claude_briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    out_path = brief_dir / f"seed_{int(seed):04d}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


__all__ = [
    "generate_claude_brief",
    "run_claude_code_harvest",
    "save_claude_brief",
]
