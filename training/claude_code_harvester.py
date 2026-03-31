from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from training.harvester_support import brief_prompt_lines, summarize_attempt_for_claude
from training.layout import use_case_layout

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_CACHE_PATH = REPO_ROOT.parent / "autoppia_rl" / "data" / "tasks" / "cache" / "autoppia_cinema_tasks.json"
WEB_REPO_ROOT = REPO_ROOT.parent / "autoppia_webs_demo" / "web_1_autocinema"


def focus_root(*, use_case: str) -> Path:
    return use_case_layout(repo_root=REPO_ROOT, web_project="autocinema", use_case=use_case).root


def _load_task_row(use_case: str) -> dict[str, Any]:
    payload = json.loads(TASK_CACHE_PATH.read_text(encoding="utf-8"))
    rows = payload["tasks"] if isinstance(payload, dict) and isinstance(payload.get("tasks"), list) else payload
    if not isinstance(rows, list):
        raise ValueError(f"unexpected task cache format: {TASK_CACHE_PATH}")
    normalized = str(use_case).strip().upper()
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_use_case = ((row.get("use_case") or {}).get("name") if isinstance(row.get("use_case"), dict) else None)
        if str(row_use_case).strip().upper() == normalized:
            return row
    raise ValueError(f"use case not found in task cache: {use_case}")


def _candidate_web_files(use_case: str) -> list[Path]:
    normalized = str(use_case).strip().upper()
    mapping: dict[str, list[str]] = {
        "CONTACT": [
            "src/app/contact/page.tsx",
            "src/components/contact/ContactSection.tsx",
            "src/dynamic/v3/data/id-variants.json",
        ],
        "LOGIN": [
            "src/app/login/page.tsx",
            "src/dynamic/v3/data/id-variants.json",
            "src/data/users.ts",
        ],
        "REGISTRATION": [
            "src/app/register/page.tsx",
            "src/dynamic/v3/data/id-variants.json",
            "src/context/AuthContext.tsx",
        ],
        "ADD_TO_WATCHLIST": [
            "src/app/movies/[movieId]/page.tsx",
            "src/components/movies/MovieDetailHero.tsx",
            "src/dynamic/v3/data/id-variants.json",
            "src/app/login/page.tsx",
        ],
        "ADD_COMMENT": [
            "src/app/movies/[movieId]/page.tsx",
            "src/components/movies/CommentsPanel.tsx",
            "src/dynamic/v3/data/id-variants.json",
        ],
        "ADD_FILM": [
            "src/app/admin/movies/new/page.tsx",
            "src/components/movies/MovieEditor.tsx",
            "src/dynamic/v3/data/id-variants.json",
            "src/app/login/page.tsx",
        ],
    }
    default_files = [
        "src/dynamic/v3/data/id-variants.json",
        "src/app/page.tsx",
    ]
    rels = mapping.get(normalized, default_files)
    return [WEB_REPO_ROOT / rel for rel in rels if (WEB_REPO_ROOT / rel).exists()]


def _use_case_keywords(use_case: str) -> list[str]:
    normalized = str(use_case).strip().upper()
    mapping: dict[str, list[str]] = {
        "CONTACT": ["contact", "name", "email", "subject", "message", "send", "submit", "sent", "success"],
        "LOGIN": ["login", "email", "password", "submit", "sign in", "success"],
        "REGISTRATION": ["register", "name", "email", "password", "confirm", "submit", "success"],
        "ADD_TO_WATCHLIST": ["watchlist", "save", "add", "login"],
        "ADD_COMMENT": ["comment", "message", "submit", "post"],
        "ADD_FILM": ["movie", "title", "description", "genre", "submit", "save"],
    }
    return mapping.get(normalized, [normalized.lower()])


def _relevant_variant_keys(use_case: str) -> list[str]:
    normalized = str(use_case).strip().upper()
    mapping: dict[str, list[str]] = {
        "CONTACT": [
            "contact-name-input",
            "contact-email-input",
            "contact-subject-input",
            "contact-message-textarea",
            "send-message-button",
        ],
        "LOGIN": [
            "login-username-input",
            "login-password-input",
            "login-sign-in-button",
        ],
        "REGISTRATION": [
            "register-username-input",
            "register-email-input",
            "register-password-input",
            "register-confirm-password-input",
            "register-submit-button",
        ],
        "ADD_TO_WATCHLIST": [
            "watchlist-button",
            "add-watchlist-button",
            "login-sign-in-button",
        ],
        "ADD_COMMENT": [
            "comment-input",
            "comment-submit-button",
        ],
        "ADD_FILM": [
            "movie-title-input",
            "movie-description-input",
            "movie-submit-button",
        ],
    }
    return mapping.get(normalized, [])


def _extract_matching_windows(text: str, *, keywords: list[str], radius: int = 8, max_chars: int = 2500) -> str:
    lines = text.splitlines()
    lowered_keywords = [str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()]
    if not lines or not lowered_keywords:
        return text[:max_chars]
    matched_indexes = [
        idx for idx, line in enumerate(lines) if any(keyword in line.lower() for keyword in lowered_keywords)
    ]
    if not matched_indexes:
        return text[:max_chars]
    ranges: list[tuple[int, int]] = []
    for idx in matched_indexes:
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    chunks: list[str] = []
    total_chars = 0
    for start, end in ranges:
        chunk = "\n".join(lines[start:end]).strip()
        if not chunk:
            continue
        if chunks:
            chunk = "\n...\n" + chunk
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        chunk = chunk[:remaining]
        chunks.append(chunk)
        total_chars += len(chunk)
        if total_chars >= max_chars:
            break
    return "".join(chunks)[:max_chars] or text[:max_chars]


def _compact_id_variants(
    path: Path,
    *,
    keywords: list[str],
    relevant_keys: list[str] | None = None,
    max_entries: int = 24,
    max_chars: int = 2500,
) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.read_text(encoding="utf-8")[:max_chars]
    if not isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False, indent=2)[:max_chars]
    lowered_keywords = [str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()]
    filtered: dict[str, Any] = {}
    exact_keys = [str(key).strip() for key in (relevant_keys or []) if str(key).strip()]
    if exact_keys:
        for key in exact_keys:
            if key in payload:
                filtered[key] = payload[key]
        if filtered:
            return json.dumps(filtered, ensure_ascii=False, indent=2)[:max_chars]
    for key, value in payload.items():
        key_lower = str(key).lower()
        if any(keyword in key_lower for keyword in lowered_keywords):
            filtered[str(key)] = value
        if len(filtered) >= max_entries:
            break
    if not filtered:
        for key, value in payload.items():
            filtered[str(key)] = value
            if len(filtered) >= min(max_entries, 8):
                break
    return json.dumps(filtered, ensure_ascii=False, indent=2)[:max_chars]


def _file_snippets(paths: list[Path], *, use_case: str, max_chars_per_file: int = 1400) -> list[dict[str, str]]:
    keywords = _use_case_keywords(use_case)
    relevant_keys = _relevant_variant_keys(use_case)
    snippets: list[dict[str, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if path.name == "id-variants.json":
            content = _compact_id_variants(
                path,
                keywords=keywords,
                relevant_keys=relevant_keys,
                max_chars=max_chars_per_file,
            )
        else:
            content = _extract_matching_windows(text, keywords=keywords, max_chars=max_chars_per_file)
        snippets.append(
            {
                "path": str(path),
                "content": content,
            }
        )
    return snippets


def _load_existing_examples(use_case: str, max_examples: int = 1) -> list[dict[str, Any]]:
    path = focus_root(use_case=use_case) / "gold" / "episodes.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if not bool(payload.get("success")):
            continue
        summary = {
            "seed": payload.get("seed"),
            "score": payload.get("score"),
            "steps": payload.get("steps"),
            "final_url": payload.get("final_url"),
        }
        result_path = str(payload.get("result_path") or "").strip()
        if result_path:
            try:
                report = json.loads(Path(result_path).read_text(encoding="utf-8"))
                episodes = report.get("episodes") if isinstance(report, dict) else None
                episode = episodes[0] if isinstance(episodes, list) and episodes else None
                guided_execution = episode.get("guided_execution") if isinstance(episode, dict) else None
                if isinstance(guided_execution, list) and guided_execution:
                    summary["guided_actions"] = [
                        item.get("planned_action")
                        for item in guided_execution[:6]
                        if isinstance(item, dict) and isinstance(item.get("planned_action"), dict)
                    ]
            except Exception:
                pass
        trace_file = str(payload.get("trace_file") or "").strip()
        if trace_file:
            try:
                trace_payload = json.loads(Path(trace_file).read_text(encoding="utf-8"))
                steps = trace_payload.get("steps") if isinstance(trace_payload, dict) else None
                if isinstance(steps, list) and steps:
                    summary["trace_actions"] = [
                        step.get("action")
                        for step in steps[:6]
                        if isinstance(step, dict) and isinstance(step.get("action"), dict)
                    ]
            except Exception:
                pass
        rows.append(summary)
        if len(rows) >= max_examples:
            break
    return rows


def _build_prompt(
    use_case: str,
    seed: int,
    task_row: dict[str, Any],
    web_files: list[Path],
    examples: list[dict[str, Any]],
    snippets: list[dict[str, str]],
    previous_attempts: list[dict[str, Any]] | None = None,
) -> str:
    prompt = str(task_row.get("prompt") or "").strip()
    constraints = ((task_row.get("use_case") or {}).get("constraints") if isinstance(task_row.get("use_case"), dict) else None)
    previous_attempts = [row for row in (previous_attempts or []) if isinstance(row, dict)]
    source_hints = {
        "web_repo_root": str(WEB_REPO_ROOT),
        "task_cache_path": str(TASK_CACHE_PATH),
        "focus_data_root": str(focus_root(use_case=use_case)),
        "useful_code_paths": [str(path) for path in web_files],
        "validation_model": {
            "judge": "stateful evaluator",
            "execution": "proposed UI actions are replayed in order against the real demo web",
            "acceptance": "trajectory is gold only if replay reaches success=true and score=1.0",
        },
        "research_goal": {
            "demo_web": "autocinema",
            "use_case": use_case,
            "seed": seed,
            "task": prompt,
        },
    }
    return f"""
You are a code-aware trajectory harvesting teacher for Autocinema.

Goal:
- Produce a concise harvesting brief for use case {use_case} and seed {seed}.
- Use the actual web app code to infer the most reliable route, fields, ids/selectors, and submit action.
- Optimize for finding a successful UI trajectory through the real browser, not for code changes.
- Return explicit executable `steps` whenever you can. The replay system prefers direct action sequences over loose hints.
- Think like a researcher: inspect the local source of the demo web, infer the likely dynamic ids/placeholders/text variants, and use previous successful trajectories from the same use case as inspiration when they help.
- The acceptance criterion is not your confidence. A trajectory is only accepted if the stateful evaluator successfully replays the proposed actions and returns score 1.0.

Task prompt:
{prompt}

Constraints:
{json.dumps(constraints, ensure_ascii=False, indent=2)}

Relevant web files:
{json.dumps([str(p) for p in web_files], ensure_ascii=False, indent=2)}

Code snippets from those files:
{json.dumps(snippets, ensure_ascii=False, indent=2)}

Existing successful examples for the same use case:
{json.dumps(examples, ensure_ascii=False, indent=2)}

Previous failed attempts for the same seed:
{json.dumps(previous_attempts, ensure_ascii=False, indent=2)}

Local source and validation hints:
{json.dumps(source_hints, ensure_ascii=False, indent=2)}

Return strict JSON with this schema:
{{
  "use_case": "{use_case}",
  "seed": {seed},
  "route": ["ordered", "urls or route hints"],
  "prompt_lines": ["short imperative hint 1", "short imperative hint 2"],
  "fields": [
    {{"name": "logical field", "ids": ["candidate-id-1"], "value": "exact text to type when known", "value_rule": "exact value guidance"}}
  ],
  "submit": {{"ids": ["candidate-button-id"], "text": ["button text"], "action": "click submit once"}},
  "success_signals": {{"texts": ["success text"], "ids": ["success-id"], "url_contains": ["path fragment"]}},
  "steps": [
    {{
      "type": "NavigateAction|TypeAction|ClickAction",
      "ids": ["candidate-id-1"],
      "text_hints": ["visible button text when relevant"],
      "text": "exact text to type for TypeAction"
    }}
  ],
  "pitfalls": ["mistake to avoid 1"],
  "action_sketch": ["step 1", "step 2", "step 3"],
  "confidence": 0.0
}}

Rules:
- Keep prompt_lines short and directly usable as extra harvesting hints.
- Use previous failed attempts to revise the plan. If an earlier attempt clicked the wrong link or filled a bad value, explicitly correct that.
- If previous failed attempts say the agent was still not on `/contact` or another target route, make the first step an explicit navigation to the target page.
- If previous failed attempts include `candidate_actions` or `guided_attempts_recent`, use them to avoid repeating the same broken sequence.
- Prefer code-backed ids and route hints over vague language.
- If the task requires a negative constraint like subject not containing a word, state an explicit safe subject example.
- Do not suggest direct backend calls or code edits.
- For form tasks, prefer a direct explicit `steps` sequence: navigate -> fill each field -> click submit.
""".strip()


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
        "required": ["use_case", "seed", "route", "prompt_lines", "fields", "submit", "success_signals", "pitfalls", "action_sketch", "confidence"],
    }


def generate_claude_brief(
    *,
    use_case: str,
    seed: int,
    model: str = "claude-sonnet-4-5",
    previous_attempts: list[dict[str, Any]] | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    task_row = _load_task_row(use_case)
    web_files = _candidate_web_files(use_case)
    examples = _load_existing_examples(use_case)
    snippets = _file_snippets(web_files, use_case=use_case)
    prompt = _build_prompt(
        use_case=use_case,
        seed=seed,
        task_row=task_row,
        web_files=web_files,
        examples=examples,
        snippets=snippets,
        previous_attempts=previous_attempts,
    )
    schema = _brief_schema()
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        model,
        "--max-budget-usd",
        "1",
        "--json-schema",
        json.dumps(schema, ensure_ascii=False),
        prompt,
    ]
    raw = subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True, timeout=max(1, int(timeout_seconds)))
    payload = json.loads(raw)
    structured = payload.get("structured_output")
    if isinstance(structured, dict):
        result = structured
    else:
        result = json.loads(str(payload.get("result") or "{}"))
    return {
        "brief": result,
        "meta": {
            "model": model,
            "total_cost_usd": float(payload.get("total_cost_usd") or 0.0),
            "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
            "session_id": payload.get("session_id"),
        },
        "web_files": [str(p) for p in web_files],
        "snippets": snippets,
        "examples": examples,
    }


def save_claude_brief(*, use_case: str, seed: int, payload: dict[str, Any], output_root: Path) -> Path:
    brief_dir = output_root / "harvester" / "claude_briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    out_path = brief_dir / f"seed_{int(seed):04d}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


__all__ = [
    "brief_prompt_lines",
    "generate_claude_brief",
    "save_claude_brief",
    "summarize_attempt_for_claude",
]
