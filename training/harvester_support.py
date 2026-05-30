from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ID_VARIANTS = REPO_ROOT.parent / "autoppia_webs_demo" / "web_1_autocinema" / "src" / "dynamic" / "v3" / "data" / "id-variants.json"


def brief_prompt_lines(payload: dict[str, Any] | None) -> list[str]:
    brief = payload.get("brief") if isinstance(payload, dict) else None
    if not isinstance(brief, dict):
        return []
    lines: list[str] = []
    route = [str(item).strip() for item in (brief.get("route") or []) if str(item).strip()]
    if route:
        lines.append("Follow this route exactly: " + " -> ".join(route))
    for line in brief.get("prompt_lines") or []:
        text = str(line).strip()
        if text:
            lines.append(text)
    for field in brief.get("fields") or []:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or "").strip()
        ids = [str(item).strip() for item in (field.get("ids") or []) if str(item).strip()]
        value = str(field.get("value") or "").strip()
        value_rule = str(field.get("value_rule") or "").strip()
        parts = []
        if name:
            parts.append(f"For {name}")
        if ids:
            parts.append("prefer ids " + ", ".join(ids[:5]))
        if value:
            parts.append(f"use exact value {value!r}")
        if value_rule:
            parts.append(value_rule)
        if parts:
            lines.append("; ".join(parts))
    submit = brief.get("submit")
    if isinstance(submit, dict):
        ids = [str(item).strip() for item in (submit.get("ids") or []) if str(item).strip()]
        text = [str(item).strip() for item in (submit.get("text") or []) if str(item).strip()]
        action = str(submit.get("action") or "").strip()
        submit_parts = ["When the workflow is complete"]
        if ids:
            submit_parts.append("click submit using ids " + ", ".join(ids[:5]))
        if text:
            submit_parts.append("or visible text " + ", ".join(text[:5]))
        if action:
            submit_parts.append(action)
        lines.append("; ".join(submit_parts))
    success_signals = brief.get("success_signals")
    if isinstance(success_signals, dict):
        texts = [str(item).strip() for item in (success_signals.get("texts") or []) if str(item).strip()]
        ids = [str(item).strip() for item in (success_signals.get("ids") or []) if str(item).strip()]
        fragments = [str(item).strip() for item in (success_signals.get("url_contains") or []) if str(item).strip()]
        signal_parts = ["Success means"]
        if texts:
            signal_parts.append("text appears: " + ", ".join(texts[:5]))
        if ids:
            signal_parts.append("success ids: " + ", ".join(ids[:5]))
        if fragments:
            signal_parts.append("url contains: " + ", ".join(fragments[:5]))
        if len(signal_parts) > 1:
            lines.append("; ".join(signal_parts))
    for line in brief.get("action_sketch") or []:
        text = str(line).strip()
        if text:
            lines.append(f"Action sketch: {text}")
    for line in brief.get("pitfalls") or []:
        text = str(line).strip()
        if text:
            lines.append(f"Avoid this mistake: {text}")
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = " ".join(line.split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def summarize_attempt_for_claude(
    *,
    attempt_name: str,
    report: dict[str, Any],
    row: dict[str, Any] | None,
) -> dict[str, Any]:
    episode = ((report.get("episodes") or [{}])[0]) if isinstance(report, dict) else {}
    if not isinstance(episode, dict):
        episode = {}
    summary: dict[str, Any] = {
        "attempt_name": str(attempt_name or "").strip(),
        "success": bool(episode.get("success")),
        "score": float(episode.get("score") or 0.0),
        "steps": int(episode.get("steps") or 0),
        "final_url": str(episode.get("final_url") or row.get("final_url") if isinstance(row, dict) else episode.get("final_url") or "").strip(),
        "final_content_excerpt": str(episode.get("final_content") or "")[:800],
        "model": str(episode.get("model") or report.get("model") or ""),
        "estimated_cost_usd": float(episode.get("estimated_cost_usd") or report.get("estimated_cost_usd") or 0.0),
    }
    if isinstance(row, dict):
        candidate_file = str(row.get("candidate_path") or "").strip()
        if candidate_file:
            summary["candidate_path"] = candidate_file
            try:
                candidate_payload = json.loads(Path(candidate_file).read_text(encoding="utf-8"))
                actions = candidate_payload.get("actions")
                if isinstance(actions, list):
                    summary["candidate_actions"] = actions[:8]
            except Exception:
                pass
    final_content = str(episode.get("final_content") or "")
    if final_content:
        summary["final_content_excerpt"] = final_content[:800]
        lowered = final_content.lower()
        hints: list[str] = []
        if "/contact" in lowered and "navigate" in lowered:
            hints.append("not_on_target_page")
        if "fill" in lowered and "contact form" in lowered:
            hints.append("form_not_completed")
        if "submit" in lowered:
            hints.append("submit_missing")
        if hints:
            summary["failure_hints"] = hints
    trace_file = Path(str((row or {}).get("trace_file") or "")).resolve() if isinstance(row, dict) and (row or {}).get("trace_file") else None
    if trace_file and trace_file.exists():
        try:
            trace_payload = json.loads(trace_file.read_text(encoding="utf-8"))
            step_traces = trace_payload.get("step_traces") if isinstance(trace_payload, dict) else None
            if isinstance(step_traces, list):
                recent_steps: list[dict[str, Any]] = []
                for step in step_traces[-4:]:
                    if not isinstance(step, dict):
                        continue
                    actions = [str((item or {}).get("type") or "") for item in (step.get("actions") or []) if isinstance(item, dict)]
                    execution = step.get("execution") if isinstance(step.get("execution"), dict) else {}
                    recent_steps.append(
                        {
                            "step_index": int(step.get("step_index") or 0),
                            "actions": actions,
                            "exec_ok": bool(execution.get("exec_ok", True)),
                            "error": str(execution.get("error") or "")[:200],
                        }
                    )
                summary["recent_steps"] = recent_steps
        except Exception:
            pass
    guided_episodes = report.get("episodes") if isinstance(report, dict) else None
    guided_episode = guided_episodes[0] if isinstance(guided_episodes, list) and guided_episodes else None
    guided_execution = guided_episode.get("guided_execution") if isinstance(guided_episode, dict) else None
    if isinstance(guided_execution, list) and guided_execution:
        recent_attempts: list[dict[str, Any]] = []
        for item in guided_execution[-4:]:
            if not isinstance(item, dict):
                continue
            recent_attempts.append(
                {
                    "planned_action": item.get("planned_action") if isinstance(item.get("planned_action"), dict) else {},
                    "selected_action": item.get("selected_action") if isinstance(item.get("selected_action"), dict) else {},
                    "success": bool(item.get("success")),
                    "score": float(item.get("score") or 0.0),
                    "url": str(item.get("url") or ""),
                }
            )
        summary["guided_attempts_recent"] = recent_attempts
    return summary


def _base_origin(task_url: str) -> str:
    parsed = urlparse(str(task_url))
    return f"{parsed.scheme}://{parsed.netloc}"


def _seeded_url(task_url: str, route_hint: str) -> str:
    parsed = urlparse(str(task_url))
    origin = _base_origin(task_url)
    hint = str(route_hint).strip()
    match = re.search(r"(\/[A-Za-z0-9_\/\-]+)", hint)
    if match:
        hint = match.group(1)
    if hint.startswith("http://") or hint.startswith("https://"):
        return hint
    path = hint if hint.startswith("/") else f"/{hint}"
    seed = parsed.query
    query = f"?{seed}" if seed else ""
    return f"{origin}{path}{query}"


def _value_from_rule(field: dict[str, Any]) -> str:
    value = str(field.get("value") or "").strip()
    if value:
        return value
    rule = str(field.get("value_rule") or "").strip()
    name = str(field.get("name") or "").strip().lower()
    quoted = re.findall(r"'([^']+)'", rule)
    if "does not contain" in rule.lower() and name == "subject":
        return "General inquiry"
    if "must contain this string" in rule.lower() and quoted:
        return quoted[0]
    if "enter exactly" in rule.lower() and quoted:
        return quoted[0]
    if quoted:
        return quoted[0]
    return ""


def _selector_candidates(
    *,
    ids: list[str] | None = None,
    classes: list[str] | None = None,
    placeholders: list[str] | None = None,
    texts: list[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in ids or []:
        value = str(value).strip()
        key = f"id:{value.lower()}"
        if value and key not in seen:
            seen.add(key)
            out.append({"type": "attributeValueSelector", "attribute": "id", "value": value, "case_sensitive": False})
    for value in classes or []:
        value = str(value).strip()
        key = f"class:{value.lower()}"
        if value and key not in seen:
            seen.add(key)
            out.append({"type": "attributeValueSelector", "attribute": "class", "value": value, "case_sensitive": False})
    for value in placeholders or []:
        value = str(value).strip()
        key = f"placeholder:{value.lower()}"
        if value and key not in seen:
            seen.add(key)
            out.append({"type": "attributeValueSelector", "attribute": "placeholder", "value": value, "case_sensitive": False})
    for value in texts or []:
        value = str(value).strip()
        key = f"text:{value.lower()}"
        if value and key not in seen:
            seen.add(key)
            out.append({"type": "tagContainsSelector", "value": value, "case_sensitive": False})
    return out


@lru_cache(maxsize=1)
def _load_id_variants() -> dict[str, list[str]]:
    if not WEB_ID_VARIANTS.exists():
        return {}
    try:
        payload = json.loads(WEB_ID_VARIANTS.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    if not isinstance(payload, dict):
        return out
    for key, value in payload.items():
        if not isinstance(value, list):
            continue
        variants = [str(item).strip() for item in value if str(item).strip()]
        out[str(key).strip()] = variants
    return out


def _expand_id_variants(ids: list[str] | None) -> list[str]:
    variants_map = _load_id_variants()
    out: list[str] = []
    seen: set[str] = set()
    for value in ids or []:
        candidate = str(value).strip()
        if not candidate:
            continue
        for item in [candidate, *variants_map.get(candidate, [])]:
            normalized = str(item).strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                out.append(normalized)
    return out


def _success_signal_hit(*, html: str, url: str, brief: dict[str, Any]) -> bool:
    signals = brief.get("success_signals")
    if not isinstance(signals, dict):
        return False
    html_lower = str(html or "").lower()
    url_lower = str(url or "").lower()
    texts = [str(item).strip().lower() for item in (signals.get("texts") or []) if str(item).strip()]
    ids = [str(item).strip().lower() for item in (signals.get("ids") or []) if str(item).strip()]
    fragments = [str(item).strip().lower() for item in (signals.get("url_contains") or []) if str(item).strip()]
    if any(text in html_lower for text in texts):
        return True
    if any(f'id="{value}"' in html_lower for value in ids):
        return True
    if any(fragment in url_lower for fragment in fragments):
        return True
    return False


def _guided_actions_from_brief(*, task_url: str, brief: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    route = [str(item).strip() for item in (brief.get("route") or []) if str(item).strip()]
    if route:
        actions.append({"type": "NavigateAction", "url": _seeded_url(task_url, route[0]), "go_back": False, "go_forward": False})
    for field in brief.get("fields") or []:
        if not isinstance(field, dict):
            continue
        value = _value_from_rule(field)
        ids = _expand_id_variants([str(item).strip() for item in (field.get("ids") or []) if str(item).strip()])
        if not value or not ids:
            continue
        actions.append(
            {
                "type": "TypeAction",
                "selector_candidates": _selector_candidates(ids=ids),
                "text": value,
                "field_name": str(field.get("name") or "").strip(),
            }
        )
    submit = brief.get("submit")
    if isinstance(submit, dict):
        ids = _expand_id_variants([str(item).strip() for item in (submit.get("ids") or []) if str(item).strip()])
        texts = [str(item).strip() for item in (submit.get("text") or []) if str(item).strip()]
        candidates = _selector_candidates(ids=ids, texts=texts)
        if candidates:
            actions.append({"type": "ClickAction", "selector_candidates": candidates, "field_name": "submit"})
    return actions


__all__ = [
    "_expand_id_variants",
    "_guided_actions_from_brief",
    "_success_signal_hit",
    "brief_prompt_lines",
    "summarize_attempt_for_claude",
]


def flatten_backend_events_from_report(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    episodes = report.get("episodes") if isinstance(report, dict) else None
    episode = episodes[0] if isinstance(episodes, list) and episodes else None
    flat: list[dict[str, Any]] = []
    if isinstance(episode, dict):
        seen = episode.get("backend_events_seen")
        if isinstance(seen, list):
            for item in seen:
                if isinstance(item, list):
                    flat.extend([ev for ev in item if isinstance(ev, dict)])
                elif isinstance(item, dict):
                    flat.append(item)
        guided = episode.get("guided_execution")
        if isinstance(guided, list):
            for step in guided:
                if not isinstance(step, dict):
                    continue
                for ev in step.get("backend_events") or []:
                    if isinstance(ev, dict):
                        flat.append(ev)
    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for ev in flat:
        key = json.dumps(ev, sort_keys=True, ensure_ascii=False)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(ev)
    return deduped


def semantic_event_validation(*, task_row: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        from autoppia_iwa.src.demo_webs.base_events import Event
        from autoppia_iwa.src.demo_webs.classes import BackendEvent
        from autoppia_iwa.src.data_generation.tests.classes import CheckEventTest
    except Exception as exc:
        return {"success": False, "reason": f"import_error:{exc}", "matched_tests": 0, "total_tests": 0}

    backend_dicts = flatten_backend_events_from_report(report)
    backend_events = [BackendEvent(**ev) for ev in backend_dicts if isinstance(ev, dict)]
    parsed_events = Event.parse_all(backend_events)
    raw_tests = task_row.get("tests") if isinstance(task_row, dict) else None
    tests = [t for t in (raw_tests or []) if isinstance(t, dict) and (t.get("type") == "CheckEventTest" or "CheckEvent" in str(t.get("type") or ""))]
    matched = 0
    for raw in tests:
        test = CheckEventTest(
            type="CheckEventTest",
            event_name=str(raw.get("event_name") or ""),
            event_criteria=raw.get("event_criteria") or {},
            description=str(raw.get("description") or "Check if specific event was triggered"),
        )
        ok = False
        for ev in parsed_events:
            if ev.event_name != test.event_name:
                continue
            try:
                criteria = ev.ValidationCriteria(**test.event_criteria)
            except Exception:
                continue
            if ev.validate_criteria(criteria):
                ok = True
                break
        if ok:
            matched += 1
    return {
        "success": bool(tests) and matched == len(tests),
        "matched_tests": matched,
        "total_tests": len(tests),
        "backend_events": backend_dicts[:20],
    }
