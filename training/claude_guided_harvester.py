from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import autoppia_iwa.src.execution.actions.actions  # noqa: F401
from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.execution.actions.base import BaseAction
from autoppia_iwa.src.web_agents.classes import replace_credential_placeholders_in_string

import training.harvester_support as harvester_support
from training.deterministic_harvester.normalizer import extract_seed_from_task_url
from training.deterministic_harvester.projects import resolve_project_id
from training.deterministic_harvester.resolvers import dataset_movie_candidates, resolve_movie_detail_url

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_CACHE = REPO_ROOT.parent / "autoppia_rl" / "data" / "tasks" / "cache" / "autoppia_cinema_tasks.json"
WEB_ID_VARIANTS = REPO_ROOT.parent / "autoppia_webs_demo" / "web_1_autocinema" / "src" / "dynamic" / "v3" / "data" / "id-variants.json"


def _load_raw_tasks(cache_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    rows: Any
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        rows = payload["tasks"]
    elif isinstance(payload, dict):
        nested_rows: list[dict[str, Any]] = []
        for value in payload.values():
            if isinstance(value, dict) and isinstance(value.get("tasks"), list):
                nested_rows.extend(row for row in value["tasks"] if isinstance(row, dict))
        rows = nested_rows if nested_rows else payload
    else:
        rows = payload
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _sanitize_task_row_for_replay(row: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(row)
    use_case_payload = sanitized.get("use_case")
    use_case_name = str(use_case_payload.get("name") or "").strip().upper() if isinstance(use_case_payload, dict) else ""
    if not use_case_name:
        return sanitized
    auth_use_cases = {"LOGIN", "LOGOUT", "REGISTRATION"}
    seed = extract_seed_from_task_url(str(sanitized.get("url") or "")) or 1
    web_agent_id = _guided_web_agent_id(seed)
    sanitized = _render_placeholders(sanitized, web_agent_id)
    tests = sanitized.get("tests")
    if not isinstance(tests, list):
        return sanitized
    for test in tests:
        if not isinstance(test, dict):
            continue
        if str(test.get("event_name") or "").strip().upper() != use_case_name:
            continue
        criteria = test.get("event_criteria")
        if not isinstance(criteria, dict):
            continue
        criteria.pop("password", None)
        if use_case_name not in auth_use_cases:
            criteria.pop("username", None)
    return sanitized


def _load_tasks(*, cache_path: Path, use_case: str, web_project_id: str, limit: int | None = 1) -> list[Task]:
    tasks: list[Task] = []
    for row in _load_raw_tasks(cache_path):
        normalized_row = _sanitize_task_row_for_replay(row)
        uc_payload = row.get("use_case")
        uc_name = str(uc_payload.get("name") or "") if isinstance(uc_payload, dict) else ""
        if use_case and str(use_case).upper() not in uc_name.upper():
            continue
        if web_project_id and str(normalized_row.get("web_project_id") or "") != str(web_project_id):
            continue
        try:
            tasks.append(Task(**normalized_row))
        except Exception:
            continue
        if limit is not None and int(limit) > 0 and len(tasks) >= int(limit):
            break
    return tasks


def _inject_seed(task: Task, seed: int) -> tuple[Task, int]:
    cloned = deepcopy(task)
    seed_i = int(seed)
    base_url = cloned.url.split("?")[0] if "?" in cloned.url else cloned.url
    cloned.url = f"{base_url}?seed={seed_i}"
    return cloned, seed_i


def _task_for_seed(*, use_case: str, seed: int | None = None, task_cache: Path | None = None, web_project_id: str | None = None):
    cache_path = Path(task_cache).resolve() if task_cache else Path(DEFAULT_TASK_CACHE).resolve()
    tasks = _load_tasks(
        cache_path=cache_path,
        use_case=use_case,
        web_project_id=resolve_project_id(explicit_project_id=web_project_id),
        limit=None,
    )
    if not tasks:
        raise ValueError(f"No task found for use_case={use_case} in {cache_path}")
    if seed is None:
        return tasks[0]
    seed_i = int(seed)
    for task in tasks:
        if extract_seed_from_task_url(str(task.url)) == seed_i:
            return task
    task, _ = _inject_seed(tasks[0], seed=seed_i)
    return task


def _guided_web_agent_id(seed: int) -> str:
    seed_i = int(seed)
    if 1 <= seed_i <= 255:
        return str(seed_i)
    normalized = ((seed_i - 1) % 255) + 1
    return str(normalized)


def _normalize_action_url(*, task_url: str, target_url: str) -> str:
    raw_target = str(target_url or "").strip()
    if not raw_target:
        return raw_target
    absolute = urljoin(task_url, raw_target)
    parsed_task = urlparse(str(task_url))
    parsed_target = urlparse(absolute)
    task_query = dict(parse_qsl(parsed_task.query or "", keep_blank_values=True))
    target_query = dict(parse_qsl(parsed_target.query or "", keep_blank_values=True))
    if task_query.get("seed") and not target_query.get("seed"):
        target_query["seed"] = task_query["seed"]
    return urlunparse(
        (
            parsed_target.scheme,
            parsed_target.netloc,
            parsed_target.path,
            parsed_target.params,
            urlencode(target_query),
            parsed_target.fragment,
        )
    )


def _dataset_movie_candidates(*, task_url: str, filters: dict[str, Any], web_project_id: str = "autocinema") -> list[str]:
    return dataset_movie_candidates(task_url=task_url, filters=filters, web_project_id=web_project_id)


def _sanitize_type_ids(ids: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    blocked_tokens = ("button", "submit", "action", "toggle", "link")
    for value in ids or []:
        candidate = str(value).strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if any(token in lowered for token in blocked_tokens):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(candidate)
    return out


def _guided_actions_from_brief(*, task_url: str, brief: dict[str, Any], web_project_id: str = "autocinema") -> list[dict[str, Any]]:
    explicit_steps = brief.get("steps")
    if isinstance(explicit_steps, list) and explicit_steps:
        actions: list[dict[str, Any]] = []
        for item in explicit_steps:
            if not isinstance(item, dict):
                continue
            payload = deepcopy(item)
            action_type = str(payload.get("type") or "").strip()
            if action_type == "NavigateAction":
                payload["url"] = _normalize_action_url(task_url=task_url, target_url=str(payload.get("url") or ""))
            if action_type in {"TypeAction", "ClickAction"}:
                ids = harvester_support._expand_id_variants([str(entry).strip() for entry in (payload.get("ids") or []) if str(entry).strip()])
                if action_type == "TypeAction":
                    ids = _sanitize_type_ids(ids)
                texts = [str(entry).strip() for entry in (payload.get("text_hints") or []) if str(entry).strip()]
                if ids or texts:
                    payload["selector_candidates"] = harvester_support._selector_candidates(ids=ids, texts=texts)
                payload.pop("ids", None)
                payload.pop("text_hints", None)
            actions.append(payload)
        if actions:
            return actions

    actions: list[dict[str, Any]] = list(harvester_support._guided_actions_from_brief(task_url=task_url, brief=brief))
    discover = brief.get("discover_target")
    if isinstance(discover, dict):
        kind = str(discover.get("kind") or "").strip().lower()
        if kind == "movie_detail":
            filters = discover.get("filters") if isinstance(discover.get("filters"), dict) else {}
            resolved_url = resolve_movie_detail_url(
                task_url=task_url,
                filters=filters,
                web_project_id=web_project_id,
            )
            if resolved_url:
                actions.append(
                    {
                        "type": "NavigateAction",
                        "url": resolved_url,
                        "go_back": False,
                        "go_forward": False,
                    }
                )
            else:
                actions.append(
                    {
                        "type": "ClickAction",
                        "selector_candidates": harvester_support._selector_candidates(
                            ids=harvester_support._expand_id_variants(["view-details-button"]),
                            texts=["View Details"],
                        ),
                        "field_name": "movie_detail",
                    }
                )
    for field in brief.get("fields") or []:
        if not isinstance(field, dict):
            continue
        value = harvester_support._value_from_rule(field)
        ids = harvester_support._expand_id_variants([str(item).strip() for item in (field.get("ids") or []) if str(item).strip()])
        if not value or not ids:
            continue
        actions.append(
            {
                "type": "TypeAction",
                "selector_candidates": harvester_support._selector_candidates(ids=ids),
                "text": value,
                "field_name": str(field.get("name") or "").strip(),
            }
        )
    submit = brief.get("submit")
    if isinstance(submit, dict):
        ids = harvester_support._expand_id_variants([str(item).strip() for item in (submit.get("ids") or []) if str(item).strip()])
        texts = [str(item).strip() for item in (submit.get("text") or []) if str(item).strip()]
        candidates = harvester_support._selector_candidates(ids=ids, texts=texts)
        if candidates:
            actions.append({"type": "ClickAction", "selector_candidates": candidates, "field_name": "submit"})
    return actions


def _expand_id_variants(ids: list[str] | None) -> list[str]:
    return harvester_support._expand_id_variants(ids)


def _success_signal_hit(*, html: str, url: str, brief: dict[str, Any]) -> bool:
    return harvester_support._success_signal_hit(html=html, url=url, brief=brief)


def _render_placeholders(payload: Any, web_agent_id: str) -> Any:
    if isinstance(payload, str):
        return replace_credential_placeholders_in_string(payload, web_agent_id)
    if isinstance(payload, list):
        return [_render_placeholders(item, web_agent_id) for item in payload]
    if isinstance(payload, dict):
        return {key: _render_placeholders(value, web_agent_id) for key, value in payload.items()}
    return payload


def _ordered_selector_candidates(
    planned_action: dict[str, Any],
    resolved_candidates: list[dict[str, Any]] | None = None,
    existing_exact_candidates: list[dict[str, Any]] | None = None,
    *,
    exact_match_found: bool = False,
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    explicit = list(planned_action.get("selector_candidates") or [])
    existing_exact = list(existing_exact_candidates or [])
    if existing_exact:
        selector_stream = existing_exact + explicit + [selector for selector in list(resolved_candidates or []) if selector not in existing_exact]
    elif explicit:
        selector_stream = explicit + list(resolved_candidates or [])
    elif resolved_candidates and not exact_match_found:
        selector_stream = list(resolved_candidates)
    else:
        selector_stream = explicit + list(resolved_candidates or [])
    for selector in selector_stream:
        if not isinstance(selector, dict):
            continue
        try:
            key = json.dumps(selector, sort_keys=True)
        except Exception:
            key = str(selector)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(selector)
    return ordered


async def _execute_action_candidates(session, planned_action: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    action_type = str(planned_action.get("type") or "").strip()
    if action_type == "NavigateAction":
        normalized_action = dict(planned_action)
        normalized_action["url"] = _normalize_action_url(
            task_url=str(getattr(session, "task", None).url if getattr(session, "task", None) is not None else ""),
            target_url=str(planned_action.get("url") or ""),
        )
        action = BaseAction.create_action(normalized_action)
        result = await session.step(action)
        execution = {
            "planned_action": planned_action,
            "attempts": [{"action": normalized_action, "success": bool(result.action_result.successfully_executed), "error": str(result.action_result.error or "") if result.action_result else ""}],
        }
        return result, execution
    if action_type == "WaitAction":
        cands = list(planned_action.get("selector_candidates") or [])
        single_sel = planned_action.get("selector")
        if single_sel and isinstance(single_sel, dict) and not cands:
            cands = [single_sel]
        ts = planned_action.get("time_seconds")
        if cands:
            w_payload: dict[str, Any] = {
                "type": "WaitAction",
                "selector": cands[0],
                "timeout_seconds": float(planned_action.get("timeout_seconds") or 5.0),
            }
            w_action = BaseAction.create_action(w_payload)
            result = await session.step(w_action)
            return result, {
                "planned_action": planned_action,
                "attempts": [
                    {
                        "action": w_payload,
                        "success": bool(result.action_result.successfully_executed),
                        "error": str(result.action_result.error or "") if result.action_result else "",
                    }
                ],
            }
        if ts is not None:
            wa = {"type": "WaitAction", "time_seconds": float(ts)}
            w_action = BaseAction.create_action(wa)
            result = await session.step(w_action)
            return result, {
                "planned_action": planned_action,
                "attempts": [
                    {
                        "action": wa,
                        "success": bool(result.action_result.successfully_executed),
                        "error": str(result.action_result.error or "") if result.action_result else "",
                    }
                ],
            }
    attempts: list[dict[str, Any]] = []
    resolved_candidates: list[dict[str, Any]] = []
    existing_exact_candidates: list[dict[str, Any]] = []
    exact_match_found = False
    page = getattr(session, "page", None)
    field_name = str(planned_action.get("field_name") or "").strip().lower()
    if page is not None:
        dom_match = None
        for _ in range(6):
            try:
                dom_match = await page.evaluate(
                    """
(payload) => {
  const type = String(payload.type || "");
  const fieldName = String(payload.field_name || "").toLowerCase();
  const idSelector = (id) => id ? ({type: "attributeValueSelector", attribute: "id", value: id, case_sensitive: false}) : null;
  const customSelector = (value) => value ? ({type: "attributeValueSelector", attribute: "custom", value, case_sensitive: false}) : null;
  const normalize = (value) => String(value || "").trim().toLowerCase();
  const textMatch = (value, target) => normalize(value).includes(normalize(target));
  const expandHints = (values) => {
    const seen = new Set();
    const out = [];
    for (const raw of Array.isArray(values) ? values : []) {
      const normalized = normalize(raw);
      if (!normalized) continue;
      const variants = [normalized];
      if (normalized.endsWith("s") && normalized.length > 1) variants.push(normalized.slice(0, -1));
      if (!normalized.endsWith("s")) variants.push(`${normalized}s`);
      for (const item of variants) {
        if (!item || seen.has(item)) continue;
        seen.add(item);
        out.push(item);
      }
    }
    return out;
  };
  const escapeCss = (value) => {
    if (typeof CSS !== "undefined" && CSS && typeof CSS.escape === "function") return CSS.escape(value);
    return String(value || "").replace(/[^a-zA-Z0-9_-]/g, (char) => `\\\\${char}`);
  };
  const selectorForElement = (element) => {
    if (!element || !element.tagName) return null;
    const id = String(element.id || "").trim();
    if (id) {
      const idPath = `#${escapeCss(id)}`;
      try {
        if (document.querySelectorAll(idPath).length === 1) return customSelector(idPath);
      } catch (_) {}
    }
    const parts = [];
    let node = element;
    while (node && node.nodeType === Node.ELEMENT_NODE && node !== document.documentElement) {
      const tag = String(node.tagName || "").toLowerCase();
      if (!tag) break;
      let index = 1;
      let sibling = node.previousElementSibling;
      while (sibling) {
        if (String(sibling.tagName || "").toLowerCase() === tag) index += 1;
        sibling = sibling.previousElementSibling;
      }
      parts.unshift(`${tag}:nth-of-type(${index})`);
      const path = parts.join(" > ");
      try {
        if (path && document.querySelectorAll(path).length === 1) return customSelector(path);
      } catch (_) {}
      node = node.parentElement;
    }
    return null;
  };
  const controls = [...document.querySelectorAll("input, textarea, button")];
  const exactIds = Array.isArray(payload.exact_ids) ? payload.exact_ids.map((value) => String(value || "").trim()).filter(Boolean) : [];
  const labelHints = Array.isArray(payload.label_hints) ? payload.label_hints.map((value) => String(value || "").trim()).filter(Boolean) : [];
  const fieldHints = expandHints([fieldName, ...labelHints]);
  const existingExact = [];

  for (const exactId of exactIds) {
    const element = document.getElementById(exactId);
    if (!element) continue;
    const tag = element.tagName.toLowerCase();
    if (type === "TypeAction" && (tag === "input" || tag === "textarea")) existingExact.push(idSelector(exactId));
    if (type === "ClickAction" && (tag === "button" || tag === "a" || tag === "input")) existingExact.push(idSelector(exactId));
  }

  let heuristic = null;
  if (type === "TypeAction") {
    const labels = [...document.querySelectorAll("label")];
    for (const label of labels) {
      const labelText = (label.textContent || "").trim().toLowerCase();
      if (!labelText || !fieldHints.some((hint) => hint && labelText.includes(hint))) continue;
      const htmlFor = String(label.getAttribute("for") || "");
      const explicit = htmlFor ? document.getElementById(htmlFor) : null;
      const nested = label.querySelector("input, textarea");
      const parent = label.parentElement;
      const sibling = parent ? [...parent.querySelectorAll("input, textarea")].find((el) => !label.contains(el)) : null;
      const control = explicit || nested || sibling || null;
      if (control) {
        heuristic = selectorForElement(control);
        break;
      }
    }
    if (!heuristic) {
      const textNodes = [...document.querySelectorAll("p, span, div, legend")];
      for (const node of textNodes) {
        const text = normalize(node.textContent || "");
        if (!text || !fieldHints.some((hint) => hint && text.includes(hint))) continue;
        const localControl = node.querySelector("input, textarea");
        const parent = node.parentElement;
        const sibling = parent ? [...parent.querySelectorAll("input, textarea")][0] : null;
        const control = localControl || sibling || null;
        if (control) {
          heuristic = selectorForElement(control);
          break;
        }
      }
    }
    if (!heuristic) {
      for (const control of controls) {
        const id = String(control.id || "");
        const placeholder = String(control.getAttribute("placeholder") || "");
        const typeAttr = String(control.getAttribute("type") || "");
        if (fieldName === "email" && typeAttr === "email" && id) {
          heuristic = selectorForElement(control);
          break;
        }
        const tag = control.tagName.toLowerCase();
        if ((fieldName === "message" || fieldName === "comment" || fieldName === "content") && tag === "textarea" && id) {
          heuristic = selectorForElement(control);
          break;
        }
        if ((fieldName === "name" || fieldName === "author" || fieldName === "commenter_name") && tag === "input" && id) {
          heuristic = selectorForElement(control);
          break;
        }
        if ((fieldHints.some((hint) => textMatch(id, hint)) || fieldHints.some((hint) => textMatch(placeholder, hint))) && (id || placeholder)) {
          heuristic = selectorForElement(control);
          break;
        }
      }
    }
  }

    if (type === "ClickAction" && !heuristic) {
    for (const control of controls) {
      const id = String(control.id || "");
      const text = (control.textContent || "").trim();
      if ((labelHints.some((hint) => hint && textMatch(text, hint)) || labelHints.some((hint) => hint && textMatch(id, hint))) && id) {
        heuristic = selectorForElement(control);
        break;
      }
    }
    if (!heuristic && fieldName !== "logout") {
      for (const control of controls) {
        const tag = control.tagName.toLowerCase();
        const typeAttr = String(control.getAttribute("type") || "");
        const id = String(control.id || "");
        if (tag === "button" && typeAttr === "submit" && id) {
          heuristic = selectorForElement(control);
          break;
        }
      }
    }
  }
  return {existing_exact: existingExact, heuristic};
}
                    """,
                    {
                        "type": action_type,
                        "field_name": field_name,
                        "exact_ids": [
                            item.get("value")
                            for item in (planned_action.get("selector_candidates") or [])
                            if isinstance(item, dict) and item.get("type") == "attributeValueSelector" and item.get("attribute") == "id"
                        ],
                        "label_hints": [item.get("value") for item in (planned_action.get("selector_candidates") or []) if isinstance(item, dict) and item.get("type") == "tagContainsSelector"],
                    },
                )
            except Exception:
                dom_match = None
            if isinstance(dom_match, dict):
                existing_exact = dom_match.get("existing_exact") or []
                if existing_exact:
                    exact_match_found = True
                for selector in existing_exact:
                    if isinstance(selector, dict) and selector:
                        existing_exact_candidates.append(selector)
                        resolved_candidates.append(selector)
                heuristic = dom_match.get("heuristic")
                if isinstance(heuristic, dict) and heuristic:
                    resolved_candidates.append(heuristic)
            if resolved_candidates:
                break
            wait_action = BaseAction.create_action({"type": "WaitAction", "time_seconds": 0.35})
            await session.step(wait_action)

    selector_candidates = _ordered_selector_candidates(
        planned_action,
        resolved_candidates,
        existing_exact_candidates,
        exact_match_found=exact_match_found,
    )
    if action_type == "ClickAction" and field_name == "logout":
        selector_candidates = selector_candidates[:6]
    last_result = None
    for selector in selector_candidates:
        payload = {"type": action_type, "selector": selector}
        if action_type == "TypeAction":
            payload["text"] = str(planned_action.get("text") or "")
        elif action_type == "SelectAction":
            payload["value"] = str(planned_action.get("value") or "")
        action = BaseAction.create_action(payload)
        result = await session.step(action)
        action_result = result.action_result
        success = bool(action_result.successfully_executed) if action_result is not None else True
        error = str(action_result.error or "") if action_result is not None and action_result.error else ""
        attempts.append({"action": payload, "success": success, "error": error})
        last_result = result
        if (
            not success
            and action_type == "ClickAction"
            and page is not None
            and field_name != "logout"
            and isinstance(selector, dict)
            and selector.get("type") == "attributeValueSelector"
            and selector.get("attribute") == "id"
            and str(selector.get("value") or "").strip()
        ):
            selector_id = str(selector.get("value") or "").strip()
            js_click_error = ""
            try:
                clicked = await page.evaluate(
                    """
(payload) => {
  const id = String(payload.id || "");
  const element = id ? document.getElementById(id) : null;
  if (!element) return {clicked: false, reason: "missing"};
  element.click();
  return {clicked: true, reason: ""};
}
                    """,
                    {"id": selector_id},
                )
            except Exception as exc:
                clicked = {"clicked": False, "reason": str(exc)}
            clicked_ok = bool(isinstance(clicked, dict) and clicked.get("clicked"))
            if not clicked_ok:
                js_click_error = str(clicked.get("reason") or "") if isinstance(clicked, dict) else ""
            else:
                wait_action = BaseAction.create_action({"type": "WaitAction", "time_seconds": 0.35})
                result = await session.step(wait_action)
                last_result = result
                attempts.append(
                    {
                        "action": {"type": "DomClickFallback", "selector_id": selector_id},
                        "success": True,
                        "error": "",
                    }
                )
                success = True
            if js_click_error:
                attempts.append(
                    {
                        "action": {"type": "DomClickFallback", "selector_id": selector_id},
                        "success": False,
                        "error": js_click_error,
                    }
                )
        if success:
            break
    if last_result is None:
        raise RuntimeError(f"No selector candidates available for action: {planned_action}")
    return last_result, {"planned_action": planned_action, "attempts": attempts}


async def _run_guided_brief_async(
    *,
    use_case: str,
    seed: int,
    brief_payload: dict[str, Any],
    task_cache: Path | None = None,
    web_project_id: str | None = None,
    max_steps: int = 12,
    allow_signal_success: bool = False,
    planned_actions_override: list[dict[str, Any]] | None = None,
    headless: bool | None = None,
) -> dict[str, Any]:
    try:
        from src.operator.eval.session import build_task_execution_session
    except ModuleNotFoundError:
        session_path = REPO_ROOT / "src" / "operator" / "eval" / "session.py"
        spec = importlib.util.spec_from_file_location("autoppia_operator_eval_session", session_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        build_task_execution_session = module.build_task_execution_session

    brief = brief_payload.get("brief") if isinstance(brief_payload, dict) else {}
    if not isinstance(brief, dict):
        raise ValueError("brief payload missing brief object")
    task = _task_for_seed(use_case=use_case, seed=seed, task_cache=task_cache, web_project_id=web_project_id)
    effective_seed = extract_seed_from_task_url(str(task.url)) or int(seed)
    web_agent_id = _guided_web_agent_id(effective_seed)
    validator_id = f"claude-guided-validator-{effective_seed}-{random.randint(1000, 9999)}"
    planned_actions_source = (
        list(planned_actions_override)
        if isinstance(planned_actions_override, list) and planned_actions_override
        else _guided_actions_from_brief(
            task_url=str(task.url),
            brief=brief,
            web_project_id=str(getattr(task, "web_project_id", "") or web_project_id or "autocinema"),
        )
    )
    planned_actions = _render_placeholders(
        planned_actions_source[: max(1, int(max_steps))],
        web_agent_id,
    )
    session = build_task_execution_session(
        task=task,
        web_agent_id=web_agent_id,
        validator_id=validator_id,
        enable_score_cheating=False,
        capture_screenshot=False,
        headless=headless,
    )
    started = time.time()
    step_result = await session.reset()
    execution_log: list[dict[str, Any]] = []
    try:
        for planned_action in planned_actions:
            step_result, execution = await _execute_action_candidates(session, planned_action)
            execution["score"] = float(step_result.score.raw_score)
            execution["success"] = bool(step_result.score.success)
            execution["url"] = str(step_result.snapshot.url)
            execution_log.append(execution)
            if bool(step_result.score.success) and float(step_result.score.raw_score) >= 1.0:
                break
        # Form submits and event logging can settle slightly after the click.
        for _ in range(6):
            html = str(step_result.snapshot.html or "")
            url = str(step_result.snapshot.url)
            if bool(step_result.score.success) and float(step_result.score.raw_score) >= 1.0:
                break
            if allow_signal_success and _success_signal_hit(html=html, url=url, brief=brief):
                break
            wait_action = BaseAction.create_action({"type": "WaitAction", "time_seconds": 0.5})
            step_result = await session.step(wait_action)
        final_url = str(step_result.snapshot.url)
        final_html = str(step_result.snapshot.html or "")
        signal_success = _success_signal_hit(html=final_html, url=final_url, brief=brief)
        success = bool(step_result.score.success)
        score = float(step_result.score.raw_score)
        report = {
            "provider": "claude-guided",
            "model": str((brief_payload.get("meta") or {}).get("model") or "claude-guided"),
            "episodes": [
                {
                    "success": success,
                    "score": score,
                    "steps": len(execution_log),
                    "seed": int(effective_seed),
                    "use_case": use_case,
                    "task_id": str(task.id),
                    "episode_task_id": f"claude-guided-{use_case.lower()}-{effective_seed}",
                    "web_agent_id": web_agent_id,
                    "validator_id": validator_id,
                    "final_url": final_url,
                    "final_content": final_html[:5000],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "task_seconds": round(time.time() - started, 4),
                    "guided_execution": execution_log,
                    "success_via_signal": signal_success,
                }
            ],
        }
        return report
    finally:
        await session.close()


def run_guided_brief(
    *,
    use_case: str,
    seed: int,
    brief_payload: dict[str, Any],
    task_cache: Path | None = None,
    web_project_id: str | None = None,
    max_steps: int = 12,
    allow_signal_success: bool = False,
    planned_actions_override: list[dict[str, Any]] | None = None,
    headless: bool | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        _run_guided_brief_async(
            use_case=use_case,
            seed=seed,
            brief_payload=brief_payload,
            task_cache=task_cache,
            web_project_id=web_project_id,
            max_steps=max_steps,
            allow_signal_success=allow_signal_success,
            planned_actions_override=planned_actions_override,
            headless=headless,
        )
    )
