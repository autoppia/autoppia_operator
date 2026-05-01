from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.operator.support.iwa import IWA_ACT_PROTOCOL_VERSION, BaseAction, StepAllowedTool, StepRequest, StepResponse, Task
from src.operator.support.utils import candidate_text, env_bool, normalize_demo_url, normalize_selector_payload


def use_vision() -> bool:
    return env_bool("USE_VISION", False) or env_bool("AGENT_USE_VISION", False)


_normalize_demo_url = normalize_demo_url
_candidate_text = candidate_text


def _is_navigate_action_type(action_type: Any) -> bool:
    return str(action_type or "").strip().lower() in {"navigateaction", "navigate"}


def _is_done_action_type(action_type: Any) -> bool:
    return str(action_type or "").strip().lower() in {"doneaction", "finishaction", "done", "finish"}


def _is_request_user_input_action_type(action_type: Any) -> bool:
    return str(action_type or "").strip().lower().replace("-", "_") in {
        "requestuserinputaction",
        "request_user_input",
        "request_user_input_action",
    }


_normalize_selector_payload = normalize_selector_payload


def _step_request_from_payload(payload: dict[str, Any]) -> Any:
    if not isinstance(payload, dict):
        payload = {}
    try:
        return StepRequest.model_validate(payload)
    except Exception:
        return SimpleNamespace(
            protocol_version=str(payload.get("protocol_version") or IWA_ACT_PROTOCOL_VERSION),
            task_id=str(payload.get("task_id") or ""),
            prompt=str(payload.get("prompt") or payload.get("task_prompt") or ""),
            url=_normalize_demo_url(str(payload.get("url") or "")),
            html=str(payload.get("html") or payload.get("snapshot_html") or ""),
            screenshot=payload.get("screenshot"),
            step_index=int(payload.get("step_index") or 0),
            history=payload.get("history") if isinstance(payload.get("history"), list) else [],
            tools=payload.get("tools") if isinstance(payload.get("tools"), list) else payload.get("allowed_tools") if isinstance(payload.get("allowed_tools"), list) else [],
            include_reasoning=bool(payload.get("include_reasoning") or payload.get("return_reasoning")),
        )


def _sanitize_action_payload(action_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(action_payload or {})
    selector = _normalize_selector_payload(payload.get("selector"))
    if isinstance(selector, dict):
        payload["selector"] = selector
    elif "selector" in payload:
        payload.pop("selector", None)
    if _is_navigate_action_type(payload.get("type")):
        payload["url"] = _normalize_demo_url(payload.get("url"))
    return payload


def _extract_result_text_from_action(action: dict[str, Any] | None) -> str | None:
    return _candidate_text(action) if isinstance(action, dict) else None


def _task_from_payload(payload: dict[str, Any]) -> Task:
    request = _step_request_from_payload(payload)
    task_payload = {
        "id": str(getattr(request, "task_id", "") or ""),
        "url": _normalize_demo_url(str(getattr(request, "url", "") or "")),
        "prompt": str(getattr(request, "prompt", "") or ""),
        "web_project_id": payload.get("web_project_id"),
    }
    try:
        if isinstance(Task, type):
            return Task(**task_payload)
    except Exception:
        pass
    return SimpleNamespace(**task_payload)


def _canonical_browser_tool_name(raw_name: str) -> str:
    name = str(raw_name or "").strip().lower()
    if not name:
        return ""
    if name == "user.request_input":
        return name
    suffix = name[8:] if name.startswith("browser.") else name
    if suffix == "evaluate":
        return ""
    alias = {
        "search": "search",
        "navigate": "navigate",
        "go_back": "go_back",
        "click": "click",
        "dblclick": "dblclick",
        "rightclick": "rightclick",
        "middleclick": "middleclick",
        "tripleclick": "tripleclick",
        "input": "input",
        "scroll": "scroll",
        "wait": "wait",
        "select_dropdown": "select_dropdown",
        "dropdown_options": "dropdown_options",
        "hover": "hover",
        "screenshot": "screenshot",
        "send_keys": "send_keys",
        "hold_key": "hold_key",
        "extract": "extract",
        "done": "done",
    }
    return f"browser.{alias.get(suffix, suffix)}"


def is_tool_enabled(name: str) -> bool:
    canonical = _canonical_browser_tool_name(name)
    if not canonical:
        return False
    return not (canonical == "browser.screenshot" and not use_vision())


def _action_to_tool_call(action: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    action_type = str(action.get("type") or "").strip()
    if not action_type or _is_done_action_type(action_type):
        return None
    arguments = {key: value for key, value in action.items() if key != "type"}
    if _is_request_user_input_action_type(action_type):
        return {"name": "user.request_input", "arguments": arguments}
    mapping = {
        "navigateaction": "browser.navigate",
        "clickaction": "browser.click",
        "gobackaction": "browser.go_back",
        "searchaction": "browser.search",
        "typeaction": "browser.input",
        "fillaction": "browser.input",
        "scrollaction": "browser.scroll",
        "waitaction": "browser.wait",
        "selectaction": "browser.select_dropdown",
        "selectdropdownoptionaction": "browser.select_dropdown",
        "hoveraction": "browser.hover",
        "screenshotaction": "browser.screenshot",
        "getdropdownoptionsaction": "browser.dropdown_options",
        "doubleclickaction": "browser.dblclick",
        "rightclickaction": "browser.rightclick",
        "middleclickaction": "browser.middleclick",
        "tripleclickaction": "browser.tripleclick",
        "holdkeyaction": "browser.hold_key",
        "sendkeysaction": "browser.send_keys",
        "sendkeysiwaaction": "browser.send_keys",
        "extractaction": "browser.extract",
    }
    tool_name = mapping.get(action_type.lower())
    if not tool_name:
        suffix = action_type[:-6] if action_type.endswith("Action") else action_type
        tool_name = f"browser.{suffix.strip().lower()}"
    tool_name = _canonical_browser_tool_name(tool_name)
    if not is_tool_enabled(tool_name):
        return None
    return _normalize_tool_call_payload({"name": tool_name, "arguments": arguments})


def _normalize_tool_call_payload(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    name = _canonical_browser_tool_name(str(tool_call.get("name") or "").strip())
    if not name or not is_tool_enabled(name):
        return None
    arguments = dict(tool_call.get("arguments") or {}) if isinstance(tool_call.get("arguments"), dict) else {}
    selector = _normalize_selector_payload(arguments.get("selector"))
    if isinstance(selector, dict):
        arguments["selector"] = selector
    elif "selector" in arguments:
        arguments.pop("selector", None)
    if name == "browser.select_dropdown":
        value = _candidate_text(arguments.get("text")) or _candidate_text(arguments.get("value"))
        if value:
            arguments["text"] = value
        arguments.pop("value", None)
    return {"name": name, "arguments": arguments}


def _serialize_use_case(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        out = {
            "name": str(raw.get("name") or "").strip()[:80],
            "description": str(raw.get("description") or "").strip()[:400],
        }
        return {k: v for k, v in out.items() if v}
    out = {
        "name": str(getattr(raw, "name", "") or "").strip()[:80],
        "description": str(getattr(raw, "description", "") or "").strip()[:400],
    }
    return {k: v for k, v in out.items() if v}


def _collect_supported_tool_definitions() -> list[dict[str, Any]]:
    defs_fn = getattr(BaseAction, "all_function_definitions", None)
    if not callable(defs_fn):
        return []
    try:
        definitions = defs_fn()
    except Exception:
        return []
    if not isinstance(definitions, list):
        return []

    out: list[dict[str, Any]] = []
    for item in definitions:
        if not isinstance(item, dict):
            continue
        function_payload = item.get("function") if isinstance(item.get("function"), dict) else {}
        raw_name = str(function_payload.get("name") or "").strip()
        if not raw_name or raw_name == "done":
            continue
        namespaced = "user.request_input" if raw_name == "request_user_input" else f"browser.{raw_name}"
        if not is_tool_enabled(namespaced):
            continue
        out.append(
            {
                "name": namespaced,
                "description": str(function_payload.get("description") or ""),
                "parameters": function_payload.get("parameters") if isinstance(function_payload.get("parameters"), dict) else {},
            }
        )
    return out


def _normalize_allowed_tool_names(allowed_tools: Any) -> set[str]:
    if not isinstance(allowed_tools, list):
        return set()
    out: set[str] = set()
    for item in allowed_tools:
        tool_payload = item if isinstance(item, dict) else None
        if tool_payload is None:
            continue
        try:
            normalized_tool = StepAllowedTool.model_validate(tool_payload)
            raw_name = str(normalized_tool.name or "").strip()
        except Exception:
            raw_name = str(tool_payload.get("name") or "").strip()
        if not raw_name and isinstance(tool_payload.get("function"), dict):
            function_name = str((tool_payload.get("function") or {}).get("name") or "").strip()
            if function_name:
                raw_name = "user.request_input" if function_name == "request_user_input" else _canonical_browser_tool_name(f"browser.{function_name}")
        if raw_name:
            canonical = _canonical_browser_tool_name(raw_name)
            if canonical and is_tool_enabled(canonical):
                out.add(canonical)
    return out


def _act_http_response(
    raw_resp: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    allowed_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    allowed = set(allowed_tool_names or [])
    content = _candidate_text(raw_resp.get("content"))
    done_from_actions = False
    tool_calls: list[dict[str, Any]] = []

    raw_tool_calls = raw_resp.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if not name or not is_tool_enabled(name) or (allowed and name not in allowed):
                continue
            normalized_call = _normalize_tool_call_payload(raw_call)
            if isinstance(normalized_call, dict):
                tool_calls.append(normalized_call)
    else:
        for action in actions:
            if not isinstance(action, dict):
                continue
            if _is_done_action_type(action.get("type")):
                done_from_actions = True
                if not content:
                    content = _extract_result_text_from_action(action)
                continue
            call = _action_to_tool_call(action)
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "").strip()
            if allowed and name not in allowed:
                continue
            tool_calls.append(call)

    out: dict[str, Any] = {
        "protocol_version": str(raw_resp.get("protocol_version") or IWA_ACT_PROTOCOL_VERSION),
        "tool_calls": tool_calls,
        "content": (content if (bool(raw_resp.get("done")) or done_from_actions) and isinstance(content, str) and content.strip() else None),
        "reasoning": str(raw_resp.get("reasoning")).strip()[:200] if isinstance(raw_resp.get("reasoning"), str) and str(raw_resp.get("reasoning")).strip() else None,
        "done": bool(raw_resp.get("done")) or done_from_actions,
    }
    execution_mode = str(raw_resp.get("execution_mode") or "").strip().lower()
    if execution_mode in {"batch", "single_step"}:
        out["execution_mode"] = execution_mode
    state_out = raw_resp.get("state_out") if isinstance(raw_resp.get("state_out"), dict) else raw_resp.get("internal_state") if isinstance(raw_resp.get("internal_state"), dict) else None
    if isinstance(state_out, dict):
        out["state_out"] = dict(state_out)
    if isinstance(raw_resp.get("error"), str) and str(raw_resp.get("error")).strip():
        out["error"] = str(raw_resp.get("error")).strip()[:400]
    if isinstance(raw_resp.get("metrics"), dict):
        out["metrics"] = raw_resp.get("metrics")
        helper_models = [str(m).strip() for m in list(raw_resp.get("helper_models") or []) if str(m).strip()]
        llm_metrics = out["metrics"].get("llm") if isinstance(out["metrics"].get("llm"), dict) else None
        if helper_models and isinstance(llm_metrics, dict):
            llm_metrics["helper_models"] = helper_models
        if isinstance(raw_resp.get("call_breakdown"), dict) and isinstance(llm_metrics, dict):
            llm_metrics["call_breakdown"] = raw_resp.get("call_breakdown")
        if isinstance(raw_resp.get("usage_breakdown"), dict) and isinstance(llm_metrics, dict):
            llm_metrics["usage_breakdown"] = raw_resp.get("usage_breakdown")
    if isinstance(raw_resp.get("usage"), dict):
        out["usage"] = raw_resp.get("usage")
    if isinstance(raw_resp.get("total_tokens"), int):
        out["total_tokens"] = int(raw_resp.get("total_tokens") or 0)
    if isinstance(raw_resp.get("model"), str) and str(raw_resp.get("model")).strip():
        out["model"] = str(raw_resp.get("model")).strip()
    if isinstance(raw_resp.get("estimated_cost_usd"), int | float):
        out["estimated_cost_usd"] = float(raw_resp.get("estimated_cost_usd"))
    if isinstance(raw_resp.get("helper_models"), list):
        out["helper_models"] = [str(m).strip() for m in raw_resp.get("helper_models") if str(m).strip()]
    if isinstance(raw_resp.get("call_breakdown"), dict):
        out["call_breakdown"] = raw_resp.get("call_breakdown")
    if isinstance(raw_resp.get("usage_breakdown"), dict):
        out["usage_breakdown"] = raw_resp.get("usage_breakdown")
    if isinstance(raw_resp.get("action_rationales"), list):
        out["action_rationales"] = raw_resp.get("action_rationales")
    if isinstance(raw_resp.get("failure_reason"), str) and str(raw_resp.get("failure_reason")).strip():
        out["failure_reason"] = str(raw_resp.get("failure_reason")).strip()[:80]
    # When autoppia_iwa is not importable, StepResponse is typing.Any and has no model_validate.
    _validate_response = getattr(StepResponse, "model_validate", None)
    if callable(_validate_response):
        validated = _validate_response({k: out[k] for k in ("protocol_version", "tool_calls", "content", "reasoning", "done", "error") if k in out})
        out.update(validated.model_dump())
    return out
