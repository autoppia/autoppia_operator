from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SECRET_KEY_RE = re.compile(r"(password|passwd|token|secret|api[_-]?key|authorization)", re.IGNORECASE)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)


def extract_task_payload(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Accept both direct task payload and wrapped payload formats."""
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("payload"), dict):
        candidate = raw["payload"]
        if isinstance(candidate.get("steps"), list):
            return candidate
    if isinstance(raw.get("steps"), list):
        return raw
    if isinstance(raw.get("data"), dict):
        data = raw["data"]
        if isinstance(data.get("payload"), dict) and isinstance(data["payload"].get("steps"), list):
            return data["payload"]
        if isinstance(data.get("steps"), list):
            return data
    return None


def _redact_scalar(value: Any) -> Any:
    if isinstance(value, str):
        redacted = _BEARER_RE.sub("Bearer <redacted>", value)
        if len(redacted) > 120 and any(tok in redacted.lower() for tok in ("sk-", "api", "token", "secret")):
            return "<redacted>"
        return redacted
    return value


def _redact_obj(value: Any, *, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _SECRET_KEY_RE.search(key):
                out[key] = "<redacted>"
                continue
            out[key] = _redact_obj(v, parent_key=key)
        return out
    if isinstance(value, list):
        return [_redact_obj(v, parent_key=parent_key) for v in value]
    if parent_key and _SECRET_KEY_RE.search(parent_key):
        return "<redacted>"
    return _redact_scalar(value)


def _normalize_action(action: Any) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, dict):
        action_dict = dict(action)
    elif hasattr(action, "model_dump"):
        try:
            action_dict = action.model_dump(mode="json", exclude_none=True)  # type: ignore[attr-defined]
        except Exception:
            action_dict = {"type": str(getattr(action, "type", "unknown"))}
    else:
        action_dict = {"type": str(getattr(action, "type", "unknown"))}

    nested = action_dict.get("action")
    if isinstance(nested, dict):
        merged = dict(nested)
        for k, v in action_dict.items():
            if k not in merged and k != "action":
                merged[k] = v
        action_dict = merged

    attrs = action_dict.get("attributes")
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            if k not in action_dict or action_dict.get(k) in (None, "", [], {}):
                action_dict[k] = v

    action_type = str(action_dict.get("type") or "unknown")
    normalized: dict[str, Any] = {"type": action_type}

    selector = action_dict.get("selector")
    if selector is not None:
        normalized["selector"] = _redact_obj(selector)

    for key in ("value", "text", "url", "x", "y", "button", "delta", "time_seconds"):
        if key in action_dict and action_dict.get(key) is not None:
            normalized[key] = _redact_obj(action_dict.get(key), parent_key=key)

    return normalized


def _clean_agent_io(obj: Any, *, keep_html: bool) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "task_id",
        "prompt",
        "url",
        "web_project_id",
        "current_url",
        "timestamp",
        "step_index",
    ):
        if key in obj and obj.get(key) is not None:
            out[key] = _redact_obj(obj.get(key), parent_key=key)
    if keep_html and obj.get("html") is not None:
        out["html"] = _redact_obj(obj.get("html"), parent_key="html")
    return out


def _step_to_clean(step: Any, *, keep_html: bool) -> dict[str, Any] | None:
    if not isinstance(step, dict):
        return None

    step_out: dict[str, Any] = {
        "step_index": int(step.get("step_index") or 0),
        "timestamp": step.get("timestamp"),
        "success": bool(step.get("success")) if step.get("success") is not None else False,
        "error": _redact_obj(step.get("error"), parent_key="error") if step.get("error") is not None else None,
        "execution_time_ms": int(step.get("execution_time_ms")) if isinstance(step.get("execution_time_ms"), int | float) else None,
        "agent_input": _clean_agent_io(step.get("agent_input"), keep_html=keep_html),
        "post_execute_output": _clean_agent_io(step.get("post_execute_output"), keep_html=keep_html),
        "llm_calls": _redact_obj(step.get("llm_calls")) if isinstance(step.get("llm_calls"), list) else [],
    }

    action = None
    if isinstance(step.get("agent_output"), dict):
        action = _normalize_action(step["agent_output"].get("action"))
    elif isinstance(step.get("action"), dict):
        action = _normalize_action(step.get("action"))

    step_out["agent_output"] = {"action": action} if action else None
    return step_out


def _dedupe_consecutive_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    prev_key: str | None = None
    for action in actions:
        key = json.dumps(action, sort_keys=True, ensure_ascii=True)
        if key == prev_key:
            continue
        deduped.append(action)
        prev_key = key
    return deduped


def _is_success(*, payload: dict[str, Any], task_meta: dict[str, Any] | None, min_eval_score: float) -> bool:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    status = str(summary.get("status") or "").lower()
    eval_score = float(summary.get("eval_score") or 0.0)

    if status == "success":
        return True
    if eval_score >= float(min_eval_score):
        return True

    if isinstance(task_meta, dict):
        if str(task_meta.get("status") or "").lower() == "completed":
            return True
        if float(task_meta.get("eval_score") or 0.0) >= float(min_eval_score):
            return True
    return False


def normalize_trajectory(
    *,
    payload: dict[str, Any],
    run_id: str,
    source_url: str,
    task_meta: dict[str, Any] | None,
    keep_html: bool,
    min_eval_score: float,
    max_steps: int,
    max_actions: int,
    dedupe_actions: bool,
) -> dict[str, Any]:
    task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    steps_raw = payload.get("steps") if isinstance(payload.get("steps"), list) else []

    task_id = str(payload.get("task_id") or (task_meta or {}).get("taskId") or "")
    prompt = str(task.get("prompt") or (task_meta or {}).get("prompt") or "")
    url = str(task.get("url") or (task_meta or {}).get("url") or "")
    website = str(task.get("website") or (task_meta or {}).get("website") or "")
    use_case = task.get("use_case")
    if not use_case:
        use_case = (task_meta or {}).get("useCase")

    clean_steps: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    for step in steps_raw[: max(0, int(max_steps))]:
        clean = _step_to_clean(step, keep_html=bool(keep_html))
        if clean is None:
            continue
        clean_steps.append(clean)
        ao = clean.get("agent_output")
        if isinstance(ao, dict) and isinstance(ao.get("action"), dict):
            actions.append(ao["action"])

    if dedupe_actions:
        actions = _dedupe_consecutive_actions(actions)

    if max_actions >= 0:
        actions = actions[: int(max_actions)]

    success = _is_success(payload=payload, task_meta=task_meta, min_eval_score=min_eval_score)
    eval_score = float(summary.get("eval_score") or (task_meta or {}).get("eval_score") or 0.0)

    unique_fingerprint = hashlib.sha256((f"{run_id}|{task_id}|{prompt}|{url}|" + json.dumps(actions, sort_keys=True, ensure_ascii=True)).encode("utf-8")).hexdigest()[:16]

    return {
        "trajectory_id": f"{run_id}:{task_id}:{unique_fingerprint}",
        "run_id": str(run_id),
        "task_id": task_id,
        "source_url": str(source_url),
        "task": {
            "prompt": _redact_obj(prompt, parent_key="prompt"),
            "url": _redact_obj(url, parent_key="url"),
            "website": _redact_obj(website, parent_key="website"),
            "use_case": _redact_obj(use_case, parent_key="use_case"),
        },
        "summary": {
            "status": str(summary.get("status") or "unknown"),
            "success": bool(success),
            "eval_score": eval_score,
            "reward": float(summary.get("reward") or 0.0),
            "eval_time_sec": float(summary.get("eval_time_sec") or 0.0),
            "steps_total": int(summary.get("steps_total") or len(clean_steps)),
            "steps_success": int(summary.get("steps_success") or sum(1 for s in clean_steps if s.get("success"))),
        },
        "actions": actions,
        "steps": clean_steps,
    }


def build_sft_record(trajectory: dict[str, Any], *, system_prompt: str | None = None) -> dict[str, Any]:
    task = trajectory.get("task") if isinstance(trajectory.get("task"), dict) else {}
    prompt = str(task.get("prompt") or "")
    url = str(task.get("url") or "")
    actions = trajectory.get("actions") if isinstance(trajectory.get("actions"), list) else []

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})

    messages.append({"role": "user", "content": f"URL: {url}\nTask: {prompt}"})
    messages.append({"role": "assistant", "content": json.dumps(actions, ensure_ascii=False)})
    return {"messages": messages}


def dedupe_trajectories(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop semantic duplicates, keeping the best score version."""
    best: dict[str, dict[str, Any]] = {}
    for item in trajectories:
        task = item.get("task") if isinstance(item.get("task"), dict) else {}
        actions = item.get("actions") if isinstance(item.get("actions"), list) else []
        key = hashlib.sha256((f"{task.get('prompt', '')}|{task.get('url', '')}|" + json.dumps(actions, sort_keys=True, ensure_ascii=True)).encode("utf-8")).hexdigest()
        prev = best.get(key)
        score = float((item.get("summary") or {}).get("eval_score") or 0.0)
        prev_score = float((prev.get("summary") or {}).get("eval_score") or 0.0) if isinstance(prev, dict) else -1.0
        if prev is None or score > prev_score:
            best[key] = item
    return list(best.values())


__all__ = [
    "build_sft_record",
    "dedupe_trajectories",
    "extract_task_payload",
    "normalize_trajectory",
]
