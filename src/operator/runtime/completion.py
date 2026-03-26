from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Tuple

from infra.llm_gateway import openai_chat_completions


@dataclass
class CompletionCheckResult:
    is_complete: bool
    confidence: float
    reason: str
    model: str
    usage: dict[str, int] | None = None


def _parse_json_obj(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _to_history_lines(history: list[dict[str, Any]] | None) -> str:
    out: list[str] = []
    for idx, h in enumerate((history or [])[-5:]):
        action = h.get("action")
        if isinstance(action, dict):
            at = str(action.get("type") or "")
        else:
            at = str(h.get("action") or "")
        ok = bool(h.get("success", h.get("exec_ok", True)))
        out.append(f"{idx + 1}. action={at} success={ok}")
    return "\n".join(out)


def run_completion_check(
    *,
    task_id: str,
    task: str,
    url: str,
    page_summary: str,
    history: list[dict[str, Any]] | None,
    page_ir_text: str = "",
) -> Tuple[CompletionCheckResult, Dict[str, Any]]:
    model = str(os.getenv("AGENT_COMPLETION_MODEL", "gpt-4o-mini")).strip()
    temperature = float(os.getenv("AGENT_COMPLETION_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("AGENT_COMPLETION_MAX_TOKENS", "120"))

    system_msg = (
        "You are a strict task completion checker for web automation.\n"
        "Determine whether the user's objective is already completed in the CURRENT page state.\n"
        "Return JSON only with keys: is_complete (bool), confidence (0..1), reason (short string).\n"
        "If uncertain, return is_complete=false."
    )
    user_msg = (
        f"TASK: {task}\n"
        f"CURRENT_URL: {url}\n"
        f"PAGE_SUMMARY: {page_summary[:2500]}\n"
        + (f"PAGE_IR: {page_ir_text[:2500]}\n" if page_ir_text else "")
        + (f"HISTORY:\n{_to_history_lines(history)}\n" if history else "")
    )
    resp = openai_chat_completions(
        task_id=task_id,
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage_raw = resp.get("usage") if isinstance(resp.get("usage"), dict) else None
    usage: dict[str, int] | None = None
    if isinstance(usage_raw, dict):
        try:
            usage = {
                "prompt_tokens": int(usage_raw.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_raw.get("completion_tokens") or 0),
                "total_tokens": int(usage_raw.get("total_tokens") or 0),
            }
        except Exception:
            usage = None

    content = ""
    try:
        content = str(resp["choices"][0]["message"]["content"] or "")
    except Exception:
        content = ""
    obj = _parse_json_obj(content)

    is_complete = bool(obj.get("is_complete", False))
    try:
        confidence = float(obj.get("confidence") if obj.get("confidence") is not None else 0.0)
    except Exception:
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    reason = str(obj.get("reason") or "").strip()[:220]

    result = CompletionCheckResult(
        is_complete=is_complete,
        confidence=confidence,
        reason=reason,
        model=model,
        usage=usage,
    )
    return result, obj
