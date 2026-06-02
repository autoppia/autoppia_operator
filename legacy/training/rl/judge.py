from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from infra.llm_gateway import openai_chat_completions

logger = logging.getLogger(__name__)


@dataclass
class JudgeScore:
    reward_delta: float
    confidence: float
    reason: str
    model: str
    raw: dict[str, Any] | None = None


def _parse_json_obj(text: str) -> dict[str, Any]:
    s = str(text or "").strip()
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


def llm_judge_enabled() -> bool:
    return str(os.getenv("CONTACT_RL_JUDGE_ENABLED", "0")).strip().lower() not in {"", "0", "false", "no"}


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), float(lo)), float(hi))


def judge_step_progress(
    *,
    task_id: str,
    task_prompt: str,
    step_index: int,
    before_url: str,
    after_url: str,
    before_html: str,
    after_html: str,
    action: dict[str, Any] | None,
    base_reward: float,
) -> JudgeScore:
    if not llm_judge_enabled():
        return JudgeScore(reward_delta=0.0, confidence=0.0, reason="judge_disabled", model="")
    if not str(os.getenv("OPENAI_API_KEY", "")).strip():
        return JudgeScore(reward_delta=0.0, confidence=0.0, reason="judge_missing_api_key", model="")

    model = str(os.getenv("CONTACT_RL_JUDGE_MODEL", "gpt-4o")).strip()
    temperature = float(os.getenv("CONTACT_RL_JUDGE_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("CONTACT_RL_JUDGE_MAX_TOKENS", "220"))
    reward_scale = float(os.getenv("CONTACT_RL_JUDGE_REWARD_SCALE", "0.5"))
    before_tail = str(before_html or "")[-1800:]
    after_tail = str(after_html or "")[-1800:]
    action_json = json.dumps(action or {}, ensure_ascii=False, sort_keys=True)

    system_msg = (
        "You are a strict reward-shaping judge for web RL.\n"
        "Score whether the latest browser action improved progress toward the task.\n"
        "Return JSON only with keys:\n"
        "reward_delta: number in [-1, 1]\n"
        "confidence: number in [0, 1]\n"
        "reason: short string.\n"
        "Use positive reward only for concrete progress visible in URL/DOM/action outcome.\n"
        "Use negative reward for clear regression, distraction, or wasted action.\n"
        "Be conservative."
    )
    user_msg = (
        f"TASK: {task_prompt[:2400]}\n"
        f"STEP_INDEX: {int(step_index)}\n"
        f"BASE_REWARD: {float(base_reward):.4f}\n"
        f"BEFORE_URL: {before_url}\n"
        f"AFTER_URL: {after_url}\n"
        f"ACTION: {action_json}\n"
        f"BEFORE_HTML_TAIL: {before_tail}\n"
        f"AFTER_HTML_TAIL: {after_tail}\n"
    )
    try:
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
    except Exception as exc:
        logger.warning("Judge call failed for task_id=%s step=%s: %s", task_id, int(step_index), exc)
        return JudgeScore(reward_delta=0.0, confidence=0.0, reason="judge_error", model=model)
    content = ""
    try:
        content = str(resp["choices"][0]["message"]["content"] or "")
    except Exception:
        content = ""
    obj = _parse_json_obj(content)
    try:
        reward_delta = _clip(float(obj.get("reward_delta", 0.0)), -1.0, 1.0) * reward_scale
    except Exception:
        reward_delta = 0.0
    try:
        confidence = _clip(float(obj.get("confidence", 0.0)), 0.0, 1.0)
    except Exception:
        confidence = 0.0
    reason = str(obj.get("reason") or "").strip()[:200]
    return JudgeScore(
        reward_delta=float(reward_delta),
        confidence=float(confidence),
        reason=reason,
        model=model,
        raw=obj or None,
    )
