"""Fine-tuned BU-30B LoRA policy for FSM action selection.

Uses the fine-tuned browser-use/bu-30b-a3b-preview model (via a local
vLLM endpoint or OpenAI-compatible API) to produce browser actions.
Falls back to the original LLM policy on parse errors.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:8000/v1"
_DEFAULT_MODEL = "browser-use/bu-30b-a3b-preview"

# Repo root for resolving relative paths
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _serialize_obs(policy_obs: Dict[str, Any]) -> str:
    """Serialize policy observation to compact text for the fine-tuned model."""
    try:
        from training.obs_serializer import serialize_observation
        return serialize_observation(policy_obs)
    except Exception:
        # Minimal fallback
        parts = []
        task = policy_obs.get("prompt", "") or policy_obs.get("task_text", "")
        if task:
            parts.append(f"Task: {task}")
        url = policy_obs.get("url", "")
        if url:
            parts.append(f"URL: {url}")
        step = policy_obs.get("step_index")
        if step is not None:
            parts.append(f"Step: {step}")
        return "\n".join(parts)


def _parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON action dict from model output."""
    raw = raw.strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try extracting first JSON object
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


class BUPolicy:
    """Policy backed by the fine-tuned BU-30B LoRA model.

    Connects to a vLLM (or OpenAI-compatible) endpoint. Falls back to
    the provided ``fallback_policy`` on connection or parse errors.
    """

    def __init__(
        self,
        fallback_policy: Any,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.fallback = fallback_policy
        self.endpoint = (
            endpoint
            or os.environ.get("BU_POLICY_ENDPOINT", "").strip()
            or _DEFAULT_ENDPOINT
        )
        self.model = (
            model
            or os.environ.get("BU_POLICY_MODEL", "").strip()
            or _DEFAULT_MODEL
        )
        self._client: Any = None
        self.debug_dir = str(os.getenv("FSM_POLICY_DEBUG_DIR", "") or "").strip()

    def _get_client(self):
        """Lazy-init an OpenAI client pointing at the vLLM endpoint."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.endpoint,
                api_key=os.environ.get("BU_POLICY_API_KEY", "none"),
            )
        except ImportError:
            logger.warning("openai package not installed; BUPolicy will use fallback")
            self._client = None
        return self._client

    def decide(
        self,
        *,
        task_id: str,
        prompt: str,
        mode: str,
        policy_obs: Dict[str, Any],
        allowed_tools: set,
        model_name: str,
        plan_model_name: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select an action using the fine-tuned model.

        Signature matches ``Policy.decide`` so it's a drop-in replacement.
        """
        # Deterministic modes bypass the model
        if mode == "POPUP":
            return {"type": "meta", "name": "META.SOLVE_POPUPS", "arguments": {}}, {"source": "deterministic"}
        if mode == "REPORT":
            facts = policy_obs.get("memory", {}).get("facts") if isinstance(policy_obs.get("memory"), dict) else []
            fact = str((facts or [""])[0] if isinstance(facts, list) and facts else "")
            content = fact or "Task appears complete."
            return {"type": "final", "done": True, "content": content}, {"source": "deterministic"}

        # Try the fine-tuned model
        try:
            return self._model_decide(
                task_id=task_id,
                prompt=prompt,
                mode=mode,
                policy_obs=policy_obs,
                allowed_tools=allowed_tools,
            )
        except Exception as exc:
            logger.warning("BUPolicy model call failed (%s); falling back to LLM policy", exc)
            return self.fallback.decide(
                task_id=task_id,
                prompt=prompt,
                mode=mode,
                policy_obs=policy_obs,
                allowed_tools=allowed_tools,
                model_name=model_name,
                plan_model_name=plan_model_name,
            )

    def _model_decide(
        self,
        *,
        task_id: str,
        prompt: str,
        mode: str,
        policy_obs: Dict[str, Any],
        allowed_tools: set,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Call the fine-tuned model and parse the response."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("No OpenAI client available")

        obs_text = _serialize_obs(policy_obs)

        system_msg = (
            "You are a browser-use web operator. "
            "Given the observation, return ONE JSON action. No markdown.\n"
            "Valid outputs:\n"
            '1) {"type":"browser","tool_call":{"name":"...","arguments":{...}}}\n'
            '2) {"type":"final","done":true,"content":"..."}\n'
            f"Allowed tools: {', '.join(sorted(allowed_tools))}\n"
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": obs_text},
            ],
            max_tokens=512,
            temperature=0.1,
        )

        raw_output = response.choices[0].message.content or ""
        usage_info = {
            "source": "bu_policy",
            "model": self.model,
            "tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
        }

        action = _parse_action(raw_output)
        if action is None:
            raise ValueError(f"Failed to parse model output: {raw_output[:200]}")

        # Validate action structure
        action_type = action.get("type", "")
        if action_type == "browser":
            tc = action.get("tool_call")
            if not isinstance(tc, dict) or "name" not in tc:
                raise ValueError(f"Invalid browser action structure: {action}")
            tool_name = tc.get("name", "")
            if tool_name not in allowed_tools and f"browser.{tool_name}" not in allowed_tools:
                raise ValueError(f"Model chose disallowed tool: {tool_name}")
        elif action_type == "final":
            action.setdefault("done", True)
            action.setdefault("content", "")
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        return action, usage_info
