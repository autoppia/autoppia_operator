from __future__ import annotations

import json
import os
from typing import Any

import httpx

from autoppia_harvester.claude_code import ACTION_SCHEMA, DEFAULT_ALLOWED_TOOLS
from autoppia_harvester.models import FindTrayectoryRequest, FindTrayectoryResponse
from autoppia_harvester.trajectory import extract_json_object, normalize_trajectory


def _build_messages(request: FindTrayectoryRequest) -> list[dict[str, str]]:
    task_payload = {
        "task": request.model_dump(mode="json"),
        "iwa_contract": {
            "endpoint": "/find_trayectory",
            "response_field": "trajectory",
            "trajectory_item_shape": {"name": "click", "arguments": {"selector": {"type": "attributeValueSelector", "attribute": "id", "value": "cta"}}},
            "allowed_tools": request.tools or request.allowed_tools or DEFAULT_ALLOWED_TOOLS,
        },
    }
    system = (
        "You are Autoppia Harvester for the IWA benchmark. "
        "Return only a JSON object matching the requested schema. "
        "Generate a short replayable trajectory; do not execute code."
    )
    user = f"""
Create a replayable trajectory for this task.

Rules:
- `trajectory` must be a list of IWA tool calls: {{"name": "...", "arguments": {{...}}}}.
- Use unprefixed IWA tool names: navigate, click, type, select_dropdown, send_keys, wait, done.
- Prefer stable selectors: id, name, data-testid, aria-label, role, href, text/tagContainsSelector.
- Selector format: {{"type":"attributeValueSelector","attribute":"id","value":"..."}} or {{"type":"tagContainsSelector","value":"..."}}.
- Include a navigate action first when the task has a start URL.
- Preserve any seed query string in navigation URLs.
- Keep the trajectory short and replayable.
- Use IWA credential placeholders from the task as literal text when needed.

Task payload:
{json.dumps(task_payload, ensure_ascii=True)}

Return JSON schema:
{json.dumps(ACTION_SCHEMA, ensure_ascii=True)}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OpenAIHarvester:
    def __init__(self) -> None:
        self.model = os.getenv("AUTOPPIA_HARVESTER_OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5"
        self.base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY") or "dummy"
        self.timeout = float(os.getenv("AUTOPPIA_HARVESTER_TIMEOUT_SECONDS", "120"))
        self.max_output_tokens = int(os.getenv("AUTOPPIA_HARVESTER_MAX_OUTPUT_TOKENS", "4096"))

    async def find_trayectory(self, request: FindTrayectoryRequest) -> FindTrayectoryResponse:
        payload = {
            "model": self.model,
            "messages": _build_messages(request),
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
        if response.status_code >= 400:
            detail = response.text[-1200:]
            raise RuntimeError(f"OpenAI harvester failed with status {response.status_code}: {detail}")

        data = response.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        message = (choices or [{}])[0].get("message") if isinstance(choices, list) and choices else {}
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, list):
            content = "\n".join(str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item) for item in content)
        parsed = extract_json_object(str(content or ""))
        trajectory = normalize_trajectory(parsed)
        usage = data.get("usage") if isinstance(data, dict) else {}
        input_tokens = int((usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens") or 0)
        output_tokens = int((usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens") or 0)

        return FindTrayectoryResponse(
            web_agent_id=os.getenv("AUTOPPIA_HARVESTER_WEB_AGENT_ID", "autoppia-harvester"),
            task_id=request.canonical_task_id,
            trajectory=trajectory,
            actions=trajectory,
            model_used=f"openai:{self.model}",
            extracted_data=parsed.get("extracted_data"),
            summary=str(parsed.get("summary") or ""),
            success=bool(parsed.get("success")) and bool(trajectory),
            failure_reason=str(parsed.get("failure_reason") or parsed.get("failureReason") or ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def harvest(self, request: FindTrayectoryRequest) -> FindTrayectoryResponse:
        return await self.find_trayectory(request)
