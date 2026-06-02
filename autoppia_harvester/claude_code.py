from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoppia_harvester.models import HarvestRequest, HarvestResponse
from autoppia_harvester.trajectory import extract_json_object, normalize_trajectory


ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "summary": {"type": "string"},
        "failure_reason": {"type": "string"},
        "trajectory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "reasoning": {"type": "string"},
                },
                "required": ["name", "arguments"],
            },
        },
        "extracted_data": {"type": "string"},
    },
    "required": ["success", "summary", "trajectory"],
}


DEFAULT_ALLOWED_TOOLS = [
    {"name": "navigate", "arguments": {"url": "https://example.com"}},
    {"name": "click", "arguments": {"selector": {"type": "attributeValueSelector", "attribute": "id", "value": "submit"}}},
    {"name": "type", "arguments": {"selector": {"type": "attributeValueSelector", "attribute": "name", "value": "q"}, "text": "text"}},
    {"name": "select_dropdown", "arguments": {"selector": {"type": "attributeValueSelector", "attribute": "name", "value": "genre"}, "value": "Sci-Fi"}},
    {"name": "send_keys", "arguments": {"keys": "Enter"}},
    {"name": "wait", "arguments": {"seconds": 1}},
    {"name": "done", "arguments": {"summary": "Task completed"}},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def harvester_root() -> Path:
    root = Path(os.getenv("AUTOPPIA_HARVESTER_WORKDIR", "/tmp/autoppia_harvester")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def build_prompt(task_file: Path, output_file: Path) -> str:
    return f"""
You are Autoppia Harvester for the IWA benchmark.

Read `{task_file}`. Discover a replayable trajectory that solves the task on the given URL.
You may create and run short Playwright/Python/Node scripts inside this workspace to inspect the page.

Return a JSON object and also write it to `{output_file}`.

Rules:
- The output must match the provided JSON schema.
- `trajectory` must be a list of IWA tool calls: {{"name": "...", "arguments": {{...}}}}.
- Use unprefixed IWA action names: navigate, click, type, select_dropdown, send_keys, wait, done.
- Prefer stable selectors: id, name, data-testid, aria-label, role, href, text/tagContainsSelector.
- Selector format: {{"type":"attributeValueSelector","attribute":"id","value":"..."}} or {{"type":"tagContainsSelector","value":"..."}}.
- Include a navigate action first when the task has a start URL.
- Do not claim success unless the trajectory is plausibly complete.
- Keep the trajectory short and replayable. Avoid task-specific Python in the returned result.
- Use IWA credential placeholders from the prompt as literal text when needed, e.g. <username>, <password>, <web_agent_id>.

Return JSON schema:
{json.dumps(ACTION_SCHEMA, ensure_ascii=True)}
""".strip()


class ClaudeCodeHarvester:
    def __init__(self) -> None:
        self.claude_bin = shutil.which(os.getenv("AUTOPPIA_HARVESTER_CLAUDE_BIN", "claude"))
        self.model = os.getenv("AUTOPPIA_HARVESTER_CLAUDE_MODEL", "sonnet")
        self.timeout = int(os.getenv("AUTOPPIA_HARVESTER_TIMEOUT_SECONDS", "900"))

    async def harvest(self, request: HarvestRequest) -> HarvestResponse:
        if not self.claude_bin:
            raise RuntimeError("Claude CLI is not installed or not on PATH")

        run_id = str(uuid.uuid4())
        run_dir = harvester_root() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        task_payload = {
            "created_at": now_iso(),
            "task": request.model_dump(mode="json"),
            "iwa_contract": {
                "endpoint": "/harvest",
                "response_field": "trajectory",
                "trajectory_item_shape": {"name": "click", "arguments": {"selector": {"type": "attributeValueSelector", "attribute": "id", "value": "cta"}}},
                "allowed_tools": request.tools or request.allowed_tools or DEFAULT_ALLOWED_TOOLS,
            },
        }
        task_file = run_dir / "task.json"
        output_file = run_dir / "result.json"
        write_json(task_file, task_payload)
        (run_dir / "README.md").write_text("Autoppia Harvester isolated workspace.\n", encoding="utf-8")

        cmd = [
            self.claude_bin,
            "--print",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(ACTION_SCHEMA, ensure_ascii=True),
            "--model",
            self.model,
            "--dangerously-skip-permissions",
            "--permission-mode",
            "bypassPermissions",
            "--add-dir",
            str(run_dir),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(run_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "AUTOPPIA_HARVESTER_RUN_DIR": str(run_dir)},
        )
        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(build_prompt(task_file, output_file).encode("utf-8")), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(f"Claude harvester timed out after {self.timeout}s") from exc

        stdout = stdout_raw.decode("utf-8", errors="replace")
        stderr = stderr_raw.decode("utf-8", errors="replace")
        (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")

        if proc.returncode != 0:
            raise RuntimeError(f"Claude harvester failed with exit code {proc.returncode}: {stderr[-1000:]}")

        output_text = output_file.read_text(encoding="utf-8") if output_file.exists() else stdout
        parsed = extract_json_object(output_text)
        trajectory = normalize_trajectory(parsed)

        return HarvestResponse(
            web_agent_id=os.getenv("AUTOPPIA_HARVESTER_WEB_AGENT_ID", "autoppia-harvester"),
            task_id=request.canonical_task_id,
            trajectory=trajectory,
            actions=trajectory,
            model_used=f"claude-code:{self.model}",
            extracted_data=parsed.get("extracted_data"),
            summary=str(parsed.get("summary") or ""),
            success=bool(parsed.get("success")) and bool(trajectory),
            failure_reason=str(parsed.get("failure_reason") or parsed.get("failureReason") or ""),
        )
