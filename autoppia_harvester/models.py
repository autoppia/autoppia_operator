from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HarvestRequest(BaseModel):
    id: str | None = None
    task_id: str | None = None
    url: str = ""
    prompt: str = ""
    web_project_id: str = ""
    title: str | None = None
    description: str | None = None
    tests: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    @property
    def canonical_task_id(self) -> str:
        return str(self.task_id or self.id or "")


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None

    model_config = {"extra": "allow"}


class HarvestResponse(BaseModel):
    web_agent_id: str = "autoppia-harvester"
    task_id: str = ""
    trajectory: list[ToolCall] = Field(default_factory=list)
    actions: list[ToolCall] = Field(default_factory=list)
    recording: Any | None = None
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model_used: str | None = None
    extracted_data: str | None = None
    summary: str = ""
    success: bool = False
    failure_reason: str = ""
