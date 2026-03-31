from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CanonicalBrowserState(BaseModel):
    task_id: str = ""
    prompt: str = ""
    url: str = ""
    snapshot_html: str = ""
    screenshot: Any = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    internal_state: dict[str, Any] = Field(default_factory=dict)
    score_feedback: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[Any] | None = None
    step_index: int = 0
    include_reasoning: bool = False
    completion_only: bool = False
    web_project_id: str = ""
    use_case: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump()
