from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskInfo:
    prompt: str = ""
    url: str = ""
    website: str = ""
    use_case: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TaskInfo:
        payload = data if isinstance(data, dict) else {}
        return cls(
            prompt=str(payload.get("prompt") or ""),
            url=str(payload.get("url") or ""),
            website=str(payload.get("website") or ""),
            use_case=payload.get("use_case"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "url": self.url,
            "website": self.website,
            "use_case": self.use_case,
        }


@dataclass
class TrajectorySummary:
    status: str = "unknown"
    success: bool = False
    eval_score: float = 0.0
    reward: float = 0.0
    eval_time_sec: float = 0.0
    steps_total: int = 0
    steps_success: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrajectorySummary:
        payload = data if isinstance(data, dict) else {}
        return cls(
            status=str(payload.get("status") or "unknown"),
            success=bool(payload.get("success")),
            eval_score=float(payload.get("eval_score") or 0.0),
            reward=float(payload.get("reward") or 0.0),
            eval_time_sec=float(payload.get("eval_time_sec") or 0.0),
            steps_total=int(payload.get("steps_total") or 0),
            steps_success=int(payload.get("steps_success") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "eval_score": self.eval_score,
            "reward": self.reward,
            "eval_time_sec": self.eval_time_sec,
            "steps_total": self.steps_total,
            "steps_success": self.steps_success,
        }


@dataclass
class ActionRecord:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ActionRecord:
        payload = dict(data) if isinstance(data, dict) else {}
        action_type = str(payload.pop("type", "unknown") or "unknown")
        return cls(type=action_type, payload=payload)

    def to_dict(self) -> dict[str, Any]:
        out = {"type": self.type}
        out.update(self.payload)
        return out


@dataclass
class StepRecord:
    step_index: int = 0
    timestamp: str | None = None
    success: bool = False
    error: Any = None
    execution_time_ms: int | None = None
    agent_input: dict[str, Any] = field(default_factory=dict)
    post_execute_output: dict[str, Any] = field(default_factory=dict)
    llm_calls: list[Any] = field(default_factory=list)
    action: ActionRecord | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StepRecord:
        payload = data if isinstance(data, dict) else {}
        action_payload = None
        if isinstance(payload.get("agent_output"), dict):
            action_payload = payload["agent_output"].get("action")
        return cls(
            step_index=int(payload.get("step_index") or 0),
            timestamp=payload.get("timestamp"),
            success=bool(payload.get("success")),
            error=payload.get("error"),
            execution_time_ms=(int(payload.get("execution_time_ms")) if isinstance(payload.get("execution_time_ms"), int | float) else None),
            agent_input=dict(payload.get("agent_input") or {}),
            post_execute_output=dict(payload.get("post_execute_output") or {}),
            llm_calls=list(payload.get("llm_calls") or []),
            action=ActionRecord.from_dict(action_payload) if isinstance(action_payload, dict) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "agent_input": self.agent_input,
            "post_execute_output": self.post_execute_output,
            "llm_calls": self.llm_calls,
            "agent_output": {"action": self.action.to_dict()} if self.action else None,
        }


@dataclass
class TrajectoryRecord:
    trajectory_id: str
    run_id: str
    task_id: str
    source_url: str
    task: TaskInfo
    summary: TrajectorySummary
    actions: list[ActionRecord] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryRecord:
        return cls(
            trajectory_id=str(data.get("trajectory_id") or ""),
            run_id=str(data.get("run_id") or ""),
            task_id=str(data.get("task_id") or ""),
            source_url=str(data.get("source_url") or ""),
            task=TaskInfo.from_dict(data.get("task") if isinstance(data, dict) else None),
            summary=TrajectorySummary.from_dict(data.get("summary") if isinstance(data, dict) else None),
            actions=[ActionRecord.from_dict(a) for a in (data.get("actions") or []) if isinstance(a, dict)],
            steps=[StepRecord.from_dict(s) for s in (data.get("steps") or []) if isinstance(s, dict)],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "source_url": self.source_url,
            "task": self.task.to_dict(),
            "summary": self.summary.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_sft_record(self, *, system_prompt: str | None = None) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append(
            {
                "role": "user",
                "content": f"URL: {self.task.url}\nTask: {self.task.prompt}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps([a.to_dict() for a in self.actions], ensure_ascii=False),
            }
        )
        return {"messages": messages}


__all__ = [
    "ActionRecord",
    "StepRecord",
    "TaskInfo",
    "TrajectoryRecord",
    "TrajectorySummary",
]
