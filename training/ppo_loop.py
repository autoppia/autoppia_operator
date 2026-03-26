from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bs4 import BeautifulSoup

from .dataset import write_jsonl


@dataclass
class PPOLoopConfig:
    num_episodes: int = 20
    max_steps_per_episode: int = 25
    epsilon_exploration: float = 0.05
    score_gain_weight: float = 1.0
    step_penalty: float = 0.01
    action_failure_penalty: float = 0.05
    success_bonus: float = 1.0
    failure_penalty: float = 0.2
    random_seed: int = 42
    web_agent_id_prefix: str = "autoppia-ppo"


@dataclass
class PolicyDecision:
    action: dict[str, Any]
    raw_response: dict[str, Any] = field(default_factory=dict)
    logprob: float | None = None


@dataclass
class PPOStepTransition:
    trajectory_id: str
    task_id: str
    step_index: int
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    score_before: float
    score_after: float
    done: bool
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "step_index": self.step_index,
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "done": self.done,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class PPOEpisode:
    trajectory_id: str
    task_id: str
    final_success: bool
    final_score: float
    total_reward: float
    steps: list[PPOStepTransition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "final_success": self.final_success,
            "final_score": self.final_score,
            "total_reward": self.total_reward,
            "steps": [s.to_dict() for s in self.steps],
        }


class PolicyAdapter(Protocol):
    def decide(self, payload: dict[str, Any]) -> PolicyDecision: ...


class OperatorLLMPolicy:
    """Uses autoppia_operator agent policy as the PPO exploration policy."""

    def __init__(self, *, model_override: str | None = None) -> None:
        self.model_override = model_override
        self._agent = None

    def _get_agent(self):
        if self._agent is not None:
            return self._agent

        from src.operator.agent import AutoppiaOperator

        self._agent = AutoppiaOperator(id="ppo", name="AutoppiaPPOPolicy")
        return self._agent

    @staticmethod
    def _run_async(coro: Any) -> Any:
        return asyncio.run(coro)

    def decide(self, payload: dict[str, Any]) -> PolicyDecision:
        req = dict(payload)
        if self.model_override:
            req["model"] = self.model_override

        agent = self._get_agent()
        response = self._run_async(agent.act_from_payload(req))
        actions = response.get("actions") if isinstance(response, dict) else []
        if not isinstance(actions, list) or not actions:
            return PolicyDecision(action={"type": "WaitAction", "time_seconds": 1.0}, raw_response=response or {})

        first = actions[0] if isinstance(actions[0], dict) else {"type": "WaitAction", "time_seconds": 1.0}
        return PolicyDecision(action=first, raw_response=response if isinstance(response, dict) else {})


def _to_task(task_payload: dict[str, Any]) -> Any:
    from autoppia_iwa.src.data_generation.tasks.classes import Task

    if "task" in task_payload and isinstance(task_payload.get("task"), dict):
        task_payload = task_payload["task"]

    if hasattr(Task, "deserialize") and callable(getattr(Task, "deserialize")):
        try:
            return Task.deserialize(task_payload)
        except Exception:
            pass

    clean = {
        "id": str(task_payload.get("id") or task_payload.get("taskId") or uuid.uuid4()),
        "url": str(task_payload.get("url") or task_payload.get("startUrl") or ""),
        "prompt": str(task_payload.get("prompt") or task_payload.get("taskPrompt") or ""),
        "web_project_id": str(task_payload.get("web_project_id") or task_payload.get("website") or ""),
        "tests": task_payload.get("tests") if isinstance(task_payload.get("tests"), list) else [],
        "is_web_real": bool(task_payload.get("is_web_real") or False),
    }
    return Task(**clean)


def load_tasks(path: Path, *, limit: int | None = None) -> list[Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
        items = raw["tasks"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Task file must contain a JSON list or {'tasks': [...]} object")

    tasks: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            tasks.append(_to_task(item))
        except Exception:
            continue
        if limit is not None and len(tasks) >= int(limit):
            break
    return tasks


def _exploration_action(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html or "", "lxml")
    candidates = soup.select("button[id], a[id], input[id], [role='button'][id]")
    if candidates:
        node = random.choice(candidates)
        node_id = str(node.get("id") or "").strip()
        if node_id:
            return {
                "type": "ClickAction",
                "selector": {
                    "type": "attributeValueSelector",
                    "attribute": "id",
                    "value": node_id,
                },
            }

    return random.choice(
        [
            {"type": "ScrollAction", "down": True, "up": False},
            {"type": "ScrollAction", "down": False, "up": True},
            {"type": "WaitAction", "time_seconds": 1.0},
        ]
    )


class IWAStatefulPPOCollector:
    """Collects PPO-style rollouts using IWA StatefulEvaluator rewards.

    This keeps policy improvement pluggable: output rollouts can be consumed by
    TRL PPO, custom policy-gradient code, or logged for offline RL.
    """

    def __init__(self, *, policy: PolicyAdapter, config: PPOLoopConfig | None = None) -> None:
        self.policy = policy
        self.config = config or PPOLoopConfig()
        random.seed(self.config.random_seed)

    @staticmethod
    def _action_to_payload(action_obj: Any) -> dict[str, Any]:
        if action_obj is None:
            return {"type": "WaitAction", "time_seconds": 1.0}
        if isinstance(action_obj, dict):
            return action_obj
        if hasattr(action_obj, "model_dump"):
            try:
                return action_obj.model_dump(exclude_none=True)
            except Exception:
                pass
        return {"type": str(getattr(action_obj, "type", "WaitAction"))}

    def _create_action(self, action_payload: dict[str, Any]):
        from autoppia_iwa.src.execution.actions.base import BaseAction

        created = BaseAction.create_action(action_payload)
        if created is None:
            created = BaseAction.create_action({"type": "WaitAction", "time_seconds": 1.0})
        return created

    def run_episode(self, task: Any, *, episode_index: int) -> PPOEpisode:
        from autoppia_iwa.src.evaluation.stateful_evaluator.evaluator import StatefulEvaluator

        trajectory_id = f"ppo:{str(getattr(task, 'id', 'task'))}:{uuid.uuid4().hex[:12]}"
        history: list[dict[str, Any]] = []
        transitions: list[PPOStepTransition] = []
        total_reward = 0.0

        web_agent_id = f"{self.config.web_agent_id_prefix}-{episode_index:05d}"
        evaluator = StatefulEvaluator(task=task, web_agent_id=web_agent_id, should_record_gif=False)

        try:
            first = evaluator.reset()
            snapshot = first.snapshot
            score_before = float(first.score.raw_score)

            for step_idx in range(int(self.config.max_steps_per_episode)):
                payload = {
                    "task_id": str(getattr(task, "id", "")),
                    "prompt": str(getattr(task, "prompt", "")),
                    "snapshot_html": snapshot.html,
                    "url": snapshot.url,
                    "step_index": step_idx,
                    "history": history,
                }

                decision = self.policy.decide(payload)
                action_payload = dict(decision.action)

                if random.random() < float(self.config.epsilon_exploration):
                    action_payload = _exploration_action(snapshot.html)

                action_obj = self._create_action(action_payload)
                step = evaluator.step(action_obj)

                score_after = float(step.score.raw_score)
                is_done_action = str(getattr(action_obj, "type", "")) == "DoneAction"
                done = bool(step.score.success) or is_done_action or (step_idx + 1 >= int(self.config.max_steps_per_episode))

                reward = (score_after - score_before) * float(self.config.score_gain_weight)
                reward -= float(self.config.step_penalty)

                action_error = None
                action_ok = True
                if step.action_result is not None:
                    action_ok = bool(getattr(step.action_result, "successfully_executed", True))
                    action_error = getattr(step.action_result, "error", None)
                if not action_ok:
                    reward -= float(self.config.action_failure_penalty)

                if done and bool(step.score.success):
                    reward += float(self.config.success_bonus)
                elif done:
                    reward -= float(self.config.failure_penalty)

                transition = PPOStepTransition(
                    trajectory_id=trajectory_id,
                    task_id=str(getattr(task, "id", "")),
                    step_index=step_idx,
                    observation={
                        "prompt": str(getattr(task, "prompt", "")),
                        "url": snapshot.url,
                        "snapshot_html": snapshot.html,
                        "history_len": len(history),
                    },
                    action=self._action_to_payload(action_obj),
                    reward=float(reward),
                    score_before=float(score_before),
                    score_after=float(score_after),
                    done=bool(done),
                    success=bool(step.score.success if done else False),
                    error=str(action_error) if action_error else None,
                )
                transitions.append(transition)
                total_reward += float(reward)

                history.append(
                    {
                        "step_index": step_idx,
                        "action": self._action_to_payload(action_obj),
                        "success": bool(action_ok),
                        "error": str(action_error) if action_error else None,
                        "current_url": step.snapshot.url,
                    }
                )

                snapshot = step.snapshot
                score_before = score_after
                if done:
                    break

            final = evaluator.get_score_details()
            return PPOEpisode(
                trajectory_id=trajectory_id,
                task_id=str(getattr(task, "id", "")),
                final_success=bool(final.success),
                final_score=float(final.raw_score),
                total_reward=float(total_reward),
                steps=transitions,
            )
        finally:
            evaluator.close()

    def collect(self, tasks: list[Any]) -> list[PPOEpisode]:
        episodes: list[PPOEpisode] = []
        if not tasks:
            return episodes

        target = int(self.config.num_episodes)
        for ep_idx in range(target):
            task = tasks[ep_idx % len(tasks)]
            episodes.append(self.run_episode(task, episode_index=ep_idx))
        return episodes


def export_ppo_collection(out_dir: Path, episodes: list[PPOEpisode]) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = out_dir / "ppo_episodes.jsonl"
    transitions_path = out_dir / "ppo_transitions.jsonl"

    write_jsonl(episodes_path, [ep.to_dict() for ep in episodes])
    all_steps = [step.to_dict() for ep in episodes for step in ep.steps]
    write_jsonl(transitions_path, all_steps)

    summary = {
        "episodes": len(episodes),
        "steps": len(all_steps),
        "successes": sum(1 for ep in episodes if ep.final_success),
        "avg_final_score": (
            sum(ep.final_score for ep in episodes) / len(episodes) if episodes else 0.0
        ),
        "avg_total_reward": (
            sum(ep.total_reward for ep in episodes) / len(episodes) if episodes else 0.0
        ),
    }

    summary_path = out_dir / "ppo_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "episodes": str(episodes_path),
        "transitions": str(transitions_path),
        "summary": str(summary_path),
    }


__all__ = [
    "IWAStatefulPPOCollector",
    "OperatorLLMPolicy",
    "PPOEpisode",
    "PPOLoopConfig",
    "PPOStepTransition",
    "PolicyDecision",
    "export_ppo_collection",
    "load_tasks",
]
