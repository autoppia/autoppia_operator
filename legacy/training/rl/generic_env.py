from __future__ import annotations

import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import autoppia_iwa.src.execution.actions  # noqa: F401 ensures registry is populated
from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.execution.actions.base import BaseAction
from autoppia_iwa.src.evaluation.stateful_evaluator import StepResult

from training.rl.reward import compute_step_reward

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_step_engine_components():
    local_src_root = REPO_ROOT / "src"
    current_src = sys.modules.get("src")
    if current_src is None or not str(getattr(current_src, "__file__", "")).startswith(str(local_src_root)):
        sys.modules.pop("src", None)
        src_init = local_src_root / "__init__.py"
        src_spec = importlib.util.spec_from_file_location(
            "src",
            src_init,
            submodule_search_locations=[str(local_src_root)],
        )
        if src_spec is None or src_spec.loader is None:
            raise ModuleNotFoundError(f"Unable to load local src package from {src_init}")
        src_module = importlib.util.module_from_spec(src_spec)
        sys.modules["src"] = src_module
        src_spec.loader.exec_module(src_module)
    import src.operator.agents.step_engine as step_engine_module
    from src.operator.agents.step_engine.engine import StepEngine as _StepEngineClass
    from src.operator.agents.step_engine.state import AgentState
    from src.operator.agents.step_engine.utils import _supported_browser_tool_names
    from src.operator.eval.session import build_task_execution_session as _build_task_execution_session

    return (
        step_engine_module._STEP_ENGINE,
        _StepEngineClass,
        AgentState,
        _supported_browser_tool_names,
        _build_task_execution_session,
    )


def _load_raw_tasks(cache_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    raw_tasks = payload.get("tasks") if isinstance(payload, dict) and isinstance(payload.get("tasks"), list) else payload
    if not isinstance(raw_tasks, list):
        return []
    return [row for row in raw_tasks if isinstance(row, dict)]


def _extract_use_case_name(task_payload: dict[str, Any]) -> str:
    use_case = task_payload.get("use_case")
    if isinstance(use_case, dict):
        return str(use_case.get("name") or "").strip()
    return str(use_case or "").strip()


def inject_seed(task: Task, seed: int | None = None) -> tuple[Task, int]:
    task_copy = Task(**task.model_dump(mode="python"))
    seed_i = int(seed) if seed is not None else random.randint(1, 100_000)
    base_url = str(task_copy.url or "")
    if "?" in base_url:
        base_url = base_url.split("?", 1)[0]
    task_copy.url = f"{base_url}?seed={seed_i}"
    return task_copy, seed_i


def build_task_from_cache(
    *,
    cache_path: str | Path,
    web_project_id: str,
    use_case: str,
    seed: int,
) -> Task:
    cache = Path(cache_path).expanduser().resolve()
    rows = _load_raw_tasks(cache)
    bucket = [
        row
        for row in rows
        if str(row.get("web_project_id") or "").strip() == str(web_project_id).strip()
        and _extract_use_case_name(row).upper() == str(use_case).strip().upper()
    ]
    if not bucket:
        raise ValueError(f"No task found for web_project_id={web_project_id} use_case={use_case} in {cache}")
    rnd = random.Random(int(seed))
    rnd.shuffle(bucket)
    task = Task(**bucket[0])
    injected, _ = inject_seed(task, seed=int(seed))
    return injected


def _extract_task_target_values(task: Task) -> list[str]:
    values: list[str] = []
    for test in getattr(task, "tests", []) or []:
        if not isinstance(test, dict):
            continue
        criteria = test.get("event_criteria")
        if not isinstance(criteria, dict):
            continue
        for raw in criteria.values():
            if isinstance(raw, dict):
                operator = str(raw.get("operator") or "").strip().lower()
                if operator in {"not_contains", "not_equal", "not_equals"}:
                    continue
                value = raw.get("value")
                if isinstance(value, (str, int, float)):
                    text = str(value).strip()
                    if text and text not in values:
                        values.append(text)
            elif isinstance(raw, (str, int, float)):
                text = str(raw).strip()
                if text and text not in values:
                    values.append(text)
    return values


@dataclass
class RLStepRecord:
    episode_step: int
    policy_input_text: str
    policy_output: dict[str, Any]
    chosen_action: dict[str, Any] | None
    env_action: dict[str, Any] | None
    exec_ok: bool
    error: str | None
    reward: dict[str, Any]
    before_url: str
    after_url: str
    before_score: float
    after_score: float
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenericRLEnv:
    def __init__(
        self,
        *,
        web_project_id: str,
        use_case: str,
        seed: int,
        task_cache_path: str | Path,
        model_override: str = "",
        max_steps: int = 12,
        capture_screenshot: bool = False,
        headless: bool | None = None,
        step_engine_instance: Any | None = None,
    ) -> None:
        self.web_project_id = str(web_project_id).strip()
        self.use_case = str(use_case).strip().upper()
        self.seed = int(seed)
        self.model_override = str(model_override or "")
        self.max_steps = max(1, int(max_steps))
        self.task = build_task_from_cache(
            cache_path=task_cache_path,
            web_project_id=self.web_project_id,
            use_case=self.use_case,
            seed=self.seed,
        )
        self.target_values = _extract_task_target_values(self.task)
        default_step_engine, self.StepEngineClass, self.AgentState, self.supported_browser_tool_names, self._build_task_execution_session = _load_step_engine_components()
        self.step_engine = step_engine_instance if step_engine_instance is not None else default_step_engine
        self.capture_screenshot = bool(capture_screenshot)
        self.headless = headless
        self.session = None
        self.internal_state: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.rollout_steps: list[RLStepRecord] = []
        self.last_step_result: StepResult | None = None
        self.web_agent_id = f"generic-rl-{self.web_project_id}-{self.use_case}-{self.seed}-{random.randint(1000, 9999)}"
        self.validator_id = f"generic-rl-validator-{self.web_project_id}-{self.use_case}-{self.seed}-{random.randint(1000, 9999)}"

    def _allowed_tools_for_step(self, step_index: int) -> list[str]:
        return sorted(self.supported_browser_tool_names())

    def _score_feedback(self, step_result: StepResult) -> dict[str, Any]:
        return {
            "raw_score": float(step_result.score.raw_score),
            "tests_passed": int(step_result.score.tests_passed),
            "total_tests": int(step_result.score.total_tests),
            "success": bool(step_result.score.success),
        }

    def _use_case_payload(self) -> dict[str, str]:
        raw = getattr(self.task, "use_case", None)
        if isinstance(raw, dict):
            return {
                "id": str(raw.get("id") or self.use_case),
                "name": str(raw.get("name") or self.use_case),
            }
        name = str(getattr(raw, "name", "") or self.use_case)
        identifier = str(getattr(raw, "id", "") or name or self.use_case)
        return {"id": identifier, "name": name or self.use_case}

    def _build_payload(self, *, step_result: StepResult, step_index: int) -> dict[str, Any]:
        return {
            "task_id": str(self.task.id),
            "prompt": str(self.task.prompt),
            "web_project_id": str(self.task.web_project_id or self.web_project_id),
            "use_case": self._use_case_payload(),
            "url": str(step_result.snapshot.url or self.task.url),
            "snapshot_html": str(step_result.snapshot.html or ""),
            "screenshot": getattr(step_result.snapshot, "screenshot", None),
            "history": list(self.history),
            "internal_state": dict(self.internal_state),
            "score_feedback": self._score_feedback(step_result),
            "allowed_tools": self._allowed_tools_for_step(int(step_index)),
            "step_index": int(step_index),
            "include_reasoning": True,
        }

    def _prepare_prompt(self, *, step_result: StepResult, step_index: int) -> str:
        state = self.AgentState.from_internal_state(self.internal_state, prompt=str(self.task.prompt))
        prepared = self.step_engine._prepare_run_context(
            task_id=str(self.task.id),
            prompt=str(self.task.prompt),
            web_project_id=str(self.task.web_project_id or self.web_project_id),
            use_case=self._use_case_payload(),
            url=str(step_result.snapshot.url or self.task.url),
            html=str(step_result.snapshot.html or ""),
            screenshot=getattr(step_result.snapshot, "screenshot", None),
            step_index=int(step_index),
            history=list(self.history),
            state=state,
            allowed=set(self._allowed_tools_for_step(int(step_index))),
            model_override=self.model_override,
        )
        policy_obs = prepared.get("policy_obs") if isinstance(prepared.get("policy_obs"), dict) else {}
        return str(policy_obs.get("policy_input_text") or "")

    async def reset(self) -> StepResult:
        self.session = self._build_task_execution_session(
            task=self.task,
            web_agent_id=self.web_agent_id,
            validator_id=self.validator_id,
            enable_score_cheating=False,
            capture_screenshot=self.capture_screenshot,
            headless=self.headless,
        )
        self.internal_state = {}
        self.history = []
        self.rollout_steps = []
        self.last_step_result = await self.session.reset()
        return self.last_step_result

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def step(self, step_index: int) -> tuple[RLStepRecord, bool]:
        if self.session is None or self.last_step_result is None:
            raise RuntimeError("Environment must be reset before stepping")
        before = self.last_step_result
        before_score = float(before.score.raw_score)
        before_url = str(before.snapshot.url or self.task.url)
        before_html = str(before.snapshot.html or "")
        policy_input_text = self._prepare_prompt(step_result=before, step_index=step_index)
        payload = self._build_payload(step_result=before, step_index=step_index)
        policy_output = self.step_engine.run(payload=payload, model_override=self.model_override)
        self.internal_state = dict(policy_output.get("internal_state") or {})
        done = bool(policy_output.get("done"))
        actions_data = policy_output.get("actions") if isinstance(policy_output.get("actions"), list) else []
        chosen_action = dict(actions_data[0]) if actions_data and isinstance(actions_data[0], dict) else None
        env_action = None
        exec_ok = True
        error = None
        after = before
        if done and not actions_data:
            after = before
        elif actions_data:
            try:
                for action_payload in actions_data:
                    if not isinstance(action_payload, dict):
                        continue
                    env_action = dict(action_payload)
                    action = BaseAction.create_action(env_action)
                    if action is None:
                        exec_ok = False
                        error = "invalid_action"
                        break
                    if str(getattr(action, "type", "")) == "DoneAction":
                        done = True
                        break
                    after = await self.session.step(action)
                    self.last_step_result = after
                    action_result = after.action_result
                    if action_result is not None and not bool(getattr(action_result, "successfully_executed", True)):
                        exec_ok = False
                        error = str(getattr(action_result, "error", None) or "")
                        break
                else:
                    after = self.last_step_result
            except Exception as exc:
                exec_ok = False
                error = f"{type(exc).__name__}: {exc}"
                after = before
                self.last_step_result = after
                done = True
        else:
            done = True
        after = self.last_step_result if self.last_step_result is not None else before
        after_score = float(after.score.raw_score)
        after_url = str(after.snapshot.url or before_url)
        after_html = str(after.snapshot.html or "")
        previous_action = self.history[-1].get("action") if self.history and isinstance(self.history[-1], dict) else None
        reward = compute_step_reward(
            step_index=int(step_index),
            prev_score=before_score,
            current_score=after_score,
            success=bool(after.score.success),
            exec_ok=bool(exec_ok),
            current_url=after_url,
            chosen_action=env_action or chosen_action,
            previous_action=previous_action if isinstance(previous_action, dict) else None,
            previous_url=before_url,
            before_html=before_html,
            after_html=after_html,
            target_values=self.target_values,
            task_prompt=str(self.task.prompt or ""),
        ).to_dict()
        executed_action = env_action or chosen_action
        self.history.append(
            {
                "step": int(step_index),
                "url": str(after_url),
                "action": dict(executed_action or {}),
                "done": bool(done),
                "exec_ok": bool(exec_ok),
                "error": str(error or ""),
                "text": str((executed_action or {}).get("text") or (executed_action or {}).get("value") or ""),
            }
        )
        record = RLStepRecord(
            episode_step=int(step_index),
            policy_input_text=policy_input_text,
            policy_output=dict(policy_output or {}),
            chosen_action=chosen_action,
            env_action=env_action,
            exec_ok=bool(exec_ok),
            error=error,
            reward=reward,
            before_url=before_url,
            after_url=after_url,
            before_score=float(before_score),
            after_score=float(after_score),
            success=bool(after.score.success),
        )
        self.rollout_steps.append(record)
        done = bool(done or after.score.success or int(step_index) + 1 >= self.max_steps)
        return record, done


__all__ = ["GenericRLEnv", "RLStepRecord", "build_task_from_cache", "inject_seed"]
