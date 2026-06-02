from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.parse import urljoin
from urllib.parse import urlsplit

import autoppia_iwa.src.execution.actions  # noqa: F401 ensures action registry is populated
from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.execution.actions.base import BaseAction
from autoppia_iwa.src.evaluation.stateful_evaluator import StepResult
from training.rl.reward import RewardBreakdown, compute_step_reward

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONTACT_TASK_CACHE_DIR = REPO_ROOT / "data" / "autocinema" / "contact" / "task_cache"
DEFAULT_CONTACT_GOLD_RUNS_DIR = REPO_ROOT / "data" / "autocinema" / "contact" / "gold" / "runs"


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
    from src.operator.eval.session import build_task_execution_session as _build_task_execution_session
    from src.operator.agents.step_engine.state import AgentState
    from src.operator.agents.step_engine.utils import _supported_browser_tool_names
    return step_engine_module._STEP_ENGINE, _StepEngineClass, AgentState, _supported_browser_tool_names, _build_task_execution_session


@dataclass
class Step0Candidate:
    action: dict[str, Any]
    label: str
    source: str
    score: float


@dataclass
class ContactFieldTarget:
    key: str
    value: str
    selector_id: str
    aliases: tuple[str, ...] = ()


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


def build_contact_task(*, seed: int, task_cache_dir: str | Path = DEFAULT_CONTACT_TASK_CACHE_DIR) -> Task:
    cache_dir = Path(task_cache_dir).expanduser().resolve()
    candidate = cache_dir / f"contact_seed_{int(seed):04d}_claude_01.json"
    if not candidate.exists():
        raise FileNotFoundError(f"CONTACT task cache missing for seed {seed}: {candidate}")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        raise ValueError(f"Invalid CONTACT task cache: {candidate}")
    for row in tasks:
        use_case = row.get("use_case") if isinstance(row.get("use_case"), dict) else {}
        tests = row.get("tests") if isinstance(row.get("tests"), list) else []
        if str(use_case.get("name") or "").upper() == "CONTACT":
            clean_row = dict(row)
            clean_row["prompt"] = _canonical_contact_prompt(clean_row)
            return Task(**clean_row)
        if any(str((test or {}).get("event_name") or "").upper() == "CONTACT" for test in tests if isinstance(test, dict)):
            clean_row = dict(row)
            clean_row["prompt"] = _canonical_contact_prompt(clean_row)
            return Task(**clean_row)
    raise ValueError(f"No CONTACT task found in cache: {candidate}")


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


def _canonical_contact_prompt(row: dict[str, Any]) -> str:
    tests = row.get("tests") if isinstance(row.get("tests"), list) else []
    criteria: dict[str, Any] = {}
    for test in tests:
        if not isinstance(test, dict):
            continue
        event_criteria = test.get("event_criteria")
        if isinstance(event_criteria, dict):
            criteria = event_criteria
            break
    name = str(criteria.get("name") or "David").strip()
    email_raw = criteria.get("email")
    if isinstance(email_raw, dict):
        email = str(email_raw.get("value") or "user1@site.com").strip()
    else:
        email = str(email_raw or "user1@site.com").strip()
    subject_raw = criteria.get("subject")
    if isinstance(subject_raw, dict):
        operator = str(subject_raw.get("operator") or "").strip().lower()
        subject_value = str(subject_raw.get("value") or "Information").strip()
        if operator in {"not_contains", "not_equal", "not_equals"}:
            subject_text = f"a subject that does NOT contain '{subject_value}'"
        else:
            subject_text = f"a subject that equals '{subject_value}'"
    else:
        subject_value = str(subject_raw or "").strip()
        subject_text = f"a subject that equals '{subject_value}'" if subject_value else "a subject field"
    message = str(criteria.get("message") or "Please provide me with more information").strip()
    return (
        f"Fill out the contact form with a name that equals '{name}', "
        f"an email that contains '{email}', "
        f"{subject_text}, "
        f"and a message that equals '{message}'."
    )


def _extract_prompt_target_map(task_prompt: str | None) -> dict[str, str]:
    prompt = str(task_prompt or "")
    out: dict[str, str] = {}
    for label, value in re.findall(
        r"\b([A-Za-z0-9 _-]+?)\s+that\s+(?:equals|contains)\s+'([^']*)'",
        prompt,
        flags=re.IGNORECASE,
    ):
        key = re.sub(r"\s+", " ", str(label or "").strip().lower())
        key = re.sub(r"^(?:a|an|the)\s+", "", key).strip()
        value = str(value or "").strip()
        if "email" in key or "e-mail" in key or key.endswith(" mail"):
            key = "email"
        elif "subject" in key or "topic" in key or "title" in key:
            key = "subject"
        elif "message" in key or "comment" in key or "details" in key or "description" in key or "body" in key:
            key = "message"
        elif re.search(r"\bname\b", key) and "username" not in key:
            key = "name"
        if key and value and key not in out:
            out[key] = value
    if "subject" not in out:
        m_subject_bad = re.search(
            r"\bsubject\b[^.]{0,80}\bdoes\s+not\s+contain\s+'([^']+)'",
            prompt,
            flags=re.IGNORECASE,
        )
        if m_subject_bad:
            forbidden = str(m_subject_bad.group(1) or "").strip().lower()
            if forbidden:
                fallback = "Inquiry"
                if forbidden in fallback.lower():
                    fallback = "Request"
                out["subject"] = fallback
    return out


def _field_targets_from_prompt(task_prompt: str | None) -> list[ContactFieldTarget]:
    targets = _extract_prompt_target_map(task_prompt)
    specs = [
        ("name", "contact-name-input", ("name",)),
        ("email", "contact-email-input", ("email", "e-mail")),
        ("subject", "contact-subject-input", ("subject", "topic", "title")),
        ("message", "contact-message-input", ("message", "comment", "details", "description", "body")),
    ]
    out: list[ContactFieldTarget] = []
    for key, selector_id, aliases in specs:
        value = str(targets.get(key) or "").strip()
        if value:
            out.append(ContactFieldTarget(key=key, value=value, selector_id=selector_id, aliases=aliases))
    return out


def _load_reference_tool_calls(*, seed: int, gold_runs_dir: str | Path = DEFAULT_CONTACT_GOLD_RUNS_DIR) -> list[dict[str, Any]]:
    path = Path(gold_runs_dir).expanduser().resolve() / f"seed_{int(seed):04d}_claude_01.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, list) or not episodes:
        return []
    guided = episodes[0].get("guided_execution") if isinstance(episodes[0], dict) else None
    if not isinstance(guided, list):
        return []
    tool_calls: list[dict[str, Any]] = []
    for step in guided:
        tool_call = step.get("policy_tool_call") if isinstance(step, dict) else None
        if isinstance(tool_call, dict):
            tool_calls.append(tool_call)
    return tool_calls


def _pathish(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return str(urlparse(value).path or "").lower()
    return value


def _submit_like(chosen_action: dict[str, Any] | None) -> bool:
    if not isinstance(chosen_action, dict):
        return False
    selector = chosen_action.get("selector") if isinstance(chosen_action.get("selector"), dict) else {}
    blob = " ".join(
        [
            str(chosen_action.get("type") or ""),
            str(chosen_action.get("url") or ""),
            str(selector.get("attribute") or ""),
            str(selector.get("value") or ""),
            str(chosen_action.get("_element_id") or ""),
        ]
    ).lower()
    return bool(re.search(r"\b(send|submit|save|continue|confirm)\b", blob))


def _teacher_shaping(
    *,
    step_index: int,
    chosen_action: dict[str, Any] | None,
    current_url: str,
    reference_tool_calls: list[dict[str, Any]],
) -> tuple[float, float, str]:
    if step_index < 0 or step_index >= len(reference_tool_calls):
        return 0.0, 0.0, ""
    if not isinstance(chosen_action, dict):
        return 0.0, 0.20, "teacher_missing_action"
    tool_call = reference_tool_calls[step_index]
    name = str(tool_call.get("name") or "").strip().lower()
    arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
    action_type = str(chosen_action.get("type") or "")
    chosen_text = str(chosen_action.get("text") or chosen_action.get("value") or "").strip().lower()
    chosen_url = str(chosen_action.get("url") or "").strip().lower()
    on_contact = "/contact" in str(current_url or "").lower()
    if name == "browser.navigate":
        expected_path = _pathish(arguments.get("url"))
        if expected_path and (
            (action_type == "NavigateAction" and expected_path in _pathish(chosen_url))
            or (action_type == "ClickAction" and expected_path in _pathish(chosen_url))
        ):
            if step_index == 0:
                return 2.50, 0.0, "teacher_route_match"
            return 1.00, 0.0, "teacher_route_match"
        if action_type == "TypeAction":
            if step_index == 0:
                return 0.0, 1.20, "teacher_route_miss_type"
            return 0.0, 0.40, "teacher_route_miss_type"
        if step_index == 0:
            return 0.0, 0.80, "teacher_route_miss"
        return 0.0, 0.25, "teacher_route_miss"
    if name == "browser.input":
        expected_text = str(arguments.get("text") or "").strip().lower()
        if on_contact and action_type == "TypeAction" and expected_text and chosen_text == expected_text:
            return 0.35, 0.0, "teacher_input_match"
        if action_type == "TypeAction" and expected_text and chosen_text and chosen_text != expected_text:
            return 0.0, 0.15, "teacher_input_wrong_text"
        return 0.0, 0.10, "teacher_input_miss"
    if name == "browser.click":
        if on_contact and action_type == "ClickAction" and _submit_like(chosen_action):
            return 0.35, 0.0, "teacher_click_match"
        return 0.0, 0.10, "teacher_click_miss"
    return 0.0, 0.0, ""


class ContactRLEnv:
    def __init__(
        self,
        *,
        seed: int,
        model_override: str = "",
        max_steps: int = 12,
        task_cache_dir: str | Path = DEFAULT_CONTACT_TASK_CACHE_DIR,
        capture_screenshot: bool = False,
        headless: bool | None = None,
        step_engine_instance: Any | None = None,
    ) -> None:
        self.seed = int(seed)
        self.model_override = str(model_override or "")
        self.max_steps = max(1, int(max_steps))
        self.task = build_contact_task(seed=self.seed, task_cache_dir=task_cache_dir)
        self.target_values = _extract_task_target_values(self.task)
        self.reference_tool_calls = _load_reference_tool_calls(seed=self.seed)
        self.capture_screenshot = bool(capture_screenshot)
        self.headless = headless
        default_step_engine, self.StepEngineClass, self.AgentState, self.supported_browser_tool_names, self._build_task_execution_session = _load_step_engine_components()
        self.step_engine = step_engine_instance if step_engine_instance is not None else default_step_engine
        self.session = None
        self.internal_state: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.rollout_steps: list[RLStepRecord] = []
        self.last_step_result: StepResult | None = None
        self.web_agent_id = f"contact-rl-{self.seed}-{random.randint(1000, 9999)}"
        self.validator_id = f"contact-rl-validator-{self.seed}-{random.randint(1000, 9999)}"

    def _allowed_tools_for_step(self, step_index: int) -> list[str]:
        allowed = sorted(self.supported_browser_tool_names())
        if int(step_index) == 0:
            blocked = {"browser.scroll", "browser.wait", "browser.search"}
            allowed = [tool for tool in allowed if tool not in blocked]
        elif self.last_step_result is not None and "/contact" in str(self.last_step_result.snapshot.url or "").lower():
            blocked = {"browser.scroll", "browser.wait", "browser.search"}
            allowed = [tool for tool in allowed if tool not in blocked]
        return allowed

    def _build_step0_candidates(self, *, current_url: str, html: str) -> list[Step0Candidate]:
        out: list[Step0Candidate] = []
        seen: set[str] = set()
        prompt = str(self.task.prompt or "").lower()

        def add(action: dict[str, Any], label: str, source: str, score: float) -> None:
            key = json.dumps(action, ensure_ascii=False, sort_keys=True)
            if key in seen:
                return
            seen.add(key)
            out.append(Step0Candidate(action=action, label=label, source=source, score=float(score)))

        route_hints = []
        if "contact" in prompt:
            route_hints.append(("/contact", 10.0))
        route_hints.extend([("/support", 5.0), ("/about", 2.0)])
        for path, bonus in route_hints:
            add(
                {"type": "NavigateAction", "url": urljoin(current_url or "http://example.invalid", path), "go_back": False, "go_forward": False},
                f"navigate:{path}",
                "route_guess",
                bonus,
            )

        if html and BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(str(html or ""), "html.parser")
            except Exception:
                soup = None
            if soup is not None:
                for link in soup.find_all("a", href=True):
                    href = str(link.get("href") or "").strip()
                    if not href:
                        continue
                    parsed_href = urlsplit(urljoin(current_url or "http://example.invalid", href))
                    text = " ".join(link.get_text(" ", strip=True).split()).strip()
                    blob = f"{text} {href}".lower()
                    score = 0.0
                    if "contact" in blob:
                        score += 12.0
                    if any(token in blob for token in ("support", "message", "email", "get in touch")):
                        score += 6.0
                    if "about" in blob:
                        score += 2.0
                    if parsed_href.query:
                        score -= 3.0
                    if "seed=" in parsed_href.query.lower():
                        score -= 4.0
                    if score <= 0.0:
                        continue
                    add(
                        {"type": "NavigateAction", "url": parsed_href.geturl(), "go_back": False, "go_forward": False},
                        f"visible_link:{text or href}",
                        "visible_link",
                        score,
                    )

        out.sort(key=lambda item: float(item.score), reverse=True)
        return out[:12]

    def _build_contact_workflow_candidates(self, *, current_url: str, html: str) -> list[Step0Candidate]:
        out: list[Step0Candidate] = []
        seen: set[str] = set()

        def add(action: dict[str, Any], label: str, source: str, score: float) -> None:
            key = json.dumps(action, ensure_ascii=False, sort_keys=True)
            if key in seen:
                return
            seen.add(key)
            out.append(Step0Candidate(action=action, label=label, source=source, score=float(score)))

        if "/contact" not in str(current_url or "").lower():
            return out
        soup = None
        if html and BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(str(html or ""), "html.parser")
            except Exception:
                soup = None

        def find_element(selector_id: str, aliases: tuple[str, ...]) -> Any | None:
            if soup is None:
                return None
            direct = soup.find(id=selector_id)
            if direct is not None:
                return direct
            for tag in soup.find_all(["input", "textarea", "button"]):
                blob = " ".join(
                    [
                        str(tag.get("id") or ""),
                        str(tag.get("name") or ""),
                        str(tag.get("placeholder") or ""),
                        " ".join(tag.get("class") or []),
                        tag.get_text(" ", strip=True),
                    ]
                ).lower()
                if any(alias in blob for alias in aliases):
                    return tag
            return None

        def element_has_value(tag: Any, target: str) -> bool:
            if tag is None:
                return False
            pieces = [
                str(tag.get("value") or ""),
                str(tag.get_text(" ", strip=True) or ""),
                str(tag.get("placeholder") or ""),
            ]
            haystack = " ".join(pieces).lower()
            return bool(target and target.lower() in haystack)

        targets = _field_targets_from_prompt(self.task.prompt)
        all_filled = True
        for idx, target in enumerate(targets):
            tag = find_element(target.selector_id, target.aliases)
            if tag is not None and element_has_value(tag, target.value):
                continue
            all_filled = False
            selector_id = str(tag.get("id") or target.selector_id).strip() if tag is not None else target.selector_id
            selector_id = selector_id or target.selector_id
            add(
                {
                    "type": "TypeAction",
                    "selector": {
                        "type": "attributeValueSelector",
                        "attribute": "id",
                        "value": selector_id,
                        "case_sensitive": False,
                    },
                    "text": target.value,
                },
                f"type:{target.key}:{selector_id}",
                "contact_form",
                20.0 - float(idx),
            )

        if all_filled:
            submit_button = soup.find(id="send-message-button") if soup is not None else None
            if submit_button is None and soup is not None:
                for tag in soup.find_all(["button", "input"]):
                    blob = " ".join(
                        [
                            str(tag.get("id") or ""),
                            str(tag.get("name") or ""),
                            str(tag.get("value") or ""),
                            tag.get_text(" ", strip=True),
                        ]
                    ).lower()
                    if any(token in blob for token in ("send", "submit", "message", "contact")):
                        submit_button = tag
                        break
            selector_id = str(submit_button.get("id") or "send-message-button").strip() if submit_button is not None else "send-message-button"
            selector_id = selector_id or "send-message-button"
            add(
                {
                    "type": "ClickAction",
                    "selector": {
                        "type": "attributeValueSelector",
                        "attribute": "id",
                        "value": selector_id,
                        "case_sensitive": False,
                    },
                },
                f"click_submit:{selector_id}",
                "contact_submit",
                25.0,
            )

        out.sort(key=lambda item: float(item.score), reverse=True)
        return out[:12]

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

    def _score_feedback(self, step_result: StepResult) -> dict[str, Any]:
        return {
            "raw_score": float(step_result.score.raw_score),
            "tests_passed": int(step_result.score.tests_passed),
            "total_tests": int(step_result.score.total_tests),
            "success": bool(step_result.score.success),
        }

    def _build_payload(self, *, step_result: StepResult, step_index: int) -> dict[str, Any]:
        return {
            "task_id": str(self.task.id),
            "prompt": str(self.task.prompt),
            "web_project_id": str(self.task.web_project_id or ""),
            "use_case": {"id": "CONTACT", "name": "CONTACT"},
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
            web_project_id=str(self.task.web_project_id or "autocinema"),
            use_case={"id": "CONTACT", "name": "CONTACT"},
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

    async def step(self, step_index: int) -> tuple[RLStepRecord, bool]:
        if self.session is None or self.last_step_result is None:
            raise RuntimeError("Environment must be reset before stepping")
        before = self.last_step_result
        before_score = float(before.score.raw_score)
        before_url = str(before.snapshot.url or self.task.url)
        before_html = str(before.snapshot.html or "")
        policy_input_text = self._prepare_prompt(step_result=before, step_index=step_index)
        payload = self._build_payload(step_result=before, step_index=step_index)
        structured_candidates: list[Step0Candidate] = []
        structured_reason = ""
        if "/contact" in before_url.lower():
            structured_candidates = self._build_contact_workflow_candidates(current_url=before_url, html=before_html)
            structured_reason = "Structured contact-workflow selector"
        elif int(step_index) == 0:
            structured_candidates = self._build_step0_candidates(current_url=before_url, html=before_html)
            structured_reason = "Structured step-0 selector"

        if structured_candidates:
            best = structured_candidates[0]
            policy_output = {
                "actions": [dict(best.action)],
                "done": False,
                "content": "",
                "reasoning": f"{structured_reason} chose {best.label} from {best.source}.",
                "step0_candidates": [
                    {"action": dict(item.action), "label": item.label, "source": item.source, "score": float(item.score)}
                    for item in structured_candidates
                ],
            }
        elif int(step_index) == 0:
            step0_candidates = self._build_step0_candidates(current_url=before_url, html=before_html)
            if step0_candidates:
                best = step0_candidates[0]
                policy_output = {
                    "actions": [dict(best.action)],
                    "done": False,
                    "content": "",
                    "reasoning": f"Structured step-0 selector chose {best.label} from {best.source}.",
                    "step0_candidates": [
                        {"action": dict(item.action), "label": item.label, "source": item.source, "score": float(item.score)}
                        for item in step0_candidates
                    ],
                }
            else:
                policy_output = self.step_engine.run(payload=payload, model_override=self.model_override)
        else:
            policy_output = self.step_engine.run(payload=payload, model_override=self.model_override)
        self.internal_state = dict(policy_output.get("internal_state") or {})
        chosen_action = None
        actions_data = policy_output.get("actions") if isinstance(policy_output.get("actions"), list) else []
        if actions_data:
            chosen_action = dict(actions_data[0] or {}) if isinstance(actions_data[0], dict) else None
            chosen_blob = json.dumps(chosen_action, ensure_ascii=False).lower() if isinstance(chosen_action, dict) else ""
            if "/contact" in before_url.lower() and any(token in chosen_blob for token in ("/register", "register-", "login", "/login")):
                structured_candidates = self._build_contact_workflow_candidates(current_url=before_url, html=before_html)
                if structured_candidates:
                    best = structured_candidates[0]
                    policy_output = {
                        "actions": [dict(best.action)],
                        "done": False,
                        "content": "",
                        "reasoning": f"Blocked non-contact action inside contact workflow; chose {best.label} from {best.source}.",
                        "step0_candidates": [
                            {"action": dict(item.action), "label": item.label, "source": item.source, "score": float(item.score)}
                            for item in structured_candidates
                        ],
                    }
                    actions_data = policy_output["actions"]
                    chosen_action = dict(actions_data[0])
        done = bool(policy_output.get("done"))
        exec_ok = True
        error = None
        env_action = None
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
                    ar = after.action_result
                    if ar is not None and not bool(getattr(ar, "successfully_executed", True)):
                        exec_ok = False
                        error = str(getattr(ar, "error", None) or "")
                        break
                else:
                    after = self.last_step_result
            except Exception as exc:
                retryable = (
                    env_action is not None
                    and str((env_action or {}).get("type") or "") in {"NavigateAction", "TypeAction", "ClickAction"}
                    and type(exc).__name__ == "TimeoutError"
                )
                if retryable:
                    try:
                        await asyncio.sleep(0.5)
                        action = BaseAction.create_action(env_action)
                        if action is None:
                            raise exc
                        after = await self.session.step(action)
                        self.last_step_result = after
                        ar = after.action_result
                        if ar is not None and not bool(getattr(ar, "successfully_executed", True)):
                            exec_ok = False
                            error = str(getattr(ar, "error", None) or "")
                        else:
                            exec_ok = True
                            error = None
                    except Exception as retry_exc:
                        recovered = False
                        if self.session is not None and type(retry_exc).__name__ == "TimeoutError":
                            try:
                                score = await self.session._score_async()
                                snapshot = await self.session._snapshot_async()
                                recovered_after = StepResult(score=score, snapshot=snapshot, action_result=None)
                                self.last_step_result = recovered_after
                                after = recovered_after
                                recovered_url = str(snapshot.url or "")
                                if "/contact" in recovered_url.lower():
                                    exec_ok = True
                                    error = None
                                    recovered = True
                            except Exception:
                                recovered = False
                        if not recovered:
                            exec_ok = False
                            error = f"{type(retry_exc).__name__}: {retry_exc}"
                            after = before
                            self.last_step_result = after
                            done = True
                else:
                    exec_ok = False
                    error = f"{type(exc).__name__}: {exc}"
                    after = before
                    self.last_step_result = after
                    done = True
        else:
            try:
                after = await self.session.step(None)
                self.last_step_result = after
            except Exception as exc:
                exec_ok = False
                error = f"{type(exc).__name__}: {exc}"
                after = before
                self.last_step_result = after
                done = True
        after = self.last_step_result if self.last_step_result is not None else before
        reward = compute_step_reward(
            step_index=int(step_index),
            prev_score=before_score,
            current_score=float(after.score.raw_score),
            success=bool(after.score.success),
            exec_ok=exec_ok,
            current_url=str(after.snapshot.url or before_url),
            chosen_action=chosen_action,
            previous_action=self.history[-1]["action_payload"] if self.history and isinstance(self.history[-1], dict) else None,
            previous_url=before_url,
            before_html=before_html,
            after_html=str(after.snapshot.html or ""),
            target_values=self.target_values,
            task_prompt=str(self.task.prompt or ""),
        )
        teacher_bonus, teacher_penalty, teacher_reason = _teacher_shaping(
            step_index=int(step_index),
            chosen_action=chosen_action,
            current_url=str(after.snapshot.url or before_url),
            reference_tool_calls=self.reference_tool_calls,
        )
        reward_dict = reward.to_dict()
        reward_dict["teacher_bonus"] = float(teacher_bonus)
        reward_dict["teacher_penalty"] = float(teacher_penalty)
        reward_dict["teacher_total"] = float(teacher_bonus - teacher_penalty)
        reward_dict["teacher_reason"] = teacher_reason
        reward_dict["total"] = float(reward_dict.get("total", 0.0) + teacher_bonus - teacher_penalty)
        history_item = {
            "step": len(self.history),
            "episode_step": int(step_index),
            "url": str(after.snapshot.url or before_url),
            "action": str((env_action or {}).get("type") or ("done" if done else "noop")),
            "action_payload": env_action,
            "exec_ok": bool(exec_ok),
            "error": error,
            "done": bool(done),
            "content": policy_output.get("content"),
            "reasoning": policy_output.get("reasoning"),
            "raw_policy_output": policy_output,
            "chosen_action": chosen_action,
            "teacher_reason": teacher_reason,
        }
        self.history.append(history_item)
        record = RLStepRecord(
            episode_step=int(step_index),
            policy_input_text=policy_input_text,
            policy_output=policy_output,
            chosen_action=chosen_action,
            env_action=env_action,
            exec_ok=bool(exec_ok),
            error=error,
            reward=reward_dict,
            before_url=before_url,
            after_url=str(after.snapshot.url or before_url),
            before_score=float(before.score.raw_score),
            after_score=float(after.score.raw_score),
            success=bool(after.score.success),
        )
        self.rollout_steps.append(record)
        return record, bool(done or after.score.success or (step_index + 1) >= self.max_steps)

    async def rollout(self) -> dict[str, Any]:
        await self.reset()
        done = False
        try:
            for step_index in range(self.max_steps):
                _, done = await self.step(step_index)
                if done:
                    break
            final = self.last_step_result
            if final is None:
                raise RuntimeError("rollout finished without final step result")
            return {
                "seed": int(self.seed),
                "task_id": str(self.task.id),
                "prompt": str(self.task.prompt),
                "url": str(self.task.url),
                "final_url": str(final.snapshot.url or ""),
                "success": bool(final.score.success),
                "score": float(final.score.raw_score),
                "tests_passed": int(final.score.tests_passed),
                "total_tests": int(final.score.total_tests),
                "steps": [item.to_dict() for item in self.rollout_steps],
                "history": list(self.history),
            }
        finally:
            await self.close()


async def rollout_contact_seed(
    *,
    seed: int,
    model_override: str = "",
    max_steps: int = 12,
    task_cache_dir: str | Path = DEFAULT_CONTACT_TASK_CACHE_DIR,
) -> dict[str, Any]:
    env = ContactRLEnv(
        seed=seed,
        model_override=model_override,
        max_steps=max_steps,
        task_cache_dir=task_cache_dir,
    )
    return await env.rollout()


def rollout_contact_seed_sync(
    *,
    seed: int,
    model_override: str = "",
    max_steps: int = 12,
    task_cache_dir: str | Path = DEFAULT_CONTACT_TASK_CACHE_DIR,
) -> dict[str, Any]:
    return asyncio.run(
        rollout_contact_seed(
            seed=seed,
            model_override=model_override,
            max_steps=max_steps,
            task_cache_dir=task_cache_dir,
        )
    )
