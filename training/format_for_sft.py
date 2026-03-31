"""Export the replayable Autocinema harvest into Browser Use SFT JSONL files.

The exporter now supports multiple policy-training surfaces:

- v2: compact observation text with corrected runtime-compatible indices.
- v3: runtime-aligned `policy_input_text` plus the same wrapper structure the
  step-engine expects at inference time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.operator.agents.step_engine.candidates import Candidate, CandidateExtractor
from src.operator.agents.step_engine.observation import ObsBuilder
from src.operator.agents.step_engine.state import AgentState

from .obs_serializer import serialize_observation

SYSTEM_PROMPT = "You are a browser-use style web agent operating on Autocinema tasks. Return the next browser tool call as JSON with the chosen tool arguments."
RUNTIME_ALIGNED_SYSTEM_PROMPT = (
    "You are a browser-use-style web automation policy.\n"
    "Given the task and the current browser state, choose the next browser step sequence.\n"
    "Return ONE JSON object only. No markdown. No prose. No chain-of-thought.\n"
    "You must choose exactly one of:\n"
    "1) browser tool_call or browser tool_calls\n"
    "2) final (done=true + content)\n\n"
    "Rules:\n"
    "- This runtime allows up to 3 browser actions per step.\n"
    "- Prefer a concrete browser action when there is a reasonable actionable target.\n"
    "- Never return more than 3 browser actions.\n"
    "- If you return multiple browser actions, they must stay within the same local workflow and should usually be a short form-filling or commit sequence.\n"
    "- If the current page already contains the answer, return final immediately.\n"
    "- Use final/done or browser.done with a concrete content string when the task is satisfied.\n"
    "- Prefer arguments.index that refers to INTERACTIVE ELEMENT SHORTLIST.\n"
    "- Never emit unavailable tools.\n"
    "- Preserve placeholders such as <username>, <password>, <signup_email> exactly when typing.\n"
    "- Before choosing actions, infer one short local workflow plan for the current page and keep it stable until that workflow is completed or visibly blocked.\n"
)
FORMAT_VERSION = "autocinema.browser_use.sft.v2"
RUNTIME_ALIGNED_FORMAT_VERSION = "autocinema.browser_use.sft.v3"
BASE_MODEL = "browser-use/bu-30b-a3b-preview"
DEFAULT_SEED = 36
DEFAULT_VAL_RATIO = 0.1

_CANDIDATE_EXTRACTOR = CandidateExtractor()
_OBS_BUILDER = ObsBuilder()


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


@dataclass
class SFTExample:
    episode_task_id: str
    trace_file: str
    use_case: str
    step_index: int
    record: dict[str, Any]


def _extract_visible_text(snapshot_html: str) -> str:
    if not snapshot_html:
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(snapshot_html)
    return parser.text()


def _recent_history_from_trace_steps(
    trace_steps: list[dict[str, Any]],
    *,
    current_step_index: int,
    limit: int = 4,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for previous in trace_steps:
        if not isinstance(previous, dict):
            continue
        step_index = int(previous.get("step_index") or 0)
        if step_index >= current_step_index:
            break
        action = previous.get("action") if isinstance(previous.get("action"), dict) else {}
        execution = previous.get("execution") if isinstance(previous.get("execution"), dict) else {}
        after = previous.get("after") if isinstance(previous.get("after"), dict) else {}
        history.append(
            {
                "step_index": step_index,
                "tool": str(action.get("tool") or ""),
                "arguments": action.get("arguments") if isinstance(action.get("arguments"), dict) else {},
                "exec_ok": str(execution.get("status") or "").lower() == "ok",
                "url": str(after.get("url") or ""),
            }
        )
    return history[-limit:]


def _selector_summary(selector: dict[str, Any] | None) -> str:
    if not isinstance(selector, dict):
        return ""
    selector_type = str(selector.get("type") or "").strip()
    if selector_type == "attributeValueSelector":
        attribute = str(selector.get("attribute") or "").strip()
        value = str(selector.get("value") or "").strip()
        if attribute and value:
            return f"{attribute}={value}"
    value = str(selector.get("value") or "").strip()
    if selector_type and value:
        return f"{selector_type}:{value[:120]}"
    return selector_type


def _candidate_obs(candidates: list[Candidate], *, limit: int = 24) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cand in enumerate(candidates[:limit]):
        row = cand.as_obs(index=index)
        row["selector_summary"] = _selector_summary(cand.selector)
        rows.append(row)
    return rows


def _history_for_policy_input(
    trace_steps: list[dict[str, Any]],
    *,
    current_step_index: int,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for previous in trace_steps:
        if not isinstance(previous, dict):
            continue
        step_index = int(previous.get("step_index") or 0)
        if step_index >= current_step_index:
            break
        action = previous.get("action") if isinstance(previous.get("action"), dict) else {}
        execution = previous.get("execution") if isinstance(previous.get("execution"), dict) else {}
        after = previous.get("after") if isinstance(previous.get("after"), dict) else {}
        error = str(execution.get("error") or "").strip()
        history.append(
            {
                "step": step_index,
                "url": str(after.get("url") or ""),
                "action": action,
                "done": bool((previous.get("act_response") or {}).get("done")) if isinstance(previous.get("act_response"), dict) else False,
                "exec_ok": str(execution.get("status") or "").lower() == "ok",
                "error": error,
                "text": str(action.get("text") or action.get("value") or ""),
            }
        )
    return history


def _build_observation(
    trace: dict[str, Any],
    step: dict[str, Any],
    *,
    candidates: list[Candidate] | None = None,
    text_ir: dict[str, Any] | None = None,
) -> str:
    request = step.get("act_request") if isinstance(step.get("act_request"), dict) else {}
    before = step.get("before") if isinstance(step.get("before"), dict) else {}
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    response = step.get("act_response") if isinstance(step.get("act_response"), dict) else {}
    snapshot_html = str(request.get("snapshot_html") or "")
    url = str(request.get("url") or before.get("url") or trace.get("task_url") or "")
    if candidates is None:
        candidates = _CANDIDATE_EXTRACTOR.extract(snapshot_html=snapshot_html, url=url)
    if text_ir is None:
        text_ir = _OBS_BUILDER.build_text_ir(snapshot_html)
    trace_steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    step_index = int(step.get("step_index") or 0)
    history_recent = _recent_history_from_trace_steps(trace_steps, current_step_index=step_index)
    use_case = str(_episode_use_case := trace.get("episode", {}).get("use_case") or trace.get("task_web_project_id") or "")

    obs = {
        "prompt": request.get("prompt") or trace.get("task_prompt") or "",
        "url": url,
        "step_index": step_index,
        "page_observations": {
            "visible_text": str(text_ir.get("visible_text") or ""),
            "title": str(text_ir.get("title") or ""),
            "headings": text_ir.get("headings") if isinstance(text_ir.get("headings"), list) else [],
            "forms": text_ir.get("forms") if isinstance(text_ir.get("forms"), list) else [],
            "page_facts": text_ir.get("page_facts") if isinstance(text_ir.get("page_facts"), list) else [],
            "value_lines": text_ir.get("value_lines") if isinstance(text_ir.get("value_lines"), list) else [],
        },
        "candidates": _candidate_obs(candidates),
        "memory": {
            "facts": [
                f"use_case={use_case}",
                f"episode_task_id={trace.get('episode', {}).get('episode_task_id', '')}",
                f"allowed_tools={','.join(str(tool) for tool in (request.get('allowed_tools') or []))}",
            ],
            "history_recent": history_recent,
            "state_in": request.get("state_in") if isinstance(request.get("state_in"), dict) else {},
            "state_out": response.get("state_out") if isinstance(response.get("state_out"), dict) else {},
            "last_action": {
                "tool": str(action.get("tool") or ""),
                "arguments": action.get("arguments") if isinstance(action.get("arguments"), dict) else {},
            },
        },
    }
    return serialize_observation(obs)


def _build_runtime_aligned_observation(
    trace: dict[str, Any],
    step: dict[str, Any],
    *,
    use_case: str,
    candidates: list[Candidate] | None = None,
    text_ir: dict[str, Any] | None = None,
) -> str:
    request = step.get("act_request") if isinstance(step.get("act_request"), dict) else {}
    before = step.get("before") if isinstance(step.get("before"), dict) else {}
    snapshot_html = str(request.get("snapshot_html") or "")
    url = str(request.get("url") or before.get("url") or trace.get("task_url") or "")
    prompt = str(request.get("prompt") or trace.get("task_prompt") or "")
    web_project_id = str(request.get("web_project_id") or trace.get("task_web_project_id") or "")
    step_index = int(step.get("step_index") or 0)
    if candidates is None:
        candidates = _CANDIDATE_EXTRACTOR.extract(snapshot_html=snapshot_html, url=url)
    if text_ir is None:
        text_ir = _OBS_BUILDER.build_text_ir(snapshot_html)
    state = AgentState.from_internal_state(request.get("state_in") or {}, prompt)
    history = _history_for_policy_input(
        trace.get("steps") if isinstance(trace.get("steps"), list) else [],
        current_step_index=step_index,
    )
    policy_obs = _OBS_BUILDER.build_policy_obs(
        task_id=str(request.get("task_id") or ""),
        prompt=prompt,
        web_project_id=web_project_id,
        use_case={"id": str(use_case or ""), "name": str(use_case or "")},
        snapshot_html=snapshot_html,
        step_index=step_index,
        url=url,
        mode=str(state.mode or "BOOTSTRAP"),
        flags={},
        state=state,
        text_ir=text_ir,
        candidates=candidates,
        history=history,
        screenshot_available=bool(request.get("screenshot")),
    )
    return str(policy_obs.get("policy_input_text") or "")


def _tool_calls_from_step(step: dict[str, Any]) -> list[dict[str, Any]]:
    response = step.get("act_response")
    if not isinstance(response, dict):
        return []
    tool_calls = response.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [call for call in tool_calls if isinstance(call, dict) and call.get("name")]


def _normalize_trace_tool_call(
    tool_call: dict[str, Any],
    *,
    candidates: list[Candidate],
    current_url: str,
) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    name = str(tool_call.get("name") or "").strip()
    arguments = dict(tool_call.get("arguments") or {}) if isinstance(tool_call.get("arguments"), dict) else {}
    if not name:
        return None

    if name == "browser.navigate":
        target_url = str(arguments.get("url") or "").strip()
        if not target_url or target_url == str(current_url or "").strip():
            return None
        return {"name": name, "arguments": arguments}

    raw_index = arguments.get("index")
    if not isinstance(raw_index, int):
        return {"name": name, "arguments": arguments}

    if name == "browser.click":
        eligible = [idx for idx, cand in enumerate(candidates) if cand.role in {"button", "link"}]
    elif name == "browser.input":
        eligible = [idx for idx, cand in enumerate(candidates) if cand.role == "input"]
    else:
        return {"name": name, "arguments": arguments}

    if 0 <= raw_index < len(candidates):
        candidate = candidates[raw_index]
        if (name == "browser.click" and candidate.role in {"button", "link"}) or (name == "browser.input" and candidate.role == "input"):
            return {"name": name, "arguments": arguments}

    if 0 <= raw_index < len(eligible):
        remapped = dict(arguments)
        remapped["index"] = int(eligible[raw_index])
        return {"name": name, "arguments": remapped}

    return None


def _guided_action_from_execution(execution: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(execution, dict):
        return None
    selected = execution.get("selected_action")
    if isinstance(selected, dict) and selected.get("type"):
        return selected
    attempts = execution.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            action = attempt.get("action")
            if isinstance(action, dict) and action.get("type"):
                return action
    planned = execution.get("planned_action")
    if isinstance(planned, dict) and planned.get("type"):
        return planned
    return None


def _build_guided_observation(
    *,
    episode_row: dict[str, Any],
    report_episode: dict[str, Any],
    execution_log: list[dict[str, Any]],
    current_index: int,
) -> str:
    previous_actions: list[dict[str, Any]] = []
    for prev in execution_log[:current_index]:
        action = _guided_action_from_execution(prev)
        if isinstance(action, dict):
            previous_actions.append(action)
    payload = {
        "mode": "guided_harvester",
        "use_case": str(episode_row.get("use_case") or report_episode.get("use_case") or ""),
        "seed": int(episode_row.get("seed") or report_episode.get("seed") or 0),
        "task_id": str(episode_row.get("task_id") or report_episode.get("task_id") or ""),
        "current_url": str(execution_log[current_index].get("url") or report_episode.get("final_url") or ""),
        "step_index": current_index,
        "previous_actions": previous_actions[-4:],
        "planned_action": execution_log[current_index].get("planned_action"),
        "attempts": execution_log[current_index].get("attempts"),
    }
    return json.dumps(payload, ensure_ascii=False)


def convert_guided_report_to_sft_examples(
    *,
    episode_row: dict[str, Any],
    report: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[SFTExample]:
    examples: list[SFTExample] = []
    episodes = report.get("episodes") if isinstance(report.get("episodes"), list) else []
    if not episodes:
        return examples
    report_episode = episodes[0] if isinstance(episodes[0], dict) else {}
    execution_log = report_episode.get("guided_execution") if isinstance(report_episode.get("guided_execution"), list) else []
    if not execution_log:
        return examples
    episode_task_id = str(episode_row.get("episode_task_id") or report_episode.get("episode_task_id") or "")
    trace_file = str(episode_row.get("result_path") or episode_row.get("trace_file") or "")
    use_case = str(episode_row.get("use_case") or report_episode.get("use_case") or "")

    for idx, execution in enumerate(execution_log):
        if not isinstance(execution, dict):
            continue
        action = _guided_action_from_execution(execution)
        if not isinstance(action, dict) or not action.get("type"):
            continue
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _build_guided_observation(
                        episode_row=episode_row,
                        report_episode=report_episode,
                        execution_log=execution_log,
                        current_index=idx,
                    ),
                },
                {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
            ],
            "metadata": {
                "episode_task_id": episode_task_id,
                "trace_file": trace_file,
                "use_case": use_case,
                "step_index": idx,
            },
        }
        examples.append(
            SFTExample(
                episode_task_id=episode_task_id,
                trace_file=trace_file,
                use_case=use_case,
                step_index=idx,
                record=record,
            )
        )
    return examples


def convert_trace_to_sft_examples(
    *,
    episode_row: dict[str, Any],
    trace: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
    runtime_aligned: bool = False,
) -> list[SFTExample]:
    examples: list[SFTExample] = []
    trace_file = str(episode_row.get("trace_file") or "")
    episode_task_id = str(episode_row.get("episode_task_id") or trace.get("episode", {}).get("episode_task_id") or "")
    use_case = str(episode_row.get("use_case") or trace.get("episode", {}).get("use_case") or "")
    if not isinstance(trace.get("steps"), list) and isinstance(trace.get("episode"), dict):
        synthetic_report = {"episodes": [trace.get("episode")]}
        return convert_guided_report_to_sft_examples(
            episode_row=episode_row,
            report=synthetic_report,
            system_prompt=system_prompt,
        )

    for step in trace.get("steps", []):
        if not isinstance(step, dict):
            continue
        tool_calls = _tool_calls_from_step(step)
        if not tool_calls:
            continue
        request = step.get("act_request") if isinstance(step.get("act_request"), dict) else {}
        before = step.get("before") if isinstance(step.get("before"), dict) else {}
        snapshot_html = str(request.get("snapshot_html") or "")
        url = str(request.get("url") or before.get("url") or trace.get("task_url") or "")
        candidates = _CANDIDATE_EXTRACTOR.extract(snapshot_html=snapshot_html, url=url)
        text_ir = _OBS_BUILDER.build_text_ir(snapshot_html)
        normalized_tool_call = _normalize_trace_tool_call(
            tool_calls[0],
            candidates=candidates,
            current_url=url,
        )
        if not isinstance(normalized_tool_call, dict):
            continue
        assistant_payload: dict[str, Any]
        assistant_payload = {"type": "browser", "tool_call": normalized_tool_call} if runtime_aligned else normalized_tool_call
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        _build_runtime_aligned_observation(
                            trace,
                            step,
                            use_case=use_case,
                            candidates=candidates,
                            text_ir=text_ir,
                        )
                        if runtime_aligned
                        else _build_observation(
                            trace,
                            step,
                            candidates=candidates,
                            text_ir=text_ir,
                        )
                    ),
                },
                {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
            ],
            "metadata": {
                "episode_task_id": episode_task_id,
                "trace_file": trace_file,
                "use_case": use_case,
                "step_index": int(step.get("step_index") or 0),
            },
        }
        examples.append(
            SFTExample(
                episode_task_id=episode_task_id,
                trace_file=trace_file,
                use_case=use_case,
                step_index=int(step.get("step_index") or 0),
                record=record,
            )
        )
    return examples


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_harvest_to_sft(
    *,
    input_path: str,
    output_dir: str,
    summary_path: str | None = None,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    system_prompt: str = SYSTEM_PROMPT,
    base_model: str = BASE_MODEL,
    train_seeds: list[int] | None = None,
    val_seeds: list[int] | None = None,
    trace_only: bool = True,
    runtime_aligned: bool = False,
) -> dict[str, Any]:
    episodes_path = Path(input_path)
    out_dir = Path(output_dir)
    summary_file = Path(summary_path) if summary_path else episodes_path.with_name("summary.json")
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}

    episode_rows = _load_jsonl(episodes_path)
    successful_rows = [row for row in episode_rows if bool(row.get("success"))]

    if train_seeds is not None or val_seeds is not None:
        train_seed_set = {int(value) for value in (train_seeds or [])}
        val_seed_set = {int(value) for value in (val_seeds or [])}
        train_rows = [row for row in successful_rows if int(row.get("seed") or -1) in train_seed_set]
        val_rows = [row for row in successful_rows if int(row.get("seed") or -1) in val_seed_set]
        overlap = {str(row.get("episode_task_id") or "") for row in train_rows if str(row.get("episode_task_id") or "") in {str(item.get("episode_task_id") or "") for item in val_rows}}
        if overlap:
            raise ValueError(f"Train/val split overlap detected for episodes: {sorted(overlap)}")
    else:
        rng = random.Random(int(seed))
        shuffled_successes = list(successful_rows)
        rng.shuffle(shuffled_successes)

        if len(shuffled_successes) <= 1:
            val_episode_count = 0
        else:
            raw_count = round(len(shuffled_successes) * max(0.0, min(val_ratio, 0.5)))
            val_episode_count = min(max(raw_count, 1), len(shuffled_successes) - 1)

        val_rows = shuffled_successes[:val_episode_count]
        train_rows = shuffled_successes[val_episode_count:]

    train_examples: list[SFTExample] = []
    val_examples: list[SFTExample] = []
    guided_fallback_episodes = 0
    skipped_without_trace = 0

    for bucket, rows in ((train_examples, train_rows), (val_examples, val_rows)):
        for row in rows:
            trace_file = Path(str(row.get("trace_file") or ""))
            result_path = Path(str(row.get("result_path") or ""))
            if trace_file.exists() and trace_file.is_file():
                trace = json.loads(trace_file.read_text(encoding="utf-8"))
                bucket.extend(
                    convert_trace_to_sft_examples(
                        episode_row=row,
                        trace=trace,
                        system_prompt=system_prompt,
                        runtime_aligned=runtime_aligned,
                    )
                )
                continue
            if trace_only:
                skipped_without_trace += 1
                continue
            if result_path.exists() and result_path.is_file():
                report = json.loads(result_path.read_text(encoding="utf-8"))
                bucket.extend(
                    convert_guided_report_to_sft_examples(
                        episode_row=row,
                        report=report,
                        system_prompt=system_prompt,
                    )
                )
                guided_fallback_episodes += 1

    train_records = [example.record for example in train_examples]
    val_records = [example.record for example in val_examples]
    usable_episode_ids = {str(example.episode_task_id) for example in train_examples + val_examples if str(example.episode_task_id)}

    if not train_records:
        raise ValueError("No train SFT examples were produced from successful episodes")
    if not val_records and len(usable_episode_ids) > 1:
        raise ValueError("Validation split is empty; reduce filtering or adjust val_ratio")

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)

    manifest = {
        "format_version": RUNTIME_ALIGNED_FORMAT_VERSION if runtime_aligned else FORMAT_VERSION,
        "base_model": base_model,
        "training_method": "lora_qlora",
        "system_prompt": system_prompt,
        "source_episodes_path": str(episodes_path),
        "source_summary_path": str(summary_file),
        "episodes_total": int(summary.get("episodes_total") or len(episode_rows)),
        "successful_episodes": int(summary.get("successes_total") or len(successful_rows)),
        "successful_episodes_total": int(summary.get("successes_total") or len(successful_rows)),
        "successful_episodes_used": len(train_rows) + len(val_rows),
        "selected_episode_ids": [str(row.get("episode_task_id") or "") for row in train_rows + val_rows],
        "train_episode_ids": [str(row.get("episode_task_id") or "") for row in train_rows],
        "val_episode_ids": [str(row.get("episode_task_id") or "") for row in val_rows],
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "selection_policy": "successful replayable episodes only; deterministic episode split before step expansion",
        "source_mode": "trace_only" if trace_only else "trace_or_guided_report",
        "runtime_aligned": bool(runtime_aligned),
        "guided_fallback_episodes": int(guided_fallback_episodes),
        "skipped_without_trace": int(skipped_without_trace),
        "split_seed": int(seed),
        "val_ratio": float(val_ratio),
        "train_seeds": sorted({int(row.get("seed") or 0) for row in train_rows}),
        "val_seeds": sorted({int(row.get("seed") or 0) for row in val_rows}),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "trace_files": sorted({str(example.trace_file) for example in train_examples + val_examples if example.trace_file}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def format_episodes_file(
    input_path: str,
    output_dir: str,
    val_ratio: float = DEFAULT_VAL_RATIO,
    trace_only: bool = True,
    runtime_aligned: bool = False,
) -> None:
    export_harvest_to_sft(
        input_path=input_path,
        output_dir=output_dir,
        val_ratio=val_ratio,
        trace_only=trace_only,
        runtime_aligned=runtime_aligned,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the replayable Autocinema harvest into SFT JSONL")
    parser.add_argument("--input", default="data/autocinema_trajectory_harvest/episodes.jsonl", help="Path to replayable harvest episodes.jsonl")
    parser.add_argument("--summary", default="data/autocinema_trajectory_harvest/summary.json", help="Path to harvest summary.json")
    parser.add_argument("--output-dir", default="data/autocinema_trajectory_harvest/sft", help="Output directory for SFT train/val/manifest")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation episode ratio")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic split seed")
    parser.add_argument("--allow-guided-fallback", action="store_true", help="Allow guided-report fallback examples when traces are missing")
    parser.add_argument("--runtime-aligned", action="store_true", help="Export runtime-aligned policy_input_text and wrapped browser decisions")
    args = parser.parse_args(argv)

    manifest = export_harvest_to_sft(
        input_path=args.input,
        output_dir=args.output_dir,
        summary_path=args.summary,
        val_ratio=args.val_ratio,
        seed=args.seed,
        trace_only=not bool(args.allow_guided_fallback),
        system_prompt=RUNTIME_ALIGNED_SYSTEM_PROMPT if bool(args.runtime_aligned) else SYSTEM_PROMPT,
        runtime_aligned=bool(args.runtime_aligned),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
