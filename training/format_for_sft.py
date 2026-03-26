"""Export the replayable Autocinema harvest into Browser Use SFT JSONL files."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .obs_serializer import serialize_observation

SYSTEM_PROMPT = (
    "You are a browser-use style web agent operating on Autocinema tasks. "
    "Return the next browser tool call as JSON with the chosen tool arguments."
)
FORMAT_VERSION = "autocinema.browser_use.sft.v1"
BASE_MODEL = "browser-use/bu-30b-a3b-preview"
DEFAULT_SEED = 36
DEFAULT_VAL_RATIO = 0.1


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


def _normalize_candidates(snapshot_html: str) -> list[dict[str, Any]]:
    if not snapshot_html:
        return []
    candidates: list[dict[str, Any]] = []
    lowered = snapshot_html.lower()
    if "<button" in lowered:
        candidates.append({"type": "button", "text": "button"})
    if "<input" in lowered:
        candidates.append({"type": "input", "text": "input"})
    if "<textarea" in lowered:
        candidates.append({"type": "textarea", "text": "textarea"})
    if "<a " in lowered or "<a>" in lowered:
        candidates.append({"type": "link", "text": "link"})
    return candidates


def _build_observation(trace: dict[str, Any], step: dict[str, Any]) -> str:
    request = step.get("act_request") if isinstance(step.get("act_request"), dict) else {}
    before = step.get("before") if isinstance(step.get("before"), dict) else {}
    snapshot_html = str(request.get("snapshot_html") or "")
    visible_text = _extract_visible_text(snapshot_html)

    obs = {
        "prompt": request.get("prompt") or trace.get("task_prompt") or "",
        "url": request.get("url") or before.get("url") or trace.get("task_url") or "",
        "step_index": step.get("step_index", 0),
        "page_observations": {
            "visible_text": visible_text,
        },
        "candidates": _normalize_candidates(snapshot_html),
        "memory": {
            "facts": [
                f"use_case={trace.get('episode', {}).get('use_case') or trace.get('task_web_project_id')}",
                f"episode_task_id={trace.get('episode', {}).get('episode_task_id', '')}",
            ]
        },
    }
    return serialize_observation(obs)


def _tool_calls_from_step(step: dict[str, Any]) -> list[dict[str, Any]]:
    response = step.get("act_response")
    if not isinstance(response, dict):
        return []
    tool_calls = response.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [call for call in tool_calls if isinstance(call, dict) and call.get("name")]


def convert_trace_to_sft_examples(
    *,
    episode_row: dict[str, Any],
    trace: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
) -> list[SFTExample]:
    examples: list[SFTExample] = []
    trace_file = str(episode_row.get("trace_file") or "")
    episode_task_id = str(episode_row.get("episode_task_id") or trace.get("episode", {}).get("episode_task_id") or "")
    use_case = str(episode_row.get("use_case") or trace.get("episode", {}).get("use_case") or "")

    for step in trace.get("steps", []):
        if not isinstance(step, dict):
            continue
        tool_calls = _tool_calls_from_step(step)
        if not tool_calls:
            continue
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_observation(trace, step)},
                {"role": "assistant", "content": json.dumps(tool_calls[0], ensure_ascii=False)},
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
) -> dict[str, Any]:
    episodes_path = Path(input_path)
    out_dir = Path(output_dir)
    summary_file = Path(summary_path) if summary_path else episodes_path.with_name("summary.json")
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}

    episode_rows = _load_jsonl(episodes_path)
    successful_rows = [row for row in episode_rows if bool(row.get("success"))]

    rng = random.Random(int(seed))
    shuffled_successes = list(successful_rows)
    rng.shuffle(shuffled_successes)

    if len(shuffled_successes) <= 1:
        val_episode_count = 0
    else:
        raw_count = int(round(len(shuffled_successes) * max(0.0, min(val_ratio, 0.5))))
        val_episode_count = min(max(raw_count, 1), len(shuffled_successes) - 1)

    val_rows = shuffled_successes[:val_episode_count]
    train_rows = shuffled_successes[val_episode_count:]

    train_examples: list[SFTExample] = []
    val_examples: list[SFTExample] = []

    for bucket, rows in ((train_examples, train_rows), (val_examples, val_rows)):
        for row in rows:
            trace_file = Path(str(row.get("trace_file") or ""))
            if not trace_file.exists():
                continue
            trace = json.loads(trace_file.read_text(encoding="utf-8"))
            bucket.extend(
                convert_trace_to_sft_examples(
                    episode_row=row,
                    trace=trace,
                    system_prompt=system_prompt,
                )
            )

    train_records = [example.record for example in train_examples]
    val_records = [example.record for example in val_examples]

    if not train_records:
        raise ValueError("No train SFT examples were produced from successful episodes")
    if not val_records and len(successful_rows) > 1:
        raise ValueError("Validation split is empty; reduce filtering or adjust val_ratio")

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)

    manifest = {
        "format_version": FORMAT_VERSION,
        "base_model": base_model,
        "training_method": "lora_qlora",
        "system_prompt": system_prompt,
        "source_episodes_path": str(episodes_path),
        "source_summary_path": str(summary_file),
        "episodes_total": int(summary.get("episodes_total") or len(episode_rows)),
        "successful_episodes_total": int(summary.get("successes_total") or len(successful_rows)),
        "successful_episodes_used": len(train_rows) + len(val_rows),
        "selected_episode_ids": [str(row.get("episode_task_id") or "") for row in train_rows + val_rows],
        "train_episode_ids": [str(row.get("episode_task_id") or "") for row in train_rows],
        "val_episode_ids": [str(row.get("episode_task_id") or "") for row in val_rows],
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "selection_policy": "successful replayable episodes only; deterministic episode split before step expansion",
        "split_seed": int(seed),
        "val_ratio": float(val_ratio),
        "trace_files": sorted(
            {
                str(example.trace_file)
                for example in train_examples + val_examples
                if example.trace_file
            }
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def format_episodes_file(
    input_path: str,
    output_dir: str,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> None:
    export_harvest_to_sft(
        input_path=input_path,
        output_dir=output_dir,
        val_ratio=val_ratio,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the replayable Autocinema harvest into SFT JSONL")
    parser.add_argument("--input", default="data/autocinema_trajectory_harvest/episodes.jsonl", help="Path to replayable harvest episodes.jsonl")
    parser.add_argument("--summary", default="data/autocinema_trajectory_harvest/summary.json", help="Path to harvest summary.json")
    parser.add_argument("--output-dir", default="data/autocinema_trajectory_harvest/sft", help="Output directory for SFT train/val/manifest")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation episode ratio")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic split seed")
    args = parser.parse_args(argv)

    manifest = export_harvest_to_sft(
        input_path=args.input,
        output_dir=args.output_dir,
        summary_path=args.summary,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
