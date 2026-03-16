"""Convert trajectory episodes to SFT training format (HuggingFace messages JSONL).

Reads episodes from episodes.jsonl and produces train/val JSONL splits
in the HuggingFace messages format.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

from .obs_serializer import serialize_observation

SYSTEM_PROMPT = (
    "You are a web agent. Complete the task by choosing browser actions. "
    "Respond with a JSON object containing tool, args, and reasoning."
)


def convert_episode_to_sft(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a single episode to a list of SFT training examples.

    Each step that made progress becomes one training example with
    system + user (observation) + assistant (action) messages.

    Args:
        episode: Episode dict from episodes.jsonl.

    Returns:
        List of dicts with {"messages": [{"role": ..., "content": ...}, ...]}.
    """
    results: List[Dict[str, Any]] = []
    task_text = episode.get("task_text", "")
    steps = episode.get("steps", [])

    for step in steps:
        # Skip no-op or error steps
        status = step.get("verifier_status", "")
        if status in ("no_op", "loop", "error"):
            continue

        # Build the observation text
        obs_dict = _step_to_obs_dict(step, task_text)
        obs_text = serialize_observation(obs_dict)

        # Build the action JSON
        action = _step_to_action(step)
        action_text = json.dumps(action, ensure_ascii=False)

        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
                {"role": "assistant", "content": action_text},
            ]
        }
        results.append(example)

    return results


def _step_to_obs_dict(step: Dict[str, Any], task_text: str) -> Dict[str, Any]:
    """Build an observation dict from a step record for serialization."""
    candidates = step.get("candidates", [])
    return {
        "prompt": task_text,
        "url": step.get("url", ""),
        "step_index": step.get("step_index", 0),
        "page_observations": {
            "visible_text": step.get("page_title", ""),
        },
        "candidates": candidates,
    }


def _step_to_action(step: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the action from a step record."""
    chosen = step.get("chosen_action", {})
    if isinstance(chosen, dict) and chosen:
        return chosen

    # Fallback: construct from action type and candidate
    action_type = step.get("chosen_action_type", "click")
    candidate_id = step.get("chosen_candidate_id", "")
    return {
        "tool": f"browser.{action_type}",
        "args": {"candidate_id": candidate_id},
    }


def format_episodes_file(
    input_path: str,
    output_dir: str,
    val_ratio: float = 0.1,
) -> None:
    """Read episodes.jsonl and produce train.jsonl + val.jsonl.

    Args:
        input_path: Path to episodes.jsonl.
        output_dir: Directory for train.jsonl and val.jsonl.
        val_ratio: Fraction of examples for validation set.
    """
    all_examples: List[Dict[str, Any]] = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            examples = convert_episode_to_sft(episode)
            all_examples.extend(examples)

    if not all_examples:
        return

    # Shuffle and split
    random.shuffle(all_examples)
    split_idx = max(1, int(len(all_examples) * (1 - val_ratio)))
    train = all_examples[:split_idx]
    val = all_examples[split_idx:]

    os.makedirs(output_dir, exist_ok=True)
    _write_jsonl(os.path.join(output_dir, "train.jsonl"), train)
    if val:
        _write_jsonl(os.path.join(output_dir, "val.jsonl"), val)


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
