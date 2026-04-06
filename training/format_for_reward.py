"""Convert trajectory episodes to preference pairs for reward model training.

Produces (state, chosen_action, rejected_action) triples from episodes
where the same state was visited with different outcomes.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from .obs_serializer import serialize_observation


def convert_episode_to_pairs(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract preference pairs from an episode.

    Strategy:
    1. Within an episode, find steps at the same URL where one made progress
       and another didn't — the progressing action is "chosen", the other is
       "rejected".
    2. For steps with multiple candidates, if the chosen action failed (no_op),
       treat any candidate that later succeeded as the preferred action.

    Args:
        episode: Episode dict from episodes.jsonl.

    Returns:
        List of preference pair dicts with keys:
        - prompt: the observation text
        - chosen: the better action (JSON string)
        - rejected: the worse action (JSON string)
    """
    pairs: List[Dict[str, Any]] = []
    steps = episode.get("steps", [])
    task_text = episode.get("task_text", "")

    if len(steps) < 2:
        return pairs

    # Group steps by URL for same-state comparison
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    for step in steps:
        url = step.get("url", "")
        if url:
            by_url.setdefault(url, []).append(step)

    # Strategy 1: Compare steps at the same URL
    for url, url_steps in by_url.items():
        good_steps = [s for s in url_steps if s.get("made_progress")]
        bad_steps = [s for s in url_steps if not s.get("made_progress")]

        for good in good_steps:
            for bad in bad_steps:
                obs_dict = _step_to_obs_dict(bad, task_text)
                obs_text = serialize_observation(obs_dict)

                chosen_action = _extract_action(good)
                rejected_action = _extract_action(bad)

                if chosen_action == rejected_action:
                    continue

                pairs.append({
                    "prompt": obs_text,
                    "chosen": json.dumps(chosen_action, ensure_ascii=False),
                    "rejected": json.dumps(rejected_action, ensure_ascii=False),
                })

    # Strategy 2: Within a single step, if it failed and had multiple candidates,
    # create a pair with the chosen (bad) vs the next successful action
    for i, step in enumerate(steps):
        if step.get("made_progress") or i + 1 >= len(steps):
            continue
        next_step = steps[i + 1]
        if not next_step.get("made_progress"):
            continue
        # Same URL means similar state
        if step.get("url", "") != next_step.get("url", ""):
            continue

        obs_dict = _step_to_obs_dict(step, task_text)
        obs_text = serialize_observation(obs_dict)

        chosen_action = _extract_action(next_step)
        rejected_action = _extract_action(step)

        if chosen_action == rejected_action:
            continue

        pair = {
            "prompt": obs_text,
            "chosen": json.dumps(chosen_action, ensure_ascii=False),
            "rejected": json.dumps(rejected_action, ensure_ascii=False),
        }
        if pair not in pairs:
            pairs.append(pair)

    return pairs


def _step_to_obs_dict(step: Dict[str, Any], task_text: str) -> Dict[str, Any]:
    """Build observation dict from step for serialization."""
    return {
        "prompt": task_text,
        "url": step.get("url", ""),
        "step_index": step.get("step_index", 0),
        "page_observations": {
            "visible_text": step.get("page_title", ""),
        },
        "candidates": step.get("candidates", []),
    }


def _extract_action(step: Dict[str, Any]) -> Dict[str, Any]:
    """Extract action dict from a step."""
    chosen = step.get("chosen_action", {})
    if isinstance(chosen, dict) and chosen:
        return chosen
    action_type = step.get("chosen_action_type", "click")
    candidate_id = step.get("chosen_candidate_id", "")
    return {
        "tool": f"browser.{action_type}",
        "args": {"candidate_id": candidate_id},
    }


def format_reward_file(
    input_path: str,
    output_path: str,
) -> None:
    """Read episodes.jsonl and produce reward training JSONL.

    Args:
        input_path: Path to episodes.jsonl.
        output_path: Path for output preference pairs JSONL.
    """
    all_pairs: List[Dict[str, Any]] = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            pairs = convert_episode_to_pairs(episode)
            all_pairs.extend(pairs)

    if not all_pairs:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
