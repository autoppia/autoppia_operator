"""Dataset exporters for training: pairwise ranking, verifier classification, SFT, DAgger."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .schema import EpisodeRecord, FullStepRecord


def load_episodes(path: str) -> List[EpisodeRecord]:
    """Load EpisodeRecords from a JSONL file."""
    episodes: List[EpisodeRecord] = []
    if not os.path.exists(path):
        return episodes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            episode = _dict_to_episode(data)
            episodes.append(episode)
    return episodes


def export_pairwise_ranking(
    episodes: List[EpisodeRecord],
    output_path: str,
) -> int:
    """Export pairwise ranking dataset from episodes.

    For each step where the chosen candidate made progress, creates
    (winner, loser) pairs from the chosen vs non-chosen candidates.

    Returns number of pairs written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for episode in episodes:
            for step in episode.steps:
                if not step.made_progress:
                    continue
                if not step.chosen_candidate_id:
                    continue

                winner_features: Dict[str, float] = {}
                loser_candidates: List[Dict[str, float]] = []

                for c in step.candidates:
                    if c.candidate_id == step.chosen_candidate_id:
                        winner_features = c.features
                    else:
                        loser_candidates.append(c.features)

                for loser_feat in loser_candidates:
                    pair = {
                        "task_id": step.task_id,
                        "task_text": step.task_text,
                        "step_index": step.step_index,
                        "winner_features": winner_features,
                        "loser_features": loser_feat,
                        "score_delta": step.score_delta,
                    }
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    count += 1
    return count


def export_verifier_classification(
    episodes: List[EpisodeRecord],
    output_path: str,
) -> int:
    """Export verifier classification dataset from episodes.

    Each step becomes a training example: features -> verifier label.
    Uses IWA validation events as ground truth.

    Returns number of examples written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for episode in episodes:
            for i, step in enumerate(episode.steps):
                label = _derive_verifier_label(step, episode, i)
                example = {
                    "task_id": step.task_id,
                    "task_text": step.task_text,
                    "step_index": step.step_index,
                    "url": step.url,
                    "url_changed": step.url_changed,
                    "dom_changed": step.dom_changed,
                    "chosen_action_type": step.chosen_action_type,
                    "candidate_count": step.candidate_count,
                    "loop_count": step.loop_count,
                    "unique_urls_visited": step.unique_urls_visited,
                    "previous_action_type": step.previous_action_type,
                    "score_delta": step.score_delta,
                    "dom_node_count": step.dom_node_count,
                    "validation_success_count": sum(
                        1 for v in step.validation_events if v.success
                    ),
                    "validation_fail_count": sum(
                        1 for v in step.validation_events if not v.success
                    ),
                    "label": label,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
    return count


def export_sft(
    episodes: List[EpisodeRecord],
    output_path: str,
) -> int:
    """Export SFT dataset: state -> best action from successful episodes.

    Only includes steps from successful episodes that made progress.

    Returns number of examples written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for episode in episodes:
            if not episode.success:
                continue
            for step in episode.steps:
                if not step.made_progress:
                    continue
                example = {
                    "task_id": step.task_id,
                    "task_text": step.task_text,
                    "url": step.url,
                    "page_title": step.page_title,
                    "candidate_count": step.candidate_count,
                    "action_type": step.chosen_action_type,
                    "action": step.chosen_action,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
    return count


def export_dagger(
    episodes: List[EpisodeRecord],
    output_path: str,
) -> int:
    """Export DAgger dataset: uncertain/failed states with expert corrections.

    Includes steps where expert_action was recorded.

    Returns number of examples written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for episode in episodes:
            for step in episode.steps:
                if step.expert_action is None:
                    continue
                example = {
                    "task_id": step.task_id,
                    "task_text": step.task_text,
                    "step_index": step.step_index,
                    "url": step.url,
                    "page_title": step.page_title,
                    "agent_action": step.chosen_action,
                    "agent_action_type": step.chosen_action_type,
                    "expert_action": step.expert_action,
                    "expert_candidate_id": step.expert_candidate_id,
                    "score_delta": step.score_delta,
                    "verifier_status": step.verifier_status,
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
    return count


def _derive_verifier_label(
    step: FullStepRecord,
    episode: EpisodeRecord,
    step_idx: int,
) -> str:
    """Derive ground-truth verifier label from step data."""
    # Check validation events for errors
    for ve in step.validation_events:
        if not ve.success and ve.source in ("frontend", "backend"):
            return "validation_error"

    # Check for dangerous action
    if step.error_type == "dangerous":
        return "dangerous_action"

    # Check for task completion (last step of successful episode)
    is_last = step_idx == len(episode.steps) - 1
    if is_last and episode.success:
        return "task_complete"

    # Loop detection
    if step.loop_count >= 3:
        return "loop"

    # Check for progress
    if step.made_progress or step.score_delta > 0:
        return "local_progress"

    return "no_op"


# Alias expected by CHECK.sh
export_ranking_pairs = export_pairwise_ranking


def _dict_to_episode(data: Dict[str, Any]) -> EpisodeRecord:
    """Reconstruct EpisodeRecord from a JSON dict."""
    from .schema import CandidateRecord, ValidationEvent

    steps: List[FullStepRecord] = []
    for sd in data.get("steps", []):
        candidates = [
            CandidateRecord(**{k: v for k, v in c.items() if k in CandidateRecord.__dataclass_fields__})
            for c in sd.get("candidates", [])
            if isinstance(c, dict)
        ]
        validation_events = [
            ValidationEvent(**{k: v for k, v in ve.items() if k in ValidationEvent.__dataclass_fields__})
            for ve in sd.get("validation_events", [])
            if isinstance(ve, dict)
        ]
        step_kwargs = {k: v for k, v in sd.items() if k in FullStepRecord.__dataclass_fields__}
        step_kwargs["candidates"] = candidates
        step_kwargs["validation_events"] = validation_events
        steps.append(FullStepRecord(**step_kwargs))

    ep_kwargs = {k: v for k, v in data.items() if k in EpisodeRecord.__dataclass_fields__ and k != "steps"}
    ep_kwargs["steps"] = steps
    return EpisodeRecord(**ep_kwargs)
