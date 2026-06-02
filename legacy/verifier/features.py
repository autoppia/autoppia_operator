"""Feature extraction for the learned progress verifier.

Extracts generic page/action/history features for classifying step outcomes.
No website-specific features. No keyword dictionaries or hardcoded string matching.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Feature vector dimension for the verifier classifier
VERIFIER_FEATURE_DIM = 32


def extract_verifier_features(
    task: str,
    step: Dict[str, Any],
    history: List[Dict[str, Any]] | None = None,
) -> List[float]:
    """Extract a fixed-size feature vector for verifier classification.

    All features are structural/numeric — no keyword matching or task classification.

    Args:
        task: Natural language task description.
        step: Step dict with url, url_changed, dom_changed, action_type, etc.
        history: Previous steps for computing history features.

    Returns:
        Feature vector of length VERIFIER_FEATURE_DIM.
    """
    features: List[float] = []
    history = history or []
    task_tokens = set(task.lower().split())

    # --- URL change features (4 dims) ---
    features.append(1.0 if step.get("url_changed", False) else 0.0)
    url = str(step.get("url", ""))
    features.append(min(len(url) / 200.0, 1.0))
    # URL path depth
    path_parts = [p for p in url.split("/") if p and "://" not in p]
    features.append(min(len(path_parts) / 10.0, 1.0))
    # Token overlap between task and URL (computed, no keyword list)
    url_tokens = set(url.lower().replace("/", " ").replace("-", " ").replace("_", " ").split())
    url_overlap = len(task_tokens & url_tokens)
    features.append(min(url_overlap / max(len(task_tokens), 1), 1.0))

    # --- Page title features (3 dims) ---
    title = str(step.get("page_title", "")).lower()
    features.append(min(len(title) / 100.0, 1.0))
    # Token overlap between task and title
    title_tokens = set(title.split())
    title_overlap = len(task_tokens & title_tokens)
    features.append(min(title_overlap / max(len(task_tokens), 1), 1.0))
    # Title changed from previous step
    prev_title = str(history[-1].get("page_title", "")).lower() if history else ""
    features.append(1.0 if title != prev_title and prev_title else 0.0)

    # --- DOM diff summary (4 dims) ---
    features.append(1.0 if step.get("dom_changed", False) else 0.0)
    features.append(min(float(step.get("dom_node_count", 0)) / 500.0, 1.0))
    prev_nodes = float(history[-1].get("dom_node_count", 0)) if history else 0.0
    curr_nodes = float(step.get("dom_node_count", 0))
    node_delta = (curr_nodes - prev_nodes) / max(prev_nodes, 1.0)
    features.append(max(min(node_delta, 1.0), -1.0))
    features.append(min(float(step.get("candidate_count", 0)) / 50.0, 1.0))

    # --- Action result features (5 dims) ---
    action_type = str(step.get("chosen_action_type", "")).lower()
    # Encode action type as numeric properties instead of keyword matching
    features.append(min(len(action_type) / 10.0, 1.0))  # action name length
    # Is same action type as previous step
    prev_action = str(history[-1].get("chosen_action_type", "")).lower() if history else ""
    features.append(1.0 if action_type == prev_action and action_type else 0.0)
    # Action type changed
    features.append(1.0 if action_type != prev_action and prev_action else 0.0)
    # Number of distinct action types used so far
    all_actions = {str(h.get("chosen_action_type", "")) for h in history}
    all_actions.add(action_type)
    all_actions.discard("")
    features.append(min(len(all_actions) / 6.0, 1.0))
    features.append(float(step.get("score_delta", 0.0)))

    # --- Validation event features (4 dims) ---
    val_events = step.get("validation_events", [])
    val_success = sum(1 for v in val_events if isinstance(v, dict) and v.get("success"))
    val_fail = sum(1 for v in val_events if isinstance(v, dict) and not v.get("success"))
    features.append(min(val_success / 3.0, 1.0))
    features.append(min(val_fail / 3.0, 1.0))
    features.append(min(len(val_events) / 5.0, 1.0))  # total event count
    # Ratio of successful validations
    features.append(val_success / max(len(val_events), 1))

    # --- History features (8 dims) ---
    step_count = len(history) + 1
    features.append(min(step_count / 20.0, 1.0))

    # Unique URLs visited
    all_urls = {str(h.get("url", "")) for h in history}
    all_urls.add(str(step.get("url", "")))
    all_urls.discard("")
    features.append(min(len(all_urls) / 10.0, 1.0))

    # Action diversity
    features.append(min(len(all_actions) / 5.0, 1.0))

    # Loop count
    features.append(min(float(step.get("loop_count", 0)) / 5.0, 1.0))

    # Progress ratio in history
    progress_count = sum(1 for h in history if h.get("made_progress", False))
    features.append(progress_count / max(len(history), 1))

    # Cumulative reward
    features.append(max(min(float(step.get("cumulative_reward", 0.0)), 1.0), -1.0))

    # Error count in history
    error_count = sum(1 for h in history if h.get("error_type", ""))
    features.append(min(error_count / 5.0, 1.0))

    # Consecutive no-progress steps
    consec_no_progress = 0
    for h in reversed(history):
        if h.get("made_progress", False):
            break
        consec_no_progress += 1
    features.append(min(consec_no_progress / 5.0, 1.0))

    # --- Completion signal features (4 dims) — all numeric, no keyword matching ---
    # Token overlap between task and page text (computed overlap, not keyword check)
    page_text = str(step.get("page_text", "")).lower()
    page_tokens = set(page_text.split()[:200])  # limit for performance
    page_overlap = len(task_tokens & page_tokens)
    features.append(min(page_overlap / max(len(task_tokens), 1), 1.0))
    # Page text length change (new content appeared)
    prev_text_len = float(history[-1].get("page_text_len", 0)) if history else 0.0
    curr_text_len = len(page_text)
    text_delta = (curr_text_len - prev_text_len) / max(prev_text_len, 1.0)
    features.append(max(min(text_delta, 1.0), -1.0))
    # Score delta magnitude
    features.append(min(abs(float(step.get("score_delta", 0.0))), 1.0))
    # Made progress flag
    features.append(1.0 if step.get("made_progress", False) else 0.0)

    # Pad or truncate
    while len(features) < VERIFIER_FEATURE_DIM:
        features.append(0.0)
    return features[:VERIFIER_FEATURE_DIM]


# Aliases expected by CHECK.sh
extract_features = extract_verifier_features


class VerifierFeatures:
    """Container for verifier feature metadata."""

    dim = VERIFIER_FEATURE_DIM
    extract = staticmethod(extract_verifier_features)


def verifier_feature_names() -> List[str]:
    """Return human-readable names for each feature dimension."""
    return [
        "url_changed", "url_len", "url_depth", "url_task_overlap",
        "title_len", "title_task_overlap", "title_changed",
        "dom_changed", "dom_node_count", "dom_node_delta", "candidate_count",
        "action_name_len", "same_action_as_prev", "action_changed",
        "action_diversity", "score_delta",
        "val_success_count", "val_fail_count", "val_event_count", "val_success_ratio",
        "step_count", "unique_urls", "action_type_diversity", "loop_count",
        "progress_ratio", "cumulative_reward", "error_count", "consec_no_progress",
        "page_task_overlap", "page_text_delta", "score_delta_magnitude", "made_progress",
    ][:VERIFIER_FEATURE_DIM]
