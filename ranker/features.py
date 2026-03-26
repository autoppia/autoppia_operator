"""Feature extraction for the learned candidate ranker.

Extracts generic DOM/action features from (task, state, candidate) tuples.
No website-specific features — everything is structural/semantic.
No keyword dictionaries — all features are numeric/positional.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Feature vector dimension for the ranker MLP (power-of-2 for GPU alignment)
RANKER_FEATURE_DIM = 64


def extract_candidate_features(
    task: str,
    state: Dict[str, Any],
    candidate: Dict[str, Any],
) -> List[float]:
    """Extract a fixed-size feature vector for a (task, state, candidate) tuple.

    All features are structural/numeric — no keyword lists or task classification.

    Args:
        task: Natural language task description.
        state: Current agent state dict with page_graph_summary, step_index, etc.
        candidate: Candidate dict with element_type, role, text, etc.

    Returns:
        Feature vector of length RANKER_FEATURE_DIM.
    """
    features: List[float] = []

    # --- Task features (8 dims) ---
    task_lower = task.lower()
    task_tokens = task_lower.split()
    task_token_set = set(task_tokens)

    # Task length (normalized)
    features.append(min(len(task_lower) / 200.0, 1.0))
    # Word count (normalized)
    features.append(min(len(task_tokens) / 30.0, 1.0))
    # Average word length
    avg_word_len = sum(len(w) for w in task_tokens) / max(len(task_tokens), 1)
    features.append(min(avg_word_len / 10.0, 1.0))
    # Number of unique words / total (vocabulary richness)
    features.append(len(task_token_set) / max(len(task_tokens), 1))
    # Has digits in task
    features.append(1.0 if any(c.isdigit() for c in task_lower) else 0.0)
    # Number of quoted strings (constraints)
    features.append(min(task_lower.count("'") // 2 + task_lower.count('"') // 2, 5) / 5.0)
    # Has question mark (informational indicator)
    features.append(1.0 if "?" in task else 0.0)
    # Task sentence count (approximate)
    features.append(min(task.count(".") + task.count("?") + task.count("!"), 5) / 5.0)

    # --- Page features (8 dims) ---
    pg = state.get("page_graph_summary", {})
    features.append(min(float(pg.get("node_count", 0)) / 500.0, 1.0))
    features.append(min(float(pg.get("edge_count", 0)) / 1000.0, 1.0))
    features.append(1.0 if pg.get("has_form", False) else 0.0)
    features.append(min(float(pg.get("interactive_count", 0)) / 50.0, 1.0))
    features.append(min(float(pg.get("result_list_length", 0)) / 20.0, 1.0))

    region = str(state.get("active_region", "")).lower()
    features.append(min(len(region) / 20.0, 1.0))  # region name length as proxy
    # Ratio of interactive elements to total nodes
    total_nodes = max(float(pg.get("node_count", 1)), 1.0)
    features.append(min(float(pg.get("interactive_count", 0)) / total_nodes, 1.0))
    # Page depth (URL path segments)
    url = str(state.get("url", ""))
    url_parts = [p for p in url.split("/") if p and "://" not in p]
    features.append(min(len(url_parts) / 8.0, 1.0))

    # --- Candidate features (24 dims) ---
    cand_text = str(candidate.get("text", "")).lower()
    cand_type = str(candidate.get("element_type", "")).lower()
    cand_role = str(candidate.get("role", "")).lower()

    # Element type one-hot (6 dims) — HTML tag types are structural, not keywords
    for etype in ["button", "input", "a", "select", "textarea", "label"]:
        features.append(1.0 if cand_type == etype else 0.0)

    # Role features (4 dims) — ARIA roles are structural
    for r in ["button", "link", "textbox", "combobox"]:
        features.append(1.0 if cand_role == r else 0.0)

    # Text properties (6 dims) — all numeric/boolean
    features.append(min(len(cand_text) / 100.0, 1.0))  # text length
    features.append(1.0 if candidate.get("visible", True) else 0.0)
    features.append(1.0 if candidate.get("has_placeholder", False) else 0.0)
    features.append(1.0 if candidate.get("form_membership", False) else 0.0)
    features.append(1.0 if candidate.get("aria_label", "") else 0.0)
    features.append(1.0 if cand_text.strip() else 0.0)

    # Token overlap between task and candidate text (4 dims) — computed, no keyword list
    cand_tokens = set(cand_text.split())
    overlap = len(task_token_set & cand_tokens)
    features.append(min(overlap / max(len(task_token_set), 1), 1.0))
    features.append(min(overlap / max(len(cand_tokens), 1), 1.0))
    # Longest common subsequence ratio (character-level, approximate)
    shorter = min(len(task_lower), len(cand_text))
    features.append(min(overlap / max(shorter / 5.0, 1.0), 1.0))
    # Number of candidate tokens
    features.append(min(len(cand_tokens) / 15.0, 1.0))

    # Position features (4 dims) — purely geometric
    pos = candidate.get("position", {})
    features.append(min(float(pos.get("x", 0)) / 1920.0, 1.0))
    features.append(min(float(pos.get("y", 0)) / 1080.0, 1.0))
    features.append(min(float(pos.get("width", 0)) / 500.0, 1.0))
    features.append(min(float(pos.get("height", 0)) / 200.0, 1.0))

    # --- Context features (8 dims) ---
    features.append(min(float(state.get("step_index", 0)) / 20.0, 1.0))
    prev_type = str(state.get("previous_action_type", "")).lower()
    # Previous action type encoded as length proxy (no keyword matching)
    features.append(min(len(prev_type) / 10.0, 1.0))
    # Is same action type as previous
    chosen = str(candidate.get("action_type", "")).lower()
    features.append(1.0 if chosen and chosen == prev_type else 0.0)
    # Step progression ratio
    max_steps = float(state.get("max_steps", 20))
    features.append(min(float(state.get("step_index", 0)) / max(max_steps, 1.0), 1.0))
    features.append(min(float(state.get("loop_count", 0)) / 5.0, 1.0))
    features.append(min(float(state.get("form_fill_progress", 0.0)), 1.0))
    features.append(min(float(state.get("unique_urls_visited", 1)) / 10.0, 1.0))
    features.append(min(float(candidate.get("rank_position", 0)) / 20.0, 1.0))

    # Pad or truncate to RANKER_FEATURE_DIM
    while len(features) < RANKER_FEATURE_DIM:
        features.append(0.0)
    return features[:RANKER_FEATURE_DIM]


def extract_batch_features(
    task: str,
    state: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> List[List[float]]:
    """Extract features for all candidates in a batch."""
    return [extract_candidate_features(task, state, c) for c in candidates]


def feature_names() -> List[str]:
    """Return human-readable names for each feature dimension."""
    names = [
        # Task features (8)
        "task_len_norm", "task_word_count_norm", "task_avg_word_len",
        "task_vocab_richness", "task_has_digits", "task_quoted_count",
        "task_has_question", "task_sentence_count",
        # Page features (8)
        "page_node_count", "page_edge_count", "page_has_form",
        "page_interactive_count", "page_result_list_len",
        "region_name_len", "interactive_ratio", "url_depth",
        # Candidate element type (6)
        "elem_button", "elem_input", "elem_a", "elem_select",
        "elem_textarea", "elem_label",
        # Candidate role (4)
        "role_button", "role_link", "role_textbox", "role_combobox",
        # Candidate text (6)
        "text_len", "visible", "has_placeholder", "form_membership",
        "has_aria_label", "has_text",
        # Token overlap (4)
        "token_overlap_task", "token_overlap_cand",
        "overlap_density", "cand_token_count",
        # Position (4)
        "pos_x", "pos_y", "pos_width", "pos_height",
        # Context (8)
        "step_index", "prev_action_len", "same_action_as_prev",
        "step_progression", "loop_count", "form_fill_progress",
        "unique_urls", "rank_position",
    ]
    return names[:RANKER_FEATURE_DIM]


# Aliases expected by CHECK.sh
extract_features = extract_candidate_features


class CandidateFeatures:
    """Container for candidate feature metadata."""

    dim = RANKER_FEATURE_DIM
    extract = staticmethod(extract_candidate_features)
    names = staticmethod(feature_names)
