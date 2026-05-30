from __future__ import annotations

from typing import Any

from training.deterministic_harvester.trajectory_selectors import planned_actions_for_iwa_use_case_enriched


def selector_steps_for_use_case(
    project_id: str,
    use_case: str,
    *,
    seed: int | None = None,
    task_url: str | None = None,
) -> list[dict[str, Any]]:
    """Return non-navigate planned steps that expose selector candidates."""
    out: list[dict[str, Any]] = []
    for step in planned_actions_for_iwa_use_case_enriched(
        str(project_id or "").strip(),
        str(use_case or "").strip().upper(),
        task_url=task_url,
        seed=seed,
    ):
        if str(step.get("type") or "").strip() == "NavigateAction":
            continue
        if not isinstance(step.get("selector_candidates"), list):
            continue
        out.append(dict(step))
    return out


def semantic_selector_candidates_for_use_case(
    project_id: str,
    use_case: str,
    *,
    seed: int | None = None,
    task_url: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return flattened selector candidates with XPath removed.

    This is useful for asserting semantic selector coverage while the runtime still
    retains XPath as the final fallback candidate.
    """
    out: list[dict[str, Any]] = []
    for step in selector_steps_for_use_case(
        project_id,
        use_case,
        seed=seed,
        task_url=task_url,
    ):
        candidates = step.get("selector_candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("type") or "").strip() == "xpathSelector":
                continue
            out.append(dict(candidate))
    return out


def has_semantic_selector_coverage(
    project_id: str,
    use_case: str,
    *,
    seed: int | None = None,
    task_url: str | None = None,
) -> bool:
    return bool(
        semantic_selector_candidates_for_use_case(
            project_id,
            use_case,
            seed=seed,
            task_url=task_url,
        )
    )


__all__ = [
    "has_semantic_selector_coverage",
    "selector_steps_for_use_case",
    "semantic_selector_candidates_for_use_case",
]
