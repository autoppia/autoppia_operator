"""Planned actions and selector dicts from IWA demo trajectories (all web projects).

Raw IWA steps carry one selector per action. This module reuses
``selector_candidates_for_ids`` / classes / placeholders / text keys from
``selectors`` so every project with ``web_N_<id>/src/dynamic/v3/data/*-variants.json``
gets the same id/class/text expansion that autocinema's hand-written helpers use.

Selectors that are not expandible here (e.g. ``xpathSelector``) are left unchanged
so the agent can still use the recorded XPath or CSS.
"""

from __future__ import annotations

import re
from typing import Any

from autoppia_iwa.src.demo_webs.trajectory_registry import get_trajectory_map

import training._iwa_path  # noqa: F401
from training.deterministic_harvester.iwa_planned_actions import (
    frontend_url_for_project,
    iwa_actions_to_planned_actions,
)

__all__ = [
    "enrich_iwa_selector_candidates",
    "enrich_planned_actions_with_variants",
    "list_iwa_use_cases",
    "planned_actions_for_iwa_use_case",
    "planned_actions_for_iwa_use_case_enriched",
    "selector_candidates_enriched_for_iwa_use_case",
    "selector_candidates_from_iwa_trajectory",
]


def _selector_dict_fingerprint(selector: dict[str, Any]) -> str:
    t = str(selector.get("type") or "").strip()
    a = str(selector.get("attribute") or "").strip().lower()
    v = str(selector.get("value") or "").strip().lower()
    return f"{t}|{a}|{v}"


_XPATH_ID_PATTERNS = (
    r"@id\s*=\s*['\"]([^'\"]+)['\"]",
    r"id\(\s*['\"]([^'\"]+)['\"]\s*\)",
)
_XPATH_PLACEHOLDER_PATTERN = r"@placeholder\s*=\s*['\"]([^'\"]+)['\"]"
_XPATH_CLASS_EQ_PATTERN = r"@class\s*=\s*['\"]([^'\"]+)['\"]"
_XPATH_CLASS_CONTAINS_PATTERN = r"contains\(\s*@class\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
_XPATH_TEXT_EQ_PATTERN = r"normalize-space\(\)\s*=\s*['\"]([^'\"]+)['\"]"
_XPATH_TEXT_CONTAINS_PATTERN = r"contains\(\s*normalize-space\(\)\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
_XPATH_TEXT_CONTAINS_DOT_PATTERN = r"contains\(\s*\.\s*,\s*['\"]([^'\"]+)['\"]\s*\)"
_XPATH_ARIA_PATTERN = r"@aria-label\s*=\s*['\"]([^'\"]+)['\"]"
_XPATH_ATTR_EQ_PATTERN = r"@([a-zA-Z0-9_-]+)\s*=\s*['\"]([^'\"]+)['\"]"
_XPATH_ATTR_CONTAINS_PATTERN = r"contains\(\s*@([a-zA-Z0-9_-]+)\s*,\s*['\"]([^'\"]+)['\"]\s*\)"


def _extract_unique(pattern: str, value: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in re.findall(pattern, value):
        candidate = str(match).strip()
        lowered = candidate.lower()
        if candidate and lowered not in seen:
            seen.add(lowered)
            out.append(candidate)
    return out


def _semantic_candidates_from_xpath(
    xpath: str,
    *,
    project_id: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    from training.deterministic_harvester.selectors import (
        selector_candidates_for_classes,
        selector_candidates_for_ids,
        selector_candidates_for_placeholders,
        selector_candidates_for_text_variant_keys,
        selector_candidates_for_texts,
    )

    raw = str(xpath or "").strip()
    if not raw:
        return []

    ids: list[str] = []
    for pattern in _XPATH_ID_PATTERNS:
        ids.extend(_extract_unique(pattern, raw))
    placeholders = _extract_unique(_XPATH_PLACEHOLDER_PATTERN, raw)
    class_eq = _extract_unique(_XPATH_CLASS_EQ_PATTERN, raw)
    class_contains = _extract_unique(_XPATH_CLASS_CONTAINS_PATTERN, raw)
    classes: list[str] = []
    seen_class: set[str] = set()
    for item in [*class_eq, *class_contains]:
        # Keep class selectors coarse (tokenized) to maximize resilience.
        tokens = [token for token in str(item).split() if token.strip()]
        for token in tokens or [item]:
            normalized = str(token).strip()
            lowered = normalized.lower()
            if normalized and lowered not in seen_class:
                seen_class.add(lowered)
                classes.append(normalized)
    texts = [
        *_extract_unique(_XPATH_TEXT_EQ_PATTERN, raw),
        *_extract_unique(_XPATH_TEXT_CONTAINS_PATTERN, raw),
        *_extract_unique(_XPATH_TEXT_CONTAINS_DOT_PATTERN, raw),
        *_extract_unique(_XPATH_ARIA_PATTERN, raw),
    ]
    generic_attrs: list[tuple[str, str]] = []
    seen_attr_pairs: set[str] = set()
    for attr, value in re.findall(_XPATH_ATTR_EQ_PATTERN, raw):
        attr_name = str(attr).strip().lower()
        attr_value = str(value).strip()
        if not attr_name or not attr_value:
            continue
        if attr_name in {"id", "class", "placeholder"}:
            continue
        key = f"{attr_name}={attr_value.lower()}"
        if key in seen_attr_pairs:
            continue
        seen_attr_pairs.add(key)
        generic_attrs.append((attr_name, attr_value))
    for attr, value in re.findall(_XPATH_ATTR_CONTAINS_PATTERN, raw):
        attr_name = str(attr).strip().lower()
        attr_value = str(value).strip()
        if not attr_name or not attr_value:
            continue
        if attr_name in {"id", "class", "placeholder"}:
            continue
        key = f"{attr_name}~={attr_value.lower()}"
        if key in seen_attr_pairs:
            continue
        seen_attr_pairs.add(key)
        generic_attrs.append((attr_name, attr_value))

    out: list[dict[str, Any]] = []
    for selector in selector_candidates_for_ids(*ids, project_id=project_id, seed=seed):
        out.append(dict(selector))
    for selector in selector_candidates_for_classes(*classes, project_id=project_id, seed=seed):
        out.append(dict(selector))
    for selector in selector_candidates_for_placeholders(*placeholders, project_id=project_id, seed=seed):
        out.append(dict(selector))
    for text in texts:
        for selector in selector_candidates_for_texts(text):
            out.append(dict(selector))
        for selector in selector_candidates_for_text_variant_keys(text, project_id=project_id, seed=seed):
            out.append(dict(selector))
    for attr_name, attr_value in generic_attrs:
        out.append(
            {
                "type": "attributeValueSelector",
                "attribute": attr_name,
                "value": attr_value,
                "case_sensitive": False,
            }
        )
    return out


def enrich_iwa_selector_candidates(
    candidates: list[dict[str, Any]] | None,
    *,
    project_id: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Expand IWA raw selector dicts using per-project id/class/placeholder/text variant JSONs."""
    if not candidates:
        return []
    from training.deterministic_harvester.selectors import (
        selector_candidates_for_classes,
        selector_candidates_for_ids,
        selector_candidates_for_placeholders,
        selector_candidates_for_text_variant_keys,
        selector_candidates_for_texts,
    )

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sel in candidates:
        if not isinstance(sel, dict):
            continue
        st = str(sel.get("type") or "").strip()
        if st == "attributeValueSelector":
            attr = str(sel.get("attribute") or "").strip().lower()
            val = str(sel.get("value") or "").strip()
            if not val:
                continue
            if attr == "id":
                expanded = selector_candidates_for_ids(val, project_id=project_id, seed=seed)
            elif attr == "class":
                expanded = selector_candidates_for_classes(val, project_id=project_id, seed=seed)
            elif attr in {"placeholder", "data-placeholder"}:
                expanded = selector_candidates_for_placeholders(val, project_id=project_id, seed=seed)
            else:
                expanded = [sel]
        elif st == "tagContainsSelector":
            val = str(sel.get("value") or "").strip()
            if not val:
                continue
            expanded = list(
                dict(s)
                for s in (
                    *selector_candidates_for_texts(val),
                    *selector_candidates_for_text_variant_keys(
                        val,
                        project_id=project_id,
                        seed=seed,
                    ),
                )
            )
        elif st == "xpathSelector":
            val = str(sel.get("value") or "").strip()
            expanded = []
            if val:
                semantic = _semantic_candidates_from_xpath(val, project_id=project_id, seed=seed)
                # XPath with axis navigation pinpoints a specific element precisely —
                # put it first so the executor doesn't waste time on coarse class candidates.
                uses_axis = any(ax in val for ax in ("following::", "following-sibling::", "preceding::", "ancestor::"))
                if uses_axis:
                    expanded.append(dict(sel))
                    expanded.extend(semantic)
                else:
                    expanded.extend(semantic)
                    expanded.append(dict(sel))
            else:
                expanded.append(dict(sel))
        else:
            expanded = [sel]
        for item in expanded:
            if not isinstance(item, dict):
                continue
            key = _selector_dict_fingerprint(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def enrich_planned_actions_with_variants(
    actions: list[dict[str, Any]],
    *,
    project_id: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Re-run IWA-based planned actions with broadened ``selector_candidates`` for each step."""
    out: list[dict[str, Any]] = []
    for step in actions:
        if not isinstance(step, dict):
            continue
        ac = step.get("selector_candidates")
        if isinstance(ac, list) and ac and all(isinstance(x, dict) for x in ac):
            merged = dict(step)
            merged["selector_candidates"] = enrich_iwa_selector_candidates(
                ac,
                project_id=project_id,
                seed=seed,
            )
            out.append(merged)
        else:
            out.append(dict(step))
    return out


_TRAJECTORY_MAPS: dict[str, dict[str, object] | None] = {}

_CURATED_USE_CASE_TEXT_FALLBACKS: dict[tuple[str, str], list[str]] = {
    ("automail", "EMAILS_NEXT_PAGE"): ["Next", "Next page"],
    ("autowork", "NAVBAR_FAVORITES_CLICK"): ["Favorites"],
    ("autolodge", "ADD_TO_WISHLIST"): ["Add to wishlist", "Wishlist"],
    ("autolodge", "REMOVE_FROM_WISHLIST"): ["Remove from wishlist", "Wishlist"],
    ("autolodge", "VIEW_HOTEL"): ["View hotel", "View details"],
    ("autoconnect", "APPLY_FOR_JOB"): ["Apply", "Apply now"],
    ("autoconnect", "CANCEL_APPLICATION"): ["Cancel application"],
    ("autoconnect", "COMMENT_ON_POST"): ["Comment", "Post comment"],
    ("autoconnect", "CONNECT_WITH_USER"): ["Connect", "Send request"],
    ("autoconnect", "FILTER_JOBS"): ["Location", "Salary"],
    ("autoconnect", "HIDE_POST"): ["Hide post"],
    ("autoconnect", "POST_STATUS"): ["Post", "Share"],
    ("autoconnect", "REMOVE_POST"): ["Remove post", "Delete post"],
    ("autoconnect", "UNFOLLOW_PAGE"): ["Unfollow"],
    ("autoconnect", "UNHIDE_POST"): ["Unhide post"],
    ("autoconnect", "VIEW_HIDDEN_POSTS"): ["Hidden posts"],
    ("autoconnect", "VIEW_JOB"): ["View job", "Details"],
    ("autolist", "AUTOLIST_ADD_TASK_CLICKED"): ["Add task", "New task"],
    ("autolist", "AUTOLIST_EDIT_TASK_MODAL_OPENED"): ["Edit task"],
}


def _trajectory_map(project_id: str) -> dict[str, object]:
    pid = str(project_id or "").strip()
    if pid not in _TRAJECTORY_MAPS or _TRAJECTORY_MAPS[pid] is None:
        loaded = get_trajectory_map(pid)
        if not loaded:
            raise RuntimeError(f"IWA returned no trajectory map for {pid!r}; check demo_webs trajectories for this project")
        _TRAJECTORY_MAPS[pid] = dict(loaded)
    return _TRAJECTORY_MAPS[pid]  # type: ignore[return-value]


def list_iwa_use_cases(project_id: str) -> frozenset[str]:
    return frozenset(_trajectory_map(project_id).keys())


def _has_semantic_candidates(actions: list[dict[str, Any]]) -> bool:
    for step in actions:
        candidates = step.get("selector_candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("type") or "").strip() != "xpathSelector":
                return True
    return False


def _apply_curated_semantic_fallbacks(
    project_id: str,
    use_case: str,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    key = (str(project_id or "").strip().lower(), str(use_case or "").strip().upper())
    labels = _CURATED_USE_CASE_TEXT_FALLBACKS.get(key) or []
    if not labels or _has_semantic_candidates(actions):
        return actions
    from training.deterministic_harvester.selectors import selector_candidates_for_texts

    fallback = [dict(item) for item in selector_candidates_for_texts(*labels)]
    if not fallback:
        return actions
    out: list[dict[str, Any]] = []
    injected = False
    for step in actions:
        step_type = str(step.get("type") or "").strip()
        if injected or step_type == "NavigateAction":
            out.append(dict(step))
            continue
        candidates = step.get("selector_candidates")
        if not isinstance(candidates, list):
            if step_type in {"ClickAction", "TypeAction", "SelectAction", "HoverAction", "ScrollAction", "WaitAction"}:
                merged = dict(step)
                merged["selector_candidates"] = list(fallback)
                out.append(merged)
                injected = True
                continue
            out.append(dict(step))
            continue
        merged = dict(step)
        merged["selector_candidates"] = [*fallback, *[dict(item) for item in candidates if isinstance(item, dict)]]
        out.append(merged)
        injected = True
    return out if injected else actions


def planned_actions_for_iwa_use_case(
    project_id: str,
    use_case: str,
    *,
    task_url: str | None = None,
) -> list[dict[str, Any]]:
    """IWA trajectory actions as operator `planned_actions` dicts (same shape as per-project IWA builders)."""
    u = str(use_case or "").strip().upper()
    traj_map = _trajectory_map(str(project_id or "").strip())
    traj = traj_map.get(u)
    if traj is None:
        raise ValueError(f"Unknown IWA use case {u!r} for project {str(project_id or '').strip()!r}. Known: {sorted(traj_map.keys())[:20]}{'...' if len(traj_map) > 20 else ''}")
    fe = frontend_url_for_project(str(project_id or "").strip())
    actions_raw = list(getattr(traj, "actions", None) or [])
    actions = iwa_actions_to_planned_actions(actions_raw, frontend_url=fe)
    if task_url and str(task_url).strip() and actions and str(actions[0].get("type") or "").strip() == "NavigateAction":
        first = dict(actions[0])
        first["url"] = str(task_url).strip()
        actions[0] = first
    return actions


def planned_actions_for_iwa_use_case_enriched(
    project_id: str,
    use_case: str,
    *,
    task_url: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """IWA planned actions for ``(project_id, use_case)`` with variant-expanded selectors."""
    base = planned_actions_for_iwa_use_case(project_id, use_case, task_url=task_url)
    enriched = enrich_planned_actions_with_variants(
        base,
        project_id=str(project_id or "").strip(),
        seed=seed,
    )
    return _apply_curated_semantic_fallbacks(project_id, use_case, enriched)


def selector_candidates_from_iwa_trajectory(
    project_id: str,
    use_case: str,
    *,
    task_url: str | None = None,
) -> list[dict[str, Any]]:
    """Flattened selector dicts from all planned steps that carry `selector_candidates`."""
    out: list[dict[str, Any]] = []
    for step in planned_actions_for_iwa_use_case(project_id, use_case, task_url=task_url):
        cands = step.get("selector_candidates")
        if not isinstance(cands, list):
            continue
        for item in cands:
            if isinstance(item, dict):
                out.append(dict(item))
    return out


def selector_candidates_enriched_for_iwa_use_case(
    project_id: str,
    use_case: str,
    *,
    task_url: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """All variant-expanded selector dicts in trajectory order (non-navigate steps)."""
    out: list[dict[str, Any]] = []
    for step in planned_actions_for_iwa_use_case_enriched(
        project_id,
        use_case,
        task_url=task_url,
        seed=seed,
    ):
        cands = step.get("selector_candidates")
        if not isinstance(cands, list):
            continue
        for item in cands:
            if isinstance(item, dict):
                out.append(dict(item))
    return out
