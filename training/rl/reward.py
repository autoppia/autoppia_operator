from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RewardBreakdown:
    total: float
    terminal_success: float
    score_delta: float
    partial_progress: float
    field_progress_bonus: float
    exact_target_bonus: float
    field_alignment_bonus: float
    submit_readiness_bonus: float
    task_hint_bonus: float
    step_cost: float
    repeat_penalty: float
    off_route_penalty: float
    task_hint_penalty: float
    generic_selector_penalty: float
    redundant_field_penalty: float
    execution_error_penalty: float
    noop_penalty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _action_fingerprint(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
    selector_key = "|".join(
        [
            str(selector.get("type") or ""),
            str(selector.get("attribute") or ""),
            str(selector.get("value") or ""),
        ]
    )
    return "|".join(
        [
            str(action.get("type") or ""),
            str(action.get("_element_id") or ""),
            selector_key,
            str(action.get("text") or action.get("value") or ""),
            str(action.get("url") or ""),
        ]
    )


def _extract_selector_value(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
    return str(selector.get("value") or "").strip().lower()


def _normalize_targets(target_values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in target_values or []:
        text = str(value or "").strip().lower()
        if text and text not in out:
            out.append(text)
    return out


def _extract_task_hints(task_prompt: str | None) -> tuple[list[str], list[str]]:
    prompt = str(task_prompt or "")
    raw_routes = re.findall(r"(?<![A-Za-z0-9_-])/([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*)", prompt)
    routes = sorted({f"/{match.strip().lower()}" for match in raw_routes if match.strip()})
    ids = sorted(
        {
            match.strip().lower()
            for match in re.findall(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){1,}\b", prompt)
            if match.strip()
        }
    )
    return routes, ids


def _action_matches_task_hints(
    chosen_action: dict[str, Any] | None,
    task_prompt: str | None,
) -> tuple[bool, bool]:
    if not isinstance(chosen_action, dict):
        return False, False
    routes, ids = _extract_task_hints(task_prompt)
    action_type = str(chosen_action.get("type") or "")
    selector_value = _extract_selector_value(chosen_action)
    action_url = str(chosen_action.get("url") or "").strip().lower()
    route_match = bool(action_url and any(route in action_url for route in routes))
    id_match = bool(selector_value and selector_value in ids)
    if action_type == "NavigateAction":
        return route_match, id_match
    return route_match, id_match


def _extract_primary_route(task_prompt: str | None) -> str:
    routes, _ = _extract_task_hints(task_prompt)
    for route in routes:
        if route and route != "/":
            return route
    return ""


def _is_on_primary_route(url: str | None, task_prompt: str | None) -> bool:
    primary = _extract_primary_route(task_prompt)
    if not primary:
        return False
    return primary in str(url or "").lower()


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


def _infer_action_slot(chosen_action: dict[str, Any] | None) -> str:
    if not isinstance(chosen_action, dict):
        return ""
    selector = chosen_action.get("selector") if isinstance(chosen_action.get("selector"), dict) else {}
    blob = " ".join(
        [
            str(chosen_action.get("type") or ""),
            str(chosen_action.get("url") or ""),
            str(selector.get("attribute") or ""),
            str(selector.get("value") or ""),
        ]
    ).lower()
    if any(token in blob for token in ("email", "e-mail", "mail")):
        return "email"
    if any(token in blob for token in ("subject", "topic", "title", "about")):
        return "subject"
    if any(token in blob for token in ("message", "textarea", "comment", "details", "description", "body", "content")):
        return "message"
    if re.search(r"\bname\b", blob) and not re.search(r"\buser ?name\b", blob):
        return "name"
    if "password" in blob or re.search(r"\bpass\b", blob):
        return "password"
    if "username" in blob or "login" in blob:
        return "username"
    if "search" in blob or "query" in blob:
        return "search"
    return ""


def _all_prompt_targets_present(prompt_targets: dict[str, str], html: str) -> bool:
    if not prompt_targets:
        return False
    haystack = str(html or "").lower()
    for value in prompt_targets.values():
        text = str(value or "").strip().lower()
        if not text or text not in haystack:
            return False
    return True


def _is_submit_like_action(chosen_action: dict[str, Any] | None) -> bool:
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


def compute_step_reward(
    *,
    step_index: int = 0,
    prev_score: float,
    current_score: float,
    success: bool,
    exec_ok: bool,
    current_url: str,
    chosen_action: dict[str, Any] | None,
    previous_action: dict[str, Any] | None,
    previous_url: str = "",
    before_html: str = "",
    after_html: str = "",
    target_values: list[str] | None = None,
    task_prompt: str = "",
    step_cost: float = 0.02,
) -> RewardBreakdown:
    step_index = max(int(step_index), 0)
    score_delta = float(current_score) - float(prev_score)
    partial_progress = max(0.0, score_delta)
    terminal_success = 1.0 if bool(success) else 0.0
    repeat_penalty = 0.10 if _action_fingerprint(chosen_action) and _action_fingerprint(chosen_action) == _action_fingerprint(previous_action) else 0.0
    execution_error_penalty = 0.20 if not bool(exec_ok) else 0.0
    action_type = str((chosen_action or {}).get("type") or "")
    noop_penalty = 0.05 if action_type == "WaitAction" or not action_type else 0.0
    url_lower = str(current_url or "").lower()
    off_route_penalty = 0.20 if any(fragment in url_lower for fragment in ("/register", "/login")) else 0.0
    before_lower = str(before_html or "").lower()
    after_lower = str(after_html or "").lower()
    targets = _normalize_targets(target_values)
    prompt_targets = _extract_prompt_target_map(task_prompt)
    newly_matched = sum(1 for value in targets if value in after_lower and value not in before_lower)
    field_progress_bonus = 0.20 * float(newly_matched)
    chosen_text = str((chosen_action or {}).get("text") or (chosen_action or {}).get("value") or "").strip().lower()
    exact_target_bonus = 0.05 if chosen_text and chosen_text in targets else 0.0
    field_alignment_bonus = 0.0
    action_slot = _infer_action_slot(chosen_action)
    if chosen_text and action_slot:
        target_text = str(prompt_targets.get(action_slot) or "").strip().lower()
        if target_text:
            if chosen_text == target_text:
                field_alignment_bonus += 0.20
            elif chosen_text in target_text or target_text in chosen_text:
                field_alignment_bonus += 0.10
        if chosen_text in after_lower and chosen_text not in before_lower:
            field_alignment_bonus += 0.05
    all_targets_ready = _all_prompt_targets_present(prompt_targets, after_lower)
    submit_readiness_bonus = 0.0
    redundant_field_penalty = 0.0
    if all_targets_ready and _is_submit_like_action(chosen_action):
        submit_readiness_bonus += 0.25
    if all_targets_ready and action_type == "TypeAction" and action_slot and chosen_text:
        target_text = str(prompt_targets.get(action_slot) or "").strip().lower()
        if target_text and chosen_text == target_text and chosen_text in before_lower:
            redundant_field_penalty += 0.15
    selector_value = _extract_selector_value(chosen_action)
    generic_selector_penalty = 0.15 if selector_value in {"input", "button", "textarea", "select", "a", "div", "span"} else 0.0
    route_match, id_match = _action_matches_task_hints(chosen_action, task_prompt)
    action_type = str((chosen_action or {}).get("type") or "")
    primary_route = _extract_primary_route(task_prompt)
    previous_url_lower = str(previous_url or "").lower()
    current_url_lower = str(current_url or "").lower()
    task_hint_bonus = 0.0
    task_hint_penalty = 0.0
    if primary_route and action_type == "NavigateAction":
        action_url = str((chosen_action or {}).get("url") or "").strip().lower()
        if primary_route in previous_url_lower and action_url and primary_route not in action_url:
            off_route_penalty += 0.30
        elif action_url and primary_route in action_url:
            task_hint_bonus += 0.35
    on_primary_route = _is_on_primary_route(current_url, task_prompt)
    if route_match:
        task_hint_bonus += 0.20
    if id_match:
        task_hint_bonus += 0.10
    if action_type == "NavigateAction" and primary_route:
        action_url = str((chosen_action or {}).get("url") or "").strip().lower()
        if action_url and primary_route not in action_url:
            task_hint_penalty += 0.20
    if action_type == "NavigateAction" and not route_match and any(route in str(task_prompt or "").lower() for route in ("/contact", "/login", "/register", "/search", "/cart")):
        task_hint_penalty += 0.20
    if action_type in {"TypeAction", "ClickAction"} and selector_value:
        if selector_value in {"input", "button", "textarea", "select", "a", "div", "span"}:
            task_hint_penalty += 0.05
        elif not id_match:
            task_hint_penalty += 0.05
    if not on_primary_route and action_type == "TypeAction":
        task_hint_penalty += 0.15
    if not on_primary_route and action_type == "ClickAction" and not route_match and not id_match:
        task_hint_penalty += 0.10
    if not on_primary_route and action_type == "ScrollAction" and primary_route and step_index <= 1:
        task_hint_penalty += 0.35
    if not on_primary_route and primary_route:
        action_url = str((chosen_action or {}).get("url") or "").strip().lower()
        early_bonus_scale = 1.0 if step_index == 0 else 0.5 if step_index == 1 else 0.0
        if action_type == "NavigateAction" and action_url and primary_route in action_url:
            task_hint_bonus += 0.75 * early_bonus_scale
        if step_index <= 1 and action_type == "TypeAction":
            task_hint_penalty += 0.25
        if step_index <= 1 and action_type == "ScrollAction":
            task_hint_penalty += 0.35
        if step_index <= 1 and action_type == "ClickAction" and not route_match and not id_match:
            task_hint_penalty += 0.20
        if step_index <= 1 and not action_type:
            noop_penalty += 0.30
            task_hint_penalty += 0.20
    if step_index == 0 and action_type == "ScrollAction" and float(current_score) <= 0.0:
        noop_penalty += 0.35
        task_hint_penalty += 0.75
    total = (
        terminal_success
        + (0.5 * score_delta)
        + (0.5 * partial_progress)
        + field_progress_bonus
        + exact_target_bonus
        + field_alignment_bonus
        + submit_readiness_bonus
        + task_hint_bonus
        - step_cost
        - repeat_penalty
        - off_route_penalty
        - task_hint_penalty
        - generic_selector_penalty
        - redundant_field_penalty
        - execution_error_penalty
        - noop_penalty
    )
    return RewardBreakdown(
        total=float(total),
        terminal_success=float(terminal_success),
        score_delta=float(score_delta),
        partial_progress=float(partial_progress),
        field_progress_bonus=float(field_progress_bonus),
        exact_target_bonus=float(exact_target_bonus),
        field_alignment_bonus=float(field_alignment_bonus),
        submit_readiness_bonus=float(submit_readiness_bonus),
        task_hint_bonus=float(task_hint_bonus),
        step_cost=float(step_cost),
        repeat_penalty=float(repeat_penalty),
        off_route_penalty=float(off_route_penalty),
        task_hint_penalty=float(task_hint_penalty),
        generic_selector_penalty=float(generic_selector_penalty),
        redundant_field_penalty=float(redundant_field_penalty),
        execution_error_penalty=float(execution_error_penalty),
        noop_penalty=float(noop_penalty),
    )
