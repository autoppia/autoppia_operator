"""Serialize policy observations to compact text for SFT training."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def serialize_observation(obs: Dict[str, Any]) -> str:
    """Convert a policy observation dict to compact text.

    Args:
        obs: Observation dict with keys like prompt, url, page_observations,
             candidates, step_index, memory, etc.

    Returns:
        Compact text representation (<8000 chars / ~2000 tokens).
    """
    parts: List[str] = []

    # Task
    task = obs.get("prompt", "") or obs.get("task_text", "")
    if task:
        parts.append(f"Task: {task}")

    # URL
    url = obs.get("url", "")
    if url:
        parts.append(f"URL: {url}")

    # Step
    step_index = obs.get("step_index")
    if step_index is not None:
        parts.append(f"Step: {step_index}")

    # Page content
    page_obs = obs.get("page_observations", {})
    if isinstance(page_obs, dict):
        title = page_obs.get("title", "")
        if title:
            parts.append(f"Title: {str(title)[:200]}")

        headings = page_obs.get("headings", [])
        if headings:
            parts.append("Headings: " + " | ".join(str(item)[:120] for item in headings[:8]))

        visible_text = page_obs.get("visible_text", "")
        if visible_text:
            compressed = _compress_text(visible_text, max_chars=800)
            parts.append(f"Page: {compressed}")

        # Forms summary
        forms = page_obs.get("forms", [])
        if forms:
            form_summary = _summarize_forms(forms)
            if form_summary:
                parts.append(f"Forms: {form_summary}")

        page_facts = page_obs.get("page_facts", [])
        if page_facts:
            parts.append("Page facts: " + " | ".join(str(fact)[:140] for fact in page_facts[:10]))

        value_lines = page_obs.get("value_lines", [])
        if value_lines:
            parts.append("Visible values: " + " | ".join(str(item)[:140] for item in value_lines[:10]))
    elif isinstance(page_obs, str) and page_obs:
        parts.append(f"Page: {_compress_text(page_obs, max_chars=500)}")

    # Memory / facts
    memory = obs.get("memory", {})
    if isinstance(memory, dict):
        facts = memory.get("facts", [])
        if facts:
            facts_text = "; ".join(str(f) for f in facts[:5])
            parts.append(f"Known facts: {facts_text}")
        history_recent = memory.get("history_recent", [])
        if history_recent:
            parts.append("Recent history:")
            for item in history_recent[-4:]:
                if not isinstance(item, dict):
                    continue
                tool = str(item.get("tool") or "")
                url = str(item.get("url") or "")
                exec_ok = bool(item.get("exec_ok", True))
                parts.append(
                    f"- step {int(item.get('step_index') or 0)}: {tool} exec_ok={str(exec_ok).lower()} url={url[:120]}"
                )
        state_in = memory.get("state_in", {})
        if isinstance(state_in, dict) and state_in:
            parts.append("State in: " + _compact_json(state_in, max_chars=300))
        state_out = memory.get("state_out", {})
        if isinstance(state_out, dict) and state_out:
            parts.append("State out: " + _compact_json(state_out, max_chars=300))

    # Candidates
    candidates = obs.get("candidates", [])
    if candidates:
        parts.append("Candidates:")
        for i, cand in enumerate(candidates):
            line = _format_candidate(i, cand)
            parts.append(line)

    result = "\n".join(parts)

    # Final truncation safety: keep under 8000 chars
    if len(result) > 8000:
        result = result[:7950] + "\n[truncated]"

    return result


def _compress_text(text: str, max_chars: int = 500) -> str:
    """Compress visible text by removing redundant whitespace and truncating."""
    # Normalize whitespace
    lines = text.split("\n")
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)

    compressed = " | ".join(cleaned)

    if len(compressed) > max_chars:
        compressed = compressed[:max_chars] + "..."

    return compressed


def _summarize_forms(forms: List[Any]) -> str:
    """Produce a compact summary of form fields."""
    field_summaries: List[str] = []
    for form in forms:
        if not isinstance(form, dict):
            continue
        fields = form.get("fields", [])
        for field in fields:
            if not isinstance(field, dict):
                continue
            label = field.get("label", "") or field.get("name", "")
            ftype = field.get("type", "text")
            if label:
                field_summaries.append(f"{label}({ftype})")
    return ", ".join(field_summaries[:10]) if field_summaries else ""


def _format_candidate(index: int, cand: Dict[str, Any]) -> str:
    """Format a single candidate as a compact indexed line."""
    ctype = cand.get("type", cand.get("element_type", "?"))
    role = cand.get("role", "")
    text = str(cand.get("text", ""))[:100]
    href = cand.get("href", "")
    field_hint = cand.get("field_hint", "")
    field_kind = cand.get("field_kind", "")
    placeholder = cand.get("placeholder", "") or (("placeholder" if cand.get("has_placeholder", False) else ""))
    aria_label = cand.get("aria_label", "")
    name_attr = cand.get("name_attr", "")
    selector_summary = cand.get("selector_summary", "")
    current_value = cand.get("current_value", "")

    extras: List[str] = []
    if role:
        extras.append(role)
    if href:
        extras.append(f"href={href}")
    if field_hint:
        extras.append(f"hint={field_hint}")
    if field_kind:
        extras.append(f"kind={field_kind}")
    if placeholder:
        extras.append(f"placeholder={str(placeholder)[:40]}")
    if aria_label:
        extras.append(f"aria={str(aria_label)[:40]}")
    if name_attr:
        extras.append(f"name={str(name_attr)[:40]}")
    if selector_summary:
        extras.append(f"selector={str(selector_summary)[:80]}")
    if current_value:
        extras.append(f"value={str(current_value)[:40]}")

    extra_str = f" ({', '.join(extras)})" if extras else ""
    text_str = f' "{text}"' if text else ""

    return f"[{index}] {ctype}{text_str}{extra_str}"


def _compact_json(payload: Dict[str, Any], *, max_chars: int) -> str:
    try:
        text = str(payload)
    except Exception:
        text = repr(payload)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text
