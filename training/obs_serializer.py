"""Serialize policy observations to compact text for SFT training.

Converts the rich observation dict from build_policy_obs() into a compact
text representation suitable for fine-tuning data (<2000 tokens target).
"""
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
        visible_text = page_obs.get("visible_text", "")
        if visible_text:
            # Compress visible text: keep first ~500 chars
            compressed = _compress_text(visible_text, max_chars=500)
            parts.append(f"Page: {compressed}")

        # Forms summary
        forms = page_obs.get("forms", [])
        if forms:
            form_summary = _summarize_forms(forms)
            if form_summary:
                parts.append(f"Forms: {form_summary}")
    elif isinstance(page_obs, str) and page_obs:
        parts.append(f"Page: {_compress_text(page_obs, max_chars=500)}")

    # Memory / facts
    memory = obs.get("memory", {})
    if isinstance(memory, dict):
        facts = memory.get("facts", [])
        if facts:
            facts_text = "; ".join(str(f) for f in facts[:5])
            parts.append(f"Known facts: {facts_text}")

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
    placeholder = cand.get("has_placeholder", False)

    extras: List[str] = []
    if role:
        extras.append(role)
    if href:
        extras.append(f"href={href}")
    if field_hint:
        extras.append(f"hint={field_hint}")
    if placeholder and not text:
        extras.append("placeholder")

    extra_str = f" ({', '.join(extras)})" if extras else ""
    text_str = f' "{text}"' if text else ""

    return f"[{index}] {ctype}{text_str}{extra_str}"
