from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from training.deterministic_harvester.normalizer import DeterministicTaskObjective
from training.deterministic_harvester.projects import seeded_url


@dataclass(frozen=True)
class DeterministicPlan:
    prompt_lines: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def navigate(route: str, *, objective: DeterministicTaskObjective) -> dict[str, Any]:
    return {
        "type": "NavigateAction",
        "url": seeded_url(
            task_url=objective.task_url,
            project_id=objective.web_project_id,
            route=route,
            seed=objective.seed,
        ),
        "go_back": False,
        "go_forward": False,
    }


def type_text(text: str, *, selectors: list[dict[str, Any]] | None = None, field_name: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "TypeAction", "text": str(text), "field_name": str(field_name or "").strip()}
    if selectors:
        payload["selector_candidates"] = selectors
    return payload


def click(*, selectors: list[dict[str, Any]] | None = None, field_name: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "ClickAction", "field_name": str(field_name or "").strip()}
    if selectors:
        payload["selector_candidates"] = selectors
    return payload


def select(value: str, *, selectors: list[dict[str, Any]] | None = None, field_name: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "SelectAction", "value": str(value), "field_name": str(field_name or "").strip()}
    if selectors:
        payload["selector_candidates"] = selectors
    return payload


def custom_selector(value: str) -> dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": "custom",
        "value": str(value),
        "case_sensitive": False,
    }


def hash_string(value: str) -> int:
    hash_value = 0
    for char in str(value or ""):
        hash_value = ((hash_value << 5) - hash_value) + ord(char)
        hash_value &= 0xFFFFFFFF
        if hash_value >= 0x80000000:
            hash_value -= 0x100000000
    return abs(hash_value)


def select_variant_index(seed: int, key: str, count: int) -> int:
    if int(count) <= 1:
        return 0
    return abs(hash_string(f"{str(key or '').strip()}:{int(seed)}")) % int(count)


def generate_hash_order(seed: int, key: str, count: int) -> list[int]:
    combined = f"{key}:{seed}"
    hash_value = 0
    for char in combined:
        hash_value = ((hash_value << 5) - hash_value) + ord(char)
        hash_value &= 0xFFFFFFFF
        if hash_value >= 0x80000000:
            hash_value -= 0x100000000
    order = list(range(count))
    for idx in range(count - 1, 0, -1):
        swap_idx = abs(hash_value + idx * 7919) % (idx + 1)
        order[idx], order[swap_idx] = order[swap_idx], order[idx]
    return order


def generate_dynamic_order(seed: int, key: str, count: int) -> list[int]:
    if count <= 1:
        return [0]
    original = list(range(count))
    if int(seed) == 1:
        return original
    variants: list[list[int]] = []
    for offset in range(count):
        variants.append([(index + offset) % count for index in range(count)])
    for index in range(count - 1):
        swapped = [idx for idx in range(count)]
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        if swapped != original:
            variants.append(swapped)
    for split in range(1, count):
        reversed_part = [split - 1 - idx for idx in range(split)] + [split + idx for idx in range(count - split)]
        if reversed_part != original:
            variants.append(reversed_part)
    hash_variant = generate_hash_order(seed, key, count)
    if hash_variant != original:
        variants.append(hash_variant)
    deduped: list[list[int]] = []
    seen: set[str] = set()
    for variant in variants:
        key_value = ",".join(str(item) for item in variant)
        if key_value in seen:
            continue
        seen.add(key_value)
        deduped.append(variant)
    if not deduped:
        return original
    variant_idx = select_variant_index(seed, key, len(deduped))
    return deduped[variant_idx]


def extract_path(route_or_url: str) -> str:
    raw = str(route_or_url or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return str(urlparse(raw).path or "").strip()
    return raw.split("?", 1)[0].strip()


__all__ = [
    "DeterministicPlan",
    "click",
    "custom_selector",
    "extract_path",
    "generate_dynamic_order",
    "generate_hash_order",
    "hash_string",
    "navigate",
    "select",
    "select_variant_index",
    "type_text",
]
