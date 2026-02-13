"""pricing.py

Token pricing helpers for local eval.

Source of truth (as of 2026-02-13):
- OpenAI pricing pages (openai.com/api/pricing, platform.openai.com/pricing)

Notes:
- This is only an estimate: it ignores cached-input discounts and any gateway-side
  adjustments, and assumes Chat Completions-style token accounting.
- We only model text input/output token prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float


# Keep this explicit; extend as needed.
# Prices are USD per 1M tokens.
_PRICES: dict[str, ModelPrice] = {
    # GPT-5.2 family
    'gpt-5.2': ModelPrice(input_per_1m=1.75, output_per_1m=14.00),
    'gpt-5.2-pro': ModelPrice(input_per_1m=21.00, output_per_1m=168.00),
    'gpt-5.2-codex': ModelPrice(input_per_1m=1.75, output_per_1m=14.00),
    'gpt-5.2-chat-latest': ModelPrice(input_per_1m=1.75, output_per_1m=14.00),

    # GPT-5 / GPT-5.1 family
    'gpt-5.1': ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    'gpt-5': ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    'gpt-5-mini': ModelPrice(input_per_1m=0.25, output_per_1m=2.00),
    'gpt-5-nano': ModelPrice(input_per_1m=0.05, output_per_1m=0.40),
    'gpt-5-pro': ModelPrice(input_per_1m=15.00, output_per_1m=120.00),
    'gpt-5.1-chat-latest': ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    'gpt-5-chat-latest': ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    'gpt-5.1-codex': ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    'gpt-5.1-codex-max': ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    'gpt-5-codex': ModelPrice(input_per_1m=1.25, output_per_1m=10.00),

    # GPT-4o family
    'gpt-4o': ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    'gpt-4o-mini': ModelPrice(input_per_1m=0.15, output_per_1m=0.60),

    # GPT-4.1 family
    'gpt-4.1': ModelPrice(input_per_1m=2.00, output_per_1m=8.00),
    'gpt-4.1-mini': ModelPrice(input_per_1m=0.40, output_per_1m=1.60),
    'gpt-4.1-nano': ModelPrice(input_per_1m=0.10, output_per_1m=0.40),

    # o-series
    'o4-mini': ModelPrice(input_per_1m=1.10, output_per_1m=4.40),
    'o3': ModelPrice(input_per_1m=2.00, output_per_1m=8.00),
    'o3-pro': ModelPrice(input_per_1m=20.00, output_per_1m=80.00),
    'o1': ModelPrice(input_per_1m=15.00, output_per_1m=60.00),
    'o1-pro': ModelPrice(input_per_1m=150.00, output_per_1m=600.00),
}


def _normalize_model(model: str) -> str:
    m = (model or '').strip().lower()
    # Strip snapshot suffix if present, keep the base alias.
    for base in sorted(_PRICES.keys(), key=len, reverse=True):
        if m == base or m.startswith(base + '-'):  # e.g. gpt-5.2-2025-12-11
            return base
    return m


def price_for_model(model: str) -> Optional[ModelPrice]:
    return _PRICES.get(_normalize_model(model))


def estimate_cost_usd(model: str, usage: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Estimate USD cost from an OpenAI-like usage dict.

    usage should have prompt_tokens and completion_tokens (ints).
    """
    p = price_for_model(model)
    pt = int(usage.get('prompt_tokens') or 0)
    ct = int(usage.get('completion_tokens') or 0)

    if not p:
        return 0.0, {
            'error': 'unknown_model',
            'model': str(model),
            'prompt_tokens': pt,
            'completion_tokens': ct,
        }

    cost = (pt / 1_000_000.0) * p.input_per_1m + (ct / 1_000_000.0) * p.output_per_1m
    return float(cost), {
        'model': _normalize_model(model),
        'prompt_tokens': pt,
        'completion_tokens': ct,
        'input_per_1m': p.input_per_1m,
        'output_per_1m': p.output_per_1m,
    }
