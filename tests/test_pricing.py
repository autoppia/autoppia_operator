import math

from pricing import estimate_cost_usd, price_for_model, _normalize_model


def test_normalize_model_strips_known_suffixes() -> None:
    """Normalize model strips known suffixes."""
    assert _normalize_model("  Claude-Sonnet-4-20250514  ") == "claude-sonnet-4-20250514"
    assert _normalize_model("claude-sonnet-4-20250514-extra") == "claude-sonnet-4-20250514"
    assert _normalize_model("gpt-5.2-preview") == "gpt-5.2"


def test_price_for_model_returns_price_for_known_alias() -> None:
    """Price for model returns price for known alias."""
    price = price_for_model("claude-sonnet-4-5-20250929-extra")

    assert price is not None
    assert price.input_per_1m == 3.0
    assert price.output_per_1m == 15.0


def test_price_for_model_returns_none_for_unknown_model() -> None:
    """Price for model returns none for unknown model."""
    assert price_for_model("unknown-model") is None


def test_estimate_cost_usd_for_known_model() -> None:
    """Estimate cost usd for known model."""
    cost, meta = estimate_cost_usd(
        "gpt-4o-mini",
        {"prompt_tokens": 2_000, "completion_tokens": 500},
    )

    assert math.isclose(cost, 0.0006, rel_tol=1e-9)
    assert meta == {
        "model": "gpt-4o-mini",
        "prompt_tokens": 2000,
        "completion_tokens": 500,
        "input_per_1m": 0.15,
        "output_per_1m": 0.6,
    }


def test_estimate_cost_usd_for_unknown_model_returns_error_payload() -> None:
    """Estimate cost usd for unknown model returns error payload."""
    cost, meta = estimate_cost_usd("custom-model", {"prompt_tokens": "10", "completion_tokens": None})

    assert cost == 0.0
    assert meta == {
        "error": "unknown_model",
        "model": "custom-model",
        "prompt_tokens": 10,
        "completion_tokens": 0,
    }
