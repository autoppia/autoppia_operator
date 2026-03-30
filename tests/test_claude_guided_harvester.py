from __future__ import annotations

from training.claude_guided_harvester import (
    _expand_id_variants,
    _guided_actions_from_brief,
    _ordered_selector_candidates,
    _success_signal_hit,
)


def test_guided_actions_from_brief_builds_seeded_contact_flow() -> None:
    brief = {
        "route": ["/contact"],
        "fields": [
            {"name": "name", "ids": ["contact-name-input"], "value": "David", "value_rule": "exact"},
            {"name": "email", "ids": ["contact-email-input"], "value": "user1@site.com", "value_rule": "exact"},
        ],
        "submit": {"ids": ["send-message-button"], "text": ["Send Message"], "action": "click once"},
    }
    actions = _guided_actions_from_brief(task_url="http://84.247.180.192:8000/?seed=1", brief=brief)

    assert actions[0]["type"] == "NavigateAction"
    assert actions[0]["url"] == "http://84.247.180.192:8000/contact?seed=1"
    assert actions[1]["type"] == "TypeAction"
    assert actions[1]["text"] == "David"
    assert actions[1]["selector_candidates"][0]["value"] == "contact-name-input"
    assert actions[-1]["type"] == "ClickAction"
    assert actions[-1]["selector_candidates"][0]["value"] == "send-message-button"


def test_success_signal_hit_detects_dom_text() -> None:
    brief = {"success_signals": {"texts": ["Message Sent!"], "ids": [], "url_contains": []}}
    assert _success_signal_hit(html="<div>Message Sent!</div>", url="http://example.com/contact", brief=brief)


def test_expand_id_variants_includes_dynamic_variants() -> None:
    variants = _expand_id_variants(["contact-name-input"])
    assert "contact-name-input" in variants
    assert any(item != "contact-name-input" for item in variants)


def test_guided_actions_from_brief_normalizes_route_and_value_rules() -> None:
    brief = {
        "route": ["Navigate to /contact"],
        "fields": [
            {"name": "name", "ids": ["contact-name-input"], "value_rule": "Enter exactly 'David'"},
            {"name": "email", "ids": ["contact-email-input"], "value_rule": "Enter 'user1@site.com' (must contain this string)"},
            {"name": "subject", "ids": ["contact-subject-input"], "value_rule": "Enter a subject that does NOT contain 'Information', e.g. 'General inquiry' or 'Question'"},
            {"name": "message", "ids": ["contact-message-textarea"], "value_rule": "Enter exactly 'Please provide me with more information'"},
        ],
        "submit": {"ids": ["send-message-button"], "text": ["Send Message"]},
    }
    actions = _guided_actions_from_brief(task_url="http://84.247.180.192:8000/?seed=51", brief=brief)

    assert actions[0]["url"] == "http://84.247.180.192:8000/contact?seed=51"
    assert actions[1]["text"] == "David"
    assert actions[2]["text"] == "user1@site.com"
    assert actions[3]["text"] == "General inquiry"
    assert actions[4]["text"] == "Please provide me with more information"


def test_guided_actions_from_brief_normalizes_explicit_step_navigation_with_seed() -> None:
    brief = {
        "steps": [
            {"type": "NavigateAction", "url": "/contact"},
            {"type": "TypeAction", "ids": ["contact-name-input"], "text": "David"},
        ]
    }
    actions = _guided_actions_from_brief(task_url="http://84.247.180.192:8000/?seed=9", brief=brief)
    assert actions[0]["type"] == "NavigateAction"
    assert actions[0]["url"] == "http://84.247.180.192:8000/contact?seed=9"


def test_ordered_selector_candidates_prefers_explicit_candidate_ids() -> None:
    planned_action = {
        "type": "TypeAction",
        "selector_candidates": [
            {"type": "attributeValueSelector", "attribute": "id", "value": "contact-name-input", "case_sensitive": False},
            {"type": "attributeValueSelector", "attribute": "id", "value": "name-field", "case_sensitive": False},
        ],
    }
    resolved_candidates = [
        {"type": "attributeValueSelector", "attribute": "id", "value": "input", "case_sensitive": False},
        {"type": "attributeValueSelector", "attribute": "id", "value": "contact-name-input", "case_sensitive": False},
    ]

    ordered = _ordered_selector_candidates(
        planned_action,
        resolved_candidates,
        [{"type": "attributeValueSelector", "attribute": "id", "value": "contact-name-input", "case_sensitive": False}],
        exact_match_found=True,
    )

    assert ordered[0]["value"] == "contact-name-input"
    assert ordered[1]["value"] == "input"
    assert len(ordered) == 2


def test_ordered_selector_candidates_prefers_dom_heuristic_when_no_exact_match_found() -> None:
    planned_action = {
        "type": "TypeAction",
        "selector_candidates": [
            {"type": "attributeValueSelector", "attribute": "id", "value": "contact-name-input", "case_sensitive": False},
            {"type": "attributeValueSelector", "attribute": "id", "value": "name-field", "case_sensitive": False},
        ],
    }
    resolved_candidates = [
        {"type": "attributeValueSelector", "attribute": "id", "value": "actual-seed-name-id", "case_sensitive": False},
    ]

    ordered = _ordered_selector_candidates(planned_action, resolved_candidates, [], exact_match_found=False)

    assert [item["value"] for item in ordered] == ["actual-seed-name-id"]


def test_ordered_selector_candidates_prefers_existing_exact_dom_ids_before_other_explicit_variants() -> None:
    planned_action = {
        "type": "TypeAction",
        "selector_candidates": [
            {"type": "attributeValueSelector", "attribute": "id", "value": "contact-name-input", "case_sensitive": False},
            {"type": "attributeValueSelector", "attribute": "id", "value": "name-field", "case_sensitive": False},
            {"type": "attributeValueSelector", "attribute": "id", "value": "name-input-field", "case_sensitive": False},
        ],
    }
    resolved_candidates = [
        {"type": "attributeValueSelector", "attribute": "id", "value": "name-input-field", "case_sensitive": False},
    ]
    existing_exact_candidates = [
        {"type": "attributeValueSelector", "attribute": "id", "value": "name-input-field", "case_sensitive": False},
    ]

    ordered = _ordered_selector_candidates(
        planned_action,
        resolved_candidates,
        existing_exact_candidates,
        exact_match_found=True,
    )

    assert [item["value"] for item in ordered] == ["name-input-field"]


def test_ordered_selector_candidates_uses_only_existing_exact_and_dom_resolved_candidates() -> None:
    planned_action = {
        "type": "ClickAction",
        "selector_candidates": [
            {"type": "attributeValueSelector", "attribute": "id", "value": "send-message-button", "case_sensitive": False},
            {"type": "attributeValueSelector", "attribute": "id", "value": "send-button", "case_sensitive": False},
            {"type": "attributeValueSelector", "attribute": "id", "value": "submit-btn", "case_sensitive": False},
        ],
    }
    resolved_candidates = [
        {"type": "attributeValueSelector", "attribute": "id", "value": "send-button", "case_sensitive": False},
    ]
    existing_exact_candidates = [
        {"type": "attributeValueSelector", "attribute": "id", "value": "send-button", "case_sensitive": False},
    ]

    ordered = _ordered_selector_candidates(
        planned_action,
        resolved_candidates,
        existing_exact_candidates,
        exact_match_found=True,
    )

    assert [item["value"] for item in ordered] == ["send-button"]
