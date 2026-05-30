from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace

import training.claude_guided_harvester as guided_module
import training.deterministic_harvester.resolvers as resolver_module
from training.claude_guided_harvester import (
    _dataset_movie_candidates,
    _execute_action_candidates,
    _expand_id_variants,
    _guided_actions_from_brief,
    _guided_web_agent_id,
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


def test_guided_actions_from_brief_does_not_emit_custom_movie_detail_action(monkeypatch) -> None:
    monkeypatch.setattr(
        guided_module,
        "resolve_movie_detail_url",
        lambda **kwargs: "http://localhost:3000/movies/movie-7?seed=9",
    )
    brief = {
        "discover_target": {
            "kind": "movie_detail",
            "filters": {"name_exact": "Dune"},
        }
    }
    actions = _guided_actions_from_brief(task_url="http://localhost:3000/?seed=9", brief=brief, web_project_id="autocinema")
    assert actions == [
        {
            "type": "NavigateAction",
            "url": "http://localhost:3000/movies/movie-7?seed=9",
            "go_back": False,
            "go_forward": False,
        }
    ]


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

    assert [item["value"] for item in ordered] == ["contact-name-input", "name-field", "input"]


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

    assert [item["value"] for item in ordered] == ["contact-name-input", "name-field", "actual-seed-name-id"]


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

    assert [item["value"] for item in ordered] == ["name-input-field", "contact-name-input", "name-field"]


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

    assert [item["value"] for item in ordered] == ["send-button", "send-message-button", "submit-btn"]


def test_dataset_movie_candidates_filters_seeded_movies(monkeypatch) -> None:
    payload = {
        "data": [
            {
                "id": "movie-1",
                "title": "Dune",
                "director": "Denis Villeneuve",
                "genres": ["Sci-Fi", "Drama"],
                "duration": 155,
                "rating": 8.2,
                "year": 2021,
            },
            {
                "id": "movie-2",
                "title": "Old Comedy",
                "director": "Someone Else",
                "genres": ["Comedy"],
                "duration": 90,
                "rating": 6.0,
                "year": 2001,
            },
        ]
    }

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def fake_urlopen(url, timeout):
        assert "seed_value=77" in url
        assert "project_key=web_2_autobooks" in url
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(resolver_module.urllib.request, "urlopen", fake_urlopen)

    candidates = _dataset_movie_candidates(
        task_url="http://example.test/search?seed=77",
        filters={"name_exact": "dune", "genre_contains": "sci", "rating_gte": 8.0, "year_gte": 2020},
        web_project_id="autobooks",
    )

    assert candidates == ["/movies/movie-1"]


def test_dataset_movie_candidates_supports_name_contains_genre_list_and_year_equals(monkeypatch) -> None:
    payload = {
        "data": [
            {
                "id": "real-movie-033",
                "title": "Go Go Melody",
                "director": "Vincente Minnelli",
                "genres": ["Music"],
                "duration": 115,
                "rating": 6.6,
                "year": 1958,
            },
            {
                "id": "real-movie-097",
                "title": "The Bridge on the River Kwai",
                "director": "David Lean",
                "genres": ["Adventure", "Drama", "War"],
                "duration": 161,
                "rating": 8.1,
                "year": 1957,
            },
        ]
    }

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    monkeypatch.setattr(
        resolver_module.urllib.request,
        "urlopen",
        lambda url, timeout: _FakeResponse(json.dumps(payload).encode("utf-8")),
    )

    candidates = _dataset_movie_candidates(
        task_url="http://localhost:8000/?seed=277",
        filters={
            "name_contains": "go",
            "genre_any_of": ["Music", "Mystery", "Animation"],
            "year_gte": 1958,
            "year_lte": 1958,
        },
        web_project_id="autocinema",
    )

    assert candidates == ["/movies/real-movie-033"]


def test_task_for_seed_supports_nested_project_task_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "nested_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "autocinema": {
                    "project_id": "autocinema",
                    "tasks": [
                        {
                            "id": "task-1",
                            "is_web_real": False,
                            "web_project_id": "autocinema",
                            "url": "http://localhost:8000/?seed=17",
                            "prompt": "Add a movie to the watchlist.",
                            "specifications": {},
                            "tests": [],
                            "use_case": {
                                "name": "ADD_TO_WATCHLIST",
                                "description": "Add a movie to the watchlist.",
                                "event": "AddToWatchlistEvent",
                                "event_source_code": True,
                                "examples": [],
                                "constraints": [],
                                "additional_prompt_info": "",
                            },
                            "should_record": False,
                            "original_prompt": "Add a movie to the watchlist.",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    task = guided_module._task_for_seed(
        use_case="ADD_TO_WATCHLIST",
        seed=9,
        task_cache=cache_path,
        web_project_id="autocinema",
    )

    assert task.url.endswith("?seed=9")
    use_case_payload = task.use_case if isinstance(task.use_case, dict) else {}
    assert use_case_payload.get("name") == "ADD_TO_WATCHLIST"


def test_task_for_seed_prefers_matching_task_url_seed(tmp_path: Path) -> None:
    cache_path = tmp_path / "tasks.json"
    cache_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "task-1",
                        "is_web_real": False,
                        "web_project_id": "autocinema",
                        "url": "http://localhost:8000/?seed=17",
                        "prompt": "First prompt",
                        "specifications": {},
                        "tests": [],
                        "use_case": {
                            "name": "ADD_TO_WATCHLIST",
                            "description": "Add a movie to the watchlist.",
                            "event": "AddToWatchlistEvent",
                            "event_source_code": True,
                            "examples": [],
                            "constraints": [],
                            "additional_prompt_info": "",
                        },
                        "should_record": False,
                        "original_prompt": "First prompt",
                    },
                    {
                        "id": "task-2",
                        "is_web_real": False,
                        "web_project_id": "autocinema",
                        "url": "http://localhost:8000/?seed=29",
                        "prompt": "Second prompt",
                        "specifications": {},
                        "tests": [],
                        "use_case": {
                            "name": "ADD_TO_WATCHLIST",
                            "description": "Add a movie to the watchlist.",
                            "event": "AddToWatchlistEvent",
                            "event_source_code": True,
                            "examples": [],
                            "constraints": [],
                            "additional_prompt_info": "",
                        },
                        "should_record": False,
                        "original_prompt": "Second prompt",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    task = guided_module._task_for_seed(
        use_case="ADD_TO_WATCHLIST",
        seed=29,
        task_cache=cache_path,
        web_project_id="autocinema",
    )

    assert task.url.endswith("?seed=29")
    assert task.prompt == "Second prompt"


def test_guided_web_agent_id_stays_in_expected_user_range() -> None:
    assert _guided_web_agent_id(1) == "1"
    assert _guided_web_agent_id(255) == "255"
    assert _guided_web_agent_id(256) == "1"
    assert _guided_web_agent_id(31000).isdigit()
    assert 1 <= int(_guided_web_agent_id(31000)) <= 255


def test_run_guided_brief_reports_seed_from_task_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        guided_module,
        "_task_for_seed",
        lambda **kwargs: SimpleNamespace(id="task-1", url="http://localhost:8000/?seed=418", web_project_id="autocinema"),
    )
    monkeypatch.setattr(guided_module, "_guided_actions_from_brief", lambda **kwargs: [])

    class FakeScore:
        success = True
        raw_score = 1.0

    class FakeSnapshot:
        url = "http://localhost:8000/profile?seed=418"
        html = "<html></html>"

    class FakeStepResult:
        score = FakeScore()
        snapshot = FakeSnapshot()

    class FakeSession:
        async def reset(self):
            return FakeStepResult()

        async def close(self):
            return None

    monkeypatch.setattr(guided_module, "build_task_execution_session", lambda **kwargs: FakeSession(), raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.operator.eval.session":

            class _Module:
                @staticmethod
                def build_task_execution_session(**kwargs):
                    return FakeSession()

            return _Module()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    report = guided_module.run_guided_brief(
        use_case="CONTACT",
        seed=96,
        brief_payload={"brief": {}},
        task_cache=tmp_path / "tasks.json",
        max_steps=1,
        headless=True,
    )

    assert report["episodes"][0]["seed"] == 418
    assert report["episodes"][0]["episode_task_id"].endswith("-418")


def test_execute_action_candidates_uses_dom_custom_selector_when_no_explicit_selector_candidates(monkeypatch) -> None:
    class FakePage:
        async def evaluate(self, script, payload):
            assert payload["type"] == "TypeAction"
            assert payload["field_name"] == "genres"
            return {
                "existing_exact": [],
                "heuristic": {
                    "type": "attributeValueSelector",
                    "attribute": "custom",
                    "value": "div:nth-of-type(3) > input:nth-of-type(1)",
                    "case_sensitive": False,
                },
            }

    class FakeSession:
        def __init__(self):
            self.page = FakePage()
            self.actions = []

        async def step(self, action):
            self.actions.append(action)
            return SimpleNamespace(
                action_result=SimpleNamespace(successfully_executed=True, error=""),
            )

    created_payloads = []

    def fake_create_action(payload):
        created_payloads.append(dict(payload))
        return payload

    monkeypatch.setattr(guided_module.BaseAction, "create_action", staticmethod(fake_create_action))

    session = FakeSession()
    planned_action = {"type": "TypeAction", "text": "Crime", "field_name": "genres"}

    result, execution = asyncio.run(_execute_action_candidates(session, planned_action))

    assert result.action_result.successfully_executed is True
    assert created_payloads == [
        {
            "type": "TypeAction",
            "selector": {
                "type": "attributeValueSelector",
                "attribute": "custom",
                "value": "div:nth-of-type(3) > input:nth-of-type(1)",
                "case_sensitive": False,
            },
            "text": "Crime",
        }
    ]
    assert execution["attempts"][0]["success"] is True


def test_sanitize_task_row_for_replay_aligns_login_criteria_with_seed_identity() -> None:
    row = {
        "url": "http://localhost:3000/login?seed=314",
        "use_case": {"name": "LOGIN"},
        "tests": [
            {
                "event_name": "LOGIN",
                "event_criteria": {"username": "<username>", "password": "<password>"},
            }
        ],
    }

    sanitized = guided_module._sanitize_task_row_for_replay(row)

    criteria = sanitized["tests"][0]["event_criteria"]
    assert criteria["username"] == "user59"
    assert "password" not in criteria


def test_sanitize_task_row_for_replay_strips_logout_password() -> None:
    row = {
        "url": "http://localhost:3000/profile?seed=314",
        "use_case": {"name": "LOGOUT"},
        "tests": [
            {
                "event_name": "LOGOUT",
                "event_criteria": {"username": "<username>", "password": "<password>"},
            }
        ],
    }

    sanitized = guided_module._sanitize_task_row_for_replay(row)

    criteria = sanitized["tests"][0]["event_criteria"]
    assert criteria["username"] == "user59"
    assert "password" not in criteria


def test_sanitize_task_row_for_replay_drops_auth_keys_for_non_auth_events() -> None:
    row = {
        "url": "http://localhost:3000/profile?seed=314",
        "use_case": {"name": "ADD_FILM"},
        "tests": [
            {
                "event_name": "ADD_FILM",
                "event_criteria": {"username": "<username>", "password": "<password>", "name": "New Film"},
            }
        ],
    }

    sanitized = guided_module._sanitize_task_row_for_replay(row)

    criteria = sanitized["tests"][0]["event_criteria"]
    assert "username" not in criteria
    assert "password" not in criteria
    assert criteria["name"] == "New Film"
