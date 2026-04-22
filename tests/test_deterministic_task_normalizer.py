from __future__ import annotations

import json
from pathlib import Path

from training.deterministic_harvester.normalizer import load_task_objective, normalize_task_row, task_seeds_for_use_case


def _task_row(*, use_case: str, prompt: str, url: str, event_criteria: dict, relevant_data: dict | None = None) -> dict:
    return {
        "id": f"{use_case.lower()}-task",
        "url": url,
        "prompt": prompt,
        "specifications": {},
        "tests": [
            {
                "type": "CheckEventTest",
                "event_name": use_case,
                "event_criteria": event_criteria,
            }
        ],
        "relevant_data": relevant_data or {},
        "use_case": {
            "name": use_case,
            "constraints": [],
        },
    }


def test_normalize_task_row_extracts_seed_filters_and_credentials() -> None:
    task_row = _task_row(
        use_case="ADD_TO_WATCHLIST",
        prompt="movie_name equals 'Dune'",
        url="http://example.test/movies?seed=17",
        event_criteria={"movie": {"name": "Dune"}},
        relevant_data={"user_for_login": {"username": "alice", "password": "secret123"}},
    )
    objective = normalize_task_row(task_row)
    assert objective.seed == 17
    assert objective.use_case == "ADD_TO_WATCHLIST"
    assert objective.field_values["username"] == "alice"
    assert objective.field_values["password"] == "secret123"
    assert objective.field_values["query"] == "Dune"
    assert objective.entity_filters["name_exact"] == "Dune"
    assert objective.auth_required is True


def test_load_task_objective_overrides_seed_in_url(tmp_path: Path) -> None:
    task_row = _task_row(
        use_case="CONTACT",
        prompt="subject equals 'Need help'",
        url="http://example.test/contact?seed=1",
        event_criteria={"subject": "Need help"},
    )
    cache_path = tmp_path / "tasks.json"
    cache_path.write_text(json.dumps({"tasks": [task_row]}), encoding="utf-8")
    objective = load_task_objective(cache_path=cache_path, use_case="CONTACT", seed=88)
    assert objective.seed == 88
    assert objective.task_url.endswith("seed=88")
    assert objective.route_target == "/contact"


def test_load_task_objective_prefers_matching_task_row_seed_from_url(tmp_path: Path) -> None:
    task_row_seed_5 = _task_row(
        use_case="CONTACT",
        prompt="subject equals 'Seed five'",
        url="http://example.test/contact?seed=5",
        event_criteria={"subject": "Seed five"},
    )
    task_row_seed_88 = _task_row(
        use_case="CONTACT",
        prompt="subject equals 'Seed eighty eight'",
        url="http://example.test/contact?seed=88",
        event_criteria={"subject": "Seed eighty eight"},
    )
    cache_path = tmp_path / "tasks.json"
    cache_path.write_text(json.dumps({"tasks": [task_row_seed_5, task_row_seed_88]}), encoding="utf-8")

    objective = load_task_objective(cache_path=cache_path, use_case="CONTACT", seed=88)

    assert objective.seed == 88
    assert objective.task_url.endswith("seed=88")
    assert objective.prompt == "subject equals 'Seed eighty eight'"
    assert objective.field_values["subject"] == "Seed eighty eight"


def test_load_task_objective_filters_by_project_id(tmp_path: Path) -> None:
    contact_autocinema = _task_row(
        use_case="CONTACT",
        prompt="subject equals 'Cinema'",
        url="http://localhost:3000/contact?seed=1",
        event_criteria={"subject": "Cinema"},
    )
    contact_autocinema["web_project_id"] = "autocinema"
    contact_autobooks = _task_row(
        use_case="CONTACT",
        prompt="subject equals 'Books'",
        url="http://localhost:3001/contact?seed=1",
        event_criteria={"subject": "Books"},
    )
    contact_autobooks["web_project_id"] = "autobooks"
    cache_path = tmp_path / "tasks.json"
    cache_path.write_text(json.dumps({"tasks": [contact_autocinema, contact_autobooks]}), encoding="utf-8")

    objective = load_task_objective(cache_path=cache_path, use_case="CONTACT", seed=11, web_project_id="autobooks")

    assert objective.web_project_id == "autobooks"
    assert objective.task_url.startswith("http://localhost:3001")


def test_load_task_objective_supports_nested_project_task_cache(tmp_path: Path) -> None:
    task_row = _task_row(
        use_case="ADD_TO_WATCHLIST",
        prompt="movie_name equals 'Dune'",
        url="http://localhost:3000/movies?seed=1",
        event_criteria={"movie": {"name": "Dune"}},
    )
    task_row["web_project_id"] = "autocinema"
    cache_path = tmp_path / "nested_tasks.json"
    cache_path.write_text(
        json.dumps(
            {
                "autocinema": {
                    "project_id": "autocinema",
                    "tasks": [task_row],
                }
            }
        ),
        encoding="utf-8",
    )

    objective = load_task_objective(
        cache_path=cache_path,
        use_case="ADD_TO_WATCHLIST",
        seed=23,
        web_project_id="autocinema",
    )

    assert objective.use_case == "ADD_TO_WATCHLIST"
    assert objective.seed == 23
    assert objective.task_url.endswith("seed=23")


def test_task_seeds_for_use_case_reads_seeds_from_task_urls(tmp_path: Path) -> None:
    cache_path = tmp_path / "tasks.json"
    cache_path.write_text(
        json.dumps(
            {
                "tasks": [
                    _task_row(
                        use_case="CONTACT",
                        prompt="subject equals 'First'",
                        url="http://example.test/contact?seed=12",
                        event_criteria={"subject": "First"},
                    ),
                    _task_row(
                        use_case="CONTACT",
                        prompt="subject equals 'Second'",
                        url="http://example.test/contact?seed=44",
                        event_criteria={"subject": "Second"},
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )

    assert task_seeds_for_use_case(cache_path=cache_path, use_case="CONTACT") == [12, 44]


def test_normalize_task_row_maps_film_detail_name_year_and_genre_list_into_entity_filters() -> None:
    task_row = _task_row(
        use_case="FILM_DETAIL",
        prompt="Navigate to a movie page where the genres is one of [Music, Mystery, Animation] and the year equals '1958' and the name contains 'go'",
        url="http://localhost:8000/?seed=277",
        event_criteria={
            "genres": {"operator": "in_list", "value": ["Music", "Mystery", "Animation"]},
            "year": 1958,
            "name": {"operator": "contains", "value": "go"},
        },
    )

    objective = normalize_task_row(task_row)

    assert objective.entity_filters["name_contains"] == "go"
    assert objective.entity_filters["genre_any_of"] == ["Music", "Mystery", "Animation"]
    assert objective.entity_filters["year_gte"] == 1958
    assert objective.entity_filters["year_lte"] == 1958


def test_normalize_task_row_resolves_login_placeholders_from_seed() -> None:
    task_row = _task_row(
        use_case="LOGIN",
        prompt="username equals <username> and password equals <password>",
        url="http://localhost:8000/login?seed=314",
        event_criteria={"username": "<username>", "password": "<password>"},
    )

    objective = normalize_task_row(task_row)

    assert objective.field_values["username"] == "user59"
    assert objective.field_values["password"] == "Passw0rd!"


def test_normalize_task_row_registration_uses_signup_defaults() -> None:
    task_row = _task_row(
        use_case="REGISTRATION",
        prompt="username equals <signup_username> and email equals <signup_email>",
        url="http://localhost:8000/register?seed=314",
        event_criteria={"username": "<signup_username>", "email": "<signup_email>", "password": "<signup_password>"},
    )

    objective = normalize_task_row(task_row)

    assert objective.field_values["username"] == "newuser59"
    assert objective.field_values["email"] == "newuser59@gmail.com"
    assert objective.field_values["password"] == "Passw0rd!"
    assert objective.field_values["confirm_password"] == "Passw0rd!"
