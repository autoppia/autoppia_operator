from __future__ import annotations

import json
from pathlib import Path

from training.deterministic_harvester.normalizer import load_task_objective, normalize_task_row


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
