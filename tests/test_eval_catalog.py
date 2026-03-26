from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "eval.py"
    spec = importlib.util.spec_from_file_location("autoppia_operator_eval", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_task_catalog_groups_projects_and_use_cases(tmp_path: Path) -> None:
    eval_mod = _load_eval_module()
    cache = tmp_path / "tasks.json"
    cache.write_text(
        json.dumps(
            {
                "tasks": [
                    {"id": "1", "web_project_id": "proj_a", "use_case": {"name": "LOGIN"}, "prompt": "p1", "url": "https://a"},
                    {"id": "2", "web_project_id": "proj_a", "use_case": {"name": "LOGIN"}, "prompt": "p2", "url": "https://a"},
                    {"id": "3", "web_project_id": "proj_a", "use_case": {"name": "SEARCH"}, "prompt": "p3", "url": "https://a"},
                    {"id": "4", "web_project_id": "proj_b", "use_case": {"name": "BUY"}, "prompt": "p4", "url": "https://b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    catalog = eval_mod.build_task_catalog(cache)
    assert set(catalog.keys()) == {"proj_a", "proj_b"}
    assert catalog["proj_a"]["count"] == 3
    assert catalog["proj_a"]["use_cases"]["LOGIN"] == 2
    assert catalog["proj_a"]["use_cases"]["SEARCH"] == 1
    assert catalog["proj_b"]["use_cases"]["BUY"] == 1


def test_select_all_use_case_tasks_returns_one_per_use_case(tmp_path: Path) -> None:
    eval_mod = _load_eval_module()
    raw_tasks = [
        {"id": "1", "web_project_id": "proj_a", "use_case": {"name": "LOGIN"}, "prompt": "p1", "url": "https://a"},
        {"id": "2", "web_project_id": "proj_a", "use_case": {"name": "LOGIN"}, "prompt": "p2", "url": "https://a"},
        {"id": "3", "web_project_id": "proj_a", "use_case": {"name": "SEARCH"}, "prompt": "p3", "url": "https://a"},
        {"id": "4", "web_project_id": "proj_b", "use_case": {"name": "BUY"}, "prompt": "p4", "url": "https://b"},
    ]
    selected = eval_mod.select_all_use_case_tasks(raw_tasks, web_project_id="proj_a", tasks_per_use_case=1, seed=7)
    selected_ids = {str(t.id) for t in selected}
    assert len(selected) == 2
    assert selected_ids <= {"1", "2", "3"}
