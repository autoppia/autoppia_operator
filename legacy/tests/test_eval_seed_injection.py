from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "eval.py"
    spec = importlib.util.spec_from_file_location("autoppia_operator_eval_seed", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inject_seed_preserves_existing_url_seed_when_no_override() -> None:
    eval_mod = _load_eval_module()
    task = SimpleNamespace(url="http://84.247.180.192:8000/contact?seed=7")
    updated, seed_used = eval_mod.inject_seed(task)
    assert seed_used == 7
    assert updated.url.endswith("?seed=7")


def test_inject_seed_override_wins_over_existing_url_seed() -> None:
    eval_mod = _load_eval_module()
    task = SimpleNamespace(url="http://84.247.180.192:8000/contact?seed=7")
    updated, seed_used = eval_mod.inject_seed(task, seed=11)
    assert seed_used == 11
    assert updated.url.endswith("?seed=11")
