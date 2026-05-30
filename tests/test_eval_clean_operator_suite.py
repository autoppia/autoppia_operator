from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'eval' / 'eval_clean_operator_suite.py'
    spec = importlib.util.spec_from_file_location('eval_clean_operator_suite', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_seed_spec_range_and_csv() -> None:
    module = _load_module()
    assert module.parse_seed_spec('1-3,7,9-8') == [1, 2, 3, 7, 9, 8]


def test_parse_seed_spec_empty_defaults_to_one() -> None:
    module = _load_module()
    assert module.parse_seed_spec('') == [1]
