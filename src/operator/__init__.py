from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["OPERATOR", "ApifiedWebAgent", "AutoppiaOperator", "FSMAgent", "StepAgent", "app", "fsm_operator", "step_engine"]


def __getattr__(name: str) -> Any:
    if name in {"ApifiedWebAgent", "AutoppiaOperator", "FSMAgent", "StepAgent", "OPERATOR"}:
        module = import_module("src.operator.entrypoint")
        return getattr(module, name)
    if name == "app":
        module = import_module("src.operator.api.server")
        return getattr(module, name)
    if name == "fsm_operator":
        return import_module("src.operator.agents.fsm")
    if name == "step_engine":
        return import_module("src.operator.agents.step_engine")
    raise AttributeError(f"module 'src.operator' has no attribute {name!r}")
