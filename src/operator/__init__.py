from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ApifiedWebAgent", "AutoppiaOperator", "FSMAgent", "OPERATOR", "app", "fsm_operator"]


def __getattr__(name: str) -> Any:
    if name in {"ApifiedWebAgent", "AutoppiaOperator", "FSMAgent", "OPERATOR"}:
        module = import_module("src.operator.entrypoint")
        return getattr(module, name)
    if name == "app":
        module = import_module("src.operator.api.server")
        return getattr(module, name)
    if name == "fsm_operator":
        return import_module("src.operator.agents.fsm")
    raise AttributeError(f"module 'src.operator' has no attribute {name!r}")
