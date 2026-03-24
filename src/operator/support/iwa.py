from __future__ import annotations

from typing import Any

try:
    import autoppia_iwa.src.execution.actions.actions  # noqa: F401
    from autoppia_iwa.src.data_generation.tasks.classes import Task
    from autoppia_iwa.src.execution.actions.base import BaseAction
    from autoppia_iwa.src.web_agents.act_protocol import (
        ACT_PROTOCOL_VERSION as IWA_ACT_PROTOCOL_VERSION,
    )
    from autoppia_iwa.src.web_agents.classes import IWebAgent

    AUTOPPIA_IWA_IMPORT_OK = True
    AUTOPPIA_IWA_IMPORT_ERROR = ""
except Exception:  # pragma: no cover
    IWebAgent = object  # type: ignore[assignment]
    Task = Any  # type: ignore[assignment]
    BaseAction = Any  # type: ignore[assignment]
    IWA_ACT_PROTOCOL_VERSION = "1.0"
    AUTOPPIA_IWA_IMPORT_OK = False
    AUTOPPIA_IWA_IMPORT_ERROR = "autoppia_iwa import failed in miner runtime"


__all__ = [
    "AUTOPPIA_IWA_IMPORT_ERROR",
    "AUTOPPIA_IWA_IMPORT_OK",
    "IWA_ACT_PROTOCOL_VERSION",
    "BaseAction",
    "IWebAgent",
    "Task",
]
