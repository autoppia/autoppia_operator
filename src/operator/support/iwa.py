from __future__ import annotations

from typing import Any

try:
    from autoppia_iwa.src.data_generation.tasks.classes import Task
    from autoppia_iwa.src.execution.actions.base import BaseAction
    from autoppia_iwa.src.web_agents.act_protocol import ActAllowedTool
    from autoppia_iwa.src.web_agents.act_protocol import ACT_PROTOCOL_VERSION as IWA_ACT_PROTOCOL_VERSION
    from autoppia_iwa.src.web_agents.act_protocol import ActRequest
    from autoppia_iwa.src.web_agents.act_protocol import ActResponse
    from autoppia_iwa.src.web_agents.act_protocol import ActToolCall
    from autoppia_iwa.src.web_agents.classes import IWebAgent
    from autoppia_iwa.src.web_agents.protocol import STEP_PROTOCOL_VERSION
    from autoppia_iwa.src.web_agents.protocol import StepAllowedTool
    from autoppia_iwa.src.web_agents.protocol import StepRequest
    from autoppia_iwa.src.web_agents.protocol import StepResponse
    from autoppia_iwa.src.web_agents.protocol import StepToolCall
    import autoppia_iwa.src.execution.actions.actions  # noqa: F401

    AUTOPPIA_IWA_IMPORT_OK = True
    AUTOPPIA_IWA_IMPORT_ERROR = ""
except Exception:  # pragma: no cover
    IWebAgent = object  # type: ignore[assignment]
    Task = Any  # type: ignore[assignment]
    BaseAction = Any  # type: ignore[assignment]
    StepRequest = Any  # type: ignore[assignment]
    StepResponse = Any  # type: ignore[assignment]
    StepToolCall = Any  # type: ignore[assignment]
    StepAllowedTool = Any  # type: ignore[assignment]
    ActRequest = Any  # type: ignore[assignment]
    ActResponse = Any  # type: ignore[assignment]
    ActToolCall = Any  # type: ignore[assignment]
    ActAllowedTool = Any  # type: ignore[assignment]
    STEP_PROTOCOL_VERSION = "1.0"
    IWA_ACT_PROTOCOL_VERSION = "1.0"
    AUTOPPIA_IWA_IMPORT_OK = False
    AUTOPPIA_IWA_IMPORT_ERROR = "autoppia_iwa import failed in miner runtime"


__all__ = [
    "AUTOPPIA_IWA_IMPORT_ERROR",
    "AUTOPPIA_IWA_IMPORT_OK",
    "ActAllowedTool",
    "ActRequest",
    "ActResponse",
    "ActToolCall",
    "BaseAction",
    "IWA_ACT_PROTOCOL_VERSION",
    "IWebAgent",
    "STEP_PROTOCOL_VERSION",
    "StepAllowedTool",
    "StepRequest",
    "StepResponse",
    "StepToolCall",
    "Task",
]
