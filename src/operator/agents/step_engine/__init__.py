from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[4] / ".env", override=False)

_SIMPLE_EXPORTS: dict[str, tuple[str, str]] = {
    "StepEngine": ("src.operator.agents.step_engine.engine", "StepEngine"),
    "FSMOperator": ("src.operator.agents.step_engine.engine", "StepEngine"),
    "CanonicalBrowserState": ("src.operator.agents.step_engine.models", "CanonicalBrowserState"),
    "MAX_INTERNAL_META_STEPS": ("src.operator.agents.step_engine.utils", "MAX_INTERNAL_META_STEPS"),
    "AgentFormProgress": ("src.operator.agents.step_engine.state", "AgentFormProgress"),
    "AgentState": ("src.operator.agents.step_engine.state", "AgentState"),
    "FlagDetector": ("src.operator.agents.step_engine.state", "FlagDetector"),
    "Candidate": ("src.operator.agents.step_engine.candidates", "Candidate"),
    "CandidateExtractor": ("src.operator.agents.step_engine.candidates", "CandidateExtractor"),
    "CandidateRanker": ("src.operator.agents.step_engine.candidates", "CandidateRanker"),
    "ObsBuilder": ("src.operator.agents.step_engine.observation", "ObsBuilder"),
    "_build_site_knowledge": ("src.operator.agents.step_engine.site_knowledge", "_build_site_knowledge"),
    "_crawl_site_routes": ("src.operator.agents.step_engine.site_knowledge", "_crawl_site_routes"),
    "_load_static_site_maps": ("src.operator.agents.step_engine.site_knowledge", "_load_static_site_maps"),
    "_load_task_cache_site_index": ("src.operator.agents.step_engine.site_knowledge", "_load_task_cache_site_index"),
    "_normalize_demo_url": ("src.operator.api.act_protocol", "_normalize_demo_url"),
    "_sanitize_action_payload": ("src.operator.api.act_protocol", "_sanitize_action_payload"),
    "_task_from_payload": ("src.operator.api.act_protocol", "_task_from_payload"),
}

__all__ = [
    "MAX_INTERNAL_META_STEPS",
    "_FSM_OPERATOR",
    "_STEP_ENGINE",
    "AgentFormProgress",
    "AgentState",
    "ApifiedWebAgent",
    "Candidate",
    "CandidateExtractor",
    "CandidateRanker",
    "CanonicalBrowserState",
    "FSMOperator",
    "FlagDetector",
    "ObsBuilder",
    "StepEngine",
    "_build_site_knowledge",
    "_crawl_site_routes",
    "_load_static_site_maps",
    "_load_task_cache_site_index",
    "_normalize_demo_url",
    "_sanitize_action_payload",
    "_task_from_payload",
]


def _build_step_engine() -> Any:
    from infra.llm_gateway import openai_chat_completions, openai_vision_chat_completions
    from src.operator.agents.step_engine.engine import StepEngine
    from src.operator.api.act_protocol import use_vision

    return StepEngine(
        llm_call=openai_chat_completions,
        vision_call=(openai_vision_chat_completions if use_vision() else None),
    )


def _build_apified_web_agent() -> type:
    from src.operator.agents.base import BaseApifiedWebAgent
    from src.operator.runtime.step_adapter import run_step_engine

    class ApifiedWebAgent(BaseApifiedWebAgent):
        @staticmethod
        def _runtime_impl() -> str:
            return "step_engine"

        async def act_from_payload(self, payload: dict[str, object]) -> dict[str, object]:
            model_override = str(payload.get("model") or "").strip()
            return run_step_engine(_get_step_engine(), payload, model_override=model_override)

    return ApifiedWebAgent


def _get_step_engine() -> Any:
    cached = globals().get("_STEP_ENGINE")
    if cached is not None:
        return cached
    engine = _build_step_engine()
    globals()["_STEP_ENGINE"] = engine
    globals()["_FSM_OPERATOR"] = engine
    return engine


def __getattr__(name: str) -> Any:
    if name == "ApifiedWebAgent":
        value = _build_apified_web_agent()
        globals()[name] = value
        return value
    if name in {"_STEP_ENGINE", "_FSM_OPERATOR"}:
        return _get_step_engine()
    try:
        module_name, attr_name = _SIMPLE_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
