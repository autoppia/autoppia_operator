from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from infra.llm_gateway import openai_chat_completions, openai_vision_chat_completions
from src.operator.agents.base import BaseApifiedWebAgent
from src.operator.api.act_protocol import (
    _normalize_demo_url,
    _sanitize_action_payload,
    _task_from_payload,
    use_vision,
)
from src.operator.runtime.step_adapter import run_step_engine

from .utils import MAX_INTERNAL_META_STEPS
from .state import AgentFormProgress, AgentState, FlagDetector
from .candidates import Candidate, CandidateExtractor, CandidateRanker
from .observation import ObsBuilder
from .site_knowledge import _build_site_knowledge, _crawl_site_routes, _load_static_site_maps, _load_task_cache_site_index
from .engine import StepEngine
from .models import CanonicalBrowserState

load_dotenv(dotenv_path=Path(__file__).resolve().parents[4] / ".env", override=False)

_STEP_ENGINE = StepEngine(
    llm_call=openai_chat_completions,
    vision_call=(openai_vision_chat_completions if use_vision() else None),
)


class ApifiedWebAgent(BaseApifiedWebAgent):
    @staticmethod
    def _runtime_impl() -> str:
        return "step_engine"

    async def act_from_payload(self, payload: dict[str, object]) -> dict[str, object]:
        model_override = str(payload.get("model") or "").strip()
        return run_step_engine(_STEP_ENGINE, payload, model_override=model_override)


FSMOperator = StepEngine
_FSM_OPERATOR = _STEP_ENGINE

__all__ = [
    "StepEngine",
    "FSMOperator",
    "CanonicalBrowserState",
    "MAX_INTERNAL_META_STEPS",
    "AgentFormProgress",
    "AgentState",
    "Candidate",
    "CandidateExtractor",
    "CandidateRanker",
    "FlagDetector",
    "ObsBuilder",
    "ApifiedWebAgent",
    "_STEP_ENGINE",
    "_FSM_OPERATOR",
    "_build_site_knowledge",
    "_crawl_site_routes",
    "_load_static_site_maps",
    "_load_task_cache_site_index",
    "_normalize_demo_url",
    "_sanitize_action_payload",
    "_task_from_payload",
]
