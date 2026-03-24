from __future__ import annotations

import os
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
from src.operator.runtime.fsm_adapter import run_fsm_operator

from .candidates import Candidate, CandidateExtractor, CandidateRanker
from .engine import FSMOperator
from .observation import ObsBuilder
from .site_knowledge import (
    _build_site_knowledge,
    _crawl_site_routes,
    _load_static_site_maps,
    _load_task_cache_site_index,
)
from .state import AgentFormProgress, AgentState, FlagDetector
from .utils import MAX_INTERNAL_META_STEPS

os.environ.setdefault("LLM_PROVIDER", "openai")
load_dotenv(dotenv_path=Path(__file__).resolve().parents[4] / ".env", override=False)

_FSM_OPERATOR = FSMOperator(
    llm_call=openai_chat_completions,
    vision_call=(openai_vision_chat_completions if use_vision() else None),
)


class ApifiedWebAgent(BaseApifiedWebAgent):
    @staticmethod
    def _runtime_impl() -> str:
        return "fsm"

    async def act_from_payload(self, payload: dict[str, object]) -> dict[str, object]:
        model_override = str(payload.get("model") or "").strip()
        return run_fsm_operator(_FSM_OPERATOR, payload, model_override=model_override)


__all__ = [
    "MAX_INTERNAL_META_STEPS",
    "_FSM_OPERATOR",
    "AgentFormProgress",
    "AgentState",
    "ApifiedWebAgent",
    "Candidate",
    "CandidateExtractor",
    "CandidateRanker",
    "FSMOperator",
    "FlagDetector",
    "ObsBuilder",
    "_build_site_knowledge",
    "_crawl_site_routes",
    "_load_static_site_maps",
    "_load_task_cache_site_index",
    "_normalize_demo_url",
    "_sanitize_action_payload",
    "_task_from_payload",
]
