from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillStep:
    action: dict[str, Any]


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str = ""
    success_criteria: str = ""
    keywords: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    steps: tuple[SkillStep, ...] = ()

    def to_tool_definition(self) -> dict[str, Any]:
        return {
            "name": f"skill.{self.id}",
            "description": self.description or self.name,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
        }


@dataclass(frozen=True)
class SkillMatch:
    skill: Skill
    score: float
    reason: str


@dataclass
class SkillRegistry:
    operator_id: str = "custom"
    operator_name: str = "Custom Operator"
    base_url: str = ""
    instructions: str = ""
    defaults: dict[str, str] = field(default_factory=dict)
    skills: tuple[Skill, ...] = ()

    def list_tool_definitions(self) -> list[dict[str, Any]]:
        return [skill.to_tool_definition() for skill in self.skills]

    def get(self, skill_id: str) -> Skill | None:
        wanted = str(skill_id or "").strip().lower()
        for skill in self.skills:
            if skill.id.lower() == wanted:
                return skill
        return None

    def match(self, prompt: str) -> SkillMatch | None:
        prompt_l = str(prompt or "").strip().lower()
        if not prompt_l:
            return None
        best: SkillMatch | None = None
        for skill in self.skills:
            score = 0.0
            reason = ""
            for phrase in skill.prompts:
                phrase_l = str(phrase or "").strip().lower()
                if phrase_l and phrase_l in prompt_l:
                    score = max(score, 1.0)
                    reason = f"matched prompt phrase '{phrase_l}'"
            for keyword in skill.keywords:
                keyword_l = str(keyword or "").strip().lower()
                if keyword_l and keyword_l in prompt_l:
                    score = max(score, 0.75)
                    reason = f"matched keyword '{keyword_l}'"
            skill_name = skill.name.strip().lower()
            if skill_name and skill_name in prompt_l:
                score = max(score, 0.85)
                reason = f"matched skill name '{skill_name}'"
            if score <= 0:
                continue
            candidate = SkillMatch(skill=skill, score=score, reason=reason or "matched skill metadata")
            if best is None or candidate.score > best.score:
                best = candidate
        return best
