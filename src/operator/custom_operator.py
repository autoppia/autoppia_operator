from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from src.operator.agents.step_engine import ApifiedWebAgent as GeneralistWebAgent
from src.operator.skills import SkillRegistry, SkillRunner, load_skill_registry


class CustomOperator(GeneralistWebAgent):
    """Generalist web operator extended with user skills.

    Skills are trajectory-backed tools. They are not the whole operator: if no skill
    applies, the request falls through to the generalist StepEngine runtime.
    """

    def __init__(
        self,
        id: str = "1",
        name: str = "CustomOperator",
        *,
        registry: SkillRegistry | None = None,
        generalist: GeneralistWebAgent | None = None,
    ) -> None:
        super().__init__(id=id, name=name)
        self.registry = registry or load_skill_registry()
        self.skill_runner = SkillRunner(self.registry)
        self.generalist = generalist or GeneralistWebAgent(id=id, name="GeneralistOperator")

    @staticmethod
    def _runtime_impl() -> str:
        return "custom_operator"

    def capabilities_payload(self) -> dict[str, Any]:
        payload = super().capabilities_payload()
        payload["name"] = self.registry.operator_name or self.name
        payload["runtime"] = "custom_operator"
        payload["base_operator"] = "generalist_step_engine"
        payload["skills"] = [
            {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "success_criteria": skill.success_criteria,
                "keywords": list(skill.keywords),
                "steps": len(skill.steps),
            }
            for skill in self.registry.skills
        ]
        payload["skill_tool_definitions"] = self.registry.list_tool_definitions()
        return payload

    async def act_from_payload(self, payload: dict[str, object]) -> dict[str, object]:
        prompt = str(payload.get("prompt") or "")
        custom_state = self._custom_state(payload)
        active = custom_state.get("active_skill") if isinstance(custom_state.get("active_skill"), dict) else None
        if active:
            skill_id = str(active.get("id") or "")
            skill = self.registry.get(skill_id)
            if skill is not None:
                step_index = int(active.get("next_step") or 0)
                if step_index < len(skill.steps):
                    action = self.skill_runner.action_at(skill, step_index, prompt=prompt)
                    if action is not None:
                        next_state = deepcopy(custom_state)
                        next_state["active_skill"] = {"id": skill.id, "next_step": step_index + 1}
                        return self._skill_response(
                            skill_id=skill.id,
                            action=action,
                            internal_state=self._merge_internal_state(payload, custom_state=next_state),
                            reasoning=f"Continuing skill {skill.id} step {step_index + 1}/{len(skill.steps)}.",
                        )
                done_state = deepcopy(custom_state)
                done_state.pop("active_skill", None)
                completed = list(done_state.get("completed_skills") or [])
                completed.append(skill.id)
                done_state["completed_skills"] = completed[-20:]
                return {
                    "protocol_version": "1.0",
                    "actions": [{"type": "DoneAction", "content": f"Skill {skill.name} completed."}],
                    "done": True,
                    "content": f"Skill {skill.name} completed.",
                    "reasoning": f"Finished skill {skill.id}; returning control to the operator.",
                    "internal_state": self._merge_internal_state(payload, custom_state=done_state),
                }

        match = self.registry.match(prompt)
        if match is not None and match.skill.steps:
            action = self.skill_runner.action_at(match.skill, 0, prompt=prompt)
            if action is not None:
                next_state = deepcopy(custom_state)
                next_state["active_skill"] = {"id": match.skill.id, "next_step": 1}
                return self._skill_response(
                    skill_id=match.skill.id,
                    action=action,
                    internal_state=self._merge_internal_state(payload, custom_state=next_state),
                    reasoning=f"Using skill {match.skill.id}: {match.reason}.",
                )

        generalist_payload = dict(payload)
        generalist_state = custom_state.get("generalist_internal_state")
        if isinstance(generalist_state, dict):
            generalist_payload["_internal_state"] = generalist_state
        if self.registry.instructions:
            generalist_payload["prompt"] = f"{prompt}\n\nCUSTOM OPERATOR INSTRUCTIONS:\n{self.registry.instructions}"
        raw = await self.generalist.act_from_payload(generalist_payload)
        raw_dict = dict(raw) if isinstance(raw, dict) else {}
        custom_state["generalist_internal_state"] = raw_dict.get("internal_state") if isinstance(raw_dict.get("internal_state"), dict) else {}
        raw_dict["internal_state"] = self._merge_internal_state(payload, custom_state=custom_state)
        if raw_dict.get("reasoning"):
            raw_dict["reasoning"] = f"Generalist fallback. {raw_dict['reasoning']}"
        else:
            raw_dict["reasoning"] = "Generalist fallback; no matching skill was selected."
        return raw_dict

    def _skill_response(self, *, skill_id: str, action: dict[str, Any], internal_state: dict[str, Any], reasoning: str) -> dict[str, Any]:
        return {
            "protocol_version": "1.0",
            "actions": [action],
            "done": False,
            "content": None,
            "reasoning": reasoning,
            "internal_state": internal_state,
            "skill_call": {"name": skill_id},
        }

    def _custom_state(self, payload: dict[str, object]) -> dict[str, Any]:
        internal = payload.get("_internal_state") if isinstance(payload.get("_internal_state"), dict) else {}
        custom = internal.get("custom_operator") if isinstance(internal.get("custom_operator"), dict) else {}
        if not custom:
            return {"completed_skills": [], "generalist_internal_state": {}}
        return deepcopy(custom)

    def _merge_internal_state(self, payload: dict[str, object], *, custom_state: dict[str, Any]) -> dict[str, Any]:
        internal = payload.get("_internal_state") if isinstance(payload.get("_internal_state"), dict) else {}
        merged = dict(internal)
        merged["custom_operator"] = custom_state
        return merged


def build_custom_operator() -> CustomOperator:
    package_dir = os.getenv("AUTOPPIA_CUSTOM_OPERATOR_PACKAGE", "")
    registry = load_skill_registry(package_dir)
    return CustomOperator(
        id=os.getenv("WEB_AGENT_ID", "1"),
        name=registry.operator_name or "CustomOperator",
        registry=registry,
    )
