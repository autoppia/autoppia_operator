from __future__ import annotations

import argparse
import asyncio
import glob
import importlib
import importlib.util
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig

from training.finetune_bu import MODEL_ID, _ensure_deps, _load_trainable_model
from training.rl.contact_env import ContactRLEnv
from training.rl.judge import judge_step_progress


def _ensure_repo_src_package() -> None:
    src_init = REPO_ROOT / "src" / "__init__.py"
    src_operator_init = REPO_ROOT / "src" / "operator" / "__init__.py"
    if "src" not in sys.modules:
        spec = importlib.util.spec_from_file_location("src", src_init, submodule_search_locations=[str(src_init.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["src"] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)
    if "src.operator" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "src.operator",
            src_operator_init,
            submodule_search_locations=[str(src_operator_init.parent)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["src.operator"] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)


_ensure_repo_src_package()
StepEngine = importlib.import_module("src.operator.agents.step_engine.engine").StepEngine

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, dict):
                out.append(str(item.get("text") or item.get("content") or ""))
            else:
                out.append(str(item or ""))
        return "\n".join(x for x in out if x)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content or "")


def _format_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|{str(msg.get('role') or 'user')}|>\n{_message_content_to_text(msg.get('content'))}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in str(raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ".." in chunk:
            start_s, end_s = chunk.split("..", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            step = 1 if end_i >= start_i else -1
            out.extend(range(start_i, end_i + step, step))
        else:
            out.append(int(chunk))
    seen: set[int] = set()
    ordered: list[int] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


@dataclass
class ExpertRecord:
    prompt_text: str
    completion_ids: list[int]
    priority: float = 1.0
    step_idx: int = 0
    route_hint: str = ""
    seed: int = -1


@dataclass
class PPOTransition:
    seed: int
    step_idx: int
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logprob: float
    value: float
    reward: float
    done: bool
    before_url: str
    after_url: str
    chosen_action: dict[str, Any] | None


@dataclass
class TeacherStep:
    seed: int
    step_idx: int
    tool_name: str
    route_hint: str = ""
    selector_attr: str = ""
    selector_value: str = ""
    completion_ids: list[int] | None = None


@dataclass
class Step0Candidate:
    tool_call: dict[str, Any]
    label: str
    source: str
    score_hint: float


def _extract_prompt_section(prompt_text: str, header: str) -> str:
    text = str(prompt_text or "")
    marker = f"{header}:\n"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    next_headers = [
        "\nTASK CONSTRAINTS:\n",
        "\nRUNTIME:\n",
        "\nCURRENT STATE:\n",
        "\nAVAILABLE TOOLS:\n",
        "\nUNAVAILABLE TOOLS:\n",
        "\nTOOL USAGE GUIDE:\n",
        "\nWORKING STATE (JSON):\n",
        "\nLOCAL WORKFLOW CLOSURE (JSON):\n",
        "\nLOCAL HTML CONTEXT (JSON):\n",
        "\nKNOWN SITE MAP (JSON):\n",
        "\nAVOID REPEATING (JSON):\n",
        "\nPREVIOUS REASONING TRACE (JSON):\n",
        "\nWORKING STATE SUMMARY:\n",
        "\nBROWSER SNAPSHOT:\n",
        "\nELEMENT TARGETING GUIDE:\n",
        "\nRAW HTML SNAPSHOT:\n",
    ]
    end = len(text)
    for h in next_headers:
        idx = text.find(h, start)
        if idx >= 0 and idx < end:
            end = idx
    return text[start:end].strip()


def _extract_current_url(prompt_text: str) -> str:
    m = re.search(r"^url=(.+)$", str(prompt_text or ""), flags=re.MULTILINE)
    return str(m.group(1) if m else "").strip()


def _extract_task_line(prompt_text: str) -> str:
    m = re.search(r"^TASK:\s*(.+)$", str(prompt_text or ""), flags=re.MULTILINE)
    return str(m.group(1) if m else "").strip()


def _candidate_tool_key(tool_call: dict[str, Any]) -> str:
    return json.dumps(tool_call, ensure_ascii=False, sort_keys=True)


def _build_step0_candidates(prompt_text: str) -> list[Step0Candidate]:
    task_text = _extract_task_line(prompt_text).lower()
    current_url = _extract_current_url(prompt_text)
    html = _extract_prompt_section(prompt_text, "RAW HTML SNAPSHOT")
    out: list[Step0Candidate] = []
    seen: set[str] = set()

    def add(tool_call: dict[str, Any], label: str, source: str, score_hint: float) -> None:
        key = _candidate_tool_key(tool_call)
        if key in seen:
            return
        seen.add(key)
        out.append(Step0Candidate(tool_call=tool_call, label=label, source=source, score_hint=float(score_hint)))

    route_matches = re.findall(r"(?<![A-Za-z0-9_-])/([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*)", task_text)
    route_hints = [f"/{match.strip().lower()}" for match in route_matches if match.strip()]
    if "/contact" not in route_hints and "contact" in task_text:
        route_hints.insert(0, "/contact")
    for path, bonus in [("/contact", 10.0), ("/support", 5.0), ("/about", 2.0)]:
        if path in route_hints or (path == "/contact" and "contact" in task_text):
            add(
                {"name": "browser.navigate", "arguments": {"url": urljoin(current_url or "http://example.invalid", path)}},
                f"navigate:{path}",
                "route_guess",
                bonus,
            )

    if html and BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            soup = None
        if soup is not None:
            for link in soup.find_all("a", href=True):
                href = str(link.get("href") or "").strip()
                if not href:
                    continue
                text = " ".join(link.get_text(" ", strip=True).split()).strip()
                blob = f"{text} {href}".lower()
                score = 0.0
                if "contact" in blob:
                    score += 12.0
                if any(token in blob for token in ("support", "message", "email", "get in touch")):
                    score += 6.0
                if "about" in blob:
                    score += 2.0
                if score <= 0.0:
                    continue
                add(
                    {"name": "browser.navigate", "arguments": {"url": urljoin(current_url or "http://example.invalid", href)}},
                    f"visible_link:{text or href}",
                    "visible_link",
                    score,
                )

    out.sort(key=lambda item: float(item.score_hint), reverse=True)
    return out[:12]


class ActorCriticPolicy(nn.Module):
    def __init__(self, model: Any, hidden_size: int) -> None:
        super().__init__()
        self.model = model
        self.value_head = nn.Linear(int(hidden_size), 1, bias=True)
        try:
            first_param = next(self.model.parameters())
            self.value_head.to(device=first_param.device, dtype=first_param.dtype)
        except StopIteration:
            pass

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


class LocalPolicyLLM:
    def __init__(
        self,
        *,
        actor_critic: ActorCriticPolicy,
        tokenizer: Any,
        max_new_tokens: int,
        max_prompt_tokens: int,
        temperature: float,
        top_p: float,
        first_step_samples: int = 1,
        first_step_temperature: float = 1.0,
        first_step_top_p: float = 0.98,
    ) -> None:
        self.actor_critic = actor_critic
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.first_step_samples = max(1, int(first_step_samples))
        self.first_step_temperature = float(first_step_temperature)
        self.first_step_top_p = float(first_step_top_p)
        self.records: list[dict[str, Any]] = []

    def _truncate_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(self.max_prompt_tokens) <= 0 or int(input_ids.shape[1]) <= int(self.max_prompt_tokens):
            return input_ids, attention_mask
        head_tokens = min(512, int(self.max_prompt_tokens) // 4)
        tail_tokens = int(self.max_prompt_tokens) - head_tokens
        return (
            torch.cat([input_ids[:, :head_tokens], input_ids[:, -tail_tokens:]], dim=1),
            torch.cat([attention_mask[:, :head_tokens], attention_mask[:, -tail_tokens:]], dim=1),
        )

    def _score_candidate(self, *, prompt_text: str, completion_text: str) -> float:
        text = str(completion_text or "").lower()
        prompt = str(prompt_text or "").lower()
        score = 0.0
        if "contact" in prompt:
            if '"name":"browser.click"' in text:
                score += 2.5
            if '"name":"browser.navigate"' in text:
                score += 2.0
            if '"name":"browser.input"' in text:
                score -= 2.0
            if '"name":"browser.scroll"' in text:
                score -= 8.0
            if '"name":"browser.wait"' in text:
                score -= 8.0
            if '"name":"browser.search"' in text:
                score -= 4.0
            if "/contact" in text:
                score += 8.0
            if any(token in text for token in ('"contact"', '"support"', '"message"', '"email"', '"send"', '"about"')):
                score += 3.0
            if '"text":"david"' in text:
                score -= 4.0
        if '"name":"browser.navigate"' in text:
            score += 3.0
        if '"name":"browser.click"' in text:
            score += 1.5
        if '"name":"browser.input"' in text:
            score -= 1.5
        if '"name":"browser.scroll"' in text:
            score -= 4.0
        if '"name":"browser.wait"' in text:
            score -= 4.0
        if '"name":"browser.search"' in text:
            score -= 2.0
        if "/contact" in text:
            score += 5.0
        if "contact" in text:
            score += 2.0
        if "contact form" in prompt and '"name":"browser.input"' in text and '"text":"david"' in text:
            score -= 3.0
        return score

    @torch.no_grad()
    def __call__(self, *, task_id: str, messages: list[dict[str, Any]], model: str, temperature: float, max_tokens: int, **_: Any) -> dict[str, Any]:
        prompt_text = _format_messages(messages)
        is_first_step = "step_index=0" in prompt_text.lower()
        if is_first_step:
            candidates = _build_step0_candidates(prompt_text)
            if candidates:
                best_cand = candidates[0]
                completion_text = json.dumps({"type": "browser", "tool_call": best_cand.tool_call}, ensure_ascii=False)
                enc_prompt = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
                prompt_ids = enc_prompt["input_ids"][0].tolist()
                completion_ids = self.tokenizer(completion_text, add_special_tokens=False)["input_ids"]
                self.records.append(
                    {
                        "task_id": str(task_id),
                        "prompt_text": prompt_text,
                        "prompt_ids": prompt_ids,
                        "completion_ids": list(completion_ids),
                        "completion_text": completion_text,
                        "messages": list(messages),
                        "selector_mode": "structured_step0",
                        "candidates": [
                            {
                                "tool_call": dict(item.tool_call),
                                "label": str(item.label),
                                "source": str(item.source),
                                "score_hint": float(item.score_hint),
                            }
                            for item in candidates
                        ],
                        "selected_score": float(best_cand.score_hint),
                    }
                )
                return {
                    "choices": [{"message": {"content": completion_text}}],
                    "usage": {
                        "prompt_tokens": int(len(prompt_ids)),
                        "completion_tokens": int(len(completion_ids)),
                        "total_tokens": int(len(prompt_ids) + len(completion_ids)),
                    },
                    "model": str(model or ""),
                }
        enc = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask = self._truncate_prompt(input_ids, attention_mask)
        input_ids = input_ids.to(self.actor_critic.device)
        attention_mask = attention_mask.to(self.actor_critic.device)
        sample_count = self.first_step_samples if is_first_step else 1
        sample_temperature = self.first_step_temperature if is_first_step else float(temperature or self.temperature)
        sample_top_p = self.first_step_top_p if is_first_step else self.top_p
        do_sample = float(sample_temperature) > 1e-5
        prompt_len = int(input_ids.shape[1])
        candidates: list[dict[str, Any]] = []
        for _sample_idx in range(sample_count):
            out = self.actor_critic.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(int(max_tokens or self.max_new_tokens), self.max_new_tokens),
                do_sample=do_sample,
                temperature=max(float(sample_temperature), 1e-5) if do_sample else 1.0,
                top_p=sample_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            completion_ids = out[0, prompt_len:].tolist()
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            candidates.append(
                {
                    "completion_ids": completion_ids,
                    "completion_text": completion_text,
                    "score": float(self._score_candidate(prompt_text=prompt_text, completion_text=completion_text)),
                }
            )
        filtered = list(candidates)
        if is_first_step:
            non_banned = [
                item
                for item in candidates
                if '"name":"browser.scroll"' not in str(item.get("completion_text") or "").lower()
                and '"name":"browser.wait"' not in str(item.get("completion_text") or "").lower()
            ]
            if non_banned:
                filtered = non_banned
        best = max(filtered, key=lambda item: float(item.get("score") or 0.0))
        completion_ids = list(best["completion_ids"])
        completion_text = str(best["completion_text"])
        self.records.append(
            {
                "task_id": str(task_id),
                "prompt_text": self.tokenizer.decode(input_ids[0], skip_special_tokens=False),
                "prompt_ids": input_ids[0].tolist(),
                "completion_ids": completion_ids,
                "completion_text": completion_text,
                "messages": list(messages),
                "candidates": [
                    {
                        "completion_text": str(item["completion_text"]),
                        "score": float(item["score"]),
                    }
                    for item in candidates
                ],
                "selected_score": float(best["score"]),
            }
        )
        return {
            "choices": [{"message": {"content": completion_text}}],
            "usage": {
                "prompt_tokens": int(prompt_len),
                "completion_tokens": int(len(completion_ids)),
                "total_tokens": int(prompt_len + len(completion_ids)),
            },
            "model": str(model or ""),
        }

    def pop_last_record(self) -> dict[str, Any] | None:
        if not self.records:
            return None
        return self.records.pop(0)


def _compute_action_stats(
    actor_critic: ActorCriticPolicy,
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    max_prompt_tokens: int,
    max_completion_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_tail = list(prompt_ids[-int(max_prompt_tokens) :]) if max_prompt_tokens > 0 else list(prompt_ids)
    completion_tail = list(completion_ids[: int(max_completion_tokens)]) if max_completion_tokens > 0 else list(completion_ids)
    if not completion_tail:
        zero = torch.zeros((), device=actor_critic.device)
        return zero, zero
    full = torch.tensor([prompt_tail + completion_tail], dtype=torch.long, device=actor_critic.device)
    attn = torch.ones_like(full)
    logits, values = actor_critic(full, attn)
    logits = logits[:, :-1, :]
    targets = full[:, 1:]
    prompt_len = len(prompt_tail)
    start = max(prompt_len - 1, 0)
    end = start + len(completion_tail)
    token_logprobs = torch.log_softmax(logits[:, start:end, :], dim=-1)
    chosen = targets[:, start:end]
    logprob = token_logprobs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1).sum()
    state_value = values[:, prompt_len - 1].squeeze(0)
    return logprob, state_value


def _load_expert_records(patterns: list[str], tokenizer: Any) -> list[ExpertRecord]:
    records: list[ExpertRecord] = []
    for pattern in patterns:
        for raw_path in sorted(glob.glob(pattern)):
            seed_match = re.search(r"seed_(\d+)", Path(raw_path).name)
            seed = int(seed_match.group(1)) if seed_match else -1
            payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            for episode in payload.get("episodes", []) or []:
                for ordinal, step in enumerate(episode.get("guided_execution", []) or []):
                    prompt_text = str(step.get("policy_input_text") or "")
                    tool_call = step.get("policy_tool_call") or {}
                    if not prompt_text or not tool_call:
                        continue
                    completion_text = json.dumps({"type": "browser", "tool_call": tool_call}, ensure_ascii=False)
                    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
                    if not completion_ids:
                        continue
                    args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                    route_hint = str(args.get("url") or "").lower()
                    priority = 3.0 if str(tool_call.get("name") or "").strip().lower() == "browser.navigate" else 1.0
                    records.append(
                        ExpertRecord(
                            prompt_text=prompt_text,
                            completion_ids=list(completion_ids),
                            priority=float(priority),
                            step_idx=int(step.get("step_idx")) if step.get("step_idx") is not None else int(ordinal),
                            route_hint=route_hint,
                            seed=int(seed),
                        )
                    )
    return records


def _load_teacher_steps(patterns: list[str], tokenizer: Any) -> dict[tuple[int, int], TeacherStep]:
    teacher_steps: dict[tuple[int, int], TeacherStep] = {}
    for pattern in patterns:
        for raw_path in sorted(glob.glob(pattern)):
            seed_match = re.search(r"seed_(\d+)", Path(raw_path).name)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))
            payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            for episode in payload.get("episodes", []) or []:
                for ordinal, step in enumerate(episode.get("guided_execution", []) or []):
                    tool_call = step.get("policy_tool_call") or {}
                    if not isinstance(tool_call, dict):
                        continue
                    args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                    selector = args.get("selector") if isinstance(args.get("selector"), dict) else {}
                    step_idx = int(step.get("step_idx")) if step.get("step_idx") is not None else int(ordinal)
                    completion_text = json.dumps({"type": "browser", "tool_call": tool_call}, ensure_ascii=False)
                    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
                    teacher_steps[(seed, step_idx)] = TeacherStep(
                        seed=seed,
                        step_idx=step_idx,
                        tool_name=str(tool_call.get("name") or "").strip().lower(),
                        route_hint=str(args.get("url") or "").strip().lower(),
                        selector_attr=str(selector.get("attribute") or "").strip().lower(),
                        selector_value=str(selector.get("value") or "").strip().lower(),
                        completion_ids=list(completion_ids),
                    )
    return teacher_steps


def _action_route_hint(action: dict[str, Any] | None) -> str:
    action = action if isinstance(action, dict) else {}
    selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
    if str(action.get("type") or "").strip().lower() == "navigateaction":
        return str(action.get("url") or "").strip().lower()
    if str(selector.get("attribute") or "").strip().lower() == "href":
        return str(selector.get("value") or "").strip().lower()
    return ""


def _teacher_reward_delta(teacher_step: TeacherStep | None, chosen_action: dict[str, Any] | None) -> tuple[float, str]:
    if teacher_step is None:
        return 0.0, "no_teacher"
    action = chosen_action if isinstance(chosen_action, dict) else {}
    action_type = str(action.get("type") or "").strip().lower()
    selector = action.get("selector") if isinstance(action.get("selector"), dict) else {}
    route_hint = _action_route_hint(action)
    teacher_route = str(teacher_step.route_hint or "").strip().lower()

    if "navigate" in teacher_step.tool_name:
        if teacher_route and teacher_route in route_hint:
            return 1.0, "teacher_route_match"
        if action_type == "clickaction" and teacher_route and teacher_route in route_hint:
            return 0.75, "teacher_route_click_match"
        return -1.0, "teacher_route_miss"

    if "click" in teacher_step.tool_name and teacher_step.selector_attr == "href":
        if teacher_step.selector_value and teacher_step.selector_value in route_hint:
            return 1.0, "teacher_click_href_match"
        return -0.75, "teacher_click_href_miss"

    selector_attr = str(selector.get("attribute") or "").strip().lower()
    selector_value = str(selector.get("value") or "").strip().lower()
    if teacher_step.selector_attr and teacher_step.selector_value:
        if selector_attr == teacher_step.selector_attr and selector_value == teacher_step.selector_value:
            return 1.0, "teacher_selector_match"
        return -0.75, "teacher_selector_miss"

    return 0.0, "teacher_no_signal"


def _compute_gae(transitions: list[PPOTransition], gamma: float, lam: float) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(transitions)
    returns = [0.0] * len(transitions)
    gae = 0.0
    next_value = 0.0
    for idx in range(len(transitions) - 1, -1, -1):
        t = transitions[idx]
        mask = 0.0 if bool(t.done) else 1.0
        delta = float(t.reward) + float(gamma) * next_value * mask - float(t.value)
        gae = delta + float(gamma) * float(lam) * mask * gae
        advantages[idx] = gae
        returns[idx] = gae + float(t.value)
        next_value = float(t.value)
    return advantages, returns


class ContactPPOTrainer:
    def __init__(
        self,
        *,
        base_model: str,
        output_dir: str,
        adapter_path: str = "",
        lr: float = 1e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 2,
        minibatch_size: int = 4,
        max_steps: int = 8,
        max_new_tokens: int = 96,
        max_prompt_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        first_step_samples: int = 1,
        first_step_temperature: float = 1.0,
        first_step_top_p: float = 0.98,
        expert_patterns: list[str] | None = None,
        expert_pretrain_updates: int = 0,
        expert_batch_size: int = 8,
        teacher_bc_weight: float = 0.0,
        teacher_bc_max_step: int = 0,
        judge_enabled: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        _ensure_deps()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_range = float(clip_range)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.max_steps = int(max_steps)
        self.judge_enabled = bool(judge_enabled)
        self.teacher_bc_weight = float(teacher_bc_weight)
        self.teacher_bc_max_step = int(teacher_bc_max_step)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = _load_trainable_model(base_model, bnb_config=bnb_config, bf16=True)
        model.config.use_cache = False
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if bool(gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        hidden_size = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "text_config", None).hidden_size)
        self.actor_critic = ActorCriticPolicy(model=model, hidden_size=hidden_size)
        self.actor_critic.train()
        self.optimizer = torch.optim.AdamW(
            [p for p in self.actor_critic.parameters() if p.requires_grad],
            lr=float(lr),
        )
        self.local_llm = LocalPolicyLLM(
            actor_critic=self.actor_critic,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            temperature=temperature,
            top_p=top_p,
            first_step_samples=first_step_samples,
            first_step_temperature=first_step_temperature,
            first_step_top_p=first_step_top_p,
        )
        self.step_engine = StepEngine(llm_call=self.local_llm, vision_call=None)
        self.expert_records = _load_expert_records(list(expert_patterns or []), self.tokenizer)
        self.teacher_steps = _load_teacher_steps(list(expert_patterns or []), self.tokenizer)
        self.expert_pretrain_updates = int(expert_pretrain_updates)
        self.expert_batch_size = int(expert_batch_size)

    @property
    def device(self) -> torch.device:
        return self.actor_critic.device

    def _sample_expert_batch(self, rng: random.Random) -> list[ExpertRecord]:
        if not self.expert_records:
            return []
        if len(self.expert_records) <= self.expert_batch_size:
            return list(self.expert_records)
        weights = [max(float(r.priority), 0.1) for r in self.expert_records]
        return rng.choices(self.expert_records, weights=weights, k=self.expert_batch_size)

    def _backward_expert_batch(self, batch: list[ExpertRecord]) -> float | None:
        if not batch:
            return None
        total = 0.0
        sample_size = max(len(batch), 1)
        for rec in batch:
            prompt_ids = self.tokenizer(rec.prompt_text, add_special_tokens=False)["input_ids"]
            logprob, _ = _compute_action_stats(
                self.actor_critic,
                prompt_ids=prompt_ids,
                completion_ids=rec.completion_ids,
                max_prompt_tokens=96,
                max_completion_tokens=16,
            )
            raw_loss = -logprob
            total += float(raw_loss.detach().cpu())
            (raw_loss / float(sample_size)).backward()
        return total / float(sample_size)

    def _rollout_seed(self, seed: int) -> tuple[dict[str, Any], list[PPOTransition]]:
        env = ContactRLEnv(seed=int(seed), model_override="", max_steps=self.max_steps, step_engine_instance=self.step_engine)
        transitions: list[PPOTransition] = []

        async def _run() -> dict[str, Any]:
            try:
                await env.reset()
                done = False
                for step_index in range(self.max_steps):
                    before_result = env.last_step_result
                    before_url = str(before_result.snapshot.url or env.task.url) if before_result is not None else str(env.task.url)
                    before_html = str(before_result.snapshot.html or "") if before_result is not None else ""
                    record, done = await env.step(step_index)
                    gen_record = self.local_llm.pop_last_record()
                    chosen_action = record.env_action or record.chosen_action or {}
                    completion_text = json.dumps(chosen_action, ensure_ascii=False, sort_keys=True)
                    completion_ids = self.tokenizer(completion_text, add_special_tokens=False)["input_ids"]
                    if gen_record is None:
                        prompt_text = str(record.policy_input_text or "")
                        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                    else:
                        prompt_ids = list(gen_record["prompt_ids"])
                    if not prompt_ids or not completion_ids:
                        logger.warning("Skipping PPO transition at seed=%s step=%s due to empty prompt/action ids", int(seed), int(step_index))
                        if done:
                            break
                        continue
                    old_logprob_t, value_t = _compute_action_stats(
                        self.actor_critic,
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        max_prompt_tokens=32,
                        max_completion_tokens=12,
                    )
                    reward_total = float((record.reward or {}).get("total") or 0.0)
                    teacher_delta = 0.0
                    teacher_reason = ""
                    teacher_step = self.teacher_steps.get((int(seed), int(step_index)))
                    if teacher_step is not None:
                        teacher_delta, teacher_reason = _teacher_reward_delta(teacher_step, chosen_action)
                        record.reward["teacher_delta"] = float(teacher_delta)
                        record.reward["teacher_reason"] = teacher_reason
                        reward_total = float(reward_total + teacher_delta)
                        record.reward["total"] = reward_total
                    judge_delta = 0.0
                    judge_reason = ""
                    if self.judge_enabled:
                        judge_score = judge_step_progress(
                            task_id=str(env.task.id),
                            task_prompt=str(env.task.prompt or ""),
                            step_index=int(step_index),
                            before_url=before_url,
                            after_url=str(record.after_url or before_url),
                            before_html=before_html,
                            after_html=str(env.last_step_result.snapshot.html or "") if env.last_step_result is not None else "",
                            action=chosen_action,
                            base_reward=reward_total,
                        )
                        judge_delta = float(judge_score.reward_delta)
                        judge_reason = judge_score.reason
                        record.reward["judge_delta"] = float(judge_delta)
                        record.reward["judge_reason"] = judge_reason
                        record.reward["total"] = float(record.reward.get("total", 0.0) + judge_delta)
                        reward_total = float(record.reward["total"])
                    transitions.append(
                        PPOTransition(
                            seed=int(seed),
                            step_idx=int(step_index),
                            prompt_ids=list(prompt_ids),
                            completion_ids=list(completion_ids),
                            old_logprob=float(old_logprob_t.detach().cpu()),
                            value=float(value_t.detach().cpu()),
                            reward=float(reward_total),
                            done=bool(done),
                            before_url=before_url,
                            after_url=str(record.after_url or before_url),
                            chosen_action=chosen_action,
                        )
                    )
                    if done:
                        break
                final = env.last_step_result
                assert final is not None
                return {
                    "seed": int(seed),
                    "task_id": str(env.task.id),
                    "prompt": str(env.task.prompt),
                    "success": bool(final.score.success),
                    "score": float(final.score.raw_score),
                    "steps": [item.to_dict() for item in env.rollout_steps],
                }
            except Exception as exc:
                final = env.last_step_result
                return {
                    "seed": int(seed),
                    "task_id": str(getattr(env.task, "id", "")),
                    "prompt": str(getattr(env.task, "prompt", "")),
                    "success": bool(final.score.success) if final is not None else False,
                    "score": float(final.score.raw_score) if final is not None else 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "steps": [item.to_dict() for item in env.rollout_steps],
                }
            finally:
                await env.close()

        return asyncio.run(_run()), transitions

    def _ppo_update(self, transitions: list[PPOTransition], *, rng: random.Random) -> dict[str, float]:
        advantages, returns = _compute_gae(transitions, gamma=self.gamma, lam=self.gae_lambda)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if len(transitions) > 1:
            adv_t = (adv_t - adv_t.mean()) / adv_t.std(unbiased=False).clamp_min(1e-6)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_logprobs_t = torch.tensor([t.old_logprob for t in transitions], dtype=torch.float32, device=self.device)
        indices = list(range(len(transitions)))
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_teacher_bc = 0.0
        total_batches = 0
        for _epoch in range(self.ppo_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                batch_idx = indices[start : start + self.minibatch_size]
                self.optimizer.zero_grad(set_to_none=True)
                policy_loss = torch.zeros((), device=self.device)
                value_loss = torch.zeros((), device=self.device)
                entropy_bonus = torch.zeros((), device=self.device)
                teacher_bc_loss = torch.zeros((), device=self.device)
                for idx in batch_idx:
                    t = transitions[idx]
                    new_logprob, new_value = _compute_action_stats(
                        self.actor_critic,
                        prompt_ids=t.prompt_ids,
                        completion_ids=t.completion_ids,
                        max_prompt_tokens=32,
                        max_completion_tokens=12,
                    )
                    ratio = torch.exp(new_logprob - old_logprobs_t[idx])
                    surr1 = ratio * adv_t[idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_t[idx]
                    policy_loss = policy_loss + (-torch.minimum(surr1, surr2))
                    value_pred_clipped = torch.tensor(float(t.value), device=self.device) + (new_value - torch.tensor(float(t.value), device=self.device)).clamp(-self.clip_range, self.clip_range)
                    vf1 = (new_value - returns_t[idx]).pow(2)
                    vf2 = (value_pred_clipped - returns_t[idx]).pow(2)
                    value_loss = value_loss + 0.5 * torch.maximum(vf1, vf2)
                    entropy_bonus = entropy_bonus + (-new_logprob)
                    if self.teacher_bc_weight > 0.0 and int(t.step_idx) <= self.teacher_bc_max_step:
                        teacher = self.teacher_steps.get((int(t.seed), int(t.step_idx)))
                        if teacher is not None and teacher.completion_ids:
                            teacher_logprob, _ = _compute_action_stats(
                                self.actor_critic,
                                prompt_ids=t.prompt_ids,
                                completion_ids=list(teacher.completion_ids),
                                max_prompt_tokens=32,
                                max_completion_tokens=12,
                            )
                            teacher_bc_loss = teacher_bc_loss + (-teacher_logprob)
                denom = float(max(len(batch_idx), 1))
                loss = (
                    (policy_loss / denom)
                    + self.vf_coef * (value_loss / denom)
                    - self.ent_coef * (entropy_bonus / denom)
                    + self.teacher_bc_weight * (teacher_bc_loss / denom)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.actor_critic.parameters() if p.requires_grad], self.max_grad_norm)
                self.optimizer.step()
                total_policy += float((policy_loss / denom).detach().cpu())
                total_value += float((value_loss / denom).detach().cpu())
                total_entropy += float((entropy_bonus / denom).detach().cpu())
                total_teacher_bc += float((teacher_bc_loss / denom).detach().cpu())
                total_batches += 1
        return {
            "policy_loss": total_policy / float(max(total_batches, 1)),
            "value_loss": total_value / float(max(total_batches, 1)),
            "entropy": total_entropy / float(max(total_batches, 1)),
            "teacher_bc_loss": total_teacher_bc / float(max(total_batches, 1)),
            "avg_return": float(sum(returns) / float(max(len(returns), 1))),
        }

    def _save_checkpoint(self) -> None:
        self.actor_critic.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        torch.save({"value_head": self.actor_critic.value_head.state_dict()}, self.output_dir / "value_head.pt")

    def train(
        self,
        *,
        train_seeds: list[int],
        dev_seeds: list[int],
        updates: int,
        episodes_per_update: int,
        episodes_per_seed: int = 1,
    ) -> dict[str, Any]:
        metrics: list[dict[str, Any]] = []
        rng = random.Random(42)
        episodes_per_seed = max(1, int(episodes_per_seed))
        _save_json(self.output_dir / "online_status.json", {"phase": "startup", "updates": int(updates)})
        for pre_idx in range(self.expert_pretrain_updates):
            _save_json(self.output_dir / "online_status.json", {"phase": "expert_pretrain", "current_update": int(pre_idx), "total_updates": int(self.expert_pretrain_updates)})
            logger.info("expert_pretrain[%s]: sampling batch", int(pre_idx))
            batch = self._sample_expert_batch(rng)
            if not batch:
                break
            _save_json(
                self.output_dir / "online_status.json",
                {
                    "phase": "expert_pretrain",
                    "current_update": int(pre_idx),
                    "total_updates": int(self.expert_pretrain_updates),
                    "subphase": "backward",
                    "expert_records": int(len(batch)),
                },
            )
            logger.info("expert_pretrain[%s]: backward batch_size=%s", int(pre_idx), int(len(batch)))
            self.optimizer.zero_grad(set_to_none=True)
            expert_loss = self._backward_expert_batch(batch)
            logger.info("expert_pretrain[%s]: optimizer step", int(pre_idx))
            torch.nn.utils.clip_grad_norm_([p for p in self.actor_critic.parameters() if p.requires_grad], self.max_grad_norm)
            self.optimizer.step()
            metric = {"phase": "expert_pretrain", "update": int(pre_idx), "loss": float(expert_loss or 0.0), "expert_records": int(len(batch))}
            metrics.append(metric)
            _save_json(self.output_dir / "online_metrics.json", {"updates": metrics})
            _save_json(
                self.output_dir / "online_status.json",
                {
                    "phase": "expert_pretrain",
                    "current_update": int(pre_idx),
                    "total_updates": int(self.expert_pretrain_updates),
                    "subphase": "checkpoint",
                    "loss": float(expert_loss or 0.0),
                    "expert_records": int(len(batch)),
                },
            )
            logger.info("expert_pretrain[%s]: save checkpoint", int(pre_idx))
            self._save_checkpoint()
        for update_idx in range(int(updates)):
            _save_json(self.output_dir / "online_status.json", {"phase": "rollout", "current_update": int(update_idx), "total_updates": int(updates)})
            selected: list[int] = []
            while len(selected) < int(episodes_per_update):
                chosen_seed = int(rng.choice(train_seeds))
                block = min(episodes_per_seed, int(episodes_per_update) - len(selected))
                selected.extend([chosen_seed] * int(block))
            logger.info("rollout[%s]: selected seeds=%s", int(update_idx), selected)
            all_transitions: list[PPOTransition] = []
            episodes: list[dict[str, Any]] = []
            for seed in selected:
                _save_json(
                    self.output_dir / "online_status.json",
                    {
                        "phase": "rollout",
                        "current_update": int(update_idx),
                        "total_updates": int(updates),
                        "subphase": "episode",
                        "seed": int(seed),
                    },
                )
                episode, transitions = self._rollout_seed(int(seed))
                episodes.append({"seed": int(seed), "success": bool(episode.get("success")), "score": float(episode.get("score") or 0.0)})
                all_transitions.extend(transitions)
                _save_json(self.output_dir / "episodes" / f"update_{int(update_idx):04d}_seed_{int(seed):04d}.json", episode)
            if not all_transitions:
                continue
            _save_json(
                self.output_dir / "online_status.json",
                {
                    "phase": "rollout",
                    "current_update": int(update_idx),
                    "total_updates": int(updates),
                    "subphase": "ppo_update",
                    "transitions": int(len(all_transitions)),
                },
            )
            logger.info("rollout[%s]: ppo update transitions=%s", int(update_idx), int(len(all_transitions)))
            ppo_stats = self._ppo_update(all_transitions, rng=rng)
            success_rate = sum(1 for e in episodes if e["success"]) / float(max(len(episodes), 1))
            avg_score = sum(float(e["score"]) for e in episodes) / float(max(len(episodes), 1))
            metric = {
                "update": int(update_idx),
                "success_rate": float(success_rate),
                "avg_score": float(avg_score),
                "policy_loss": float(ppo_stats["policy_loss"]),
                "value_loss": float(ppo_stats["value_loss"]),
                "entropy": float(ppo_stats["entropy"]),
                "teacher_bc_loss": float(ppo_stats["teacher_bc_loss"]),
                "avg_return": float(ppo_stats["avg_return"]),
                "transitions": int(len(all_transitions)),
                "judge_enabled": bool(self.judge_enabled),
            }
            metrics.append(metric)
            _save_json(self.output_dir / "online_metrics.json", {"updates": metrics})
            _save_json(self.output_dir / "online_status.json", {"phase": "done_update", "current_update": int(update_idx), "success_rate": success_rate, "avg_score": avg_score})
            self._save_checkpoint()
        summary = {"output_dir": str(self.output_dir), "updates": metrics, "dev_seeds": list(dev_seeds)}
        _save_json(self.output_dir / "online_summary.json", summary)
        _save_json(self.output_dir / "online_status.json", {"phase": "done", "updates": len(metrics)})
        return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="PPO trainer for CONTACT with optional GPT-4o judge shaping.")
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-seeds", default="1..50")
    parser.add_argument("--dev-seeds", default="51..60")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--first-step-samples", type=int, default=1)
    parser.add_argument("--first-step-temperature", type=float, default=1.0)
    parser.add_argument("--first-step-top-p", type=float, default=0.98)
    parser.add_argument("--expert-run-glob", action="append", default=[])
    parser.add_argument("--expert-pretrain-updates", type=int, default=0)
    parser.add_argument("--expert-batch-size", type=int, default=8)
    parser.add_argument("--teacher-bc-weight", type=float, default=0.0)
    parser.add_argument("--teacher-bc-max-step", type=int, default=0)
    parser.add_argument("--judge-enabled", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    trainer = ContactPPOTrainer(
        base_model=str(args.base_model),
        output_dir=str(args.output_dir),
        adapter_path=str(args.adapter_path or ""),
        lr=float(args.lr),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_range),
        vf_coef=float(args.vf_coef),
        ent_coef=float(args.ent_coef),
        max_grad_norm=float(args.max_grad_norm),
        ppo_epochs=int(args.ppo_epochs),
        minibatch_size=int(args.minibatch_size),
        max_steps=int(args.max_steps),
        max_new_tokens=int(args.max_new_tokens),
        max_prompt_tokens=int(args.max_prompt_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        first_step_samples=int(args.first_step_samples),
        first_step_temperature=float(args.first_step_temperature),
        first_step_top_p=float(args.first_step_top_p),
        expert_patterns=list(args.expert_run_glob or []),
        expert_pretrain_updates=int(args.expert_pretrain_updates),
        expert_batch_size=int(args.expert_batch_size),
        teacher_bc_weight=float(args.teacher_bc_weight),
        teacher_bc_max_step=int(args.teacher_bc_max_step),
        judge_enabled=bool(args.judge_enabled),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    summary = trainer.train(
        train_seeds=_parse_seeds(str(args.train_seeds)),
        dev_seeds=_parse_seeds(str(args.dev_seeds)),
        updates=int(args.updates),
        episodes_per_update=int(args.episodes_per_update),
        episodes_per_seed=int(args.episodes_per_seed),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
