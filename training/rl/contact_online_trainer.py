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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig

from training.finetune_bu import MODEL_ID, _ensure_deps, _load_trainable_model
from training.rl.contact_env import ContactRLEnv, _load_reference_tool_calls

def _ensure_repo_src_package() -> None:
    src_init = REPO_ROOT / "src" / "__init__.py"
    src_operator_init = REPO_ROOT / "src" / "operator" / "__init__.py"
    if "src" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "src",
            src_init,
            submodule_search_locations=[str(src_init.parent)],
        )
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


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item or ""))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content or "")


def _format_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = _message_content_to_text(msg.get("content"))
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _extract_json_object(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return "{}"
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + float(gamma) * running
        out[idx] = running
    return out


@dataclass
class GenerationRecord:
    task_id: str
    messages: list[dict[str, Any]]
    prompt_text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_text: str


@dataclass
class ExpertRecord:
    prompt_text: str
    completion_ids: list[int]
    priority: float = 1.0
    step_idx: int = 0
    route_hint: str = ""


class LocalPolicyLLM:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 256,
        max_prompt_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.records: list[GenerationRecord] = []

    def _truncate_encoded_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(self.max_prompt_tokens) <= 0:
            return input_ids, attention_mask
        seq_len = int(input_ids.shape[1])
        if seq_len <= int(self.max_prompt_tokens):
            return input_ids, attention_mask
        head_tokens = min(512, int(self.max_prompt_tokens) // 4)
        tail_tokens = int(self.max_prompt_tokens) - head_tokens
        truncated_ids = torch.cat([input_ids[:, :head_tokens], input_ids[:, -tail_tokens:]], dim=1)
        truncated_mask = torch.cat([attention_mask[:, :head_tokens], attention_mask[:, -tail_tokens:]], dim=1)
        logger.info(
            "Truncating prompt for generation from %s to %s tokens (head=%s tail=%s)",
            seq_len,
            int(truncated_ids.shape[1]),
            head_tokens,
            tail_tokens,
        )
        return truncated_ids, truncated_mask

    @torch.no_grad()
    def __call__(self, *, task_id: str, messages: list[dict[str, Any]], model: str, temperature: float, max_tokens: int, **_: Any) -> dict[str, Any]:
        prompt_text = _format_messages(messages)
        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask = self._truncate_encoded_prompt(input_ids, attention_mask)
        encoded = {"input_ids": input_ids, "attention_mask": attention_mask}
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        do_sample = float(temperature or self.temperature) > 1e-5
        out = self.model.generate(
            **encoded,
            max_new_tokens=min(int(max_tokens or self.max_new_tokens), self.max_new_tokens),
            do_sample=do_sample,
            temperature=max(float(temperature or self.temperature), 1e-5) if do_sample else 1.0,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = int(encoded["input_ids"].shape[1])
        completion_ids = out[0, prompt_len:].tolist()
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        truncated_prompt_text = self.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
        self.records.append(
            GenerationRecord(
                task_id=str(task_id),
                messages=list(messages),
                prompt_text=truncated_prompt_text,
                prompt_ids=encoded["input_ids"][0].tolist(),
                completion_ids=completion_ids,
                completion_text=completion_text,
            )
        )
        usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(len(completion_ids)),
            "total_tokens": int(prompt_len + len(completion_ids)),
        }
        return {"choices": [{"message": {"content": completion_text}}], "usage": usage, "model": str(model or "")}

    def pop_last_record(self) -> GenerationRecord | None:
        if not self.records:
            return None
        return self.records.pop(0)


def _compute_completion_logprob(
    model: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    max_prompt_tokens: int = 128,
    max_completion_tokens: int = 16,
) -> torch.Tensor:
    if not completion_ids:
        return torch.zeros((), device=model.device)
    prompt_tail = list(prompt_ids[-int(max_prompt_tokens) :]) if max_prompt_tokens > 0 else list(prompt_ids)
    completion_tail = list(completion_ids[: int(max_completion_tokens)]) if max_completion_tokens > 0 else list(completion_ids)
    if not completion_tail:
        return torch.zeros((), device=model.device)
    torch.cuda.empty_cache()
    full_ids = torch.tensor([prompt_tail + completion_tail], dtype=torch.long, device=model.device)
    outputs = model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :]
    targets = full_ids[:, 1:]
    prompt_len = len(prompt_tail)
    start = max(prompt_len - 1, 0)
    end = start + len(completion_tail)
    token_logprobs = torch.log_softmax(logits[:, start:end, :], dim=-1)
    chosen = targets[:, start:end]
    return token_logprobs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1).sum()


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_status(path: Path, **payload: Any) -> None:
    _save_json(path, payload)


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


def _load_expert_records(patterns: list[str], tokenizer: Any) -> list[ExpertRecord]:
    records: list[ExpertRecord] = []
    for pattern in patterns:
        for raw_path in sorted(glob.glob(pattern)):
            path = Path(raw_path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Failed to read expert run %s", path)
                continue
            for episode in payload.get("episodes", []) or []:
                for step in episode.get("guided_execution", []) or []:
                    prompt_text = str(step.get("policy_input_text") or "")
                    tool_call = step.get("policy_tool_call") or {}
                    if not prompt_text or not tool_call:
                        continue
                    target_obj = {"type": "browser", "tool_call": tool_call}
                    completion_text = json.dumps(target_obj, ensure_ascii=False)
                    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
                    if not completion_ids:
                        continue
                    tool_name = str(tool_call.get("tool") or "").strip().lower()
                    arguments = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                    route_hint = str(arguments.get("url") or "").lower()
                    priority = 1.0
                    if tool_name == "browser.navigate":
                        priority = 3.0
                    elif tool_name == "browser.click" and any(token in route_hint for token in ("/contact", "/login", "/register", "/search", "/cart")):
                        priority = 2.0
                    records.append(
                        ExpertRecord(
                            prompt_text=prompt_text,
                            completion_ids=list(completion_ids),
                            priority=float(priority),
                            step_idx=int(step.get("step_idx") or 0),
                            route_hint=str(route_hint),
                        )
                    )
    return records


def _pathish(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        from urllib.parse import urlparse

        return str(urlparse(value).path or "").lower()
    return value


class ContactOnlineTrainer:
    def __init__(
        self,
        *,
        base_model: str,
        output_dir: str,
        adapter_path: str = "",
        lr: float = 5e-6,
        gamma: float = 0.99,
        max_steps: int = 12,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_prompt_tokens: int = 2048,
        bf16: bool = True,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        expert_patterns: list[str] | None = None,
        expert_weight: float = 0.0,
        expert_batch_size: int = 4,
        expert_pretrain_updates: int = 0,
        expert_logprob_prompt_tokens: int = 96,
        expert_logprob_completion_tokens: int = 12,
        rollout_logprob_prompt_tokens: int = 32,
        rollout_logprob_completion_tokens: int = 8,
        train_prefix_steps: int = 2,
        advantage_clip: float = 10.0,
        expert_max_step_idx: int = 0,
        expert_route_contains: str = "",
        teacher_correction_weight: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        _ensure_deps()
        os.environ.setdefault("FSM_OBS_EXTRACT_MODE", "off")
        self.base_model = str(base_model)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        self.expert_weight = float(expert_weight)
        self.expert_batch_size = max(int(expert_batch_size), 1)
        self.expert_pretrain_updates = max(int(expert_pretrain_updates), 0)
        self.expert_logprob_prompt_tokens = max(int(expert_logprob_prompt_tokens), 16)
        self.expert_logprob_completion_tokens = max(int(expert_logprob_completion_tokens), 4)
        self.rollout_logprob_prompt_tokens = max(int(rollout_logprob_prompt_tokens), 8)
        self.rollout_logprob_completion_tokens = max(int(rollout_logprob_completion_tokens), 4)
        self.train_prefix_steps = max(int(train_prefix_steps), 1)
        self.advantage_clip = max(float(advantage_clip), 0.0)
        self.expert_max_step_idx = max(int(expert_max_step_idx), 0)
        self.expert_route_contains = str(expert_route_contains or "").strip().lower()
        self.teacher_correction_weight = max(float(teacher_correction_weight), 0.0)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = _load_trainable_model(self.base_model, bnb_config=bnb_config, bf16=bf16)
        model.config.use_cache = False
        if adapter_path:
            logger.info("Loading bootstrap adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora_rank),
                lora_alpha=int(lora_alpha),
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
        self.model = model
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if bool(gradient_checkpointing) and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=float(lr),
        )
        self.local_llm = LocalPolicyLLM(
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=int(max_new_tokens),
            max_prompt_tokens=int(max_prompt_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        self.step_engine = StepEngine(llm_call=self.local_llm, vision_call=None)
        self.expert_records = _load_expert_records(list(expert_patterns or []), self.tokenizer)
        if self.expert_records:
            logger.info("Loaded %s expert records", len(self.expert_records))

    def _sample_expert_batch(self, rng: random.Random) -> list[ExpertRecord]:
        if not self.expert_records:
            return []
        eligible = [record for record in self.expert_records if int(record.step_idx) <= int(self.expert_max_step_idx)]
        if self.expert_route_contains:
            routed = [record for record in eligible if self.expert_route_contains in str(record.route_hint or "").lower()]
            if routed:
                eligible = routed
        if not eligible:
            eligible = list(self.expert_records)
        if len(eligible) <= self.expert_batch_size:
            return list(eligible)
        weights = [max(float(record.priority), 0.1) for record in eligible]
        return rng.choices(eligible, weights=weights, k=self.expert_batch_size)

    def _backward_expert_batch(self, sample: list[ExpertRecord], *, scale: float = 1.0) -> float | None:
        if not sample:
            return None
        sample_size = max(len(sample), 1)
        total_raw_loss = 0.0
        for record in sample:
            prompt_ids = self.tokenizer(record.prompt_text, add_special_tokens=False)["input_ids"]
            expert_logprob = _compute_completion_logprob(
                self.model,
                prompt_ids,
                record.completion_ids,
                max_prompt_tokens=self.expert_logprob_prompt_tokens,
                max_completion_tokens=self.expert_logprob_completion_tokens,
            )
            raw_loss = -expert_logprob
            total_raw_loss += float(raw_loss.detach().cpu())
            scaled_loss = raw_loss * (float(scale) / float(sample_size))
            scaled_loss.backward()
            del expert_logprob, raw_loss, scaled_loss
        return total_raw_loss / float(sample_size)

    def _rollout_seed(self, seed: int) -> tuple[dict[str, Any], list[GenerationRecord]]:
        env = ContactRLEnv(
            seed=int(seed),
            model_override="",
            max_steps=self.max_steps,
            step_engine_instance=self.step_engine,
        )
        episode_records: list[GenerationRecord] = []

        async def _run() -> dict[str, Any]:
            await env.reset()
            try:
                done = False
                for step_index in range(self.max_steps):
                    started_at = time.perf_counter()
                    _, done = await env.step(step_index)
                    elapsed = time.perf_counter() - started_at
                    generation_record = self.local_llm.pop_last_record()
                    if generation_record is not None:
                        episode_records.append(generation_record)
                    step_payload = env.rollout_steps[-1].to_dict() if env.rollout_steps else {}
                    logger.info(
                        "rollout seed=%s step=%s elapsed=%.2fs action=%s reward=%.3f after_url=%s",
                        int(seed),
                        int(step_index),
                        float(elapsed),
                        json.dumps((step_payload.get("chosen_action") or {}), ensure_ascii=False, sort_keys=True),
                        float((((step_payload.get("reward") if isinstance(step_payload, dict) else {}) or {}).get("total") or 0.0)),
                        str(step_payload.get("after_url") or ""),
                    )
                    if done:
                        break
                final = env.last_step_result
                if final is None:
                    raise RuntimeError("missing final step result")
                return {
                    "seed": int(seed),
                    "task_id": str(env.task.id),
                    "prompt": str(env.task.prompt),
                    "success": bool(final.score.success),
                    "score": float(final.score.raw_score),
                    "steps": [item.to_dict() for item in env.rollout_steps],
                }
            finally:
                await env.close()

        return (asyncio.run(_run()), episode_records)

    def train(self, *, train_seeds: list[int], dev_seeds: list[int], updates: int, episodes_per_update: int) -> dict[str, Any]:
        metrics: list[dict[str, Any]] = []
        rng = random.Random(42)
        _save_status(
            self.output_dir / "online_status.json",
            phase="startup",
            base_model=self.base_model,
            output_dir=str(self.output_dir),
            expert_records=int(len(self.expert_records)),
            train_seeds=list(train_seeds),
            dev_seeds=list(dev_seeds),
            updates=int(updates),
            episodes_per_update=int(episodes_per_update),
        )
        for pre_idx in range(self.expert_pretrain_updates):
            _save_status(
                self.output_dir / "online_status.json",
                phase="expert_pretrain",
                current_update=int(pre_idx),
                total_updates=int(self.expert_pretrain_updates),
            )
            sample = self._sample_expert_batch(rng)
            if not sample:
                break
            self.optimizer.zero_grad(set_to_none=True)
            expert_loss_value = self._backward_expert_batch(sample, scale=1.0)
            if expert_loss_value is None:
                break
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
            self.optimizer.step()
            metric = {
                "phase": "expert_pretrain",
                "update": int(pre_idx),
                "loss": float(expert_loss_value),
                "success_rate": None,
                "avg_score": None,
                "expert_records": int(len(sample)),
            }
            metrics.append(metric)
            _save_json(self.output_dir / "online_metrics.json", {"updates": metrics})
            logger.info("expert_pretrain=%s loss=%.4f", pre_idx, metric["loss"])
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
        for update_idx in range(int(updates)):
            _save_status(
                self.output_dir / "online_status.json",
                phase="rollout",
                current_update=int(update_idx),
                total_updates=int(updates),
            )
            selected = [rng.choice(train_seeds) for _ in range(int(episodes_per_update))]
            step_examples: list[tuple[list[int], list[int], float, int]] = []
            teacher_correction_examples: list[tuple[list[int], list[int], int]] = []
            update_summary = {"update": int(update_idx), "seeds": selected, "episodes": []}
            for seed in selected:
                episode, generation_records = self._rollout_seed(int(seed))
                reference_tool_calls = _load_reference_tool_calls(seed=int(seed))
                _save_json(self.output_dir / "episodes" / f"update_{int(update_idx):04d}_seed_{int(seed):04d}.json", episode)
                _save_status(
                    self.output_dir / "online_status.json",
                    phase="rollout_done",
                    current_update=int(update_idx),
                    seed=int(seed),
                    success=bool(episode.get("success")),
                    score=float(episode.get("score") or 0.0),
                )
                rewards = [float(((step.get("reward") if isinstance(step, dict) else {}) or {}).get("total") or 0.0) for step in episode.get("steps", [])]
                returns = _discounted_returns(rewards, self.gamma)
                update_summary["episodes"].append({"seed": int(seed), "success": bool(episode.get("success")), "score": float(episode.get("score") or 0.0)})
                step_rows = list(episode.get("steps", []))
                for idx, (step_payload, ret) in enumerate(zip(step_rows, returns, strict=False)):
                    gen_record = generation_records[idx] if idx < len(generation_records) else None
                    chosen_action = (step_payload or {}).get("env_action") or (step_payload or {}).get("chosen_action") or {}
                    if gen_record is not None:
                        prompt_ids = list(gen_record.prompt_ids)
                    else:
                        prompt_text = str((step_payload or {}).get("policy_input_text") or "")
                        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                    # Train against the canonical executed action rather than the
                    # raw free-form completion text, because reward is assigned to
                    # the parsed browser action, not arbitrary raw text variants.
                    completion_text = json.dumps(chosen_action, ensure_ascii=False, sort_keys=True)
                    completion_ids = self.tokenizer(completion_text, add_special_tokens=False)["input_ids"]
                    if not completion_ids:
                        continue
                    step_examples.append((prompt_ids, completion_ids, float(ret), int(idx)))
                    reward_payload = (step_payload or {}).get("reward") if isinstance(step_payload, dict) else {}
                    teacher_reason = str(
                        (step_payload or {}).get("teacher_reason")
                        or ((reward_payload or {}).get("teacher_reason") if isinstance(reward_payload, dict) else "")
                        or ""
                    )
                    teacher_mismatch = bool(teacher_reason and "match" not in teacher_reason)
                    if idx < len(reference_tool_calls):
                        ref_call = reference_tool_calls[idx]
                        ref_name = str(ref_call.get("name") or "").strip().lower()
                        ref_args = ref_call.get("arguments") if isinstance(ref_call.get("arguments"), dict) else {}
                        if idx == 0 and ref_name == "browser.navigate":
                            expected_path = _pathish(ref_args.get("url"))
                            chosen_type = str(chosen_action.get("type") or "")
                            chosen_url = _pathish(str(chosen_action.get("url") or ""))
                            if expected_path and not (
                                (chosen_type == "NavigateAction" and expected_path in chosen_url)
                                or (chosen_type == "ClickAction" and expected_path in chosen_url)
                            ):
                                teacher_mismatch = True
                    if (
                        self.teacher_correction_weight > 0.0
                        and idx < int(self.train_prefix_steps)
                        and idx < len(reference_tool_calls)
                        and teacher_mismatch
                    ):
                        reference_tool_call = reference_tool_calls[idx]
                        target_obj = {"type": "browser", "tool_call": reference_tool_call}
                        target_text = json.dumps(target_obj, ensure_ascii=False)
                        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
                        if target_ids:
                            teacher_correction_examples.append((prompt_ids, target_ids, int(idx)))
            if not step_examples:
                logger.warning("No valid losses at update %s", update_idx)
                continue
            step_examples = [item for item in step_examples if int(item[3]) < int(self.train_prefix_steps)]
            if not step_examples:
                logger.warning("No valid prefix-step losses at update %s", update_idx)
                continue
            returns_tensor = torch.tensor([item[2] for item in step_examples], dtype=torch.float32, device=self.model.device)
            returns_mean = returns_tensor.mean()
            returns_std = returns_tensor.std(unbiased=False)
            # Keep a non-zero policy gradient signal when prefix training leaves us
            # with a single step or near-constant returns.
            if int(len(step_examples)) <= 1 or float(returns_std.detach().cpu()) <= 1e-6:
                advantages = returns_tensor.clone()
            else:
                advantages = (returns_tensor - returns_mean) / returns_std.clamp_min(1e-6)
            if self.advantage_clip > 0.0:
                advantages = advantages.clamp(min=-self.advantage_clip, max=self.advantage_clip)
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            sample_size = max(len(step_examples), 1)
            rollout_loss_value = 0.0
            for (prompt_ids, completion_ids, _, _step_idx), advantage in zip(step_examples, advantages, strict=False):
                logprob = _compute_completion_logprob(
                    self.model,
                    prompt_ids,
                    completion_ids,
                    max_prompt_tokens=self.rollout_logprob_prompt_tokens,
                    max_completion_tokens=self.rollout_logprob_completion_tokens,
                )
                step_loss = -(advantage.detach()) * logprob
                rollout_loss_value += float(step_loss.detach().cpu())
                scaled_step_loss = step_loss / float(sample_size)
                scaled_step_loss.backward()
                del logprob, step_loss, scaled_step_loss
            expert_loss_value = None
            expert_sample: list[ExpertRecord] = []
            if self.expert_records and self.expert_weight > 0.0:
                expert_sample = self._sample_expert_batch(rng)
                expert_loss_value = self._backward_expert_batch(expert_sample, scale=self.expert_weight)
            teacher_correction_loss_value = None
            if teacher_correction_examples and self.teacher_correction_weight > 0.0:
                sample_size = max(len(teacher_correction_examples), 1)
                total_raw_loss = 0.0
                for prompt_ids, completion_ids, _step_idx in teacher_correction_examples:
                    teacher_logprob = _compute_completion_logprob(
                        self.model,
                        prompt_ids,
                        completion_ids,
                        max_prompt_tokens=self.expert_logprob_prompt_tokens,
                        max_completion_tokens=self.expert_logprob_completion_tokens,
                    )
                    raw_loss = -teacher_logprob
                    total_raw_loss += float(raw_loss.detach().cpu())
                    scaled_loss = raw_loss * (float(self.teacher_correction_weight) / float(sample_size))
                    scaled_loss.backward()
                    del teacher_logprob, raw_loss, scaled_loss
                teacher_correction_loss_value = total_raw_loss / float(sample_size)
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
            self.optimizer.step()
            success_rate = sum(1 for item in update_summary["episodes"] if bool(item["success"])) / max(len(update_summary["episodes"]), 1)
            avg_score = sum(float(item["score"]) for item in update_summary["episodes"]) / max(len(update_summary["episodes"]), 1)
            total_metric_loss = float(rollout_loss_value) / float(sample_size)
            if expert_loss_value is not None:
                total_metric_loss += float(self.expert_weight) * float(expert_loss_value)
            if teacher_correction_loss_value is not None:
                total_metric_loss += float(self.teacher_correction_weight) * float(teacher_correction_loss_value)
            metric = {
                "update": int(update_idx),
                "loss": float(total_metric_loss),
                "success_rate": float(success_rate),
                "avg_score": float(avg_score),
                "avg_return": float(returns_mean.detach().cpu()),
                "adv_mean": float(advantages.mean().detach().cpu()),
                "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
                "rollout_examples": int(len(step_examples)),
                "expert_records": int(len(expert_sample)),
                "teacher_corrections": int(len(teacher_correction_examples)),
            }
            metrics.append(metric)
            _save_json(self.output_dir / "online_metrics.json", {"updates": metrics})
            logger.info("update=%s loss=%.4f success_rate=%.3f avg_score=%.3f", update_idx, metric["loss"], success_rate, avg_score)
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
        summary = {"base_model": self.base_model, "output_dir": str(self.output_dir), "updates": metrics, "dev_seeds": dev_seeds}
        _save_json(self.output_dir / "online_summary.json", summary)
        _save_status(self.output_dir / "online_status.json", phase="done", updates=len(metrics))
        return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Online CONTACT RL trainer (REINFORCE) on the real stateful evaluator.")
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-seeds", default="1..50")
    parser.add_argument("--dev-seeds", default="51..60")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--expert-run-glob", action="append", default=[])
    parser.add_argument("--expert-weight", type=float, default=0.0)
    parser.add_argument("--expert-batch-size", type=int, default=4)
    parser.add_argument("--expert-pretrain-updates", type=int, default=0)
    parser.add_argument("--expert-logprob-prompt-tokens", type=int, default=96)
    parser.add_argument("--expert-logprob-completion-tokens", type=int, default=12)
    parser.add_argument("--rollout-logprob-prompt-tokens", type=int, default=32)
    parser.add_argument("--rollout-logprob-completion-tokens", type=int, default=8)
    parser.add_argument("--train-prefix-steps", type=int, default=2)
    parser.add_argument("--advantage-clip", type=float, default=10.0)
    parser.add_argument("--expert-max-step-idx", type=int, default=0)
    parser.add_argument("--expert-route-contains", default="")
    parser.add_argument("--teacher-correction-weight", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()

    trainer = ContactOnlineTrainer(
        base_model=str(args.base_model),
        output_dir=str(args.output_dir),
        adapter_path=str(args.adapter_path or ""),
        lr=float(args.lr),
        gamma=float(args.gamma),
        max_steps=int(args.max_steps),
        max_new_tokens=int(args.max_new_tokens),
        max_prompt_tokens=int(args.max_prompt_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        bf16=not bool(args.no_bf16),
        expert_patterns=list(args.expert_run_glob or []),
        expert_weight=float(args.expert_weight),
        expert_batch_size=int(args.expert_batch_size),
        expert_pretrain_updates=int(args.expert_pretrain_updates),
        expert_logprob_prompt_tokens=int(args.expert_logprob_prompt_tokens),
        expert_logprob_completion_tokens=int(args.expert_logprob_completion_tokens),
        rollout_logprob_prompt_tokens=int(args.rollout_logprob_prompt_tokens),
        rollout_logprob_completion_tokens=int(args.rollout_logprob_completion_tokens),
        train_prefix_steps=int(args.train_prefix_steps),
        advantage_clip=float(args.advantage_clip),
        expert_max_step_idx=int(args.expert_max_step_idx),
        expert_route_contains=str(args.expert_route_contains or ""),
        teacher_correction_weight=float(args.teacher_correction_weight),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    summary = trainer.train(
        train_seeds=_parse_seeds(str(args.train_seeds)),
        dev_seeds=_parse_seeds(str(args.dev_seeds)),
        updates=int(args.updates),
        episodes_per_update=int(args.episodes_per_update),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
