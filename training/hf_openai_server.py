#!/usr/bin/env python3
"""Serve the fine-tuned Browser Use adapter through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from training.serve_model import DEFAULT_ADAPTER_PATH, DEFAULT_BASE_MODEL, validate_adapter_artifacts


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type == "text":
                    parts.append(str(item.get("text") or ""))
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        parts.append(f"[image:{image_url.get('url') or ''}]")
                    else:
                        parts.append(f"[image:{image_url or ''}]")
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return _flatten_message_content(content.get("text") or content.get("content"))
    return str(content or "")


class HFOpenAIServer:
    def __init__(self, *, base_model: str, adapter_path: Path, served_model_name: str) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.served_model_name = served_model_name
        self._lock = threading.Lock()
        self._model, self._processor = self._load_model()
        self.app = FastAPI(title="Autoppia HF OpenAI server")
        self._register_routes()

    def _load_model(self) -> tuple[Any, Any]:
        from peft import PeftModel

        config = AutoConfig.from_pretrained(self.base_model, trust_remote_code=True)
        common_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
        }
        if getattr(config, "model_type", "") == "qwen3_vl_moe":
            model = AutoModelForImageTextToText.from_pretrained(self.base_model, **common_kwargs)
            processor = AutoProcessor.from_pretrained(self.base_model, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.base_model, **common_kwargs)
            processor = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, str(self.adapter_path))
        model.eval()
        return model, processor

    def _generate(self, *, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        prompt_messages = [
            {
                "role": str(message.get("role") or "user"),
                "content": _flatten_message_content(message.get("content")),
            }
            for message in messages
        ]
        chat_text = self._processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(text=[chat_text], return_tensors="pt").to(self._model.device)
        with self._lock, torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _register_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "ok",
                "backend": "hf",
                "base_model": self.base_model,
                "adapter_path": str(self.adapter_path),
                "served_model_name": self.served_model_name,
            }

        @self.app.get("/v1/models")
        async def models() -> dict[str, Any]:
            now = int(time.time())
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.served_model_name,
                        "object": "model",
                        "created": now,
                        "owned_by": "autoppia",
                    }
                ],
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise HTTPException(status_code=400, detail="messages must be a non-empty list")

            max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 256)
            max_tokens = max(1, min(max_tokens, 512))
            content = await asyncio.to_thread(
                self._generate,
                messages=messages,
                max_new_tokens=max_tokens,
            )
            prompt_chars = sum(len(_flatten_message_content(message.get("content"))) for message in messages)
            completion_chars = len(content)
            prompt_tokens = max(1, prompt_chars // 4) if prompt_chars else 0
            completion_tokens = max(1, completion_chars // 4) if completion_chars else 0
            return {
                "id": f"chatcmpl-autoppia-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve BU-30B + LoRA via a Hugging Face OpenAI-compatible API")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--served-model-name", default="autoppia")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path).resolve()
    try:
        validate_adapter_artifacts(adapter_path)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    server = HFOpenAIServer(
        base_model=args.base_model,
        adapter_path=adapter_path,
        served_model_name=args.served_model_name,
    )
    uvicorn.run(server.app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
