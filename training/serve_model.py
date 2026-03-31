#!/usr/bin/env python3
"""Serve fine-tuned BU-30B with LoRA adapter via HF or vLLM.

Usage:
    python -m training.serve_model [--port 8000] [--adapter-path models/bu-30b-lora]

This starts an OpenAI-compatible API server that the BUPolicy can query.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASE_MODEL = "browser-use/bu-30b-a3b-preview"
_DEFAULT_ADAPTER_PATH = str(_REPO_ROOT / "models" / "bu-30b-lora")
DEFAULT_BASE_MODEL = _DEFAULT_BASE_MODEL
DEFAULT_ADAPTER_PATH = _DEFAULT_ADAPTER_PATH


def validate_adapter_artifacts(adapter_path: Path) -> None:
    config_path = adapter_path / "adapter_config.json"
    model_path = adapter_path / "adapter_model.safetensors"
    metrics_path = adapter_path / "train_metrics.json"

    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("stub") is True:
        raise RuntimeError(f"Refusing to serve stub adapter declared in {config_path}")

    if model_path.stat().st_size < 1024:
        raise RuntimeError(
            f"Refusing to serve suspiciously small adapter weights from {model_path} "
            f"({model_path.stat().st_size} bytes)"
        )

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("stub") is True or "stub" in str(metrics.get("note", "")).lower():
            raise RuntimeError(f"Refusing to serve adapter with stub training metrics in {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve BU-30B + LoRA via HF or vLLM")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--base-model", default=_DEFAULT_BASE_MODEL, help="Base model name/path")
    parser.add_argument("--adapter-path", default=_DEFAULT_ADAPTER_PATH, help="LoRA adapter directory")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="Serving backend")
    parser.add_argument(
        "--served-model-name",
        default="autoppia",
        help="Model identifier exposed by the OpenAI-compatible API",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory fraction")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--max-num-seqs", type=int, default=0, help="Optional vLLM max concurrent sequences")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=0,
        help="Optional vLLM max batched tokens override",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and cudagraph capture for unstable pod runtimes",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path).resolve()
    try:
        validate_adapter_artifacts(adapter_path)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.backend == "hf":
        cmd = [
            sys.executable,
            "-m",
            "training.hf_openai_server",
            "--base-model",
            args.base_model,
            "--adapter-path",
            str(adapter_path),
            "--port",
            str(args.port),
            "--served-model-name",
            args.served_model_name,
        ]
    else:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.base_model,
            "--served-model-name", args.served_model_name,
            "--enable-lora",
            "--lora-modules", f"{args.served_model_name}={adapter_path}",
            "--port", str(args.port),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--trust-remote-code",
        ]
        if args.max_num_seqs > 0:
            cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
        if args.max_num_batched_tokens > 0:
            cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
        if args.enforce_eager:
            cmd.append("--enforce-eager")

    print(f"Starting {args.backend} server: {' '.join(cmd)}")
    print(f"Adapter: {adapter_path}")
    print(f"API endpoint: http://localhost:{args.port}/v1")
    print()
    print("Usage with operator:")
    print(f"  export FSM_POLICY=learned")
    print(f"  export BU_POLICY_ENDPOINT=http://localhost:{args.port}/v1")
    print(f"  export BU_POLICY_MODEL={args.served_model_name}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(f"Error: {args.backend} backend dependencies are missing", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
