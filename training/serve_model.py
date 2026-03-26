#!/usr/bin/env python3
"""Serve fine-tuned BU-30B with LoRA adapter via vLLM.

Usage:
    python -m training.serve_model [--port 8000] [--adapter-path models/bu-30b-lora]

This starts an OpenAI-compatible API server that the BUPolicy can query.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASE_MODEL = "browser-use/bu-30b-a3b-preview"
_DEFAULT_ADAPTER_PATH = str(_REPO_ROOT / "models" / "bu-30b-lora")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve BU-30B + LoRA via vLLM")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--base-model", default=_DEFAULT_BASE_MODEL, help="Base model name/path")
    parser.add_argument("--adapter-path", default=_DEFAULT_ADAPTER_PATH, help="LoRA adapter directory")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory fraction")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max sequence length")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path).resolve()
    if not (adapter_path / "adapter_config.json").exists():
        print(f"Error: adapter_config.json not found in {adapter_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.base_model,
        "--enable-lora",
        "--lora-modules", f"autoppia={adapter_path}",
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code",
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    print(f"Adapter: {adapter_path}")
    print(f"API endpoint: http://localhost:{args.port}/v1")
    print()
    print("Usage with operator:")
    print(f"  export FSM_POLICY=learned")
    print(f"  export BU_POLICY_ENDPOINT=http://localhost:{args.port}/v1")
    print(f"  export BU_POLICY_MODEL=autoppia")
    print()

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: vllm not installed. Install with: pip install vllm", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
