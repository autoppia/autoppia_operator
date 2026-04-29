#!/usr/bin/env python3
"""Serve fine-tuned BU-30B with LoRA adapter via HF or vLLM.

Usage:
    python -m training.serve_model [--port 8000] [--adapter-path models/bu-30b-lora]

This starts an OpenAI-compatible API server that the operator can query
through OPENAI_BASE_URL / OPENAI_MODEL.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASE_MODEL = "browser-use/bu-30b-a3b-preview"
_DEFAULT_ADAPTER_PATH = str(_REPO_ROOT / "models" / "bu-30b-lora")
_LEGACY_TRUSTED_ADAPTER_PATH = str(_REPO_ROOT / "models" / "bu-30b-login-500-lora")
DEFAULT_BASE_MODEL = _DEFAULT_BASE_MODEL
DEFAULT_ADAPTER_PATH = _DEFAULT_ADAPTER_PATH
DEFAULT_ADAPTER_CANDIDATES = (
    Path(_DEFAULT_ADAPTER_PATH),
    Path(_LEGACY_TRUSTED_ADAPTER_PATH),
)


def _looks_like_default_adapter_request(adapter_path: Path) -> bool:
    try:
        return adapter_path.resolve() == Path(DEFAULT_ADAPTER_PATH).resolve()
    except FileNotFoundError:
        return str(adapter_path) == DEFAULT_ADAPTER_PATH


def resolve_adapter_path(adapter_path: str) -> Path:
    candidate = Path(adapter_path).expanduser()
    if not candidate.is_absolute():
        candidate = (_REPO_ROOT / candidate).resolve()
    if not _looks_like_default_adapter_request(candidate):
        return candidate
    for known in DEFAULT_ADAPTER_CANDIDATES:
        if known.exists():
            return known.resolve()
    return candidate


def _adapter_recovery_hint(adapter_path: Path) -> str:
    looked_in = ", ".join(str(path) for path in DEFAULT_ADAPTER_CANDIDATES)
    return (
        "Expected a real LoRA bundle containing adapter_config.json and "
        "adapter_model.safetensors.\n"
        f"Resolved adapter path: {adapter_path}\n"
        f"Known local adapter locations: {looked_in}\n"
        "If the bundle was trained on RunPod, re-download it with "
        "`python -m training.runpod_job --existing-pod-id <pod-id> "
        "--output-dir models/bu-30b-lora --remote-adapter-dir "
        "/workspace/autoppia_operator_run/models/bu-30b-lora --download-only --keep-pod`."
    )


def missing_runtime_dependencies(backend: str) -> list[str]:
    required = ["fastapi", "uvicorn"]
    if backend == "hf":
        required.extend(["torch", "transformers", "peft"])
    elif backend == "vllm":
        required.append("vllm")
    missing: list[str] = []
    for module_name in required:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def build_preflight_report(*, backend: str, adapter_path: Path) -> dict[str, object]:
    missing_adapter_files = [name for name in ("adapter_config.json", "adapter_model.safetensors") if not (adapter_path / name).exists()]
    return {
        "backend": backend,
        "adapter_path": str(adapter_path),
        "adapter_exists": adapter_path.exists(),
        "missing_adapter_files": missing_adapter_files,
        "missing_runtime_dependencies": missing_runtime_dependencies(backend),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "default_adapter_candidates": [str(path) for path in DEFAULT_ADAPTER_CANDIDATES],
    }


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
        raise RuntimeError(f"Refusing to serve suspiciously small adapter weights from {model_path} ({model_path.stat().st_size} bytes)")

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
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate adapter/runtime prerequisites and exit without starting the server",
    )
    args = parser.parse_args()

    adapter_path = resolve_adapter_path(args.adapter_path)
    preflight = build_preflight_report(backend=args.backend, adapter_path=adapter_path)
    missing_deps = list(preflight["missing_runtime_dependencies"])
    if args.preflight:
        try:
            validate_adapter_artifacts(adapter_path)
            preflight["adapter_ready"] = True
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
            preflight["adapter_ready"] = False
            preflight["adapter_error"] = str(exc)
        print(json.dumps(preflight, indent=2))
        sys.exit(0 if preflight.get("adapter_ready") and not missing_deps else 1)

    if missing_deps:
        print(
            f"Error: missing runtime dependencies for {args.backend} backend: {', '.join(missing_deps)}",
            file=sys.stderr,
        )
        print(
            "Install the local ML serving stack in the operator environment before starting the model server.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        validate_adapter_artifacts(adapter_path)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(_adapter_recovery_hint(adapter_path), file=sys.stderr)
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
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.base_model,
            "--served-model-name",
            args.served_model_name,
            "--enable-lora",
            "--lora-modules",
            f"{args.served_model_name}={adapter_path}",
            "--port",
            str(args.port),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--max-model-len",
            str(args.max_model_len),
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
    print(f"  export OPENAI_BASE_URL=http://127.0.0.1:{args.port}/v1")
    print(f"  export OPENAI_MODEL={args.served_model_name}")
    print(f"  export AGENT_COMPLETION_MODEL={args.served_model_name}")
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
