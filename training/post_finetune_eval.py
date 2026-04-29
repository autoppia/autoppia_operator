"""Run Autocinema eval against a real fine-tuned Browser Use adapter.

This command is intentionally strict:
- it refuses to run against stub adapters
- it writes machine-readable raw + summary artifacts only after a real eval run

Usage:
    python -m training.post_finetune_eval \
        --launch-server \
        --project-id autocinema \
        --out data/autocinema_trajectory_harvest/post_finetune_eval.json \
        --summary-out data/autocinema_trajectory_harvest/post_finetune_eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from training.serve_model import (
    _DEFAULT_ADAPTER_PATH as DEFAULT_ADAPTER_PATH,
    DEFAULT_BASE_MODEL as SERVE_BASE_MODEL,
    resolve_adapter_path,
    validate_adapter_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "autocinema_trajectory_harvest" / "post_finetune_eval.json"
DEFAULT_SUMMARY_OUT = REPO_ROOT / "data" / "autocinema_trajectory_harvest" / "post_finetune_eval_summary.json"


def _wait_for_server(url: str, timeout_s: float = 180.0) -> None:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except urllib.error.URLError:
            pass
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for model server at {url}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_summary(
    *,
    endpoint: str,
    adapter_path: Path,
    raw_report: dict[str, Any],
    provider: str,
    model: str,
    raw_report_path: Path,
    success_threshold: float,
    avg_score_threshold: float,
) -> dict[str, Any]:
    episodes = raw_report.get("episodes") if isinstance(raw_report.get("episodes"), list) else []
    num_tasks = int(raw_report.get("num_tasks") or len(episodes))
    successes = int(raw_report.get("successes") or sum(1 for episode in episodes if bool(episode.get("success"))))
    avg_score = float(raw_report.get("avg_score") or 0.0)
    success_rate = float(raw_report.get("success_rate") or (successes / num_tasks if num_tasks else 0.0))
    failures = max(0, num_tasks - successes)
    successful_episode_ids = [
        str(episode.get("episode_task_id") or episode.get("task_id") or episode.get("id") or "") for episode in episodes if isinstance(episode, dict) and bool(episode.get("success"))
    ]
    successful_episode_ids = [episode_id for episode_id in successful_episode_ids if episode_id]
    use_cases = sorted({str(episode.get("use_case") or "") for episode in episodes if isinstance(episode, dict) and episode.get("use_case")})

    return {
        "model_endpoint": endpoint,
        "provider": provider,
        "served_model_identifier": model,
        "adapter_path": str(adapter_path),
        "raw_eval_artifact": str(raw_report_path),
        "tasks_evaluated": num_tasks,
        "successes": successes,
        "failures": failures,
        "avg_score": avg_score,
        "success_rate": success_rate,
        "success_threshold": success_threshold,
        "avg_score_threshold": avg_score_threshold,
        "passed_gate": success_rate >= success_threshold and avg_score >= avg_score_threshold,
        "use_cases_evaluated": use_cases,
        "successful_episode_ids": successful_episode_ids,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _validate_real_success(summary: dict[str, Any]) -> None:
    successful_episode_ids = summary.get("successful_episode_ids")
    if not isinstance(successful_episode_ids, list) or not successful_episode_ids:
        raise RuntimeError("Post-finetune eval did not produce any real successful Autocinema episode; refusing to write summary.")


def build_operator_env(
    *,
    endpoint: str,
    served_model_id: str,
    completion_model: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = endpoint
    env["OPENAI_MODEL"] = served_model_id
    env["AGENT_COMPLETION_MODEL"] = str(completion_model or served_model_id)
    # Compatibility-only exports for older wrappers that still read BU_POLICY_*.
    env["BU_POLICY_ENDPOINT"] = endpoint
    env["BU_POLICY_MODEL"] = served_model_id
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-finetune Autocinema eval against a real adapter")
    parser.add_argument("--project-id", default="autocinema")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--base-model", default=SERVE_BASE_MODEL)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model-id", default="autoppia")
    parser.add_argument("--completion-model", default="", help="Optional completion-check model override (defaults to served model)")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--server-backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--use-case")
    parser.add_argument("--all-use-cases", action="store_true")
    parser.add_argument("--tasks-per-use-case", type=int, default=1)
    parser.add_argument("--distinct-use-cases", action="store_true")
    parser.add_argument("--task-cache")
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--success-threshold", type=float, default=0.70)
    parser.add_argument("--avg-score-threshold", type=float, default=0.60)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    args = parser.parse_args()

    adapter_path = resolve_adapter_path(args.adapter_path)
    validate_adapter_artifacts(adapter_path)
    endpoint = args.endpoint
    if args.launch_server and args.endpoint == "http://127.0.0.1:8000/v1":
        endpoint = f"http://127.0.0.1:{args.port}/v1"

    env = build_operator_env(
        endpoint=endpoint,
        served_model_id=args.served_model_id,
        completion_model=(args.completion_model or None),
    )

    server_proc: subprocess.Popen[str] | None = None
    try:
        if args.launch_server:
            server_cmd = [
                sys.executable,
                "-m",
                "training.serve_model",
                "--backend",
                args.server_backend,
                "--base-model",
                args.base_model,
                "--adapter-path",
                str(adapter_path),
                "--port",
                str(args.port),
                "--served-model-name",
                args.served_model_id,
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-model-len",
                str(args.max_model_len),
            ]
            if args.max_num_seqs > 0:
                server_cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
            if args.max_num_batched_tokens > 0:
                server_cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
            if args.enforce_eager:
                server_cmd.append("--enforce-eager")
            server_proc = subprocess.Popen(
                server_cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _wait_for_server(f"http://127.0.0.1:{args.port}/health")

        out_path = Path(args.out).resolve()
        summary_out_path = Path(args.summary_out).resolve()
        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "sn36_ops.py"),
            "eval",
            "--project-id",
            args.project_id,
            "--provider",
            args.provider,
            "--model",
            args.served_model_id,
            "--num-tasks",
            str(args.num_tasks),
            "--max-steps",
            str(args.max_steps),
            "--repeat",
            str(args.repeat),
            "--task-concurrency",
            str(args.task_concurrency),
            "--success-threshold",
            str(args.success_threshold),
            "--avg-score-threshold",
            str(args.avg_score_threshold),
            "--out",
            str(out_path),
        ]
        if args.all_use_cases:
            eval_cmd.extend(["--all-use-cases", "--tasks-per-use-case", str(args.tasks_per_use_case)])
        if args.use_case:
            eval_cmd.extend(["--use-case", args.use_case])
        if args.distinct_use_cases:
            eval_cmd.append("--distinct-use-cases")
        if args.task_cache:
            eval_cmd.extend(["--task-cache", args.task_cache])

        subprocess.run(eval_cmd, cwd=REPO_ROOT, env=env, check=True)

        raw_report = _load_json(out_path)
        summary = _build_summary(
            endpoint=endpoint,
            adapter_path=adapter_path,
            raw_report=raw_report,
            provider=args.provider,
            model=args.served_model_id,
            raw_report_path=out_path,
            success_threshold=args.success_threshold,
            avg_score_threshold=args.avg_score_threshold,
        )
        _validate_real_success(summary)
        _write_json(summary_out_path, summary)
        print(json.dumps(summary, indent=2))
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
