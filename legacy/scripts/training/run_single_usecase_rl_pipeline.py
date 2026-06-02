#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.format_for_sft import RUNTIME_ALIGNED_SYSTEM_PROMPT, export_harvest_to_sft
from training.harvester import write_harvest_artifacts


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _episode_row_from_report(report: dict[str, Any], *, use_case: str, web_project_id: str, attempt_name: str, trace_dir: Path, out_path: Path) -> dict[str, Any] | None:
    episodes = report.get("episodes") if isinstance(report.get("episodes"), list) else []
    if not episodes:
        return None
    episode = episodes[0] if isinstance(episodes[0], dict) else {}
    episode_task_id = str(episode.get("episode_task_id") or "")
    trace_file = trace_dir / "episodes" / f"{episode_task_id}.json"
    return {
        "web_project_id": str(web_project_id),
        "task_id": str(episode.get("task_id") or ""),
        "episode_task_id": episode_task_id,
        "use_case": str(use_case).upper(),
        "seed": int(episode.get("seed") or 0),
        "success": bool(episode.get("success")),
        "score": float(episode.get("score") or 0.0),
        "steps": int(episode.get("steps") or 0),
        "model": str(episode.get("model") or ""),
        "prompt_tokens": int(episode.get("prompt_tokens") or 0),
        "completion_tokens": int(episode.get("completion_tokens") or 0),
        "total_tokens": int(episode.get("total_tokens") or 0),
        "estimated_cost_usd": float(episode.get("estimated_cost_usd") or 0.0),
        "result_path": str(trace_file if trace_file.exists() else out_path),
        "trace_file": str(trace_file),
        "trace_dir": str(trace_dir),
        "trace_root": str(trace_dir),
        "attempt_name": str(attempt_name),
        "harvest_mode": "heuristic_runtime",
        "notes": f"{web_project_id} {use_case} seed={int(episode.get('seed') or 0)}",
        "final_url": str(episode.get("final_url") or ""),
    }


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, **(env or {})}, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-use-case pipeline: heuristic harvest -> SFT -> PPO on clean inference")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--web-project-id", required=True)
    parser.add_argument("--use-case", required=True)
    parser.add_argument("--task-cache", required=True)
    parser.add_argument("--harvest-seeds", default="1..20")
    parser.add_argument("--holdout-seeds", default="21..25")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--heuristic-runtime", default="heuristic_structured")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--sft-base-model", default="browser-use/bu-30b-a3b-preview")
    parser.add_argument("--sft-mode", choices=["local", "runpod", "export_only"], default="local")
    parser.add_argument("--runpod-existing-pod-id", default="")
    parser.add_argument("--runpod-inventory-snapshot-path", default="")
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--sft-lora-rank", type=int, default=32)
    parser.add_argument("--sft-batch-size", type=int, default=1)
    parser.add_argument("--sft-grad-accum", type=int, default=16)
    parser.add_argument("--skip-harvest", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--ppo-base-model", default="browser-use/bu-30b-a3b-preview")
    parser.add_argument("--ppo-updates", type=int, default=4)
    parser.add_argument("--ppo-episodes-per-update", type=int, default=4)
    parser.add_argument("--ppo-episodes-per-seed", type=int, default=1)
    parser.add_argument("--ppo-max-steps", type=int, default=8)
    parser.add_argument("--ppo-samples-per-step", type=int, default=2)
    parser.add_argument("--ppo-judge-enabled", action="store_true")
    parser.add_argument("--ppo-target-success-rate", type=float, default=0.8)
    parser.add_argument("--ppo-target-avg-score", type=float, default=0.8)
    args = parser.parse_args()

    web_project_id = str(args.web_project_id).strip()
    use_case = str(args.use_case).strip().upper()
    run_root = ROOT / "data" / "single_usecase_rl" / web_project_id / use_case.lower() / str(args.run_name).strip()
    harvest_root = run_root / "harvest"
    sft_root = run_root / "sft"
    sft_adapter_dir = run_root / "models" / "sft_adapter"
    ppo_adapter_dir = run_root / "models" / "ppo_adapter"

    harvest_seeds = _parse_seeds(str(args.harvest_seeds))
    holdout_seeds = _parse_seeds(str(args.holdout_seeds))
    harvest_rows: list[dict[str, Any]] = []

    if not bool(args.skip_harvest):
        attempts_root = harvest_root / "gold" / "runs"
        traces_root = harvest_root / "gold" / "traces"
        attempts_root.mkdir(parents=True, exist_ok=True)
        traces_root.mkdir(parents=True, exist_ok=True)
        for seed in harvest_seeds:
            attempt_name = "heuristic_teacher"
            out_path = attempts_root / f"seed_{int(seed):04d}_{attempt_name}.json"
            trace_dir = traces_root / f"seed_{int(seed):04d}_{attempt_name}"
            cmd = [
                sys.executable,
                str(ROOT / "eval.py"),
                "--provider", str(args.provider),
                "--model", str(args.model),
                "--web-project-id", web_project_id,
                "--use-case", use_case,
                "--num-tasks", "1",
                "--seed", str(int(seed)),
                "--max-steps", str(int(args.max_steps)),
                "--task-cache", str(Path(args.task_cache).expanduser().resolve()),
                "--out", str(out_path),
                "--save-act-traces",
                "--trace-dir", str(trace_dir),
                "--trace-full-payloads",
                "--include-reasoning",
                "--no-failure-judge",
            ]
            _run(cmd, env={"WEB_AGENT_RUNTIME": str(args.heuristic_runtime)})
            report = json.loads(out_path.read_text(encoding="utf-8"))
            row = _episode_row_from_report(report, use_case=use_case, web_project_id=web_project_id, attempt_name=attempt_name, trace_dir=trace_dir, out_path=out_path)
            if isinstance(row, dict):
                harvest_rows.append(row)
        episodes_path, harvest_summary_path, harvest_summary = write_harvest_artifacts(output_root=harvest_root, use_case=use_case, target_seeds=harvest_seeds, rows=harvest_rows, merge_existing=False)
    else:
        episodes_path = harvest_root / "gold" / "episodes.jsonl"
        harvest_summary_path = harvest_root / "gold" / "summary.json"
        harvest_summary = json.loads(harvest_summary_path.read_text(encoding="utf-8"))

    sft_manifest: dict[str, Any] = {}
    if not bool(args.skip_sft):
        try:
            sft_manifest = export_harvest_to_sft(input_path=str(episodes_path), summary_path=str(harvest_summary_path), output_dir=str(sft_root), seed=36, val_ratio=0.2, trace_only=True, system_prompt=RUNTIME_ALIGNED_SYSTEM_PROMPT, runtime_aligned=True, train_seeds=harvest_seeds, val_seeds=holdout_seeds or None)
        except ValueError as exc:
            if "Validation split is empty" not in str(exc):
                raise
            fallback_val_seed = harvest_seeds[-1:] if len(harvest_seeds) > 1 else []
            fallback_train_seeds = harvest_seeds[:-1] if len(harvest_seeds) > 1 else harvest_seeds
            sft_manifest = export_harvest_to_sft(
                input_path=str(episodes_path),
                summary_path=str(harvest_summary_path),
                output_dir=str(sft_root),
                seed=36,
                val_ratio=0.0,
                trace_only=True,
                system_prompt=RUNTIME_ALIGNED_SYSTEM_PROMPT,
                runtime_aligned=True,
                train_seeds=fallback_train_seeds,
                val_seeds=fallback_val_seed or None,
            )
            sft_manifest["val_split_fallback"] = {
                "reason": "requested holdout seeds produced empty validation split",
                "requested_holdout_seeds": list(holdout_seeds),
                "fallback_train_seeds": list(fallback_train_seeds),
                "fallback_val_seeds": list(fallback_val_seed),
            }
        if str(args.sft_mode) == "local":
            sft_cmd = [
                sys.executable,
                str(ROOT / "training" / "finetune_bu.py"),
                "--data", str(sft_root / "train.jsonl"),
                "--val-data", str(sft_root / "val.jsonl"),
                "--base-model", str(args.sft_base_model),
                "--output-dir", str(sft_adapter_dir),
                "--epochs", str(int(args.sft_epochs)),
                "--lora-rank", str(int(args.sft_lora_rank)),
                "--batch-size", str(int(args.sft_batch_size)),
                "--grad-accum", str(int(args.sft_grad_accum)),
            ]
            _run(sft_cmd)
        elif str(args.sft_mode) == "runpod":
            sft_cmd = [
                sys.executable,
                str(ROOT / "training" / "runpod_job.py"),
                "--data", str(sft_root / "train.jsonl"),
                "--val-data", str(sft_root / "val.jsonl"),
                "--base-model", str(args.sft_base_model),
                "--output-dir", str(sft_adapter_dir),
            ]
            if str(args.runpod_existing_pod_id or "").strip():
                sft_cmd.extend(["--existing-pod-id", str(args.runpod_existing_pod_id).strip()])
            if str(args.runpod_inventory_snapshot_path or "").strip():
                sft_cmd.extend(["--inventory-snapshot-path", str(Path(args.runpod_inventory_snapshot_path).expanduser().resolve())])
            _run(sft_cmd)

    ppo_summary: dict[str, Any] = {}
    if not bool(args.skip_ppo):
        ppo_cmd = [
            sys.executable,
            str(ROOT / "training" / "rl" / "generic_ppo_trainer.py"),
            "--web-project-id", web_project_id,
            "--use-case", use_case,
            "--task-cache", str(Path(args.task_cache).expanduser().resolve()),
            "--base-model", str(args.ppo_base_model),
            "--adapter-path", str(sft_adapter_dir),
            "--output-dir", str(ppo_adapter_dir),
            "--train-seeds", str(args.harvest_seeds),
            "--dev-seeds", str(args.holdout_seeds),
            "--updates", str(int(args.ppo_updates)),
            "--episodes-per-update", str(int(args.ppo_episodes_per_update)),
            "--episodes-per-seed", str(int(args.ppo_episodes_per_seed)),
            "--max-steps", str(int(args.ppo_max_steps)),
            "--samples-per-step", str(int(args.ppo_samples_per_step)),
            "--teacher-bc-weight", "0.25",
            "--teacher-bc-max-step", str(int(args.ppo_max_steps)),
            "--expert-run-glob", str(harvest_root / "gold" / "runs" / "seed_*.json"),
            "--expert-pretrain-updates", "1",
            "--target-success-rate", str(float(args.ppo_target_success_rate)),
            "--target-avg-score", str(float(args.ppo_target_avg_score)),
        ]
        if bool(args.ppo_judge_enabled):
            ppo_cmd.append("--judge-enabled")
        _run(ppo_cmd)
        ppo_summary_path = ppo_adapter_dir / "online_summary.json"
        if ppo_summary_path.exists():
            ppo_summary = json.loads(ppo_summary_path.read_text(encoding="utf-8"))

    result = {
        "run_root": str(run_root),
        "web_project_id": web_project_id,
        "use_case": use_case,
        "harvest": {
            "episodes_path": str(episodes_path),
            "summary_path": str(harvest_summary_path),
            "summary": harvest_summary,
        },
        "sft": sft_manifest,
        "sft_adapter_dir": str(sft_adapter_dir),
        "ppo_adapter_dir": str(ppo_adapter_dir),
        "ppo_summary": ppo_summary,
    }
    _write_json(run_root / "pipeline_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
