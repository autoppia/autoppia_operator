#!/usr/bin/env python3
"""Helpers for Codex-driven sn36 operator iteration.

Commands:
- preflight: run repo checks.
- eval: run local evaluation and apply pass/fail thresholds.
- iwap: query IWAP agent-run data.
- submit: call autoppia-miner-cli with validated arguments.
- cycle: run eval, gate, optional submit, optional IWAP probe.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = {
    "success_threshold": 0.70,
    "avg_score_threshold": 0.60,
}


def _run_command(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)

    kwargs: dict[str, Any] = {
        "check": False,
        "cwd": str(cwd),
        "env": env_vars,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True

    result = subprocess.run(cmd, **kwargs)

    if check and result.returncode != 0:
        if capture:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")

    return result


def _load_eval_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise RuntimeError(f"eval report is not a JSON object: {path}")
    return payload


def _success_ratio(report: dict[str, Any], fallback_tasks: int) -> float:
    if report.get("success_rate") is not None:
        try:
            return float(report["success_rate"])
        except Exception:
            pass
    success_count = float(report.get("successes") or 0)
    total = float(report.get("num_tasks") or fallback_tasks)
    if total <= 0:
        return 0.0
    return success_count / total


def _avg_score(report: dict[str, Any], fallback_tasks: int) -> float:
    if "avg_score" in report:
        try:
            return float(report.get("avg_score"))
        except Exception:
            pass

    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return 0.0

    total = 0.0
    count = 0
    for item in episodes:
        if isinstance(item, dict):
            try:
                total += float(item.get("score") or 0.0)
                count += 1
            except Exception:
                pass
    if count <= 0:
        return 0.0
    return total / float(count)


def _is_writable_file(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8"):
            pass
        return True
    except Exception:
        return False


def cmd_preflight(_: argparse.Namespace) -> int:
    print("[sn36 preflight] python check.py")
    _run_command([sys.executable, str(ROOT / "check.py")])

    print("[sn36 preflight] python scripts/deploy_check.py")
    _run_command([sys.executable, str(ROOT / "scripts" / "deploy_check.py")])
    print("[sn36 preflight] OK")
    return 0


@dataclass
class EvalGateResult:
    report: dict[str, Any]
    success_rate: float
    avg_score: float
    passed: bool


def _run_eval(
    num_tasks: int,
    web_project_id: str | None,
    use_case: str | None,
    task_id: str | None,
    repeat: int,
    success_threshold: float,
    avg_score_threshold: float,
    provider: str,
    model: str,
    max_steps: int,
    all_use_cases: bool,
    tasks_per_use_case: int,
    distinct_use_cases: bool,
    task_cache: str | None,
    task_concurrency: int,
    out_file: Path,
) -> EvalGateResult:
    cmd = [
        sys.executable,
        str(ROOT / "eval.py"),
        "--provider",
        provider,
        "--model",
        model,
        "--num-tasks",
        str(int(num_tasks)),
        "--max-steps",
        str(int(max_steps)),
        "--out",
        str(out_file),
    ]
    if web_project_id:
        cmd.extend(["--web-project-id", web_project_id])
    if use_case:
        cmd.extend(["--use-case", str(use_case)])
    if task_id:
        cmd.extend(["--task-id", str(task_id)])
    if repeat > 1:
        cmd.extend(["--repeat", str(max(1, int(repeat)))])
    if all_use_cases:
        cmd.append("--all-use-cases")
        cmd.extend(["--tasks-per-use-case", str(max(1, int(tasks_per_use_case)))])
    if distinct_use_cases:
        cmd.append("--distinct-use-cases")
    if task_cache:
        cmd.extend(["--task-cache", str(task_cache)])
    if task_concurrency > 1:
        cmd.extend(["--task-concurrency", str(max(1, int(task_concurrency)))])

    print(f"[sn36 eval] running: {' '.join(cmd)}")
    _run_command(cmd)

    report = _load_eval_report(out_file)
    success_rate = _success_ratio(report, fallback_tasks=num_tasks)
    avg_score = _avg_score(report, fallback_tasks=num_tasks)

    passed = (success_rate >= success_threshold) and (avg_score >= avg_score_threshold)

    print(
        f"[sn36 eval] tasks={report.get('num_tasks', num_tasks)} "
        f"success_rate={success_rate:.4f} avg_score={avg_score:.4f} "
        f"pass={passed}"
    )
    return EvalGateResult(report=report, success_rate=success_rate, avg_score=avg_score, passed=passed)


def cmd_eval(args: argparse.Namespace) -> int:
    out_file = Path(args.out or (ROOT / "data" / "sn36_eval_report.json")).resolve()
    if not _is_writable_file(out_file):
        raise RuntimeError(f"cannot write eval output to {out_file}")

    result = _run_eval(
        num_tasks=args.num_tasks,
        web_project_id=args.project_id,
        use_case=args.use_case,
        task_id=args.task_id,
        repeat=args.repeat,
        success_threshold=args.success_threshold,
        avg_score_threshold=args.avg_score_threshold,
        provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        all_use_cases=bool(args.all_use_cases),
        tasks_per_use_case=max(1, int(args.tasks_per_use_case)),
        distinct_use_cases=bool(args.distinct_use_cases),
        task_cache=args.task_cache,
        task_concurrency=max(1, int(args.task_concurrency)),
        out_file=out_file,
    )

    print(f"[sn36 eval] report_file={out_file}")
    if args.json:
        payload = dict(result.report)
        payload["success_threshold"] = args.success_threshold
        payload["avg_score_threshold"] = args.avg_score_threshold
        payload["success_rate"] = result.success_rate
        payload["avg_score"] = result.avg_score
        payload["passed"] = result.passed
        payload["out_file"] = str(out_file)
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0 if result.passed else 2


def _iws_headers(token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _extract_data(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        data = payload.get("data")
        if isinstance(data, (list, dict)):
            return data
    if isinstance(payload, dict) and "result" in payload:
        data = payload.get("result")
        if isinstance(data, (list, dict)):
            return data
    return payload


def _extract_run_rows(payload: Any) -> list[dict[str, Any]]:
    data = _extract_data(payload)
    if isinstance(data, dict):
        rows = data.get("runs")
        if isinstance(rows, list):
            return rows
        rows = data.get("items")
        if isinstance(rows, list):
            return rows
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _safe_id(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _extract_score_from_task(payload: dict[str, Any]) -> float:
    for key in ("eval_score", "score", "raw_score", "reward"):
        raw = payload.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except Exception:
            pass
    return 0.0


def _fetch_iwap_json(client: httpx.Client, path: str, params: dict[str, Any] | None = None) -> Any:
    response = client.get(path, params=params)
    response.raise_for_status()
    try:
        return response.json()
    except Exception as exc:
        if response.text.strip():
            raise RuntimeError(f"non-json IWAP response for {path}: {exc}") from exc
        return []


def cmd_iwap(args: argparse.Namespace) -> int:
    if not args.base_url:
        raise RuntimeError("IWAP base URL is required via --base-url or IWAP_BASE_URL")

    base = args.base_url.rstrip("/")
    limit = max(1, int(args.limit))
    token = args.token or os.getenv("IWAP_API_TOKEN")
    headers = _iws_headers(token)

    with httpx.Client(base_url=base, timeout=45.0, headers=headers) as client:
        payload = _fetch_iwap_json(
            client,
            "/api/v1/agent-runs",
            {
                "page": 1,
                "limit": limit,
                "includeUnfinished": bool(args.include_unfinished),
                "include_unfinished": bool(args.include_unfinished),
                "sortBy": "startTime",
                "sortOrder": "desc",
            },
        )
        rows = _extract_run_rows(payload)

        selected: list[dict[str, Any]] = []
        for idx, run in enumerate(rows):
            if idx >= limit:
                break
            if not isinstance(run, dict):
                continue

            run_id = _safe_id(run, "runId", "run_id", "id")
            run_payload = {
                "run_id": run_id,
                "agent_run_id": run_id,
                "status": run.get("status") or run.get("state") or "unknown",
                "miner": run.get("miner") or run.get("miner_identity") or {},
                "snapshot": run.get("snapshot") or run.get("miner_snapshot") or {},
                "tasks": [],
                "task_count": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
            }

            try:
                tasks_payload = _fetch_iwap_json(client, f"/api/v1/agent-runs/{run_id}/tasks")
                tasks = _extract_data(tasks_payload)
                task_rows = tasks.get("tasks") if isinstance(tasks, dict) else []
                task_rows = task_rows if isinstance(task_rows, list) else []
                run_payload["task_count"] = len(task_rows)

                scores: list[float] = []
                for task in task_rows:
                    if not isinstance(task, dict):
                        continue
                    score = _extract_score_from_task(task)
                    scores.append(score)
                if scores:
                    run_payload["avg_score"] = sum(scores) / len(scores)
                    run_payload["max_score"] = max(scores)
            except Exception:
                pass

            selected.append(run_payload)

        summary = {
            "base_url": base,
            "run_count": len(selected),
            "runs": selected,
        }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = args.out
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[sn36 iwap] wrote summary to {out_path}")

    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    cli = args.cli_binary or "autoppia-miner-cli"
    command = [
        cli,
        "submit",
        "--github",
        args.github_url,
        "--agent.name",
        args.agent_name,
        "--wallet.name",
        args.wallet_name,
        "--wallet.hotkey",
        args.wallet_hotkey,
        "--subtensor.network",
        args.network,
        "--netuid",
        str(int(args.netuid)),
    ]

    if args.agent_image:
        command.extend(["--agent.image", args.agent_image])

    if args.season:
        command.extend(["--season", str(int(args.season))])

    if args.target_round:
        command.extend(["--target_round", str(int(args.target_round))])

    if args.chain_endpoint:
        command.extend(["--subtensor.chain_endpoint", args.chain_endpoint])

    print(f"[sn36 submit] {' '.join(command)}")
    _run_command(command)
    return 0


def cmd_cycle(args: argparse.Namespace) -> int:
    print("[sn36 cycle] preflight")
    cmd_preflight(args)

    print("[sn36 cycle] local eval gate")
    out_file = Path(args.out or (ROOT / "data" / "sn36_cycle_eval.json")).resolve()
    eval_result = _run_eval(
        num_tasks=args.num_tasks,
        web_project_id=args.project_id,
        use_case=args.use_case,
        task_id=args.task_id,
        repeat=args.repeat,
        success_threshold=args.success_threshold,
        avg_score_threshold=args.avg_score_threshold,
        provider=args.provider,
        model=args.model,
        max_steps=args.max_steps,
        all_use_cases=bool(args.all_use_cases),
        tasks_per_use_case=max(1, int(args.tasks_per_use_case)),
        distinct_use_cases=bool(args.distinct_use_cases),
        task_cache=args.task_cache,
        task_concurrency=max(1, int(args.task_concurrency)),
        out_file=out_file,
    )

    if not eval_result.passed:
        print("[sn36 cycle] eval gate failed", file=sys.stderr)
        return 2

    if args.submit:
        submit_args = argparse.Namespace(
            cli_binary=args.cli_binary,
            github_url=args.github_url,
            agent_name=args.agent_name,
            agent_image=args.agent_image,
            wallet_name=args.wallet_name,
            wallet_hotkey=args.wallet_hotkey,
            network=args.network,
            netuid=args.netuid,
            chain_endpoint=args.chain_endpoint,
            season=args.season,
            target_round=args.target_round,
        )
        cmd_submit(submit_args)

    if args.iwap_url:
        probe_args = argparse.Namespace(
            base_url=args.iwap_url,
            token=args.iwap_token,
            limit=max(1, int(args.iwap_limit)),
            include_unfinished=bool(args.include_unfinished),
            out=args.iwap_out,
        )
        cmd_iwap(probe_args)

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subsidy-friendly sn36 automation utilities")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="run local repo checks")
    preflight.set_defaults(func=cmd_preflight)

    eval_cmd = subparsers.add_parser("eval", help="run local eval with guardrails")
    eval_cmd.add_argument("--num-tasks", type=int, default=20)
    eval_cmd.add_argument("--project-id", default=None)
    eval_cmd.add_argument("--use-case", default=None)
    eval_cmd.add_argument("--task-id", default=None)
    eval_cmd.add_argument("--repeat", type=int, default=1)
    eval_cmd.add_argument("--max-steps", type=int, default=12)
    eval_cmd.add_argument("--provider", default="chutes")
    eval_cmd.add_argument("--model", default="deepseek-ai/DeepSeek-V3-0324")
    eval_cmd.add_argument("--all-use-cases", action="store_true")
    eval_cmd.add_argument("--tasks-per-use-case", type=int, default=1)
    eval_cmd.add_argument("--distinct-use-cases", action="store_true")
    eval_cmd.add_argument("--task-cache", default=None)
    eval_cmd.add_argument("--task-concurrency", type=int, default=1)
    eval_cmd.add_argument("--success-threshold", type=float, default=DEFAULT_METRICS["success_threshold"])
    eval_cmd.add_argument("--avg-score-threshold", type=float, default=DEFAULT_METRICS["avg_score_threshold"])
    eval_cmd.add_argument("--out", default=str(ROOT / "data" / "sn36_eval_report.json"))
    eval_cmd.add_argument("--json", action="store_true", help="print parsed JSON with guardrail result")
    eval_cmd.set_defaults(func=cmd_eval)

    iwap = subparsers.add_parser("iwap", help="read recent IWAP runs and summarize")
    iwap.add_argument(
        "--base-url",
        required=False,
        default=os.getenv("IWAP_BASE_URL"),
        help="IWAP API base URL, e.g. https://api.example.com",
    )
    iwap.add_argument("--token", default=None, help="Optional bearer token")
    iwap.add_argument("--limit", type=int, default=20)
    iwap.add_argument("--include-unfinished", action="store_true")
    iwap.add_argument("--out", default="")
    iwap.set_defaults(func=cmd_iwap)

    submit = subparsers.add_parser("submit", help="submit miner metadata via autoppia-miner-cli")
    submit.add_argument("--github-url", required=True)
    submit.add_argument("--agent-name", required=True)
    submit.add_argument("--agent-image", default="")
    submit.add_argument("--wallet-name", default=os.getenv("SN36_COLDKEY", "default"))
    submit.add_argument("--wallet-hotkey", default=os.getenv("SN36_HOTKEY", "default"))
    submit.add_argument("--network", default=os.getenv("SN36_NETWORK", "finney"))
    submit.add_argument("--netuid", type=int, default=int(os.getenv("SN36_NETUID", "36")))
    submit.add_argument("--chain-endpoint", default="")
    submit.add_argument("--season", type=int, default=0)
    submit.add_argument("--target-round", type=int, default=0)
    submit.add_argument("--cli-binary", default="autoppia-miner-cli")
    submit.set_defaults(func=cmd_submit)

    cycle = subparsers.add_parser("cycle", help="run eval gate and optional submit + IWAP check")
    cycle.add_argument("--github-url", required=True)
    cycle.add_argument("--agent-name", required=True)
    cycle.add_argument("--agent-image", default="")
    cycle.add_argument("--wallet-name", default=os.getenv("SN36_COLDKEY", "default"))
    cycle.add_argument("--wallet-hotkey", default=os.getenv("SN36_HOTKEY", "default"))
    cycle.add_argument("--network", default=os.getenv("SN36_NETWORK", "finney"))
    cycle.add_argument("--netuid", type=int, default=int(os.getenv("SN36_NETUID", "36")))
    cycle.add_argument("--chain-endpoint", default="")
    cycle.add_argument("--season", type=int, default=0)
    cycle.add_argument("--target-round", type=int, default=0)
    cycle.add_argument("--cli-binary", default="autoppia-miner-cli")

    cycle.add_argument("--num-tasks", type=int, default=20)
    cycle.add_argument("--project-id", default=None)
    cycle.add_argument("--max-steps", type=int, default=12)
    cycle.add_argument("--provider", default="chutes")
    cycle.add_argument("--model", default="deepseek-ai/DeepSeek-V3-0324")
    cycle.add_argument("--use-case", default=None)
    cycle.add_argument("--task-id", default=None)
    cycle.add_argument("--repeat", type=int, default=1)
    cycle.add_argument("--all-use-cases", action="store_true")
    cycle.add_argument("--tasks-per-use-case", type=int, default=1)
    cycle.add_argument("--distinct-use-cases", action="store_true")
    cycle.add_argument("--task-cache", default=None)
    cycle.add_argument("--task-concurrency", type=int, default=1)
    cycle.add_argument("--success-threshold", type=float, default=DEFAULT_METRICS["success_threshold"])
    cycle.add_argument("--avg-score-threshold", type=float, default=DEFAULT_METRICS["avg_score_threshold"])
    cycle.add_argument("--out", default=str(ROOT / "data" / "sn36_cycle_eval.json"))
    cycle.add_argument("--submit", action="store_true", help="submit commit through miner-cli after passing local eval")

    cycle.add_argument("--iwap-url", default="")
    cycle.add_argument("--iwap-token", default=None)
    cycle.add_argument("--iwap-limit", type=int, default=20)
    cycle.add_argument("--include-unfinished", action="store_true")
    cycle.add_argument("--iwap-out", default="")

    cycle.set_defaults(func=cmd_cycle)

    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    func = args.func
    try:
        return int(func(args))
    except RuntimeError as exc:
        print(f"[sn36] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
