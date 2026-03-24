#!/usr/bin/env python3
"""MCP-compatible wrapper for operator sn36 tools.

This module implements a small JSON-RPC 2.0 server over stdio with:
- tools/list
- tools/call
- initialize

It dispatches each tool call to existing CLI helpers in this repository:
- tools/bittensor_tools.py
- tools/iwap_tools.py
- scripts/sn36_ops.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_DOTENV_LOADED = False


def _load_env_file() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except Exception:
        # Optional integration; fail soft if dependency unavailable.
        return


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _arg_to_cmd_flag(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _run_command(argv: list[str], timeout: int = 300) -> dict[str, Any]:
    process = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    stdout = (process.stdout or "").strip()
    stderr = (process.stderr or "").strip()
    ok = process.returncode == 0

    return {
        "ok": ok,
        "return_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "command": " ".join(argv),
    }


def _build_args(prefix: list[str], args: dict[str, Any], *, boolean_flags: set[str] | None = None) -> list[str]:
    boolean_flags = boolean_flags or set()
    argv = list(prefix)
    for key, value in args.items():
        if key == "_":
            continue
        if value is None:
            continue
        flag = _arg_to_cmd_flag(key)
        if isinstance(value, bool):
            if key in boolean_flags:
                if value:
                    argv.append(flag)
                else:
                    argv.append(f"--no{flag}")
            elif value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    return argv


def _run_bittensor_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "bittensor_tools.py"),
        tool_name.replace(".", "_"),
    ]
    return _run_command(_build_args(cmd, args))


def _run_iwap_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "iwap_tools.py"),
        tool_name.replace(".", "_"),
    ]
    return _run_command(_build_args(cmd, args), timeout=180)


def _run_sn36_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    cmd = [sys.executable, str(ROOT / "scripts" / "sn36_ops.py"), tool_name]
    return _run_command(_build_args(cmd, args), timeout=600)


def _run_runpod_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "runpod_tools.py"),
        tool_name.replace(".", "_"),
    ]
    return _run_command(_build_args(cmd, args), timeout=600)


def _run_smtp_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    cmd = [sys.executable, str(ROOT / "tools" / "smtp_tools.py"), tool_name]
    return _run_command(_build_args(cmd, args), timeout=60)


def _tool_bittensor_metagraph(args: dict[str, Any]) -> dict[str, Any]:
    return _run_bittensor_tool("metagraph", args)


def _tool_bittensor_uid(args: dict[str, Any]) -> dict[str, Any]:
    if (args.get("uid") is None) and (not args.get("hotkey")):
        return {
            "ok": False,
            "error": "--uid or --hotkey is required for bittensor.uid",
        }
    return _run_bittensor_tool("uid", args)


def _tool_bittensor_uid_stats(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("uid") is None:
        return {"ok": False, "error": "--uid is required for bittensor.uid-stats"}
    return _run_bittensor_tool("uid-stats", args)


def _tool_bittensor_my_miner(args: dict[str, Any]) -> dict[str, Any]:
    return _run_bittensor_tool("my-miner", args)


def _tool_iwap_last_round(args: dict[str, Any]) -> dict[str, Any]:
    return _run_iwap_tool("last-round", args)


def _tool_iwap_season_results(args: dict[str, Any]) -> dict[str, Any]:
    return _run_iwap_tool("season-results", args)


def _tool_iwap_top_uids(args: dict[str, Any]) -> dict[str, Any]:
    return _run_iwap_tool("top-uids", args)


def _tool_iwap_rounds(args: dict[str, Any]) -> dict[str, Any]:
    return _run_iwap_tool("rounds", args)


def _tool_iwap_season_tasks(args: dict[str, Any]) -> dict[str, Any]:
    return _run_iwap_tool("season-tasks", args)


def _tool_sn36_preflight(args: dict[str, Any]) -> dict[str, Any]:
    return _run_sn36_tool("preflight", args)


def _tool_sn36_eval(args: dict[str, Any]) -> dict[str, Any]:
    payload = dict(args)
    payload["project_id"] = payload.pop("project_id", None)
    payload["task_id"] = payload.pop("task_id", None)
    return _run_sn36_tool("eval", payload)


def _tool_sn36_submit(args: dict[str, Any]) -> dict[str, Any]:
    return _run_sn36_tool("submit", args)


def _tool_sn36_cycle(args: dict[str, Any]) -> dict[str, Any]:
    return _run_sn36_tool("cycle", args)


def _tool_sn36_iwap(args: dict[str, Any]) -> dict[str, Any]:
    return _run_sn36_tool("iwap", args)


def _tool_runpod_list_pods(args: dict[str, Any]) -> dict[str, Any]:
    return _run_runpod_tool("list-pods", args)


def _tool_runpod_get_pod(args: dict[str, Any]) -> dict[str, Any]:
    return _run_runpod_tool("get-pod", args)


def _tool_runpod_create_pod(args: dict[str, Any]) -> dict[str, Any]:
    return _run_runpod_tool("create-pod", args)


def _tool_runpod_stop_pod(args: dict[str, Any]) -> dict[str, Any]:
    return _run_runpod_tool("stop-pod", args)


def _tool_runpod_resume_pod(args: dict[str, Any]) -> dict[str, Any]:
    return _run_runpod_tool("resume-pod", args)


def _tool_runpod_terminate_pod(args: dict[str, Any]) -> dict[str, Any]:
    return _run_runpod_tool("terminate-pod", args)


def _tool_runpod_get_balance(args: dict[str, Any]) -> dict[str, Any]:
    return _run_runpod_tool("get-balance", args)


def _tool_runpod_list_gpu_types(args: dict[str, Any]) -> dict[str, Any]:
    return _run_runpod_tool("list-gpu-types", args)


def _tool_runpod_graphql(args: dict[str, Any]) -> dict[str, Any]:
    payload = dict(args)
    if not payload.get("query"):
        return {"ok": False, "error": "--query is required for runpod.graphql"}
    return _run_runpod_tool("graphql", payload)


def _tool_smtp_send(args: dict[str, Any]) -> dict[str, Any]:
    return _run_smtp_tool("send", args)


def _tool_smtp_check(args: dict[str, Any]) -> dict[str, Any]:
    return _run_smtp_tool("check", args)


def build_tool_registry() -> dict[str, ToolSpec]:
    return {
        "bittensor.metagraph": ToolSpec(
            name="bittensor.metagraph",
            description="Fetch subnet metagraph snapshot and return uid/hotkey/incentive/stake rows.",
            input_schema={
                "type": "object",
                "properties": {
                    "netuid": {"type": "integer", "default": 36},
                    "network": {"type": "string", "default": "finney"},
                    "chain_endpoint": {"type": "string", "default": ""},
                    "limit": {"type": "integer", "default": 20},
                    "include_all": {"type": "boolean", "default": False},
                },
            },
            handler=_tool_bittensor_metagraph,
        ),
        "bittensor.uid": ToolSpec(
            name="bittensor.uid",
            description="Resolve metagraph row by uid or hotkey.",
            input_schema={
                "type": "object",
                "properties": {
                    "netuid": {"type": "integer", "default": 36},
                    "network": {"type": "string", "default": "finney"},
                    "chain_endpoint": {"type": "string", "default": ""},
                    "uid": {"type": ["integer", "null"], "default": None},
                    "hotkey": {"type": ["string", "null"], "default": None},
                },
            },
            handler=_tool_bittensor_uid,
        ),
        "bittensor.uid-stats": ToolSpec(
            name="bittensor.uid-stats",
            description="Get incentive/stake distribution summary for one miner uid.",
            input_schema={
                "type": "object",
                "properties": {
                    "netuid": {"type": "integer", "default": 36},
                    "network": {"type": "string", "default": "finney"},
                    "chain_endpoint": {"type": "string", "default": ""},
                    "uid": {"type": "integer"},
                },
                "required": ["uid"],
            },
            handler=_tool_bittensor_uid_stats,
        ),
        "bittensor.my-miner": ToolSpec(
            name="bittensor.my-miner",
            description="Resolve current SN36_HOTKEY to uid(s) using env/local hotkey or provided hotkey.",
            input_schema={
                "type": "object",
                "properties": {
                    "hotkey": {"type": ["string", "null"], "default": None},
                    "netuid": {"type": "integer", "default": 36},
                    "network": {"type": "string", "default": "finney"},
                    "chain_endpoint": {"type": "string", "default": ""},
                },
            },
            handler=_tool_bittensor_my_miner,
        ),
        "iwap.last-round": ToolSpec(
            name="iwap.last-round",
            description="Read latest IWAP round summary (mock mode unless live endpoint data exists).",
            input_schema={
                "type": "object",
                "properties": {
                    "mock_data": {"type": "string", "default": ""},
                },
            },
            handler=_tool_iwap_last_round,
        ),
        "iwap.season-results": ToolSpec(
            name="iwap.season-results",
            description="Aggregate mock IWAP results by UID for a season.",
            input_schema={
                "type": "object",
                "properties": {
                    "season_id": {"type": ["integer", "null"], "default": None},
                    "mock_data": {"type": "string", "default": ""},
                },
            },
            handler=_tool_iwap_season_results,
        ),
        "iwap.top-uids": ToolSpec(
            name="iwap.top-uids",
            description="Top UIDs by average score in available IWAP rounds.",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "mock_data": {"type": "string", "default": ""},
                },
            },
            handler=_tool_iwap_top_uids,
        ),
        "iwap.rounds": ToolSpec(
            name="iwap.rounds",
            description="List recent IWAP rounds from mock payload.",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 0},
                    "mock_data": {"type": "string", "default": ""},
                },
            },
            handler=_tool_iwap_rounds,
        ),
        "iwap.season-tasks": ToolSpec(
            name="iwap.season-tasks",
            description="Get per-task season rows with optional live IWAP fallback and mock fallback.",
            input_schema={
                "type": "object",
                "properties": {
                    "season_id": {"type": ["integer", "null"], "default": None},
                    "uid": {"type": ["integer", "null"], "default": None},
                    "min_score": {"type": ["number", "null"], "default": None},
                    "max_score": {"type": ["number", "null"], "default": None},
                    "limit": {"type": "integer", "default": 0},
                    "mock_data": {"type": "string", "default": ""},
                    "base_url": {"type": "string", "default": ""},
                    "token": {"type": "string", "default": ""},
                    "prefer_live": {"type": "boolean", "default": False},
                },
            },
            handler=_tool_iwap_season_tasks,
        ),
        "sn36.preflight": ToolSpec(
            name="sn36.preflight",
            description="Run local preflight checks (check.py + deploy_check.py).",
            input_schema={"type": "object", "properties": {}},
            handler=_tool_sn36_preflight,
        ),
        "sn36.eval": ToolSpec(
            name="sn36.eval",
            description="Run local evaluation gate with project/use-case/task selection and thresholds.",
            input_schema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "default": "chutes"},
                    "model": {
                        "type": "string",
                        "default": "deepseek-ai/DeepSeek-V3-0324",
                    },
                    "num_tasks": {"type": "integer", "default": 20},
                    "project_id": {"type": ["string", "null"], "default": None},
                    "use_case": {"type": ["string", "null"], "default": None},
                    "task_id": {"type": ["string", "null"], "default": None},
                    "repeat": {"type": "integer", "default": 1},
                    "max_steps": {"type": "integer", "default": 12},
                    "success_threshold": {"type": "number", "default": 0.7},
                    "avg_score_threshold": {"type": "number", "default": 0.6},
                    "all_use_cases": {"type": "boolean", "default": False},
                    "tasks_per_use_case": {"type": "integer", "default": 1},
                    "distinct_use_cases": {"type": "boolean", "default": False},
                    "task_cache": {"type": ["string", "null"], "default": None},
                    "task_concurrency": {"type": "integer", "default": 1},
                    "out": {
                        "type": "string",
                        "default": str(ROOT / "data" / "sn36_eval_report.json"),
                    },
                    "json": {"type": "boolean", "default": False},
                },
            },
            handler=_tool_sn36_eval,
        ),
        "sn36.submit": ToolSpec(
            name="sn36.submit",
            description="Submit validator metadata via autoppia-miner-cli (requires env/wallet context).",
            input_schema={
                "type": "object",
                "properties": {
                    "github_url": {"type": "string"},
                    "agent_name": {"type": "string"},
                    "agent_image": {"type": "string", "default": ""},
                    "wallet_name": {
                        "type": "string",
                        "default": os.getenv("SN36_COLDKEY", "default"),
                    },
                    "wallet_hotkey": {
                        "type": "string",
                        "default": os.getenv("SN36_HOTKEY", "default"),
                    },
                    "network": {
                        "type": "string",
                        "default": os.getenv("SN36_NETWORK", "finney"),
                    },
                    "netuid": {
                        "type": "integer",
                        "default": int(os.getenv("SN36_NETUID", "36")),
                    },
                    "chain_endpoint": {"type": "string", "default": ""},
                    "season": {"type": "integer", "default": 0},
                    "target_round": {"type": "integer", "default": 0},
                    "cli_binary": {"type": "string", "default": "autoppia-miner-cli"},
                },
                "required": ["github_url", "agent_name"],
            },
            handler=_tool_sn36_submit,
        ),
        "sn36.cycle": ToolSpec(
            name="sn36.cycle",
            description="Run preflight + eval gate; optionally submit and poll IWAP.",
            input_schema={
                "type": "object",
                "properties": {
                    "github_url": {"type": "string"},
                    "agent_name": {"type": "string"},
                    "agent_image": {"type": "string", "default": ""},
                    "wallet_name": {
                        "type": "string",
                        "default": os.getenv("SN36_COLDKEY", "default"),
                    },
                    "wallet_hotkey": {
                        "type": "string",
                        "default": os.getenv("SN36_HOTKEY", "default"),
                    },
                    "network": {
                        "type": "string",
                        "default": os.getenv("SN36_NETWORK", "finney"),
                    },
                    "netuid": {
                        "type": "integer",
                        "default": int(os.getenv("SN36_NETUID", "36")),
                    },
                    "chain_endpoint": {"type": "string", "default": ""},
                    "season": {"type": "integer", "default": 0},
                    "target_round": {"type": "integer", "default": 0},
                    "cli_binary": {"type": "string", "default": "autoppia-miner-cli"},
                    "provider": {"type": "string", "default": "chutes"},
                    "model": {
                        "type": "string",
                        "default": "deepseek-ai/DeepSeek-V3-0324",
                    },
                    "num_tasks": {"type": "integer", "default": 20},
                    "project_id": {"type": ["string", "null"], "default": None},
                    "use_case": {"type": ["string", "null"], "default": None},
                    "task_id": {"type": ["string", "null"], "default": None},
                    "repeat": {"type": "integer", "default": 1},
                    "max_steps": {"type": "integer", "default": 12},
                    "success_threshold": {"type": "number", "default": 0.7},
                    "avg_score_threshold": {"type": "number", "default": 0.6},
                    "all_use_cases": {"type": "boolean", "default": False},
                    "tasks_per_use_case": {"type": "integer", "default": 1},
                    "distinct_use_cases": {"type": "boolean", "default": False},
                    "task_cache": {"type": ["string", "null"], "default": None},
                    "task_concurrency": {"type": "integer", "default": 1},
                    "submit": {"type": "boolean", "default": False},
                    "iwap_url": {"type": "string", "default": ""},
                    "iwap_token": {"type": ["string", "null"], "default": None},
                    "iwap_limit": {"type": "integer", "default": 20},
                    "include_unfinished": {"type": "boolean", "default": False},
                    "iwap_out": {"type": "string", "default": ""},
                    "out": {
                        "type": "string",
                        "default": str(ROOT / "data" / "sn36_cycle_eval.json"),
                    },
                },
                "required": ["github_url", "agent_name"],
            },
            handler=_tool_sn36_cycle,
        ),
        "sn36.iwap": ToolSpec(
            name="sn36.iwap",
            description="Query IWAP run summary from API endpoint.",
            input_schema={
                "type": "object",
                "properties": {
                    "base_url": {
                        "type": "string",
                        "default": os.getenv("IWAP_BASE_URL", ""),
                    },
                    "token": {"type": "string", "default": ""},
                    "limit": {"type": "integer", "default": 20},
                    "include_unfinished": {"type": "boolean", "default": False},
                    "out": {"type": "string", "default": ""},
                },
            },
            handler=_tool_sn36_iwap,
        ),
        "runpod.list_pods": ToolSpec(
            name="runpod.list_pods",
            description="List RunPod pods with status and runtime summary.",
            input_schema={"type": "object", "properties": {}},
            handler=_tool_runpod_list_pods,
        ),
        "runpod.get_pod": ToolSpec(
            name="runpod.get_pod",
            description="Get detailed information for a specific RunPod pod.",
            input_schema={
                "type": "object",
                "properties": {
                    "pod_id": {"type": "string"},
                },
                "required": ["pod_id"],
            },
            handler=_tool_runpod_get_pod,
        ),
        "runpod.create_pod": ToolSpec(
            name="runpod.create_pod",
            description="Create a pod (side effects enabled required).",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "gpu_type_id": {"type": "string"},
                    "gpu_count": {"type": "integer", "default": 1},
                    "template_id": {"type": ["string", "null"], "default": None},
                    "docker_image": {
                        "type": "string",
                        "default": "runpod/pytorch:latest",
                    },
                    "volume_in_gb": {"type": "integer", "default": 20},
                    "cloud_type": {"type": "string", "default": "SECURE"},
                    "allow_side_effects": {"type": "boolean", "default": False},
                },
                "required": ["name", "gpu_type_id"],
            },
            handler=_tool_runpod_create_pod,
        ),
        "runpod.stop_pod": ToolSpec(
            name="runpod.stop_pod",
            description="Stop a RunPod pod (side effects enabled required).",
            input_schema={
                "type": "object",
                "properties": {
                    "pod_id": {"type": "string"},
                    "allow_side_effects": {"type": "boolean", "default": False},
                },
                "required": ["pod_id"],
            },
            handler=_tool_runpod_stop_pod,
        ),
        "runpod.resume_pod": ToolSpec(
            name="runpod.resume_pod",
            description="Resume a RunPod pod (side effects enabled required).",
            input_schema={
                "type": "object",
                "properties": {
                    "pod_id": {"type": "string"},
                    "gpu_count": {"type": "integer", "default": 1},
                    "allow_side_effects": {"type": "boolean", "default": False},
                },
                "required": ["pod_id"],
            },
            handler=_tool_runpod_resume_pod,
        ),
        "runpod.terminate_pod": ToolSpec(
            name="runpod.terminate_pod",
            description="Terminate a RunPod pod (side effects enabled required).",
            input_schema={
                "type": "object",
                "properties": {
                    "pod_id": {"type": "string"},
                    "allow_side_effects": {"type": "boolean", "default": False},
                },
                "required": ["pod_id"],
            },
            handler=_tool_runpod_terminate_pod,
        ),
        "runpod.balance": ToolSpec(
            name="runpod.balance",
            description="Get RunPod account balance/credit info.",
            input_schema={"type": "object", "properties": {}},
            handler=_tool_runpod_get_balance,
        ),
        "runpod.list_gpu_types": ToolSpec(
            name="runpod.list_gpu_types",
            description="List available GPU types, clouds, and base prices.",
            input_schema={"type": "object", "properties": {}},
            handler=_tool_runpod_list_gpu_types,
        ),
        "runpod.graphql": ToolSpec(
            name="runpod.graphql",
            description="Run arbitrary RunPod GraphQL query. Mutations require allow_side_effects=true.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "variables": {"type": "string", "default": ""},
                    "variables_file": {"type": "string", "default": ""},
                    "timeout": {"type": "integer", "default": 45},
                    "allow_side_effects": {"type": "boolean", "default": False},
                },
                "required": ["query"],
            },
            handler=_tool_runpod_graphql,
        ),
        "smtp.send": ToolSpec(
            name="smtp.send",
            description="Send a notification email via configured SMTP.",
            input_schema={
                "type": "object",
                "properties": {
                    "to_addresses": {"type": "string", "default": ""},
                    "cc": {"type": "string", "default": ""},
                    "bcc": {"type": "string", "default": ""},
                    "reply_to": {"type": "string", "default": ""},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "from_address": {"type": "string", "default": ""},
                    "is_html": {"type": "boolean", "default": False},
                    "dry_run": {"type": "boolean", "default": False},
                },
                "required": ["subject", "body"],
            },
            handler=_tool_smtp_send,
        ),
        "smtp.check": ToolSpec(
            name="smtp.check",
            description="Check SMTP server connectivity/HELO-style responsiveness.",
            input_schema={"type": "object", "properties": {}},
            handler=_tool_smtp_check,
        ),
    }


def _normalize_args(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"None", "null"}:
            continue
        if value is None:
            continue
        normalized[str(key)] = value
    return normalized


def _handle_initialize(req_id: Any, _registry: dict[str, ToolSpec]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2025-03-26",
            "serverInfo": {
                "name": "miner_mcp",
                "version": "1.0.0",
            },
            "capabilities": {"tools": {}},
        },
    }


def _handle_tools_list(req_id: Any, registry: dict[str, ToolSpec]) -> dict[str, Any]:
    tools_payload = []
    for tool in registry.values():
        tools_payload.append(
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
        )
    return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools_payload}}


def _handle_tools_call(req_id: Any, registry: dict[str, ToolSpec], params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    if name not in registry:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {name}"},
        }
    tool = registry[name]
    arguments = _normalize_args(params.get("arguments"))
    result = tool.handler(arguments)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ]
        },
    }


def _handle_request(raw: str, registry: dict[str, ToolSpec]) -> dict[str, Any] | None:
    try:
        req = json.loads(raw)
    except Exception:
        return None

    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return _handle_initialize(req_id, registry)
    if method == "tools/list":
        return _handle_tools_list(req_id, registry)
    if method == "tools/call":
        return _handle_tools_call(req_id, registry, params or {})
    if method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"ok": True, "status": "pong"},
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unsupported method: {method}"},
    }


def run_stdio() -> int:
    registry = build_tool_registry()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = _handle_request(line, registry)
        if response is None:
            continue
        sys.stdout.write(json.dumps(response))
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


def run_list() -> int:
    registry = build_tool_registry()
    payload = [
        {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.input_schema,
        }
        for spec in registry.values()
    ]
    print(json.dumps(payload, indent=2))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="sn36 tools MCP wrapper")
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="print tool descriptions and exit",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="repo root override (for tests/dev)",
    )
    return parser


def main() -> int:
    global ROOT
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.root:
        ROOT = Path(args.root).resolve()
    _load_env_file()
    if args.list_tools:
        return run_list()

    return run_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
