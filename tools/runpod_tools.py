#!/usr/bin/env python3
"""RunPod helper tools for subnet operator workflows.

This module mirrors the existing RunPod integration contract used in the daryxx MCP
workspace. It intentionally keeps credentials out of code and reads credentials from
environment variables only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


BASE_URL = "https://api.runpod.io/graphql"


class RunPodError(Exception):
    pass


def _run_query(api_key: str, query: str, variables: dict[str, Any] | None = None, timeout: int = 45) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables

    try:
        response = httpx.post(
            f"{BASE_URL}?api_key={api_key}",
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
    except Exception as exc:
        raise RunPodError(f"RunPod request failed: {exc}") from exc

    if response.status_code >= 400:
        raise RunPodError(f"RunPod HTTP error ({response.status_code}): {response.text[:512]}")

    try:
        data = response.json()
    except Exception as exc:
        raise RunPodError(f"Invalid RunPod JSON response: {response.text[:512]}") from exc

    if "errors" in data:
        first = data["errors"][0] if data["errors"] else {}
        raise RunPodError(f"RunPod GraphQL error: {first.get('message', 'Unknown error')}")

    return data.get("data", {})


def _api_key() -> str:
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not key:
        raise RunPodError("RUNPOD_API_KEY is required for RunPod operations.")
    return key


def _read_variables(raw: str | None, path: str | None) -> dict[str, Any] | None:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif raw:
        payload = json.loads(raw)
    else:
        return None

    if not isinstance(payload, dict):
        raise RunPodError("variables must be a JSON object.")
    return payload


def _print(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_list_pods(_: argparse.Namespace) -> int:
    query = """
    query {
        myself {
            pods {
                id
                name
                runtime { uptimeInSeconds }
                desiredStatus
                costPerHr
                gpuCount
                machineId
                machine { gpuDisplayName }
            }
        }
    }
    """
    data = _run_query(_api_key(), query)
    pods = data.get("myself", {}).get("pods", [])

    rows = [
        {
            "id": pod.get("id"),
            "name": pod.get("name"),
            "status": pod.get("desiredStatus"),
            "gpu_count": pod.get("gpuCount"),
            "gpu_type": pod.get("machine", {}).get("gpuDisplayName"),
            "cost_per_hour": pod.get("costPerHr"),
            "uptime_seconds": pod.get("runtime", {}).get("uptimeInSeconds"),
            "machine_id": pod.get("machineId"),
        }
        for pod in pods
    ]
    return _print({"ok": True, "pods": rows, "count": len(rows)})


def cmd_get_pod(args: argparse.Namespace) -> int:
    query = """
    query ($podId: String!) {
        pod(input: {podId: $podId}) {
            id
            name
            runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } }
            desiredStatus
            costPerHr
            gpuCount
            memoryInGb
            vcpuCount
            machine { gpuDisplayName diskMb memoryInGb }
        }
    }
    """
    data = _run_query(_api_key(), query, {"podId": args.pod_id})
    pod = data.get("pod", {})
    payload = {
        "ok": True,
        "id": pod.get("id"),
        "name": pod.get("name"),
        "status": pod.get("desiredStatus"),
        "gpu_count": pod.get("gpuCount"),
        "vcpu_count": pod.get("vcpuCount"),
        "memory_gb": pod.get("memoryInGb"),
        "gpu_type": pod.get("machine", {}).get("gpuDisplayName"),
        "cost_per_hour": pod.get("costPerHr"),
        "uptime_seconds": pod.get("runtime", {}).get("uptimeInSeconds"),
        "gpu_utilization": [
            {"gpu_util": item.get("gpuUtilPercent"), "memory_util": item.get("memoryUtilPercent")}
            for item in pod.get("runtime", {}).get("gpus", [])
            if isinstance(item, dict)
        ],
    }
    return _print(payload)


def cmd_create_pod(args: argparse.Namespace) -> int:
    if not args.allow_side_effects:
        raise RunPodError("Create pod requires --allow-side-effects=true")

    query = """
    mutation ($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
        }
    }
    """
    variables = {
        "input": {
            "name": args.name,
            "gpuTypeId": args.gpu_type_id,
            "gpuCount": args.gpu_count,
            "imageName": args.docker_image,
            "volumeInGb": args.volume_in_gb,
            "cloudType": args.cloud_type,
        }
    }
    if args.template_id:
        variables["input"]["templateId"] = args.template_id

    data = _run_query(_api_key(), query, variables)
    pod = data.get("podFindAndDeployOnDemand", {})
    return _print({"ok": True, "id": pod.get("id"), "name": pod.get("name"), "status": pod.get("desiredStatus")})


def cmd_stop_pod(args: argparse.Namespace) -> int:
    if not args.allow_side_effects:
        raise RunPodError("Stop pod requires --allow-side-effects=true")

    query = """
    mutation ($input: PodStopInput!) {
        podStop(input: $input) {
            id
            desiredStatus
        }
    }
    """
    data = _run_query(_api_key(), query, {"input": {"podId": args.pod_id}})
    pod = data.get("podStop", {})
    return _print({"ok": True, "id": pod.get("id"), "status": pod.get("desiredStatus")})


def cmd_resume_pod(args: argparse.Namespace) -> int:
    if not args.allow_side_effects:
        raise RunPodError("Resume pod requires --allow-side-effects=true")

    query = """
    mutation ($input: PodResumeInput!) {
        podResume(input: $input) {
            id
            desiredStatus
        }
    }
    """
    input_data = {"podId": args.pod_id}
    if args.gpu_count:
        input_data["gpuCount"] = args.gpu_count
    data = _run_query(_api_key(), query, {"input": input_data})
    pod = data.get("podResume", {})
    return _print({"ok": True, "id": pod.get("id"), "status": pod.get("desiredStatus")})


def cmd_terminate_pod(args: argparse.Namespace) -> int:
    if not args.allow_side_effects:
        raise RunPodError("Terminate pod requires --allow-side-effects=true")

    query = """
    mutation ($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    _run_query(_api_key(), query, {"input": {"podId": args.pod_id}})
    return _print({"ok": True, "id": args.pod_id, "status": "terminated"})


def cmd_get_balance(_: argparse.Namespace) -> int:
    query = """
    query {
        myself {
            currentSpendPerHr
            creditBalance
            totalSpent
        }
    }
    """
    data = _run_query(_api_key(), query)
    payload = data.get("myself", {})
    return _print(
        {
            "ok": True,
            "credit_balance": payload.get("creditBalance"),
            "current_spend_per_hour": payload.get("currentSpendPerHr"),
            "total_spent": payload.get("totalSpent"),
        }
    )


def cmd_list_gpu_types(_: argparse.Namespace) -> int:
    query = """
    query {
        gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            lowestPrice { minimumBidPrice uninterruptablePrice }
        }
    }
    """
    data = _run_query(_api_key(), query)
    items = data.get("gpuTypes", [])
    rows = [
        {
            "id": item.get("id"),
            "name": item.get("displayName"),
            "memory_gb": item.get("memoryInGb"),
            "secure_cloud": item.get("secureCloud"),
            "community_cloud": item.get("communityCloud"),
            "min_bid_price": item.get("lowestPrice", {}).get("minimumBidPrice"),
            "on_demand_price": item.get("lowestPrice", {}).get("uninterruptablePrice"),
        }
        for item in items
    ]
    return _print({"ok": True, "gpu_types": rows, "count": len(rows)})


def cmd_graphql(args: argparse.Namespace) -> int:
    query = args.query.strip()
    if not query:
        raise RunPodError("--query is required.")
    lowered = query.lower()
    if "mutation" in lowered and not args.allow_side_effects:
        raise RunPodError("GraphQL mutations require --allow-side-effects=true")

    vars_payload = _read_variables(args.variables, args.variables_file)
    data = _run_query(_api_key(), query, vars_payload, timeout=args.timeout)
    return _print({"ok": True, "result": data})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod management tools for sn36 ops")
    sub = parser.add_subparsers(dest="command", required=True)

    list_pods = sub.add_parser("list-pods", help="list all pods")
    list_pods.set_defaults(func=cmd_list_pods)

    get_pod = sub.add_parser("get-pod", help="get pod details")
    get_pod.add_argument("--pod-id", required=True)
    get_pod.set_defaults(func=cmd_get_pod)

    create_pod = sub.add_parser("create-pod", help="create and deploy pod (requires side effects)")
    create_pod.add_argument("--name", required=True)
    create_pod.add_argument("--gpu-type-id", required=True)
    create_pod.add_argument("--gpu-count", type=int, default=1)
    create_pod.add_argument("--template-id", default=None)
    create_pod.add_argument("--docker-image", default="runpod/pytorch:latest")
    create_pod.add_argument("--volume-in-gb", type=int, default=20)
    create_pod.add_argument("--cloud-type", default="SECURE")
    create_pod.add_argument("--allow-side-effects", action="store_true", help="required for create/stop/resume/terminate")
    create_pod.set_defaults(func=cmd_create_pod)

    stop_pod = sub.add_parser("stop-pod", help="stop a pod (requires side effects)")
    stop_pod.add_argument("--pod-id", required=True)
    stop_pod.add_argument("--allow-side-effects", action="store_true", help="required for stop/resume/terminate")
    stop_pod.set_defaults(func=cmd_stop_pod)

    resume_pod = sub.add_parser("resume-pod", help="resume a pod (requires side effects)")
    resume_pod.add_argument("--pod-id", required=True)
    resume_pod.add_argument("--gpu-count", type=int, default=1)
    resume_pod.add_argument("--allow-side-effects", action="store_true", help="required for stop/resume/terminate")
    resume_pod.set_defaults(func=cmd_resume_pod)

    terminate_pod = sub.add_parser("terminate-pod", help="terminate a pod (requires side effects)")
    terminate_pod.add_argument("--pod-id", required=True)
    terminate_pod.add_argument("--allow-side-effects", action="store_true", help="required for stop/resume/terminate")
    terminate_pod.set_defaults(func=cmd_terminate_pod)

    get_balance = sub.add_parser("get-balance", help="get account balance")
    get_balance.set_defaults(func=cmd_get_balance)

    gpu_types = sub.add_parser("list-gpu-types", help="list available GPU types and prices")
    gpu_types.set_defaults(func=cmd_list_gpu_types)

    graphql = sub.add_parser("graphql", help="run arbitrary runpod GraphQL (mutations need allow-side-effects)")
    graphql.add_argument("--query", required=True)
    graphql.add_argument("--variables", default=None, help='JSON object, e.g. "{\"podId\":\"...\"}"')
    graphql.add_argument("--variables-file", default=None, help="path to JSON file with GraphQL variables")
    graphql.add_argument("--timeout", type=int, default=45)
    graphql.add_argument("--allow-side-effects", action="store_true")
    graphql.set_defaults(func=cmd_graphql)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except RunPodError as exc:
        print(f"[runpod_tools] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
