#!/usr/bin/env python3
"""Autocinema trajectory harvesting entrypoint.

Builds a replayable dataset of successful and failed episodes across all
Autocinema use cases from one or more eval result JSON files. It can also run
fresh eval collection with persisted traces, then emit training-oriented
artifacts:

- summary.json
- episodes.jsonl
- collection_manifest.json
- golden_seeds.json
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUTOCINEMA_USE_CASES = [
    "ADD_COMMENT",
    "ADD_FILM",
    "ADD_TO_WATCHLIST",
    "CONTACT",
    "DELETE_FILM",
    "EDIT_FILM",
    "EDIT_USER",
    "FILM_DETAIL",
    "FILTER_FILM",
    "LOGIN",
    "LOGOUT",
    "REGISTRATION",
    "REMOVE_FROM_WATCHLIST",
    "SEARCH_FILM",
    "SHARE_MOVIE",
    "WATCH_TRAILER",
]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "autocinema_trajectory_harvest"
DEFAULT_RESULT_GLOB = str(ROOT / "data" / "eval_parallel_gpt-5.2_autocinema_16x10.json")
DEFAULT_IWA_REPO = Path("/home/usuario1/daryxx/autoppia/operator/autoppia_iwa")
DEFAULT_TRACE_ROOT_GLOBS = [
    str(ROOT / "data" / "act_traces" / "*"),
    str(DEFAULT_OUT_DIR / "raw_eval_runs" / "traces_*"),
]


@dataclass
class HarvestArtifacts:
    summary: dict[str, Any]
    episodes: list[dict[str, Any]]
    manifest: dict[str, Any]
    golden_seeds: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_branch(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_eval_once(
    *,
    provider: str,
    model: str,
    project_id: str,
    repeat: int,
    seed_start: int,
    max_steps: int,
    task_concurrency: int,
    result_path: Path,
    trace_dir: Path,
    task_cache: str | None,
    include_reasoning: bool,
    use_site_knowledge: bool,
    use_local_html_context: bool,
    failure_judge: bool,
    temperature: float,
    use_case: str | None,
    eval_timeout_seconds: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "eval.py"),
        "--provider",
        provider,
        "--model",
        model,
        "--web-project-id",
        project_id,
        "--repeat",
        str(max(1, int(repeat))),
        "--seed",
        str(int(seed_start)),
        "--max-steps",
        str(max(1, int(max_steps))),
        "--task-concurrency",
        str(max(1, int(task_concurrency))),
        "--temperature",
        str(float(temperature)),
        "--save-act-traces",
        "--trace-dir",
        str(trace_dir),
        "--out",
        str(result_path),
    ]
    if use_case:
        cmd.extend(["--use-case", str(use_case)])
        cmd.extend(["--num-tasks", "1"])
    else:
        cmd.extend(["--all-use-cases", "--tasks-per-use-case", "1"])
    if include_reasoning:
        cmd.append("--include-reasoning")
    if use_site_knowledge:
        cmd.append("--use-site-knowledge")
    if use_local_html_context:
        cmd.append("--use-local-html-context")
    if not failure_judge:
        cmd.append("--no-failure-judge")
    if task_cache:
        cmd.extend(["--task-cache", str(task_cache)])
    print(f"[harvest] running eval: {' '.join(cmd)}")
    subprocess.run(
        cmd,
        check=True,
        cwd=str(ROOT),
        timeout=(None if not eval_timeout_seconds else int(eval_timeout_seconds)),
    )
    return [str(part) for part in cmd]


def _failure_category(ep: dict[str, Any], near_miss_threshold: float) -> str:
    if bool(ep.get("success")):
        return "SUCCESS"
    for key in ("judge_failure_category", "failure_category"):
        value = ep.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    score = float(ep.get("score") or 0.0)
    if score >= float(near_miss_threshold):
        return "NEAR_MISS"
    if score <= 0.0:
        return "ZERO_SCORE"
    return "PARTIAL_FAILURE"


def _load_result_paths(glob_patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in glob_patterns:
        if any(ch in pattern for ch in "*?[]"):
            if pattern.startswith("/"):
                matches = sorted(Path(p) for p in glob.glob(pattern))
            else:
                matches = sorted(ROOT.glob(pattern))
            paths.extend(p.resolve() for p in matches if p.is_file())
            continue
        p = Path(pattern)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.is_file():
            paths.append(p)
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _load_trace_roots(glob_patterns: list[str]) -> list[Path]:
    roots: list[Path] = []
    for pattern in glob_patterns:
        if any(ch in pattern for ch in "*?[]"):
            if pattern.startswith("/"):
                matches = sorted(Path(p) for p in glob.glob(pattern))
            else:
                matches = sorted(ROOT.glob(pattern))
            roots.extend(p.resolve() for p in matches if p.is_dir())
            continue
        p = Path(pattern)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if p.is_dir():
            roots.append(p)
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root)
        if key in seen or not (root / "trace_index.json").is_file():
            continue
        seen.add(key)
        out.append(root)
    return out


def _trace_root_for_result(
    result_path: Path,
    explicit_trace_roots: list[Path],
    episode_task_ids: set[str],
) -> Path | None:
    stem = result_path.stem

    def _root_key(root: Path) -> str:
        key = root.name
        for prefix in ("traces_", "trace_"):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.endswith("_traces"):
            key = key[: -len("_traces")]
        return key

    result_keys = {stem}
    if stem.startswith("eval_"):
        result_keys.add(stem[len("eval_") :])
    valid_roots = [root for root in explicit_trace_roots if (root / "trace_index.json").is_file()]
    if valid_roots:
        matching_roots = []
        for root in valid_roots:
            key = _root_key(root)
            if key in result_keys or any(key.endswith(k) for k in result_keys):
                matching_roots.append(root)
        if len(matching_roots) == 1:
            return matching_roots[0]
        if len(matching_roots) > 1:
            return sorted(
                matching_roots,
                key=lambda r: (r / "trace_index.json").stat().st_mtime,
                reverse=True,
            )[0]

        overlap_matches: list[tuple[int, int, float, Path]] = []
        if episode_task_ids:
            for root in valid_roots:
                try:
                    payload = _json_load(root / "trace_index.json")
                except Exception:
                    continue
                rows = payload.get("episodes") if isinstance(payload, dict) else None
                if not isinstance(rows, list):
                    continue
                root_ids = {
                    str(row.get("episode_task_id"))
                    for row in rows
                    if isinstance(row, dict) and str(row.get("episode_task_id") or "").strip()
                }
                overlap = len(episode_task_ids & root_ids)
                if overlap <= 0:
                    continue
                overlap_matches.append(
                    (
                        overlap,
                        len(root_ids),
                        (root / "trace_index.json").stat().st_mtime,
                        root,
                    )
                )
            if overlap_matches:
                overlap_matches.sort(key=lambda item: (item[0], -item[1], item[2]), reverse=True)
                best_overlap, _, _, best_root = overlap_matches[0]
                if best_overlap == len(episode_task_ids):
                    return best_root
        if overlap_matches:
            return overlap_matches[0][3]
        if len(valid_roots) == 1:
            return valid_roots[0]

    candidate_dirs = [
        result_path.parent / f"traces_{stem}",
        result_path.parent / f"{stem}_traces",
        result_path.parent / "traces",
    ]
    for candidate in candidate_dirs:
        if (candidate / "trace_index.json").is_file():
            return candidate
    return None


def _load_trace_index(trace_root: Path | None) -> dict[str, dict[str, Any]]:
    if trace_root is None:
        return {}
    index_path = trace_root / "trace_index.json"
    if not index_path.is_file():
        return {}
    payload = _json_load(index_path)
    rows = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        episode_task_id = str(row.get("episode_task_id") or "").strip()
        if episode_task_id:
            out[episode_task_id] = row
    return out


def _load_eval_episodes(
    result_paths: list[Path],
    project_id: str,
    trace_roots: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    run_sources: list[dict[str, Any]] = []
    for result_path in result_paths:
        payload = _json_load(result_path)
        if not isinstance(payload, dict):
            continue
        rows = payload.get("episodes")
        if not isinstance(rows, list):
            continue
        episode_task_ids = {
            str(ep.get("episode_task_id"))
            for ep in rows
            if isinstance(ep, dict) and str(ep.get("episode_task_id") or "").strip()
        }
        trace_root = _trace_root_for_result(result_path, trace_roots, episode_task_ids)
        trace_index = _load_trace_index(trace_root)
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        run_sources.append(
            {
                "result_path": str(result_path),
                "trace_root": str(trace_root) if trace_root else None,
                "episodes": len(rows),
                "repeat": meta.get("repeat"),
                "seed_start": meta.get("seed_start"),
                "max_steps": meta.get("max_steps"),
            }
        )
        for ep in rows:
            if not isinstance(ep, dict):
                continue
            use_case = str(ep.get("use_case") or "").strip().upper()
            if not use_case:
                continue
            web_project_id = str(ep.get("web_project_id") or payload.get("web_project_id") or "").strip() or project_id
            if web_project_id != project_id:
                continue
            episode_task_id = str(ep.get("episode_task_id") or "").strip()
            trace_item = trace_index.get(episode_task_id) if episode_task_id else None
            trace_file_rel = str(trace_item.get("file") or "").strip() if isinstance(trace_item, dict) else ""
            trace_file = (trace_root / trace_file_rel).resolve() if trace_root and trace_file_rel else None
            episodes.append(
                {
                    **ep,
                    "use_case": use_case,
                    "web_project_id": web_project_id,
                    "_result_path": str(result_path),
                    "_trace_root": str(trace_root) if trace_root else None,
                    "_trace_file": str(trace_file) if trace_file and trace_file.is_file() else None,
                    "_trace_index_item": trace_item,
                }
            )
    return episodes, run_sources


def build_harvest_artifacts(
    *,
    project_id: str,
    result_paths: list[Path],
    trace_roots: list[Path],
    near_miss_threshold: float,
    branch: str,
    iwa_branch: str,
    require_trace_files: bool,
    command_sources: list[dict[str, Any]] | None = None,
) -> HarvestArtifacts:
    raw_episodes, run_sources = _load_eval_episodes(result_paths, project_id=project_id, trace_roots=trace_roots)
    per_use_case: dict[str, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []

    for use_case in AUTOCINEMA_USE_CASES:
        per_use_case[use_case] = {
            "attempted": 0,
            "successes": 0,
            "failures": 0,
            "near_miss": 0,
            "distinct_seeds": [],
            "result_paths": [],
            "trace_files": [],
            "failure_categories": {},
            "golden_seed_count": 0,
        }

    for ep in raw_episodes:
        use_case = str(ep.get("use_case") or "").strip().upper()
        if use_case not in per_use_case:
            continue
        success = bool(ep.get("success"))
        score = float(ep.get("score") or 0.0)
        seed = ep.get("seed")
        task_id = ep.get("task_id")
        episode_task_id = str(ep.get("episode_task_id") or "")
        result_path = str(ep.get("_result_path") or "")
        trace_root = str(ep.get("_trace_root") or "") or None
        trace_file = str(ep.get("_trace_file") or "") or None
        if require_trace_files and not trace_file:
            continue
        category = _failure_category(ep, near_miss_threshold=near_miss_threshold)
        trace_index_item = ep.get("_trace_index_item") if isinstance(ep.get("_trace_index_item"), dict) else {}
        final_url = trace_index_item.get("final_url") if isinstance(trace_index_item, dict) else None

        row = {
            "web_project_id": str(ep.get("web_project_id") or project_id),
            "use_case": use_case,
            "seed": (int(seed) if seed is not None else None),
            "task_id": (str(task_id) if task_id else None),
            "episode_task_id": (episode_task_id or None),
            "success": success,
            "score": score,
            "trace_root": trace_root,
            "trace_file": trace_file,
            "trace_dir": trace_root,
            "trace_ref": episode_task_id or f"{task_id}:{seed}",
            "result_path": result_path,
            "failure_category": (None if success else category),
            "judge_failure_category": ep.get("judge_failure_category"),
            "judge_failure_reasoning": ep.get("judge_failure_reasoning"),
            "steps": int(ep.get("steps") or 0),
            "model": ep.get("model"),
            "final_url": final_url,
            "notes": (None if success else f"failure_category={category}"),
        }
        normalized.append(row)

        bucket = per_use_case[use_case]
        bucket["attempted"] += 1
        if seed is not None and int(seed) not in bucket["distinct_seeds"]:
            bucket["distinct_seeds"].append(int(seed))
        if success:
            bucket["successes"] += 1
            if score >= 1.0:
                bucket["golden_seed_count"] += 1
        else:
            bucket["failures"] += 1
            if score >= float(near_miss_threshold):
                bucket["near_miss"] += 1
            categories = bucket["failure_categories"]
            categories[category] = int(categories.get(category) or 0) + 1
        if result_path and result_path not in bucket["result_paths"]:
            bucket["result_paths"].append(result_path)
        if trace_file and len(bucket["trace_files"]) < 25 and trace_file not in bucket["trace_files"]:
            bucket["trace_files"].append(trace_file)

    for use_case in AUTOCINEMA_USE_CASES:
        per_use_case[use_case]["distinct_seeds"] = sorted(per_use_case[use_case]["distinct_seeds"])

    seeds = sorted({int(ep["seed"]) for ep in normalized if ep.get("seed") is not None})
    successes = sum(1 for ep in normalized if bool(ep.get("success")))
    failures = sum(1 for ep in normalized if not bool(ep.get("success")))
    covered_use_cases = [uc for uc in AUTOCINEMA_USE_CASES if int(per_use_case[uc]["attempted"]) > 0]
    replayable_episodes = sum(1 for ep in normalized if ep.get("trace_file"))

    summary = {
        "project_id": project_id,
        "branch": branch,
        "iwa_branch": iwa_branch,
        "generated_at": _utc_now_iso(),
        "use_cases": covered_use_cases,
        "required_use_cases": list(AUTOCINEMA_USE_CASES),
        "seeds": seeds,
        "episodes_total": len(normalized),
        "successes_total": successes,
        "failures_total": failures,
        "replayable_episodes_total": replayable_episodes,
        "require_trace_files": bool(require_trace_files),
        "result_files": [str(p) for p in result_paths],
        "per_use_case": per_use_case,
    }

    golden_by_use_case: dict[str, list[dict[str, Any]]] = {}
    for use_case in AUTOCINEMA_USE_CASES:
        winners = [
            {
                "seed": ep["seed"],
                "task_id": ep["task_id"],
                "episode_task_id": ep["episode_task_id"],
                "score": ep["score"],
                "trace_file": ep["trace_file"],
                "result_path": ep["result_path"],
                "verify_command": [
                    sys.executable,
                    str(ROOT / "eval.py"),
                    "--provider",
                    str(ep.get("model") or "openai"),
                ],
            }
            for ep in normalized
            if ep["use_case"] == use_case and bool(ep["success"]) and float(ep.get("score") or 0.0) >= 1.0
        ]
        for winner in winners:
            winner["verify_command"] = [
                sys.executable,
                str(ROOT / "eval.py"),
                "--provider",
                "openai",
                "--model",
                str(next((ep.get("model") for ep in normalized if ep.get("episode_task_id") == winner["episode_task_id"]), "gpt-5.2")),
                "--task-id",
                str(winner["task_id"]),
                "--seed",
                str(winner["seed"]),
                "--repeat",
                "1",
                "--max-steps",
                "12",
                "--save-act-traces",
            ]
        if winners:
            golden_by_use_case[use_case] = winners

    manifest = {
        "generated_at": _utc_now_iso(),
        "project_id": project_id,
        "branch": branch,
        "iwa_branch": iwa_branch,
        "require_trace_files": bool(require_trace_files),
        "run_sources": run_sources,
        "command_sources": command_sources or [],
        "result_files": [str(p) for p in result_paths],
        "trace_roots": sorted({str(ep.get("trace_root")) for ep in normalized if ep.get("trace_root")}),
    }

    golden_seeds = {
        "generated_at": _utc_now_iso(),
        "project_id": project_id,
        "branch": branch,
        "iwa_branch": iwa_branch,
        "golden_by_use_case": golden_by_use_case,
    }
    return HarvestArtifacts(summary=summary, episodes=normalized, manifest=manifest, golden_seeds=golden_seeds)


def write_harvest_artifacts(*, artifacts: HarvestArtifacts, out_dir: Path) -> tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    episodes_path = out_dir / "episodes.jsonl"
    manifest_path = out_dir / "collection_manifest.json"
    golden_path = out_dir / "golden_seeds.json"
    _json_dump(summary_path, artifacts.summary)
    _json_dump(manifest_path, artifacts.manifest)
    _json_dump(golden_path, artifacts.golden_seeds)
    with episodes_path.open("w", encoding="utf-8") as f:
        for row in artifacts.episodes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary_path, episodes_path, manifest_path, golden_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest replayable Autocinema trajectories into a reusable dataset")
    parser.add_argument("--project-id", default="autocinema")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--result-glob",
        action="append",
        default=[],
        help="Result JSON path/glob to aggregate (repeatable).",
    )
    parser.add_argument(
        "--trace-root",
        action="append",
        default=[],
        help="Trace root directory containing trace_index.json (repeatable).",
    )
    parser.add_argument("--near-miss-threshold", type=float, default=0.5)
    parser.add_argument("--require-trace-files", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--run-eval", action="store_true", help="Run fresh eval.py first and include produced result JSON")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--repeat", type=int, default=3, help="Episodes per use case when --run-eval is enabled")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--task-concurrency", type=int, default=1)
    parser.add_argument("--task-cache", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--include-reasoning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-site-knowledge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-local-html-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--failure-judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-case",
        action="append",
        default=[],
        help="Use-case filter for --run-eval (repeatable). If omitted, harvest runs all use cases.",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=int,
        default=0,
        help="Per eval.py invocation timeout when --run-eval is enabled (0 disables timeout).",
    )
    parser.add_argument(
        "--iwa-repo",
        default=str(DEFAULT_IWA_REPO),
        help="Path used only to record IWA branch in summary metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    harvest_run_dir = out_dir / "raw_eval_runs"
    harvest_run_dir.mkdir(parents=True, exist_ok=True)

    result_patterns = list(args.result_glob) if args.result_glob else [DEFAULT_RESULT_GLOB]
    generated_result_paths: list[Path] = []
    trace_roots = _load_trace_roots(list(args.trace_root) if args.trace_root else list(DEFAULT_TRACE_ROOT_GLOBS))
    command_sources: list[dict[str, Any]] = []

    if bool(args.run_eval):
        requested_use_cases = [str(uc).strip().upper() for uc in (args.use_case or []) if str(uc).strip()]
        eval_units = requested_use_cases if requested_use_cases else [None]
        for idx, use_case in enumerate(eval_units):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            uc_suffix = f"_{str(use_case).lower()}" if use_case else ""
            result_path = harvest_run_dir / f"eval_{args.project_id}{uc_suffix}_{stamp}.json"
            trace_dir = harvest_run_dir / f"traces_eval_{args.project_id}{uc_suffix}_{stamp}"
            seed_start = int(args.seed_start) + idx * 1000
            command_row: dict[str, Any] = {
                "type": "fresh_eval",
                "use_case": use_case,
                "result_path": str(result_path),
                "trace_root": str(trace_dir),
            }
            try:
                cmd = _run_eval_once(
                    provider=str(args.provider),
                    model=str(args.model),
                    project_id=str(args.project_id),
                    repeat=max(1, int(args.repeat)),
                    seed_start=seed_start,
                    max_steps=max(1, int(args.max_steps)),
                    task_concurrency=max(1, int(args.task_concurrency)),
                    result_path=result_path,
                    trace_dir=trace_dir,
                    task_cache=args.task_cache,
                    include_reasoning=bool(args.include_reasoning),
                    use_site_knowledge=bool(args.use_site_knowledge),
                    use_local_html_context=bool(args.use_local_html_context),
                    failure_judge=bool(args.failure_judge),
                    temperature=float(args.temperature),
                    use_case=use_case,
                    eval_timeout_seconds=(None if int(args.eval_timeout_seconds) <= 0 else int(args.eval_timeout_seconds)),
                )
                command_row["command"] = cmd
                command_row["status"] = "ok"
                if result_path.is_file():
                    generated_result_paths.append(result_path)
                if (trace_dir / "trace_index.json").is_file():
                    trace_roots.append(trace_dir)
            except subprocess.TimeoutExpired:
                command_row["status"] = "timeout"
            except subprocess.CalledProcessError as exc:
                command_row["status"] = "error"
                command_row["returncode"] = int(exc.returncode)
            command_sources.append(command_row)

    resolved_result_paths = _load_result_paths(result_patterns)
    resolved_result_paths.extend(generated_result_paths)
    seen = set()
    final_result_paths: list[Path] = []
    for path in resolved_result_paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        final_result_paths.append(path.resolve())
    if not final_result_paths:
        print("[harvest] no result files found; provide --result-glob or enable --run-eval", file=sys.stderr)
        return 2

    artifacts = build_harvest_artifacts(
        project_id=str(args.project_id),
        result_paths=final_result_paths,
        trace_roots=trace_roots,
        near_miss_threshold=float(args.near_miss_threshold),
        branch=_git_branch(ROOT),
        iwa_branch=_git_branch(Path(args.iwa_repo)),
        require_trace_files=bool(args.require_trace_files),
        command_sources=command_sources,
    )
    summary_path, episodes_path, manifest_path, golden_path = write_harvest_artifacts(artifacts=artifacts, out_dir=out_dir)

    missing = [uc for uc in AUTOCINEMA_USE_CASES if int(artifacts.summary["per_use_case"].get(uc, {}).get("attempted", 0)) <= 0]
    low_seed = [
        uc
        for uc in AUTOCINEMA_USE_CASES
        if len(artifacts.summary["per_use_case"].get(uc, {}).get("distinct_seeds") or []) < 2
    ]
    no_gold = [
        uc
        for uc in AUTOCINEMA_USE_CASES
        if int(artifacts.summary["per_use_case"].get(uc, {}).get("golden_seed_count", 0)) <= 0
    ]
    print(f"[harvest] summary:   {summary_path}")
    print(f"[harvest] episodes:  {episodes_path}")
    print(f"[harvest] manifest:  {manifest_path}")
    print(f"[harvest] golden:    {golden_path}")
    print(
        f"[harvest] totals episodes={artifacts.summary['episodes_total']} "
        f"successes={artifacts.summary['successes_total']} failures={artifacts.summary['failures_total']} "
        f"replayable={artifacts.summary['replayable_episodes_total']}"
    )
    if missing:
        print(f"[harvest] missing use-cases: {', '.join(missing)}")
    if low_seed:
        print(f"[harvest] use-cases with <2 distinct seeds: {', '.join(low_seed)}")
    if no_gold:
        print(f"[harvest] use-cases with 0 golden seeds: {', '.join(no_gold)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
