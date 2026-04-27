#!/usr/bin/env python3
"""Export operator gold episodes to the IWA harvested_trajectories format.

The operator's gold/episodes.jsonl rows are flat (no task/tests/actions inline).
IWA's verify script expects rows with nested `task`, top-level `tests`, and `actions`.
The operator uses `selector_candidates` (array); IWA expects `selector` (single object).

This script bridges the two formats so that harvested gold trajectories can be
verified and replayed by the IWA benchmark.

Usage:
    python scripts/eval/export_gold_to_iwa.py \\
        --project-id autocinema \\
        --data-root data \\
        --task-cache data/task_cache/tasks_cache.json \\
        --iwa-root ../autoppia_iwa \\
        [--use-cases CONTACT,LOGOUT] \\
        [--limit 0]

Output:
    autoppia_iwa/src/demo_webs/projects/p01_autocinema/
        harvested_trajectories/<USE_CASE>/successful_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# IWA project slug → folder name mapping (pNN_<slug>)
# ---------------------------------------------------------------------------
_IWA_PROJECT_FOLDER: dict[str, str] = {
    "autocinema": "p01_autocinema",
    "autobooks": "p02_autobooks",
    "autozone": "p03_autozone",
    "autodining": "p04_autodining",
    "autocrm": "p05_autocrm",
    "automail": "p06_automail",
    "autodelivery": "p07_autodelivery",
    "autolodge": "p08_autolodge",
    "autoconnect": "p09_autoconnect",
    "autowork": "p10_autowork",
    "autocalendar": "p11_autocalendar",
    "autolist": "p12_autolist",
    "autodrive": "p13_autodrive",
    "autohealth": "p14_autohealth",
    "autostats": "p15_autostats",
    "autodiscord": "p16_autodiscord",
}


# ---------------------------------------------------------------------------
# Action conversion: selector_candidates → selector
# ---------------------------------------------------------------------------

def _convert_action(action: dict[str, Any]) -> dict[str, Any]:
    """Convert an operator-format action to IWA format.

    The operator uses `selector_candidates` (list of fallback selectors tried
    in order by the step engine). IWA's action models expect a single `selector`
    field. We pick the first candidate — typically the most specific one.
    """
    candidates = action.get("selector_candidates")
    result = {k: v for k, v in action.items() if k != "selector_candidates"}
    if candidates and isinstance(candidates, list) and "selector" not in result:
        result["selector"] = candidates[0]
    return result


# ---------------------------------------------------------------------------
# Task cache helpers
# ---------------------------------------------------------------------------

def _load_task_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Return a mapping of USE_CASE_NAME → task dict from the task cache."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = payload.get("tasks", [])
    lookup: dict[str, dict[str, Any]] = {}
    for task in tasks:
        use_case = task.get("use_case", {})
        if isinstance(use_case, dict):
            name = str(use_case.get("name", "")).strip().upper()
        else:
            name = str(use_case).strip().upper()
        if name:
            lookup[name] = task
    return lookup


# ---------------------------------------------------------------------------
# Episode loading
# ---------------------------------------------------------------------------

def _read_episodes(episodes_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_candidate_actions(candidate_path_str: str) -> list[dict[str, Any]]:
    candidate_file = Path(candidate_path_str)
    if not candidate_file.exists():
        return []
    payload = json.loads(candidate_file.read_text(encoding="utf-8"))
    actions = payload.get("actions", [])
    return [dict(a) for a in actions if isinstance(a, dict)]


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _start_url_from_actions(actions: list[dict[str, Any]], fallback: str) -> str:
    """Return a reachable starting URL for the task.

    The verify script calls _url_available(task.url) before executing any actions.
    We use the task cache base URL (e.g. http://localhost:8000/) with the seed
    appended as a query param — this is always reachable.

    The NavigateActions in the actions list will handle the actual navigation
    during playback; task.url only needs to be a valid starting point.
    """
    # Extract the seed from the first NavigateAction URL if present
    for action in actions:
        if str(action.get("type", "")).lower() in ("navigateaction", "navigate"):
            url = str(action.get("url", "")).strip()
            if url:
                # Use the origin (scheme + host + port) from the first navigate action
                # combined with the seed query param, to guarantee reachability
                from urllib.parse import urlparse, urlencode
                parsed = urlparse(url)
                origin = f"{parsed.scheme}://{parsed.netloc}/"
                seed_param = None
                if parsed.query:
                    for part in parsed.query.split("&"):
                        if part.startswith("seed="):
                            seed_param = part
                            break
                if seed_param:
                    return f"{origin}?{seed_param}"
                return origin
    return fallback
    return fallback


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_iwa_row(
    *,
    episode: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
    task_template: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a row in the IWA harvested_trajectories JSONL format."""
    use_case = str(episode.get("use_case", "")).strip().upper()
    seed = int(episode.get("seed", 0))
    task_id = str(episode.get("task_id", "") or f"{use_case.lower()}_{seed}")

    # Derive the starting URL from the first NavigateAction (seed-specific)
    fallback_url = str(episode.get("final_url", "http://localhost:8000/"))
    start_url = _start_url_from_actions(candidate_actions, fallback_url)

    # Task template provides prompt and tests
    prompt = ""
    tests: list[dict[str, Any]] = []
    if task_template:
        prompt = str(task_template.get("prompt", ""))
        tests = [dict(t) for t in (task_template.get("tests") or []) if isinstance(t, dict)]

    # Convert actions to IWA format
    iwa_actions = [_convert_action(a) for a in candidate_actions]

    return {
        "task": {
            "id": task_id,
            "url": start_url,
            "prompt": prompt,
            "web_project_id": str(episode.get("web_project_id", "autocinema")),
            "is_web_real": False,
            "specifications": {},
        },
        "tests": tests,
        "actions": iwa_actions,
        # Metadata kept for traceability
        "use_case": use_case,
        "seed": seed,
        "score": float(episode.get("score", 1.0)),
        "attempt_name": str(episode.get("attempt_name", "")),
        "episode_task_id": str(episode.get("episode_task_id", "")),
    }


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def export_project(
    *,
    project_id: str,
    data_root: Path,
    task_cache_path: Path,
    iwa_root: Path,
    use_case_filter: set[str],
    limit: int,
) -> dict[str, int]:
    iwa_folder = _IWA_PROJECT_FOLDER.get(project_id.lower())
    if not iwa_folder:
        print(f"ERROR unknown project_id '{project_id}'. Known: {list(_IWA_PROJECT_FOLDER)}", file=sys.stderr)
        return {}

    task_lookup = _load_task_cache(task_cache_path)
    harvested_root = iwa_root / "autoppia_iwa" / "src" / "demo_webs" / "projects" / iwa_folder / "harvested_trajectories"
    project_data_root = data_root / project_id

    if not project_data_root.exists():
        print(f"ERROR project data dir not found: {project_data_root}", file=sys.stderr)
        return {}

    stats: dict[str, int] = {}

    for episodes_path in sorted(project_data_root.rglob("gold/episodes.jsonl")):
        use_case_dir = episodes_path.parent.parent.name.upper()
        use_case = use_case_dir

        if use_case_filter and use_case not in use_case_filter:
            continue

        episodes = _read_episodes(episodes_path)
        if not episodes:
            print(f"  SKIP {use_case}: no gold episodes found")
            continue

        # Apply limit
        if limit > 0:
            episodes = episodes[:limit]

        task_template = task_lookup.get(use_case)
        if not task_template:
            print(f"  WARN {use_case}: no task template in task cache — tests will be empty")

        rows: list[dict[str, Any]] = []
        skipped = 0
        for episode in episodes:
            candidate_path_str = str(episode.get("candidate_path", ""))
            if not candidate_path_str:
                skipped += 1
                continue

            candidate_actions = _load_candidate_actions(candidate_path_str)
            if not candidate_actions:
                skipped += 1
                print(f"  WARN {use_case} seed={episode.get('seed')}: candidate has no actions, skipping")
                continue

            rows.append(_build_iwa_row(
                episode=episode,
                candidate_actions=candidate_actions,
                task_template=task_template,
            ))

        if not rows:
            print(f"  SKIP {use_case}: all {len(episodes)} episodes failed to build rows (skipped={skipped})")
            continue

        # Write JSONL
        out_dir = harvested_root / use_case
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "successful_trajectories.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats[use_case] = len(rows)
        print(f"  OK  {use_case}: {len(rows)} trajectories → {out_path}")
        if skipped:
            print(f"      (skipped {skipped} episodes with missing candidate)")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export operator gold episodes to IWA harvested_trajectories format.",
    )
    parser.add_argument("--project-id", default="autocinema", help="Web project ID (e.g. autocinema)")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Operator data root dir (default: data/)",
    )
    parser.add_argument(
        "--task-cache",
        default="data/task_cache/tasks_cache.json",
        help="Path to the task cache JSON",
    )
    parser.add_argument(
        "--iwa-root",
        default="../autoppia_iwa",
        help="Path to the autoppia_iwa repo root (default: ../autoppia_iwa)",
    )
    parser.add_argument(
        "--use-cases",
        default="",
        help="Optional comma-separated USE_CASE filter (e.g. CONTACT,LOGOUT). Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max episodes per use case. 0 = all (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root).resolve()
    task_cache_path = Path(args.task_cache).resolve()
    iwa_root = Path(args.iwa_root).resolve()

    if not task_cache_path.exists():
        print(f"ERROR task cache not found: {task_cache_path}", file=sys.stderr)
        raise SystemExit(2)

    if not iwa_root.exists():
        print(f"ERROR iwa_root not found: {iwa_root}", file=sys.stderr)
        raise SystemExit(2)

    use_case_filter: set[str] = {
        value.strip().upper()
        for value in str(args.use_cases).split(",")
        if value.strip()
    }

    print(f"Exporting gold trajectories from {data_root}/{args.project_id} → IWA")
    print(f"  iwa_root:   {iwa_root}")
    print(f"  task_cache: {task_cache_path}")
    if use_case_filter:
        print(f"  filter:     {sorted(use_case_filter)}")
    if args.limit:
        print(f"  limit:      {args.limit} per use case")
    print()

    stats = export_project(
        project_id=args.project_id,
        data_root=data_root,
        task_cache_path=task_cache_path,
        iwa_root=iwa_root,
        use_case_filter=use_case_filter,
        limit=args.limit,
    )

    print()
    total = sum(stats.values())
    print(f"Export complete: {len(stats)} use cases, {total} total trajectories")
    if not stats:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
