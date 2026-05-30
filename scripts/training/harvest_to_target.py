#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
IWA_ROOT = REPO_ROOT.parent / 'autoppia_iwa'
USE_CASE_FILES = {
    'autocinema': IWA_ROOT / 'autoppia_iwa' / 'src' / 'demo_webs' / 'projects' / 'p01_autocinema' / 'use_cases.py',
    'autobooks': IWA_ROOT / 'autoppia_iwa' / 'src' / 'demo_webs' / 'projects' / 'p02_autobooks' / 'use_cases.py',
}

os.environ['DEMO_WEBS_ENDPOINT'] = 'http://84.247.180.192'
os.environ['DEMO_WEB_SERVICE_PORT'] = '8090'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Harvest gold trajectories until target count per use case')
    ap.add_argument('--project-id', required=True, choices=sorted(USE_CASE_FILES))
    ap.add_argument('--target', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=5)
    ap.add_argument('--prompts-per-use-case', type=int, default=30)
    ap.add_argument('--task-cache', default='')
    ap.add_argument('--use-case', action='append', default=[])
    ap.add_argument('--timeout-seconds', type=int, default=180)
    ap.add_argument('--max-attempts', type=int, default=1)
    ap.add_argument('--stop-on-failure', action='store_true')
    return ap.parse_args()


def parse_use_cases(project_id: str) -> list[str]:
    text = USE_CASE_FILES[project_id].read_text(encoding='utf-8')
    names = re.findall(r'name="([A-Z_]+)"', text)
    seen: list[str] = []
    for name in names:
        if name == 'SHOPPING_CART':
            continue
        if name not in seen:
            seen.append(name)
    return seen


def task_cache_path(project_id: str, explicit: str) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (REPO_ROOT / 'tmp' / f'{project_id}_tasks_cache_full.json').resolve()


def ensure_task_cache(project_id: str, cache_path: Path, prompts_per_use_case: int, use_cases: list[str]) -> Path:
    if cache_path.exists():
        return cache_path
    fallback = REPO_ROOT / 'tmp' / f'{project_id}_tasks_cache.json'
    if fallback.exists():
        return fallback.resolve()
    raw_path = cache_path.with_suffix('.raw.json')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(IWA_ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    cmd = [
        sys.executable,
        '-m',
        'autoppia_iwa.entrypoints.generate_tasks.run',
        '-p', project_id,
        '-n', str(int(prompts_per_use_case)),
        '-o', str(raw_path),
    ]
    for uc in use_cases:
        cmd.extend(['-u', uc])
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'Unable to generate task cache for {project_id}. Existing cache not found and task generation failed: {exc}'
        ) from exc
    payload = json.loads(raw_path.read_text(encoding='utf-8'))
    project_payload = payload.get(project_id)
    if not isinstance(project_payload, dict):
        raise RuntimeError(f'project {project_id} missing from generated cache')
    flat = {
        'project_id': project_payload.get('project_id', project_id),
        'project_name': project_payload.get('project_name', project_id),
        'tasks': project_payload.get('tasks', []),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return cache_path


def load_tasks(cache_path: Path) -> list[dict]:
    payload = json.loads(cache_path.read_text(encoding='utf-8'))
    rows = payload.get('tasks') if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError(f'unexpected task cache format: {cache_path}')
    return [row for row in rows if isinstance(row, dict)]


def extract_seed(url: str) -> int | None:
    try:
        parsed = urlparse(str(url))
        value = (parse_qs(parsed.query).get('seed') or [None])[0]
        return int(value) if value is not None else None
    except Exception:
        return None


def candidate_seeds(tasks: list[dict], project_id: str, use_case: str, fallback_count: int) -> list[int]:
    seeds: list[int] = []
    matched_rows = 0
    for row in tasks:
        if str(row.get('web_project_id') or '').strip() != project_id:
            continue
        uc = ((row.get('use_case') or {}).get('name') if isinstance(row.get('use_case'), dict) else None)
        if str(uc or '').strip().upper() != str(use_case).strip().upper():
            continue
        matched_rows += 1
        seed = extract_seed(str(row.get('url') or ''))
        if seed is not None:
            seeds.append(seed)
    uniq = sorted(set(seeds))
    if uniq:
        return uniq
    if matched_rows:
        return list(range(1, int(fallback_count) + 1))
    return []


def existing_gold_seeds(project_id: str, use_case: str) -> list[int]:
    path = REPO_ROOT / 'data' / project_id / use_case.lower() / 'gold' / 'episodes.jsonl'
    if not path.exists():
        return []
    seeds: set[int] = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        if not bool(row.get('success')) or float(row.get('score') or 0.0) < 1.0:
            continue
        try:
            seeds.add(int(row.get('seed')))
        except Exception:
            continue
    return sorted(seeds)


def run_harvest(project_id: str, use_case: str, seeds: list[int], cache_path: Path, timeout_seconds: int, max_attempts: int) -> int:
    seed_arg = ','.join(str(seed) for seed in seeds)
    cmd = [
        sys.executable,
        'scripts/training/harvest_gold.py',
        '--web-project-id', project_id,
        '--use-case', use_case,
        '--seeds', seed_arg,
        '--task-cache', str(cache_path),
        '--timeout-seconds', str(int(timeout_seconds)),
        '--max-attempts', str(int(max_attempts)),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    requested_use_cases = [uc.strip().upper() for uc in args.use_case if uc.strip()]
    all_use_cases = parse_use_cases(args.project_id)
    selected_use_cases = requested_use_cases or all_use_cases
    cache_path = ensure_task_cache(args.project_id, task_cache_path(args.project_id, args.task_cache), args.prompts_per_use_case, selected_use_cases)
    tasks = load_tasks(cache_path)
    report: list[dict] = []

    for use_case in selected_use_cases:
        available = candidate_seeds(tasks, args.project_id, use_case, max(int(args.target), int(args.prompts_per_use_case)))
        gold = existing_gold_seeds(args.project_id, use_case)
        missing = [seed for seed in available if seed not in set(gold)]
        needed = max(0, int(args.target) - len(gold))
        batches = [missing[i:i + int(args.batch_size)] for i in range(0, min(len(missing), needed), int(args.batch_size))]
        row = {
            'use_case': use_case,
            'available_task_seeds': len(available),
            'existing_gold': len(gold),
            'needed': needed,
            'planned_batches': len(batches),
        }
        print(json.dumps({'stage': 'plan', **row}, ensure_ascii=False))
        report.append(row)
        for batch in batches:
            print(json.dumps({'stage': 'run', 'use_case': use_case, 'batch': batch}, ensure_ascii=False), flush=True)
            code = run_harvest(args.project_id, use_case, batch, cache_path, args.timeout_seconds, args.max_attempts)
            gold = existing_gold_seeds(args.project_id, use_case)
            print(json.dumps({'stage': 'post', 'use_case': use_case, 'batch': batch, 'returncode': code, 'gold_now': len(gold)}, ensure_ascii=False), flush=True)
            if code != 0 and args.stop_on_failure:
                return code
            if len(gold) >= int(args.target):
                break
    final = []
    for use_case in selected_use_cases:
        final.append({'use_case': use_case, 'gold': len(existing_gold_seeds(args.project_id, use_case))})
    print(json.dumps({'project_id': args.project_id, 'target': int(args.target), 'final': final}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
