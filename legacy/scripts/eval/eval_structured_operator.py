#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval.py against the structured inference operator")
    parser.add_argument('--project-id', default='autocinema')
    parser.add_argument('--use-case', required=True)
    parser.add_argument('--num-tasks', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--task-concurrency', type=int, default=1)
    parser.add_argument('--provider', default='openai')
    parser.add_argument('--model', default='gpt-5-mini')
    parser.add_argument('--out', default='tmp/structured_eval.json')
    args, extra = parser.parse_known_args()

    env = os.environ.copy()
    env['WEB_AGENT_RUNTIME'] = 'structured'
    cmd = [
        sys.executable,
        str(ROOT / 'eval.py'),
        '--provider', args.provider,
        '--model', args.model,
        '--web-project-id', args.project_id,
        '--use-case', args.use_case,
        '--num-tasks', str(int(args.num_tasks)),
        '--repeat', str(int(args.repeat)),
        '--seed', str(int(args.seed)),
        '--max-steps', str(int(args.max_steps)),
        '--task-concurrency', str(int(args.task_concurrency)),
        '--out', str(args.out),
    ]
    cmd.extend(extra)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == '__main__':
    raise SystemExit(main())
