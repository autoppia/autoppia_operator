#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a multi-use-case autocinema training campaign')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--use-cases', default='all')
    parser.add_argument('--existing-pod-id', required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lora-rank', type=int, default=32)
    parser.add_argument('--all-use-cases-eval-tasks', type=int, default=3)
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/eval/run_autocinema_campaign.py'),
        '--run-name', args.run_name,
        '--use-cases', args.use_cases,
        '--existing-pod-id', args.existing_pod_id,
        '--epochs', str(int(args.epochs)),
        '--lora-rank', str(int(args.lora_rank)),
        '--all-use-cases-eval-tasks', str(int(args.all_use_cases_eval_tasks)),
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == '__main__':
    raise SystemExit(main())
