#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OPERATOR_ROOT = SCRIPT_DIR.parents[1]
if str(OPERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_ROOT))

from training.exporters import export_sft, load_cleaned_trajectories


DEFAULT_SYSTEM_PROMPT = (
    "You are a web automation planner. Given a target URL and a natural-language task, "
    "return only a JSON array of actions to complete the task."
)


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export SFT train/val JSONL from cleaned trajectory records")
    ap.add_argument("--cleaned-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--no-system-prompt", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _args()
    records = load_cleaned_trajectories(Path(args.cleaned_jsonl))

    result = export_sft(
        records=records,
        out_dir=Path(args.out_dir).resolve(),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        system_prompt=None if args.no_system_prompt else str(args.system_prompt),
    )

    manifest_path = Path(args.out_dir).resolve() / "sft_manifest.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = dict(result.get("stats") or {})
    payload["manifest"] = str(manifest_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
