#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
OPERATOR_ROOT = SCRIPT_DIR.parent
if str(OPERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_ROOT))

from training import (
    S3TrajectorySource,
    TrajectoryBuildConfig,
    export_training_bundle,
    ingest_from_iwap_api,
    ingest_from_s3,
)


SYSTEM_PROMPT_DEFAULT = (
    "You are a web automation planner. Given a target URL and a natural-language task, "
    "return only a JSON array of actions to complete the task."
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build training datasets from IWAP trajectories. "
            "Supports IWAP API run logs and direct S3 task-log ingestion."
        )
    )

    ap.add_argument("--source", choices=["iwap-api", "s3"], default="iwap-api", help="Trajectory source backend")

    # IWAP API mode
    ap.add_argument("--base-url", default=None, help="IWAP API base URL (required for --source iwap-api)")
    ap.add_argument("--token", default=None, help="Optional bearer token (or set IWAP_API_TOKEN)")
    ap.add_argument("--max-runs", type=int, default=None, help="Optional cap on number of runs")
    ap.add_argument("--page-limit", type=int, default=50, help="Run list page size")
    ap.add_argument("--include-unfinished", action="store_true", help="Include non-finalized runs")

    # S3 mode
    ap.add_argument("--s3-bucket", default=None, help="S3 bucket name for --source s3")
    ap.add_argument("--s3-prefix", default="", help="S3 key prefix for --source s3")
    ap.add_argument("--s3-region", default=None, help="Optional S3 region")
    ap.add_argument("--s3-profile", default=None, help="Optional AWS profile name")
    ap.add_argument("--s3-endpoint-url", default=None, help="Optional custom S3 endpoint (MinIO, etc.)")
    ap.add_argument("--s3-max-objects", type=int, default=None, help="Optional cap on S3 objects to ingest")

    # Normalization/filtering
    ap.add_argument("--out-dir", default="data/training/iwap", help="Output directory")
    ap.add_argument("--min-eval-score", type=float, default=0.5, help="Success threshold when status is unavailable")
    ap.add_argument("--min-actions", type=int, default=1, help="Minimum actions required to keep a trajectory")
    ap.add_argument("--max-actions", type=int, default=80, help="Max actions to keep per trajectory")
    ap.add_argument("--max-steps", type=int, default=300, help="Max steps to keep per trajectory")
    ap.add_argument("--keep-html", action="store_true", help="Keep HTML in step snapshots (can be very large)")
    ap.add_argument("--include-failed", action="store_true", help="Include failed trajectories (default keeps only successful)")
    ap.add_argument("--no-dedupe-actions", action="store_true", help="Do not dedupe consecutive duplicate actions")
    ap.add_argument("--no-dedupe-trajectories", action="store_true", help="Do not dedupe semantically duplicate trajectories")

    # Export settings
    ap.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    ap.add_argument("--system-prompt", default=SYSTEM_PROMPT_DEFAULT, help="System prompt for SFT records")
    ap.add_argument("--no-system-prompt", action="store_true", help="Do not add system prompt to SFT records")

    return ap.parse_args()


def main() -> None:
    load_dotenv(OPERATOR_ROOT / ".env", override=False)
    args = _parse_args()

    token = args.token
    if not token:
        import os

        token = os.getenv("IWAP_API_TOKEN")

    cfg = TrajectoryBuildConfig(
        min_eval_score=float(args.min_eval_score),
        min_actions=int(args.min_actions),
        max_actions=int(args.max_actions),
        max_steps=int(args.max_steps),
        keep_html=bool(args.keep_html),
        dedupe_actions=not bool(args.no_dedupe_actions),
        dedupe_trajectories=not bool(args.no_dedupe_trajectories),
        only_successful=not bool(args.include_failed),
    )

    if args.source == "iwap-api":
        if not args.base_url:
            raise SystemExit("--base-url is required when --source iwap-api")
        records, stats = ingest_from_iwap_api(
            base_url=str(args.base_url),
            token=token,
            max_runs=args.max_runs,
            page_limit=int(args.page_limit),
            include_unfinished=bool(args.include_unfinished),
            cfg=cfg,
        )
    else:
        if not args.s3_bucket:
            raise SystemExit("--s3-bucket is required when --source s3")
        source = S3TrajectorySource(
            bucket=str(args.s3_bucket),
            prefix=str(args.s3_prefix or ""),
            region_name=args.s3_region,
            profile_name=args.s3_profile,
            endpoint_url=args.s3_endpoint_url,
        )
        records, stats = ingest_from_s3(
            source=source,
            cfg=cfg,
            max_objects=args.s3_max_objects,
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() / stamp
    artifacts = export_training_bundle(
        out_dir=out_dir,
        records=records,
        stats=stats,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        system_prompt=None if args.no_system_prompt else str(args.system_prompt),
    )

    manifest = json.loads(Path(artifacts.manifest).read_text(encoding="utf-8"))
    print(json.dumps(manifest.get("stats") or {}, ensure_ascii=False, indent=2))
    print(f"[done] dataset written to {artifacts.output_dir}")


if __name__ == "__main__":
    main()
