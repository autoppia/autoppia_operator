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

from training import (
    S3TrajectorySource,
    TrajectoryBuildConfig,
    ingest_from_s3,
    write_jsonl,
)


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Normalize S3 task-log payloads into cleaned trajectory classes")
    ap.add_argument("--s3-bucket", required=True)
    ap.add_argument("--s3-prefix", default="")
    ap.add_argument("--s3-region", default=None)
    ap.add_argument("--s3-profile", default=None)
    ap.add_argument("--s3-endpoint-url", default=None)
    ap.add_argument("--max-objects", type=int, default=None)

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-eval-score", type=float, default=0.5)
    ap.add_argument("--min-actions", type=int, default=1)
    ap.add_argument("--max-actions", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--keep-html", action="store_true")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--no-dedupe-actions", action="store_true")
    ap.add_argument("--no-dedupe-trajectories", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _args()
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

    source = S3TrajectorySource(
        bucket=str(args.s3_bucket),
        prefix=str(args.s3_prefix),
        region_name=args.s3_region,
        profile_name=args.s3_profile,
        endpoint_url=args.s3_endpoint_url,
    )

    records, stats = ingest_from_s3(source=source, cfg=cfg, max_objects=args.max_objects)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_dir / "cleaned_trajectories.jsonl"
    manifest_path = out_dir / "normalize_manifest.json"

    write_jsonl(cleaned_path, [r.to_dict() for r in records])

    manifest = {
        "stats": {
            "source": stats.source,
            "runs_seen": stats.runs_seen,
            "tasks_seen": stats.tasks_seen,
            "tasks_with_payload_url": stats.tasks_with_payload_url,
            "payload_downloaded": stats.payload_downloaded,
            "payload_parse_errors": stats.payload_parse_errors,
            "trajectories_kept": stats.trajectories_kept,
            "trajectories_filtered": stats.trajectories_filtered,
            "trajectories_deduped": stats.trajectories_deduped,
            "records_total": len(records),
        },
        "files": {"cleaned_trajectories": str(cleaned_path)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {**manifest["stats"], "manifest": str(manifest_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
