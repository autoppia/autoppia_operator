#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OPERATOR_ROOT = SCRIPT_DIR.parent
if str(OPERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(OPERATOR_ROOT))

from training import S3TrajectorySource


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="List trajectory payload objects from S3 and export a manifest JSONL")
    ap.add_argument("--s3-bucket", required=True)
    ap.add_argument("--s3-prefix", default="")
    ap.add_argument("--s3-region", default=None)
    ap.add_argument("--s3-profile", default=None)
    ap.add_argument("--s3-endpoint-url", default=None)
    ap.add_argument("--max-objects", type=int, default=None)
    ap.add_argument("--out", required=True, help="Output JSONL path")
    return ap.parse_args()


def main() -> None:
    args = _args()
    source = S3TrajectorySource(
        bucket=str(args.s3_bucket),
        prefix=str(args.s3_prefix),
        region_name=args.s3_region,
        profile_name=args.s3_profile,
        endpoint_url=args.s3_endpoint_url,
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ref in source.iter_objects(max_objects=args.max_objects):
            payload = {
                "bucket": ref.bucket,
                "key": ref.key,
                "uri": ref.uri,
                "size": ref.size,
                "etag": ref.etag,
                "last_modified": ref.last_modified.isoformat() if ref.last_modified else None,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1

    print(json.dumps({"objects": count, "manifest": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
