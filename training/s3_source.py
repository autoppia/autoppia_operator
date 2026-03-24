from __future__ import annotations

import contextlib
import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str
    size: int | None = None
    etag: str | None = None
    last_modified: datetime | None = None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    raw = str(uri).strip()
    if not raw.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri}")
    tail = raw[5:]
    if "/" not in tail:
        raise ValueError(f"S3 URI missing key path: {uri}")
    bucket, key = tail.split("/", 1)
    bucket = bucket.strip()
    key = key.lstrip("/").strip()
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def decode_json_blob(blob: bytes, *, key_hint: str = "") -> dict[str, Any]:
    if not isinstance(blob, bytes | bytearray) or not blob:
        raise ValueError("S3 object is empty")

    payload = bytes(blob)
    should_try_gzip = key_hint.endswith(".gz") or payload[:2] == b"\x1f\x8b"
    if should_try_gzip:
        with contextlib.suppress(Exception):
            payload = gzip.decompress(payload)

    try:
        parsed = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to parse JSON payload ({type(exc).__name__})") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


class S3TrajectorySource:
    """Small S3 object source with lazy boto3 import.

    Keeps boto3 optional for the miner runtime; users can install boto3 only
    on training hosts where S3 ingestion is needed.
    """

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        region_name: str | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self.bucket = str(bucket).strip()
        self.prefix = str(prefix or "").lstrip("/")
        self.region_name = region_name
        self.profile_name = profile_name
        self.endpoint_url = endpoint_url

    def _client(self):
        try:
            import boto3
        except Exception as exc:
            raise ImportError("boto3 is required for S3 ingestion. Install with: pip install boto3") from exc

        session = boto3.session.Session(profile_name=self.profile_name) if self.profile_name else boto3.session.Session()
        return session.client("s3", region_name=self.region_name, endpoint_url=self.endpoint_url)

    def iter_objects(
        self,
        *,
        max_objects: int | None = None,
        suffixes: tuple[str, ...] = (".json", ".json.gz", ".gz"),
    ) -> Iterable[S3ObjectRef]:
        cli = self._client()
        continuation_token: str | None = None
        produced = 0

        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": self.prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            resp = cli.list_objects_v2(**kwargs)
            contents = resp.get("Contents") if isinstance(resp, dict) else []
            for item in contents if isinstance(contents, list) else []:
                key = str(item.get("Key") or "")
                if not key:
                    continue
                if suffixes and not key.endswith(suffixes):
                    continue
                ref = S3ObjectRef(
                    bucket=self.bucket,
                    key=key,
                    size=int(item.get("Size")) if isinstance(item.get("Size"), int) else None,
                    etag=str(item.get("ETag")).strip('"') if item.get("ETag") is not None else None,
                    last_modified=item.get("LastModified"),
                )
                yield ref
                produced += 1
                if max_objects is not None and produced >= int(max_objects):
                    return

            if not resp.get("IsTruncated"):
                return
            continuation_token = resp.get("NextContinuationToken")
            if not continuation_token:
                return

    def fetch_object_bytes(self, ref: S3ObjectRef) -> bytes:
        cli = self._client()
        resp = cli.get_object(Bucket=ref.bucket, Key=ref.key)
        body = resp.get("Body")
        if body is None:
            raise ValueError(f"S3 get_object returned no Body for {ref.uri}")
        data = body.read()
        if not isinstance(data, bytes | bytearray):
            raise ValueError(f"S3 get_object body for {ref.uri} is not bytes")
        return bytes(data)

    def fetch_json(self, ref: S3ObjectRef) -> dict[str, Any]:
        return decode_json_blob(self.fetch_object_bytes(ref), key_hint=ref.key)


__all__ = [
    "S3ObjectRef",
    "S3TrajectorySource",
    "decode_json_blob",
    "parse_s3_uri",
]
