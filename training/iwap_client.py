from __future__ import annotations

import gzip
import json
from typing import Any, Iterable, Optional

import httpx


class IWAPClient:
    """Small IWAP API client for run/task/log discovery + task-log payload download."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout_s: float = 45.0,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.token = str(token).strip() if token else None
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=float(timeout_s),
            verify=bool(verify_ssl),
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "IWAPClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get_json(self, path_or_url: str, *, params: dict[str, Any] | None = None) -> Any:
        is_absolute = str(path_or_url).startswith("http://") or str(path_or_url).startswith("https://")
        response = self._client.get(path_or_url if is_absolute else str(path_or_url), params=params, headers=self._headers())
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_data(payload: Any) -> Any:
        if isinstance(payload, dict) and isinstance(payload.get("data"), (dict, list)):
            return payload["data"]
        return payload

    def list_runs(
        self,
        *,
        page: int = 1,
        limit: int = 50,
        include_unfinished: bool = False,
        sort_by: str = "startTime",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        payload = self._get_json(
            "/api/v1/agent-runs",
            params={
                "page": int(page),
                "limit": int(limit),
                "includeUnfinished": bool(include_unfinished),
                "sortBy": str(sort_by),
                "sortOrder": str(sort_order),
            },
        )
        data = self._extract_data(payload)
        if not isinstance(data, dict):
            return [], 0
        runs = data.get("runs") if isinstance(data.get("runs"), list) else []
        total = int(data.get("total") or len(runs))
        return [r for r in runs if isinstance(r, dict)], total

    def iter_run_ids(
        self,
        *,
        max_runs: int | None = None,
        page_limit: int = 50,
        include_unfinished: bool = False,
    ) -> Iterable[str]:
        page = 1
        yielded = 0
        while True:
            runs, _ = self.list_runs(page=page, limit=page_limit, include_unfinished=include_unfinished)
            if not runs:
                break
            for run in runs:
                run_id = run.get("runId") or run.get("run_id") or run.get("agent_run_id")
                if not run_id:
                    continue
                yield str(run_id)
                yielded += 1
                if max_runs is not None and yielded >= int(max_runs):
                    return
            if len(runs) < int(page_limit):
                return
            page += 1

    def get_run_tasks(self, run_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(f"/api/v1/agent-runs/{run_id}/tasks")
        data = self._extract_data(payload)
        if isinstance(data, dict):
            tasks = data.get("tasks")
            if isinstance(tasks, list):
                return [t for t in tasks if isinstance(t, dict)]
        return []

    def get_run_logs(self, run_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(f"/api/v1/agent-runs/{run_id}/logs")
        data = self._extract_data(payload)
        if isinstance(data, dict):
            entries = data.get("entries")
            if isinstance(entries, list):
                return [e for e in entries if isinstance(e, dict)]
            logs = data.get("logs")
            if isinstance(logs, list):
                return [e for e in logs if isinstance(e, dict)]
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict)]
        return []

    def fetch_task_log_payload(self, url: str) -> dict[str, Any]:
        response = self._client.get(str(url), headers=self._headers())
        response.raise_for_status()
        body = response.content
        if not body:
            raise ValueError(f"Empty response body for task log: {url}")

        encoding = str(response.headers.get("Content-Encoding") or "").lower()
        content_type = str(response.headers.get("Content-Type") or "").lower()
        should_try_gzip = str(url).endswith(".gz") or "gzip" in encoding or "application/gzip" in content_type

        if should_try_gzip or body[:2] == b"\x1f\x8b":
            try:
                body = gzip.decompress(body)
            except Exception:
                # Some endpoints already transparently decompress.
                pass

        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Could not parse JSON task log from {url}: {type(exc).__name__}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Unexpected task log payload type from {url}: {type(parsed).__name__}")
        return parsed


__all__ = ["IWAPClient"]
