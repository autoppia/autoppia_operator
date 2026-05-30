from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from autoppia_iwa.src.demo_webs.config import demo_web_projects


@dataclass(frozen=True)
class DeterministicProjectConfig:
    project_id: str
    frontend_url: str
    project_key: str
    web_folder: str
    id_variants_path: Path
    class_variants_path: Path
    text_variants_path: Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEBS_DEMO_ROOT = REPO_ROOT.parent / "autoppia_webs_demo"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def normalize_demo_url(raw_url: str | None, *, force_localhost: bool | None = None) -> str:
    normalized = str(raw_url or "").strip()
    if not normalized:
        return normalized
    if force_localhost is None:
        force_localhost = _env_bool("AGENT_FORCE_LOCALHOST_URLS", False)
    if not force_localhost:
        return normalized
    try:
        if "://" not in normalized:
            if not normalized.startswith("/"):
                if "." in normalized or ":" in normalized:
                    parsed = urlsplit(f"http://{normalized}")
                    path = parsed.path or ""
                    return urlunsplit(("http", "localhost", path, parsed.query, parsed.fragment))
                normalized = f"/{normalized}"
            return f"http://localhost{normalized}"
        parsed = urlsplit(normalized)
        path = parsed.path or ""
        return urlunsplit(("http", "localhost", path, parsed.query, parsed.fragment))
    except Exception:
        return "http://localhost"


@lru_cache(maxsize=1)
def _project_map() -> dict[str, DeterministicProjectConfig]:
    out: dict[str, DeterministicProjectConfig] = {}
    for index, project in enumerate(demo_web_projects, start=1):
        project_id = str(getattr(project, "id", "") or "").strip()
        if not project_id:
            continue
        web_folder = f"web_{index}_{project_id}"
        out[project_id] = DeterministicProjectConfig(
            project_id=project_id,
            frontend_url=str(getattr(project, "frontend_url", "") or "").strip(),
            project_key=web_folder,
            web_folder=web_folder,
            id_variants_path=WEBS_DEMO_ROOT / web_folder / "src" / "dynamic" / "v3" / "data" / "id-variants.json",
            class_variants_path=WEBS_DEMO_ROOT / web_folder / "src" / "dynamic" / "v3" / "data" / "class-variants.json",
            text_variants_path=WEBS_DEMO_ROOT / web_folder / "src" / "dynamic" / "v3" / "data" / "text-variants.json",
        )
    return out


def resolve_project_id(task_row: dict[str, Any] | None = None, *, explicit_project_id: str | None = None) -> str:
    explicit = str(explicit_project_id or "").strip()
    if explicit:
        return explicit
    if isinstance(task_row, dict):
        project_id = str(task_row.get("web_project_id") or task_row.get("project_id") or "").strip()
        if project_id:
            return project_id
    return "autocinema"


def project_config(project_id: str | None) -> DeterministicProjectConfig:
    resolved = resolve_project_id(explicit_project_id=project_id)
    config = _project_map().get(resolved)
    if config is None:
        web_folder = f"web_unknown_{resolved}"
        return DeterministicProjectConfig(
            project_id=resolved,
            frontend_url="http://localhost",
            project_key=web_folder,
            web_folder=web_folder,
            id_variants_path=WEBS_DEMO_ROOT / web_folder / "src" / "dynamic" / "v3" / "data" / "id-variants.json",
            class_variants_path=WEBS_DEMO_ROOT / web_folder / "src" / "dynamic" / "v3" / "data" / "class-variants.json",
            text_variants_path=WEBS_DEMO_ROOT / web_folder / "src" / "dynamic" / "v3" / "data" / "text-variants.json",
        )
    return config


def normalized_origin(*, task_url: str | None = None, project_id: str | None = None) -> str:
    raw_url = str(task_url or "").strip() or project_config(project_id).frontend_url
    normalized = normalize_demo_url(raw_url)
    parsed = urlsplit(normalized)
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
    fallback = normalize_demo_url(project_config(project_id).frontend_url)
    fallback_parsed = urlsplit(fallback)
    return urlunsplit((fallback_parsed.scheme or "http", fallback_parsed.netloc or "localhost", "", "", "")).rstrip("/")


def seeded_url(*, task_url: str | None, project_id: str | None, route: str, seed: int) -> str:
    raw_route = str(route or "").strip()
    if raw_route.startswith("http://") or raw_route.startswith("https://"):
        absolute = normalize_demo_url(raw_route)
        parsed = urlsplit(absolute)
        query = dict(parse_qsl(parsed.query or "", keep_blank_values=True))
        query.setdefault("seed", str(int(seed)))
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
    path = raw_route if raw_route.startswith("/") else f"/{raw_route}"
    return f"{normalized_origin(task_url=task_url, project_id=project_id)}{path}?seed={int(seed)}"


__all__ = [
    "DeterministicProjectConfig",
    "normalized_origin",
    "project_config",
    "resolve_project_id",
    "seeded_url",
]
