from __future__ import annotations

import json
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .projects import project_config, seeded_url


def base_origin(task_url: str) -> str:
    parsed = urlparse(str(task_url))
    return f"{parsed.scheme}://{parsed.netloc}"


def seed_from_task_url(task_url: str) -> int:
    parsed = urlparse(str(task_url))
    query = parsed.query or ""
    for part in query.split("&"):
        if not part.startswith("seed="):
            continue
        try:
            return int(part.split("=", 1)[1])
        except Exception:
            return 1
    return 1


def dataset_movie_candidates(*, task_url: str, filters: dict[str, Any], web_project_id: str) -> list[str]:
    origin = base_origin(task_url)
    seed = seed_from_task_url(task_url)
    project_key = project_config(web_project_id).project_key
    params = f"project_key={project_key}&entity_type=movies&seed_value={seed}&limit=50&method=distribute&filter_key=category"
    url = f"{origin}/api/datasets/load?{params}"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            payload = json.load(response)
    except Exception:
        return []
    movies = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(movies, list):
        return []

    name_contains = str(filters.get("name_contains") or "").strip().lower()
    name_exact = str(filters.get("name_exact") or "").strip().lower()
    name_not_contains = str(filters.get("name_not_contains") or "").strip().lower()
    director_exact = str(filters.get("director_exact") or "").strip().lower()
    director_contains = str(filters.get("director_contains") or "").strip().lower()
    genre_exact = str(filters.get("genre_exact") or "").strip().lower()
    genre_contains = str(filters.get("genre_contains") or "").strip().lower()
    genre_not_contains = str(filters.get("genre_not_contains") or "").strip().lower()
    duration_gte = int(filters.get("duration_gte") or 0) if str(filters.get("duration_gte") or "").strip() else 0
    duration_lte = int(filters.get("duration_lte") or 0) if str(filters.get("duration_lte") or "").strip() else 0
    rating_gte = float(filters.get("rating_gte") or 0) if str(filters.get("rating_gte") or "").strip() else 0.0
    rating_lte = float(filters.get("rating_lte") or 0) if str(filters.get("rating_lte") or "").strip() else 0.0
    year_gte = int(filters.get("year_gte") or 0) if str(filters.get("year_gte") or "").strip() else 0
    year_lte = int(filters.get("year_lte") or 0) if str(filters.get("year_lte") or "").strip() else 0

    candidates: list[str] = []
    for movie in movies:
        if not isinstance(movie, dict):
            continue
        movie_id = str(movie.get("id") or "").strip()
        if not movie_id:
            continue
        title = str(movie.get("title") or "").strip().lower()
        director = str(movie.get("director") or "").strip().lower()
        genres = [str(item).strip().lower() for item in (movie.get("genres") or []) if str(item).strip()]
        try:
            duration = int(float(movie.get("duration") or 0))
        except Exception:
            duration = 0
        try:
            rating = float(movie.get("rating") or 0)
        except Exception:
            rating = 0.0
        try:
            year = int(float(movie.get("year") or 0))
        except Exception:
            year = 0
        if name_exact and title != name_exact:
            continue
        if name_contains and name_contains not in title:
            continue
        if name_not_contains and name_not_contains in title:
            continue
        if director_exact and director != director_exact:
            continue
        if director_contains and director_contains not in director:
            continue
        if genre_exact and genre_exact not in genres:
            continue
        if genre_contains and not any(genre_contains in genre for genre in genres):
            continue
        if genre_not_contains and any(genre_not_contains in genre for genre in genres):
            continue
        if duration_gte and duration < duration_gte:
            continue
        if duration_lte and duration > duration_lte:
            continue
        if rating_gte and rating < rating_gte:
            continue
        if rating_lte and rating > rating_lte:
            continue
        if year_gte and year < year_gte:
            continue
        if year_lte and year > year_lte:
            continue
        candidates.append(f"/movies/{movie_id}")
    return candidates


def resolve_movie_detail_url(*, task_url: str, filters: dict[str, Any], web_project_id: str) -> str | None:
    candidates = dataset_movie_candidates(task_url=task_url, filters=filters, web_project_id=web_project_id)
    if not candidates:
        return None
    return seeded_url(task_url=task_url, project_id=web_project_id, route=candidates[0], seed=seed_from_task_url(task_url))


__all__ = [
    "base_origin",
    "dataset_movie_candidates",
    "resolve_movie_detail_url",
    "seed_from_task_url",
]
