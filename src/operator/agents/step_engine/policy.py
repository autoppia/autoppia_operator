from __future__ import annotations

from .candidates import *
from .meta_tools import *
from .observation import *
from .state import *
from .utils import *


@lru_cache(maxsize=1)
def _success_examples_by_project() -> dict[str, list[dict[str, Any]]]:
    manifests = sorted((_REPO_ROOT / "data").glob("*_trajectory_harvest/sft/manifest.json"))
    out: dict[str, list[dict[str, Any]]] = {}
    for manifest_path in manifests[:24]:
        project_dir = manifest_path.parent.parent.name
        project_id = project_dir[: -len("_trajectory_harvest")] if project_dir.endswith("_trajectory_harvest") else project_dir
        project_id = str(project_id or "").strip().lower()
        if not project_id:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        examples = out.setdefault(project_id, [])
        for trace_file in list(manifest.get("trace_files") or [])[:64]:
            trace_path = Path(str(trace_file)).expanduser()
            if not trace_path.exists():
                continue
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            episode = trace.get("episode") if isinstance(trace.get("episode"), dict) else {}
            episode_project = str(episode.get("web_project_id") or episode.get("project_id") or project_id).strip().lower() or project_id
            if episode_project != project_id:
                continue
            use_case = str(episode.get("use_case") or "")[:64]
            for step in list(trace.get("steps") or [])[:6]:
                if not isinstance(step, dict):
                    continue
                request = step.get("act_request") if isinstance(step.get("act_request"), dict) else {}
                response = step.get("act_response") if isinstance(step.get("act_response"), dict) else {}
                tool_calls = response.get("tool_calls") if isinstance(response.get("tool_calls"), list) else []
                if not tool_calls:
                    continue
                url = str(request.get("url") or trace.get("task_url") or "")
                examples.append(
                    {
                        "web_project_id": project_id,
                        "use_case": use_case,
                        "url_path": str(urlsplit(url).path or "/").rstrip("/") or "/",
                        "step_index": int(step.get("step_index") or 0),
                        "prompt": str(request.get("prompt") or trace.get("task_prompt") or "")[:280],
                        "tool_calls": tool_calls[:3],
                    }
                )
    return out


def _project_success_examples(project_id: str) -> list[dict[str, Any]]:
    pid = str(project_id or "").strip().lower()
    if not pid:
        return []
    return list((_success_examples_by_project().get(pid) or [])[:128])


def _autocinema_success_examples() -> list[dict[str, Any]]:
    return _project_success_examples("autocinema")


def _policy_use_case_name(prompt: str, policy_obs: Dict[str, Any]) -> str:
    explicit = policy_obs.get("use_case") or policy_obs.get("active_objective", {}).get("use_case") or policy_obs.get("working_state", {}).get("active_workflow")
    if isinstance(explicit, dict):
        explicit = explicit.get("name")
    explicit_text = str(explicit or "").strip().upper().replace(" ", "_")
    return explicit_text


def _infer_autocinema_use_case(prompt: str, policy_obs: Dict[str, Any]) -> str:
    explicit_text = _policy_use_case_name(prompt, policy_obs)
    if explicit_text:
        return explicit_text

    prompt_text = str(prompt or "").lower()
    heuristic_map = [
        ("remove from watchlist", "REMOVE_FROM_WATCHLIST"),
        ("remove from wishlist", "REMOVE_FROM_WATCHLIST"),
        ("add to watchlist", "ADD_TO_WATCHLIST"),
        ("add to wishlist", "ADD_TO_WATCHLIST"),
        ("watch trailer", "WATCH_TRAILER"),
        ("trailer", "WATCH_TRAILER"),
        ("share", "SHARE_MOVIE"),
        ("comment", "ADD_COMMENT"),
        ("review", "ADD_COMMENT"),
        ("registration", "REGISTRATION"),
        ("register", "REGISTRATION"),
        ("sign up", "REGISTRATION"),
        ("logout", "LOGOUT"),
        ("log out", "LOGOUT"),
        ("login", "LOGIN"),
        ("log in", "LOGIN"),
        ("contact", "CONTACT"),
        ("search", "SEARCH_FILM"),
        ("filter", "FILTER_FILM"),
        ("edit user", "EDIT_USER"),
        ("edit profile", "EDIT_USER"),
        ("edit film", "EDIT_FILM"),
        ("add film", "ADD_FILM"),
        ("delete", "DELETE_FILM"),
        ("remove", "DELETE_FILM"),
        ("watchlist", "ADD_TO_WATCHLIST"),
        ("details", "FILM_DETAIL"),
        ("detail", "FILM_DETAIL"),
    ]
    for needle, use_case in heuristic_map:
        if needle in prompt_text:
            return use_case
    return ""


def _related_autocinema_use_cases(use_case: str) -> set[str]:
    groups = [
        {"LOGIN", "LOGOUT", "REGISTRATION"},
        {"ADD_TO_WATCHLIST", "REMOVE_FROM_WATCHLIST", "LOGIN"},
        {"ADD_FILM", "EDIT_FILM", "DELETE_FILM", "EDIT_USER", "LOGIN", "REGISTRATION"},
        {"FILM_DETAIL", "WATCH_TRAILER", "SHARE_MOVIE", "ADD_COMMENT", "ADD_TO_WATCHLIST"},
        {"SEARCH_FILM", "FILTER_FILM", "FILM_DETAIL"},
        {"CONTACT"},
    ]
    normalized = str(use_case or "").strip().upper()
    for group in groups:
        if normalized in group:
            return set(group)
    return {normalized} if normalized else set()


def _autocinema_example_block(prompt: str, policy_obs: Dict[str, Any]) -> list[str]:
    project_id = str(policy_obs.get("web_project_id") or "").strip().lower() or "autocinema"
    examples = _autocinema_success_examples() if project_id == "autocinema" else _project_success_examples(project_id)
    if not examples:
        return []

    inferred_use_case = _policy_use_case_name(prompt, policy_obs)
    if not inferred_use_case and project_id == "autocinema":
        inferred_use_case = _infer_autocinema_use_case(prompt, policy_obs)
    related_use_cases = _related_autocinema_use_cases(inferred_use_case) if project_id == "autocinema" else {inferred_use_case} if inferred_use_case else set()
    current_url = str(policy_obs.get("url") or "")
    current_path = str(urlsplit(current_url).path or "/").rstrip("/") or "/"
    current_step = int(policy_obs.get("step_index") or 0)
    prompt_tokens = _tokenize(prompt)

    ranked: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for example in examples:
        example_use_case = str(example.get("use_case") or "").strip().upper()
        if inferred_use_case and example_use_case == inferred_use_case:
            use_case_score = 0
        elif example_use_case in related_use_cases:
            use_case_score = 1
        else:
            use_case_score = 2
        path_score = 0 if current_path != "/" and example.get("url_path") == current_path else 1
        step_score = abs(int(example.get("step_index") or 0) - current_step)
        prompt_score = 0
        if prompt_tokens:
            prompt_score = max(
                0,
                8 - len(prompt_tokens.intersection(_tokenize(str(example.get("prompt") or "").lower()))),
            )
        if use_case_score >= 2 and path_score == 1:
            continue
        ranked.append(((use_case_score, path_score, prompt_score, step_score), example))
    ranked.sort(key=lambda item: item[0])

    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    for _, example in ranked:
        key = (str(example.get("url_path") or ""), int(example.get("step_index") or 0))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(example)
        if len(selected) >= 2:
            break
    if not selected:
        return []

    lines = [
        "RETRIEVED SUCCESSFUL AUTOCINEMA EXAMPLES:" if project_id == "autocinema" else "RETRIEVED SUCCESSFUL TRACE EXAMPLES:",
        "- These are short action snippets from successful harvest traces. Reuse the same local workflow only when the current page state matches.",
    ]
    for idx, example in enumerate(selected, start=1):
        lines.extend(
            [
                f"Example {idx}: use_case={example.get('use_case') or 'unknown'} path={example.get('url_path') or '/'} step={int(example.get('step_index') or 0)}",
                f"Prompt excerpt: {str(example.get('prompt') or '')[:180]}",
                "Successful action JSON:",
                json.dumps(example.get("tool_calls") or [], ensure_ascii=False),
            ]
        )
    lines.append("")
    return lines


def _site_knowledge_route_for_section(policy_obs: Dict[str, Any], *, section_id: str, current_path: str) -> str:
    site_knowledge = policy_obs.get("site_knowledge") if isinstance(policy_obs.get("site_knowledge"), dict) else {}
    routes = site_knowledge.get("routes") if isinstance(site_knowledge.get("routes"), list) else []
    preferred: list[tuple[int, str]] = []
    for route in routes[:16]:
        if not isinstance(route, dict):
            continue
        if str(route.get("section_id") or "").strip().lower() != section_id:
            continue
        path = str(route.get("path") or "").strip() or "/"
        if path == current_path:
            continue
        label = str(route.get("label") or "").strip().lower()
        score = 0
        if (
            (section_id == "auth" and any(token in path for token in ("/login", "/signin", "/register", "/signup")))
            or (section_id == "catalog" and any(token in path for token in ("/search", "/browse", "/catalog")))
            or (section_id == "account" and any(token in path for token in ("/profile", "/account", "/watchlist", "/wishlist")))
            or (section_id == "form" and any(token in path for token in ("/contact", "/create", "/add", "/edit")))
        ):
            score -= 3
        if label and ("login" in label or "search" in label or "profile" in label or "contact" in label):
            score -= 1
        preferred.append((score, path))
    preferred.sort(key=lambda item: (item[0], len(item[1])))
    return preferred[0][1] if preferred else ""


def _preferred_site_knowledge_navigation(
    prompt: str,
    policy_obs: Dict[str, Any],
    *,
    current_url: str,
    current_path: str,
) -> Dict[str, Any] | None:
    site_knowledge = policy_obs.get("site_knowledge") if isinstance(policy_obs.get("site_knowledge"), dict) else {}
    current_task = site_knowledge.get("current_task_routing") if isinstance(site_knowledge.get("current_task_routing"), dict) else {}
    best_section = str(current_task.get("likely_best_section") or "").strip().lower()
    if best_section not in {"auth", "catalog", "form", "account", "info"}:
        return None
    if best_section == "auth" and current_path.startswith(("/login", "/register", "/signup", "/signin", "/auth")):
        return None
    if best_section == "catalog" and current_path.startswith(("/search", "/browse", "/catalog")):
        return None
    if best_section == "account" and current_path.startswith(("/profile", "/account", "/watchlist", "/wishlist", "/saved")):
        return None
    if best_section == "form" and re.search(r"/(contact|create|add|edit|delete|checkout|reserve|booking)", current_path):
        return None
    if best_section == "info" and re.search(r"/(about|help|support|faq|policy|contact)", current_path):
        return None
    route_path = _site_knowledge_route_for_section(policy_obs, section_id=best_section, current_path=current_path)
    if not route_path:
        return None
    return {
        "type": "browser",
        "tool_call": {
            "name": "browser.navigate",
            "arguments": {
                "url": _safe_url(route_path, base=current_url),
                "go_back": False,
                "go_forward": False,
            },
        },
    }


def _prompt_prefers_text_input(prompt: str, policy_obs: Dict[str, Any]) -> bool:
    use_case = _infer_autocinema_use_case(prompt, policy_obs)
    if use_case in {"SEARCH_FILM", "LOGIN", "REGISTRATION", "CONTACT", "ADD_COMMENT"}:
        return True
    prompt_text = str(prompt or "").lower()
    return any(
        needle in prompt_text
        for needle in (
            "search for",
            "look up",
            "find the movie",
            "log in",
            "login",
            "register",
            "sign up",
            "contact",
            "comment",
        )
    )


def _preferred_prompt_navigation(
    prompt: str,
    policy_obs: Dict[str, Any],
    *,
    allowed_tools: set[str],
) -> Dict[str, Any] | None:
    if allowed_tools and "browser.navigate" not in allowed_tools:
        return None
    current_url = str(policy_obs.get("url") or "").strip().lower()
    if current_url not in {"", "about:blank"}:
        return None
    prompt_text = str(prompt or "").strip()
    lowered = prompt_text.lower()
    if not re.search(r"\b(open|visit|go to|navigate to)\b", lowered):
        return None
    match = re.search(
        r"\b((?:https?://)?(?:www\.)?[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?:/[^\s]*)?)",
        prompt_text,
        flags=re.I,
    )
    safe_target = ""
    if match is not None:
        target = str(match.group(1) or "").rstrip(".,);:]!?")
        safe_target = _safe_url(target)
    if not safe_target:
        named_sites = {
            "wikipedia": "https://www.wikipedia.org",
            "google": "https://www.google.com",
            "github": "https://github.com",
            "youtube": "https://www.youtube.com",
            "stackoverflow": "https://stackoverflow.com",
        }
        for label, target_url in named_sites.items():
            if re.search(rf"\b{re.escape(label)}\b", lowered):
                safe_target = target_url
                break
    if not safe_target.startswith(("http://", "https://")):
        return None
    return {
        "type": "browser",
        "tool_call": {
            "name": "browser.navigate",
            "arguments": {
                "url": safe_target,
                "go_back": False,
                "go_forward": False,
            },
        },
    }


def _autocinema_task_intent_tags(prompt: str, policy_obs: Dict[str, Any]) -> set[str]:
    use_case = _infer_autocinema_use_case(prompt, policy_obs)
    tags: set[str] = set()
    tags.update(
        {
            "ADD_TO_WATCHLIST": {"watchlist_add"},
            "REMOVE_FROM_WATCHLIST": {"watchlist_remove"},
            "WATCH_TRAILER": {"trailer"},
            "SHARE_MOVIE": {"share"},
            "ADD_COMMENT": {"comment"},
            "FILM_DETAIL": {"detail"},
        }.get(use_case, set())
    )
    text = str(prompt or "").lower()
    if ("watchlist" in text or "wishlist" in text) and re.search(r"\b(remove|delete|drop)\b", text):
        tags.add("watchlist_remove")
    elif "watchlist" in text or "wishlist" in text:
        tags.add("watchlist_add")
    if re.search(r"\bwatch trailer\b|\btrailer\b", text):
        tags.add("trailer")
    if re.search(r"\bshare\b", text):
        tags.add("share")
    if re.search(r"\b(comment|review|note|feedback|reply)\b", text):
        tags.add("comment")
    if re.search(r"\b(detail|details|view detail)\b", text):
        tags.add("detail")
    return tags


def _obs_candidate_intent_tags(item: Dict[str, Any]) -> set[str]:
    blob = " ".join(
        [
            str(item.get("text") or ""),
            str(item.get("href") or ""),
            str(item.get("field_hint") or ""),
            str(item.get("field_kind") or ""),
            str(item.get("group_label") or ""),
            str(item.get("context") or ""),
            str(item.get("aria_label") or ""),
            str(item.get("placeholder") or ""),
            str(item.get("ui_state") or ""),
            str(item.get("current_value") or ""),
        ]
    ).lower()
    tags: set[str] = set()
    if re.search(r"\b(remove|delete)\s+(from\s+)?(watchlist|wishlist)\b", blob):
        tags.add("watchlist_remove")
    elif re.search(r"\b(add|save)\s+(to\s+)?(watchlist|wishlist)\b", blob):
        tags.add("watchlist_add")
    elif re.search(r"\b(in|on|inside)\s+(the\s+)?(watchlist|wishlist)\b", blob):
        tags.add("watchlist_remove")
    elif "watchlist" in blob or "wishlist" in blob:
        if re.search(r"\b(active|selected|pressed|saved|added)\b", blob):
            tags.add("watchlist_remove")
        else:
            tags.add("watchlist_add")
    if re.search(r"\bwatch trailer\b|\btrailer\b", blob):
        tags.add("trailer")
    if re.search(r"\bshare\b", blob):
        tags.add("share")
    if re.search(r"\b(comment|review|note|feedback|reply)\b", blob):
        tags.add("comment")
    if re.search(r"\bview details?\b|\bmovie details?\b", blob) or "/movies/" in str(item.get("href") or "").lower():
        tags.add("detail")
    return tags


def _obs_candidate_primary_intent_tags(item: Dict[str, Any]) -> set[str]:
    blob = " ".join(
        [
            str(item.get("text") or ""),
            str(item.get("field_hint") or ""),
            str(item.get("aria_label") or ""),
            str(item.get("placeholder") or ""),
        ]
    ).lower()
    return _obs_candidate_intent_tags({"text": blob})


def _extract_seed_from_url(url: str) -> str:
    try:
        query_pairs = dict(parse_qsl(urlsplit(str(url or "")).query, keep_blank_values=True))
        seed = query_pairs.get("seed", "")
    except Exception:
        seed = ""
    seed_text = str(seed or "").strip()
    return seed_text if seed_text else "999"


def _extract_prompt_title_literal(prompt: str) -> str:
    text = str(prompt or "")
    patterns = [
        r"\bname\s+equals\s+'([^']+)'",
        r"\btitle\s+equals\s+'([^']+)'",
        r"\bmovie\s+titled\s+'([^']+)'",
        r"\bfilm\s+titled\s+'([^']+)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = str(match.group(1) or "").strip()
            if value:
                return value[:160]
    return ""


def _page_mentions_title(policy_obs: Dict[str, Any], title_literal: str) -> bool:
    needle = str(title_literal or "").strip().lower()
    if not needle:
        return False
    haystacks = [
        str(policy_obs.get("page_ir_text") or ""),
        str(policy_obs.get("browser_state_snapshot") or ""),
        str(policy_obs.get("snapshot_html") or ""),
        str(policy_obs.get("html") or ""),
        str(policy_obs.get("url") or ""),
    ]
    page_obs = policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}
    haystacks.extend(str(line or "") for line in (page_obs.get("relevant_lines") or [])[:16])
    haystacks.extend(str(line or "") for line in (page_obs.get("likely_answers") or [])[:8])
    blob = " \n".join(haystacks).lower()
    return needle in blob


def _policy_markup(policy_obs: Dict[str, Any]) -> str:
    for key in ("snapshot_html", "html", "browser_state_snapshot", "page_ir_text"):
        value = str(policy_obs.get(key) or "")
        if value.strip():
            return value
    return ""


def _selector_for_html_id(markup: str, element_id: str) -> Dict[str, Any] | None:
    if not markup or not element_id:
        return None
    if re.search(rf'id=["\']{re.escape(element_id)}["\']', markup, flags=re.I):
        return {
            "type": "attributeValueSelector",
            "attribute": "id",
            "value": element_id,
            "case_sensitive": False,
        }
    return None


def _selector_for_visible_button_text(markup: str, button_text: str) -> Dict[str, Any] | None:
    if not markup or not button_text:
        return None
    if re.search(rf"<button\b[^>]*>\s*(?:<[^>]+>\s*)*{re.escape(button_text)}\s*</button>", markup, flags=re.I):
        return {
            "type": "tagContainsSelector",
            "value": button_text,
            "case_sensitive": False,
        }
    return None


def _preferred_direct_intent_action_from_markup(
    prompt: str,
    policy_obs: Dict[str, Any],
    *,
    allowed_tools: set[str],
) -> Dict[str, Any] | None:
    if allowed_tools and "browser.click" not in allowed_tools:
        return None
    current_path = str(urlsplit(str(policy_obs.get("url") or "")).path or "").rstrip("/") or "/"
    if not current_path.startswith("/movies/"):
        return None
    markup = _policy_markup(policy_obs)
    if not markup:
        return None
    title_literal = _extract_prompt_title_literal(prompt)
    if title_literal and not _page_mentions_title(policy_obs, title_literal):
        return None
    task_intents = _autocinema_task_intent_tags(prompt, policy_obs)
    selector_specs = [
        ("watchlist_remove", "remove-list-btn", "Remove from watchlist"),
        ("watchlist_remove", "", "In Watchlist"),
        ("watchlist_add", "add-list-btn", "Add to watchlist"),
        ("trailer", "play-trailer", "Watch trailer"),
        ("share", "share-widget", "Share"),
    ]
    for intent, preferred_id, button_text in selector_specs:
        if intent not in task_intents:
            continue
        selector = _selector_for_html_id(markup, preferred_id) if preferred_id else None
        if selector is None:
            selector = _selector_for_visible_button_text(markup, button_text)
        if selector is None:
            continue
        return {
            "type": "browser",
            "tool_call": {
                "name": "browser.click",
                "arguments": {"selector": selector},
            },
        }
    return None


def _seeded_search_url(current_url: str, seed: str, title_literal: str) -> str:
    query_items: list[tuple[str, str]] = []
    if seed:
        query_items.append(("seed", seed))
    title_text = str(title_literal or "").strip()
    if title_text:
        query_items.append(("search", title_text))
    query = urlencode(query_items)
    return _safe_url(f"/?{query}" if query else "/", base=current_url)


def _candidate_mentions_title(item: Dict[str, Any], title_literal: str) -> bool:
    needle = str(title_literal or "").strip().lower()
    if not needle:
        return False
    blob = " ".join(
        [
            str(item.get("text") or ""),
            str(item.get("href") or ""),
            str(item.get("field_hint") or ""),
            str(item.get("group_label") or ""),
            str(item.get("context") or ""),
            str(item.get("aria_label") or ""),
        ]
    ).lower()
    return needle in blob


def _preferred_title_result_action(
    prompt: str,
    policy_obs: Dict[str, Any],
    *,
    allowed_tools: set[str],
) -> Dict[str, Any] | None:
    if allowed_tools and "browser.click" not in allowed_tools:
        return None
    use_case = _infer_autocinema_use_case(prompt, policy_obs)
    if use_case not in {
        "ADD_TO_WATCHLIST",
        "REMOVE_FROM_WATCHLIST",
        "WATCH_TRAILER",
        "SHARE_MOVIE",
        "ADD_COMMENT",
        "FILM_DETAIL",
    }:
        return None
    title_literal = _extract_prompt_title_literal(prompt)
    if not title_literal:
        return None
    current_url = str(policy_obs.get("url") or "")
    current_path = str(urlsplit(current_url).path or "").rstrip("/") or "/"
    capability_gap = (
        policy_obs.get("page_observations", {}).get("capability_gap")
        if isinstance(policy_obs.get("page_observations"), dict) and isinstance(policy_obs.get("page_observations", {}).get("capability_gap"), dict)
        else {}
    )
    auth_gated_mutation_use_cases = {
        "ADD_TO_WATCHLIST",
        "REMOVE_FROM_WATCHLIST",
        "ADD_FILM",
        "EDIT_FILM",
        "DELETE_FILM",
        "EDIT_USER",
    }
    if bool(capability_gap.get("read_only_for_task")) and use_case in auth_gated_mutation_use_cases:
        return None
    if current_path.startswith("/movies/") and _page_mentions_title(policy_obs, title_literal):
        return None
    candidates = policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else []
    ranked: list[tuple[tuple[int, int, int], Dict[str, Any]]] = []
    for item in candidates[:32]:
        if not isinstance(item, dict):
            continue
        if not _candidate_mentions_title(item, title_literal):
            continue
        role = str(item.get("role") or "").strip().lower()
        href = str(item.get("href") or "").strip().lower()
        blob = " ".join(
            [
                str(item.get("text") or ""),
                str(item.get("field_hint") or ""),
                str(item.get("context") or ""),
                str(item.get("aria_label") or ""),
            ]
        ).lower()
        if role not in {"button", "link"}:
            continue
        if re.search(r"\b(home|about|contact|login|log in|register|sign up)\b", blob):
            continue
        if re.fullmatch(r"/(?:\?.*)?", href):
            continue
        item_id = str(item.get("id") or item.get("element_id") or item.get("_element_id") or "").strip().lower()
        if item_id.startswith("related-card-"):
            continue
        detail_bias = 0 if "/movies/" in href else 1
        role_bias = 0 if role == "link" else 1
        index_bias = 0 if isinstance(item.get("index"), int) else 1
        ranked.append(((detail_bias, role_bias, index_bias), item))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    chosen = ranked[0][1]
    args: Dict[str, Any] = {}
    if isinstance(chosen.get("index"), int):
        args["index"] = int(chosen["index"])
    elif isinstance(chosen.get("selector"), dict):
        args["selector"] = chosen["selector"]
    else:
        chosen_id = str(chosen.get("id") or chosen.get("element_id") or chosen.get("_element_id") or "").strip()
        if not chosen_id:
            return None
        args["element_id"] = chosen_id
    return {"type": "browser", "tool_call": {"name": "browser.click", "arguments": args}}


def _preferred_seed_stable_navigation(
    prompt: str,
    policy_obs: Dict[str, Any],
    *,
    allowed_tools: set[str],
) -> Dict[str, Any] | None:
    if allowed_tools and "browser.navigate" not in allowed_tools:
        return None
    current_url = str(policy_obs.get("url") or "")
    current_path = str(urlsplit(current_url).path or "").rstrip("/") or "/"
    seed = _extract_seed_from_url(current_url)
    use_case = _infer_autocinema_use_case(prompt, policy_obs)
    title_literal = _extract_prompt_title_literal(prompt)
    public_detail_use_cases = {
        "ADD_TO_WATCHLIST",
        "REMOVE_FROM_WATCHLIST",
        "WATCH_TRAILER",
        "SHARE_MOVIE",
        "ADD_COMMENT",
        "FILM_DETAIL",
    }
    auth_gated_mutation_use_cases = {
        "ADD_TO_WATCHLIST",
        "REMOVE_FROM_WATCHLIST",
        "ADD_FILM",
        "EDIT_FILM",
        "DELETE_FILM",
        "EDIT_USER",
    }
    capability_gap = (
        policy_obs.get("page_observations", {}).get("capability_gap")
        if isinstance(policy_obs.get("page_observations"), dict) and isinstance(policy_obs.get("page_observations", {}).get("capability_gap"), dict)
        else {}
    )

    if use_case == "LOGIN" and seed is not None and current_path not in {"/login"}:
        return {
            "type": "browser",
            "tool_call": {
                "name": "browser.navigate",
                "arguments": {
                    "url": _safe_url(f"/login?seed={seed}", base=current_url),
                    "go_back": False,
                    "go_forward": False,
                },
            },
        }

    if bool(capability_gap.get("read_only_for_task")) and (use_case not in public_detail_use_cases or use_case in auth_gated_mutation_use_cases):
        preferred_transition = str(capability_gap.get("preferred_transition") or "").strip().lower()
        transition_targets = {
            "login": f"/login?seed={seed}",
            "register": f"/register?seed={seed}",
            "manage": f"/profile?seed={seed}",
        }
        target_url = transition_targets.get(preferred_transition)
        if target_url and current_path not in {"/login", "/register", "/profile"}:
            return {
                "type": "browser",
                "tool_call": {
                    "name": "browser.navigate",
                    "arguments": {"url": _safe_url(target_url, base=current_url), "go_back": False, "go_forward": False},
                },
            }

    title_focused_use_cases = {
        "ADD_TO_WATCHLIST",
        "REMOVE_FROM_WATCHLIST",
        "SHARE_MOVIE",
        "WATCH_TRAILER",
        "ADD_COMMENT",
        "FILM_DETAIL",
        "SEARCH_FILM",
    }
    off_target_detail = current_path.startswith("/movies/") and not _page_mentions_title(policy_obs, title_literal) if use_case in title_focused_use_cases and title_literal else False
    if use_case in title_focused_use_cases and (current_path in {"/", "/about", "/contact", "/login", "/register"} or off_target_detail):
        return {
            "type": "browser",
            "tool_call": {
                "name": "browser.navigate",
                "arguments": {"url": _seeded_search_url(current_url, seed, title_literal), "go_back": False, "go_forward": False},
            },
        }
    generic_route = _preferred_site_knowledge_navigation(
        prompt,
        policy_obs,
        current_url=current_url,
        current_path=current_path,
    )
    if generic_route is not None:
        return generic_route
    return None


def _direct_intent_region_bias(item: Dict[str, Any], task_intents: set[str], current_path: str) -> tuple[int, int]:
    blob = " ".join(
        [
            str(item.get("text") or ""),
            str(item.get("href") or ""),
            str(item.get("field_hint") or ""),
            str(item.get("group_label") or ""),
            str(item.get("context") or ""),
            str(item.get("aria_label") or ""),
        ]
    ).lower()
    href = str(item.get("href") or "").strip().lower()
    cluster_bonus = 0
    penalty = 0

    if any(tag in task_intents for tag in {"watchlist_add", "watchlist_remove", "share", "trailer"}):
        if re.search(r"\bwatch trailer\b|\bwatchlist\b|\bwishlist\b|\bshare\b", blob):
            cluster_bonus = -1
        if re.search(r"\b(comment|review|note|feedback|reply)\b", blob):
            penalty += 2
        if re.search(r"\b(login|log in|sign in|register|sign up|profile|account|search|filter)\b", blob):
            penalty += 2
        if href and not href.startswith(current_path) and not href.startswith("#"):
            penalty += 3

    if "comment" not in task_intents and re.search(r"\b(comment|review|note|feedback|reply)\b", blob):
        penalty += 1
    if "detail" not in task_intents and re.search(r"\bview details?\b|\bmovie details?\b", blob):
        penalty += 1
    return cluster_bonus, penalty


def _candidate_matches_tool_args(item: Dict[str, Any], args: Dict[str, Any]) -> bool:
    try:
        item_index = int(item.get("index"))
    except (TypeError, ValueError):
        item_index = None
    try:
        arg_index = int(args.get("index"))
    except (TypeError, ValueError):
        arg_index = None
    if item_index is not None and arg_index is not None and item_index == arg_index:
        return True
    item_id = str(item.get("id") or item.get("element_id") or item.get("_element_id") or "").strip()
    if item_id and item_id in {
        str(args.get("element_id") or "").strip(),
        str(args.get("_element_id") or "").strip(),
    }:
        return True
    item_selector = item.get("selector") if isinstance(item.get("selector"), dict) else {}
    arg_selector = args.get("selector") if isinstance(args.get("selector"), dict) else {}
    if item_selector and arg_selector:
        return (
            str(item_selector.get("type") or "") == str(arg_selector.get("type") or "")
            and str(item_selector.get("attribute") or "") == str(arg_selector.get("attribute") or "")
            and str(item_selector.get("value") or "") == str(arg_selector.get("value") or "")
        )
    return False


def _tool_call_matches(preferred: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    preferred_name = str(preferred.get("name") or "").strip()
    actual_name = str(actual.get("name") or "").strip()
    if preferred_name != actual_name:
        return False
    preferred_args = preferred.get("arguments") if isinstance(preferred.get("arguments"), dict) else {}
    actual_args = actual.get("arguments") if isinstance(actual.get("arguments"), dict) else {}
    if preferred_name == "browser.navigate":
        return str(preferred_args.get("url") or "").strip() == str(actual_args.get("url") or "").strip()
    if preferred_name == "browser.click":
        preferred_item = {
            "index": preferred_args.get("index"),
            "selector": preferred_args.get("selector"),
            "id": preferred_args.get("element_id"),
            "element_id": preferred_args.get("element_id"),
            "_element_id": preferred_args.get("_element_id"),
        }
        return _candidate_matches_tool_args(preferred_item, actual_args)
    return preferred_args == actual_args


def _is_demo_execution_profile(profile: Any) -> bool:
    text = str(profile or "").strip().lower()
    return text.startswith("demo_")


def _effective_execution_profile(policy_obs: Dict[str, Any]) -> str:
    explicit = str(policy_obs.get("execution_profile") or "").strip().lower()
    if explicit:
        return explicit
    prompt = str(policy_obs.get("prompt") or "").lower()
    current_url = str(policy_obs.get("url") or "")
    path = str(urlsplit(current_url).path or "").lower()
    snapshot_html = str(policy_obs.get("snapshot_html") or "").lower()
    candidate_blob = " ".join(
        [str((item or {}).get("text") or "") for item in (policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else [])[:24] if isinstance(item, dict)]
    ).lower()
    combined = " ".join([prompt, path, snapshot_html, candidate_blob])
    if path.startswith("/movies/") or any(token in combined for token in ("watchlist", "wishlist", "watch trailer", "movie page", "film detail", "view detail")):
        return "demo_catalog_navigation"
    if any(token in combined for token in ("log in", "login", "sign in", "register", "sign up", "signup")):
        return "demo_auth_flow"
    return "general_web"


def _preferred_direct_intent_action(
    prompt: str,
    policy_obs: Dict[str, Any],
    *,
    allowed_tools: set[str],
) -> tuple[Dict[str, Any], Dict[str, Any]] | None:
    if allowed_tools and "browser.click" not in allowed_tools:
        return None
    current_path = str(urlsplit(str(policy_obs.get("url") or "")).path or "").rstrip("/") or "/"
    task_intents = _autocinema_task_intent_tags(prompt, policy_obs).intersection({"watchlist_add", "watchlist_remove", "trailer", "share", "comment", "detail"})
    if not task_intents or not current_path.startswith("/movies/"):
        return None
    candidates = policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else []
    intent_priority = ["watchlist_remove", "watchlist_add", "trailer", "share", "comment", "detail"]
    ranked: list[tuple[tuple[int, int, int], Dict[str, Any]]] = []
    for item in candidates[:24]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"button", "link", "input", "textarea"}:
            continue
        matched = task_intents.intersection(_obs_candidate_intent_tags(item))
        if not matched:
            continue
        best_intent = min(intent_priority.index(tag) for tag in matched if tag in intent_priority)
        exact_match = 0 if task_intents.intersection(_obs_candidate_primary_intent_tags(item)) else 1
        role_bias = 0 if role in {"button", "link"} else 1
        has_index = 0 if isinstance(item.get("index"), int) else 1
        cluster_bias, drift_penalty = _direct_intent_region_bias(item, task_intents, current_path)
        ranked.append(((best_intent, exact_match, cluster_bias, role_bias, drift_penalty, has_index), item))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    chosen = ranked[0][1]
    args: Dict[str, Any] = {}
    if isinstance(chosen.get("index"), int):
        args["index"] = int(chosen["index"])
    else:
        chosen_id = str(chosen.get("id") or chosen.get("element_id") or chosen.get("_element_id") or "").strip()
        if chosen_id:
            args["element_id"] = chosen_id
        elif isinstance(chosen.get("selector"), dict):
            args["selector"] = chosen["selector"]
    return {"name": "browser.click", "arguments": args}, chosen


class Policy:
    def __init__(self, llm_call: Callable[..., Dict[str, Any]]) -> None:
        self.llm_call = llm_call
        self.debug_dir = str(os.getenv("FSM_POLICY_DEBUG_DIR", "") or "").strip()

    def _debug_log(self, task_id: str, payload: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return
        try:
            base = Path(self.debug_dir).expanduser().resolve()
            safe_task = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(task_id or "task"))[:120] or "task"
            _append_jsonl(base / f"{safe_task}.jsonl", payload)
        except Exception:
            return

    def _repair_enabled(self) -> bool:
        return _env_bool("FSM_POLICY_REPAIR", False)

    def decide(
        self,
        *,
        task_id: str,
        prompt: str,
        mode: str,
        policy_obs: Dict[str, Any],
        allowed_tools: set[str],
        model_name: str,
        plan_model_name: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        max_actions_per_step = max(1, min(_env_int("FSM_MAX_ACTIONS_PER_STEP", 3), 5))
        execution_profile = _effective_execution_profile(policy_obs)
        goal_state = policy_obs.get("goal_state") if isinstance(policy_obs.get("goal_state"), dict) else {}
        goal_evaluation = policy_obs.get("goal_evaluation") if isinstance(policy_obs.get("goal_evaluation"), dict) else {}
        extra_rules: list[str] = []
        if bool(goal_state.get("stop_on_page_match")):
            extra_rules.append("- If GOAL STATE is already satisfied on the current page, finish immediately.")
            extra_rules.append("- Once the target page is open and constraints match, do not open secondary local controls.")
        if execution_profile == "demo_catalog_navigation":
            extra_rules.append("- On demo catalog tasks, prefer visible search/filter controls before opening result cards.")
            extra_rules.append("- Stop as soon as the matching detail page is open; do not continue into trailer/share/comment/watchlist actions.")
        if mode == "POPUP":
            return {"type": "meta", "name": "META.SOLVE_POPUPS", "arguments": {}}, {"source": "deterministic"}
        if mode == "REPORT":
            facts = policy_obs.get("memory", {}).get("facts") if isinstance(policy_obs.get("memory"), dict) else []
            fact = _candidate_text((facts or [""])[0] if isinstance(facts, list) and facts else "")
            content = fact or "Task appears complete."
            return {"type": "final", "done": True, "content": content}, {"source": "deterministic"}

        meta_enabled = any(str(tool or "").startswith("META.") for tool in allowed_tools)
        direct_mode = mode == "DIRECT"
        if direct_mode:
            system = (
                "You are a browser-use-style web operator.\n"
                "You have the user task, the current browser state, the indexed interactive elements, the allowed browser tools, and the unavailable tools.\n"
                "Choose the next browser step sequence.\n"
                "Return ONE JSON object only. No markdown. No prose.\n"
                "Valid outputs:\n"
                '1) {"type":"browser","tool_call":{...}}\n'
                '2) {"type":"browser","tool_calls":[{...},{...}]}\n'
                '3) {"type":"final","done":true,"content":"..."}\n'
                "Rules:\n"
                f"- This runtime allows up to {max_actions_per_step} browser actions per step.\n"
                "- First decide whether the current page already satisfies the task. If yes, finish immediately with final or browser.done.\n"
                "- content must be the actual user-facing answer, result, or extracted value.\n"
                "- Do not keep exploring when the current page already satisfies the task.\n"
                f"- Never return more than {max_actions_per_step} browser actions.\n"
                "- If you return multiple actions, they must belong to the same local workflow and be safe to execute consecutively without re-observing.\n"
                "- Prefer arguments.index that refers to INTERACTIVE ELEMENT SHORTLIST.\n"
                "- For browser.select_dropdown, include a non-empty arguments.text.\n"
                "- browser.done is the standard way to finish once the page already satisfies the task.\n"
                "- Never emit unavailable tools.\n"
                "- If the task includes filters or explicit constraints, use visible controls first before opening result items.\n"
                "- Preserve placeholders such as <username>, <password>, <signup_email> exactly when typing.\n"
                "- When useful, include reasoning as a short human-readable operator note grounded in visible page evidence.\n"
                "- reasoning must be 1-2 short sentences, concrete, and suitable for product UI.\n"
                "- reasoning must say what is visible now and why the chosen next action or final answer follows.\n"
                "- reasoning must not contain chain-of-thought, filler, generic status text, or speculation without visible support.\n"
                "- Before choosing actions, infer one short local workflow plan for the current page and keep it stable until that workflow is completed or visibly blocked.\n"
                "- Update reasoning_trace.current_subgoal and reasoning_trace.plan to reflect the current local milestone, not the whole task from scratch.\n"
                "- Do not replan the whole task every step unless the page changed materially or the current workflow clearly failed.\n"
                "- If SCORE FEEDBACK is present in state and marks success=true or score=1.0, treat it as strong completion evidence and prefer final/browser.done unless visible evidence clearly contradicts it.\n"
                "- If a login or registration form is visible, do not submit until the visible credential fields are filled.\n"
                "- If the task shows empty quoted credentials, replace them with placeholders such as <username>, <password>, <signup_username>, <signup_email>, or <signup_password> instead of empty strings.\n"
                + ("\n".join(extra_rules) + "\n" if extra_rules else "")
            )
        else:
            system = (
                "You are a browser-use-style web automation policy.\n"
                "Given the task and the current browser state, choose the next browser step sequence.\n"
                "Return ONE JSON object only. No markdown. No prose. No chain-of-thought.\n"
                "You must choose exactly one of:\n"
                "1) browser tool_call or browser tool_calls\n" + ("2) meta_tool\n" if meta_enabled else "") + f"{'3' if meta_enabled else '2'}) final (done=true + content)\n\n"
                "Rules:\n"
                f"- This runtime allows up to {max_actions_per_step} browser actions per step.\n"
                "- Prefer a concrete browser action when there is a reasonable actionable target.\n"
                + ("- Use a meta_tool only when inspection/disambiguation materially improves the next browser action.\n" if meta_enabled else "")
                + f"- Never return more than {max_actions_per_step} browser actions.\n"
                + "- If you return multiple browser actions, they must stay within the same local workflow and should usually be a short form-filling or commit sequence.\n"
                "- If the current page already contains the answer, return final immediately.\n"
                "- For question-answering and data-extraction tasks, DONE is the correct action once the answer is visible on the current page.\n"
                "- Do NOT keep exploring once the current page already answers the task.\n"
                "- Use final/done or browser.done with a concrete content string when the task is satisfied.\n"
                "- content must be the actual answer for the user, not a status message.\n"
                "- Avoid repeating low-value actions when the page did not materially change.\n"
                "- Prefer arguments.index that refers to INTERACTIVE ELEMENT SHORTLIST.\n"
                "- For browser.select_dropdown, provide arguments.text with the option text/value to choose.\n"
                "- Never emit unavailable tools.\n"
                "- Preserve placeholders such as <username>, <password>, <signup_email> exactly when typing.\n"
                "- For informational tasks, use the current visible content before navigating more.\n"
                "- When useful, include reasoning as a short human-readable operator note grounded in visible page evidence.\n"
                "- reasoning must be 1-2 short sentences, concrete, and suitable for product UI.\n"
                "- reasoning must say what is visible now and why the chosen next action or final answer follows.\n"
                "- reasoning must not contain chain-of-thought, filler, generic status text, or speculation without visible support.\n"
                "- Before choosing actions, infer one short local workflow plan for the current page and keep it stable until that workflow is completed or visibly blocked.\n"
                "- Update reasoning_trace.current_subgoal and reasoning_trace.plan to reflect the current local milestone, not the whole task from scratch.\n"
                "- Do not replan the whole task every step unless the page changed materially or the current workflow clearly failed.\n"
                "- If SCORE FEEDBACK is present in state and marks success=true or score=1.0, treat it as strong completion evidence and prefer final/browser.done unless visible evidence clearly contradicts it.\n"
                "- If a login or registration form is visible, do not submit until the visible credential fields are filled.\n"
                "- If the task shows empty quoted credentials, replace them with placeholders such as <username>, <password>, <signup_username>, <signup_email>, or <signup_password> instead of empty strings.\n"
                + ("\n".join(extra_rules) + "\n" if extra_rules else "")
            )
        autoplay_examples = _autocinema_example_block(prompt, policy_obs) if _is_demo_execution_profile(execution_profile) else []
        if direct_mode:
            user_parts = [
                "Choose the next browser step sequence.",
                f"TASK: {str(policy_obs.get('prompt') or '')[:1600]}",
                *autoplay_examples,
                "TASK CONSTRAINTS:",
                json.dumps(
                    (policy_obs.get("task_constraints") if isinstance(policy_obs.get("task_constraints"), dict) else {}),
                    ensure_ascii=False,
                ),
                f"STEP: {int(policy_obs.get('step_index') or 0)}",
                f"URL: {str(policy_obs.get('url') or '')[:1000]}",
                "",
                "RUNTIME:",
                "browser-use-like operator runtime",
                f"max_actions_per_step={max_actions_per_step}",
                "tabs_supported=false",
                "file_tools_supported=false",
                "",
                "UNAVAILABLE TOOLS:",
                ", ".join(policy_obs.get("unavailable_browser_tools") if isinstance(policy_obs.get("unavailable_browser_tools"), list) else list(_unavailable_browser_tools())),
                "",
                "TOOL USAGE GUIDE:",
                "- browser.click: buttons, links, toggles, tabs, submit controls, checkboxes, radios.",
                "- browser.input: text-entry fields only.",
                "- browser.select_dropdown: only when a concrete option text is known.",
                "- browser.dropdown_options: inspect a select before choosing if the option is unclear.",
                "- browser.extract: deterministic extraction from visible page content.",
                "- browser.search: external web search, not in-page site search boxes.",
                "",
                "REASONING FIELD CONTRACT:",
                "- You may include reasoning as a short operator note for humans.",
                "- reasoning must be 1-2 short sentences, max about 220 characters total.",
                "- reasoning must mention visible evidence or current page state and the next action or final answer.",
                "- Good reasoning example: 'The comment form is already open and only the message field is still empty, so fill it and submit.'",
                "- Bad reasoning example: 'I think this is probably the right thing to do next because it seems correct.'",
                "- Do not include hidden reasoning, self-talk, generic filler, or unsupported guesses.",
                "",
                "PLAN MAINTENANCE CONTRACT:",
                "- First infer a short local workflow for the current page before selecting actions.",
                "- Keep that workflow stable across steps unless visible evidence shows it is blocked or no longer relevant.",
                "- reasoning_trace.current_subgoal should name the current local milestone.",
                "- reasoning_trace.plan should describe the next short sequence on this page, not the whole task history.",
                "- If a form/filter/comment workflow is already active, prefer finishing it before exploring unrelated controls.",
                "",
                "REASONING TRACE CONTRACT:",
                "- You may include reasoning_trace as a short JSON object with keys task_interpretation, success_state, current_subgoal, next_expected_proof, drift_risks, where_am_i, state_assessment, plan.",
                "- Keep each field short and concrete.",
                "- reasoning_trace must describe task meaning and success, not hidden chain-of-thought.",
                "",
                "WORKING STATE CONTRACT:",
                "- You should include working_state as a short JSON object with keys current_page_kind, active_region, active_workflow, completed_fields, pending_fields, completion_evidence_missing, next_milestone, completion_state.",
                "- working_state should describe the current operational state, not hidden reasoning.",
                "",
                "ACTIVE OBJECTIVE (JSON):",
                json.dumps(policy_obs.get("active_objective") if isinstance(policy_obs.get("active_objective"), dict) else {}, ensure_ascii=False),
                "",
                f"EXECUTION PROFILE: {execution_profile}",
                "",
                "GOAL STATE (JSON):",
                json.dumps(goal_state, ensure_ascii=False),
                "",
                "GOAL EVALUATION (JSON):",
                json.dumps(goal_evaluation, ensure_ascii=False),
                "",
                "WORKING STATE (JSON):",
                json.dumps(policy_obs.get("working_state") if isinstance(policy_obs.get("working_state"), dict) else {}, ensure_ascii=False),
                "",
                *(
                    [
                        f"LLMJudgeEvaluator says score {policy_obs.get('state_score')}",
                        "",
                    ]
                    if policy_obs.get("state_score") is not None
                    else []
                ),
                "SCORE FEEDBACK (JSON):",
                json.dumps(policy_obs.get("score_feedback") if isinstance(policy_obs.get("score_feedback"), dict) else {}, ensure_ascii=False),
                "",
                "LOCAL WORKFLOW CLOSURE (JSON):",
                json.dumps(policy_obs.get("local_workflow_closure") if isinstance(policy_obs.get("local_workflow_closure"), dict) else {}, ensure_ascii=False),
                "",
                "LOCAL HTML CONTEXT (JSON):",
                json.dumps(policy_obs.get("local_html_context") if isinstance(policy_obs.get("local_html_context"), dict) else {}, ensure_ascii=False),
                "",
                "KNOWN SITE MAP (JSON):",
                json.dumps(policy_obs.get("site_knowledge") if isinstance(policy_obs.get("site_knowledge"), dict) else {}, ensure_ascii=False),
                "",
                "AVOID REPEATING (JSON):",
                json.dumps(policy_obs.get("avoid_repeating") if isinstance(policy_obs.get("avoid_repeating"), dict) else {}, ensure_ascii=False),
                "",
                "PREVIOUS REASONING TRACE (JSON):",
                json.dumps(policy_obs.get("reasoning_trace") if isinstance(policy_obs.get("reasoning_trace"), dict) else {}, ensure_ascii=False),
                "",
                "DONE / CONTENT CONTRACT:",
                "- If the answer or completed result is already visible on the current page, return final now.",
                "- final.content or browser.done.arguments.content must be the concrete answer for the user.",
                "- Do not return generic status text like 'task complete' or 'done'.",
                "- Before taking another action, check BROWSER SNAPSHOT, VISIBLE TEXT / PAGE SUMMARY, and INTERACTIVE ELEMENT SHORTLIST.",
                "- If a select already shows the target value, do not select the same value again.",
                "- Prefer arguments.index when targeting an interactive element.",
                "- Follow ACTIVE OBJECTIVE unless visible evidence proves the answer is already on the page.",
                "- Respect AVOID REPEATING unless the page materially changed.",
                "- If LOCAL WORKFLOW CLOSURE shows ready_to_commit=true and a visible commit control exists, strongly prefer finishing that local workflow before exploring unrelated controls.",
                "- Use LOCAL HTML CONTEXT to understand which inputs and commit controls belong to the same active form or region.",
                "- If multiple actions are returned, explain the local workflow in reasoning_trace.plan.",
                "- If SCORE FEEDBACK is present and success=true or score=1.0, prefer finishing now unless the page visibly contradicts that signal.",
                "",
                "BROWSER SNAPSHOT:",
                str(policy_obs.get("browser_state_snapshot") or "")[:3000],
                "",
                "VISIBLE TEXT / PAGE SUMMARY:",
                str(policy_obs.get("page_ir_text") or "")[:14000],
                "",
                "VISIBLE EVIDENCE (JSON):",
                json.dumps(
                    {
                        "likely_answers": (
                            policy_obs.get("page_observations", {}).get("likely_answers")
                            if isinstance(policy_obs.get("page_observations"), dict) and isinstance(policy_obs.get("page_observations", {}).get("likely_answers"), list)
                            else []
                        )[:8],
                        "relevant_lines": (
                            policy_obs.get("page_observations", {}).get("relevant_lines")
                            if isinstance(policy_obs.get("page_observations"), dict) and isinstance(policy_obs.get("page_observations", {}).get("relevant_lines"), list)
                            else []
                        )[:16],
                    },
                    ensure_ascii=False,
                ),
                "",
                "RECENT ACTIONS AND RESULTS (JSON):",
                json.dumps(policy_obs.get("history_recent") if isinstance(policy_obs.get("history_recent"), list) else [], ensure_ascii=False),
                "",
                "PAGE OBSERVATIONS (JSON):",
                json.dumps(policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}, ensure_ascii=False),
                "",
                "INTERACTIVE ELEMENTS (indexed tree-style):",
                str(policy_obs.get("browser_state_text") or "")[:16000],
                "",
                "INTERACTIVE ELEMENT SHORTLIST (JSON):",
                json.dumps(
                    (policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else [])[:48],
                    ensure_ascii=False,
                ),
                "",
                "ALLOWED BROWSER TOOLS: " + ", ".join(sorted([t for t in list(allowed_tools) if str(t).startswith("browser.")]) if allowed_tools else []),
                "",
                "Output schema examples:",
                '{"type":"browser","tool_call":{"name":"browser.click","arguments":{"index":0}}}',
                '{"type":"browser","reasoning":"The comment form is already open and only the message field is still missing, so complete it and submit on this page.","reasoning_trace":{"task_interpretation":"Open the correct movie and submit a valid comment.","success_state":"A new comment is visibly posted on the target movie.","current_subgoal":"Finish the visible comment form.","next_expected_proof":"A post/submit action or the new comment appears.","drift_risks":"Opening related movies or share widgets.","where_am_i":"A movie detail page with a visible comment form.","state_assessment":"The form is ready and only needs local completion.","plan":"Fill the visible fields and submit without leaving this page."},"working_state":{"current_page_kind":"movie detail page","active_region":"comment form","active_workflow":"submit a valid movie comment","completed_fields":["name"],"pending_fields":["comment"],"completion_evidence_missing":["posted comment visible"],"next_milestone":"Submit the visible comment form.","completion_state":"awaiting_local_completion"},"tool_calls":[{"name":"browser.input","arguments":{"index":0,"text":"Jordan"}},{"name":"browser.input","arguments":{"index":1,"text":"Great pacing and atmosphere."}},{"name":"browser.click","arguments":{"index":2}}]}',
                '{"type":"browser","tool_call":{"name":"browser.done","arguments":{"content":"The total value is 2844."}}}',
                '{"type":"final","done":true,"content":"The total value is 2844.","reasoning":"The total value is already visible on the current page, so return it directly."}',
            ]
        else:
            user_parts = [
                "You have a task and must choose the next browser step sequence.",
                f"TASK: {str(policy_obs.get('prompt') or '')[:1600]}",
                *autoplay_examples,
                f"STEP: {int(policy_obs.get('step_index') or 0)}",
                f"MODE: {mode}",
                f"URL: {str(policy_obs.get('url') or '')[:1000]}",
                f"SCREENSHOT_AVAILABLE: {bool(policy_obs.get('screenshot_available'))}",
                "",
                "RUNTIME:",
                "browser-use-like operator runtime",
                f"max_actions_per_step={max_actions_per_step}",
                "tabs_supported=false",
                "file_tools_supported=false",
                "",
                "TOOL USAGE GUIDE:",
                "- browser.click: buttons, links, toggles, tabs, submit controls, checkboxes, radios.",
                "- browser.input: text-entry fields only.",
                "- browser.select_dropdown: only when a concrete option text is known.",
                "- browser.dropdown_options: inspect a select before choosing if the option is unclear.",
                "- browser.extract: deterministic extraction from visible page content.",
                "- browser.search: external web search, not in-page site search boxes.",
                "",
                "REASONING FIELD CONTRACT:",
                "- You may include reasoning as a short operator note for humans.",
                "- reasoning must be 1-2 short sentences, max about 220 characters total.",
                "- reasoning must mention visible evidence or current page state and the next action or final answer.",
                "- Good reasoning example: 'The filter panel is already visible and the apply button is in the same group, so finish this local filter sequence first.'",
                "- Bad reasoning example: 'I will think step by step and try something that might work.'",
                "- Do not include hidden reasoning, self-talk, generic filler, or unsupported guesses.",
                "",
                "PLAN MAINTENANCE CONTRACT:",
                "- First infer a short local workflow for the current page before selecting actions.",
                "- Keep that workflow stable across steps unless visible evidence shows it is blocked or no longer relevant.",
                "- reasoning_trace.current_subgoal should name the current local milestone.",
                "- reasoning_trace.plan should describe the next short sequence on this page, not the whole task history.",
                "- If a form/filter/comment workflow is already active, prefer finishing it before exploring unrelated controls.",
                "",
                "REASONING TRACE CONTRACT:",
                "- You may include reasoning_trace as a short JSON object with keys task_interpretation, success_state, current_subgoal, next_expected_proof, drift_risks, where_am_i, state_assessment, plan.",
                "- Keep each field short and concrete.",
                "- reasoning_trace must describe task meaning and success, not hidden chain-of-thought.",
                "",
                "WORKING STATE CONTRACT:",
                "- You should include working_state as a short JSON object with keys current_page_kind, active_region, active_workflow, completed_fields, pending_fields, completion_evidence_missing, next_milestone, completion_state.",
                "- working_state should describe the current operational state, not hidden reasoning.",
                "",
                "ACTIVE OBJECTIVE (JSON):",
                json.dumps(policy_obs.get("active_objective") if isinstance(policy_obs.get("active_objective"), dict) else {}, ensure_ascii=False),
                "",
                f"EXECUTION PROFILE: {execution_profile}",
                "",
                "GOAL STATE (JSON):",
                json.dumps(goal_state, ensure_ascii=False),
                "",
                "GOAL EVALUATION (JSON):",
                json.dumps(goal_evaluation, ensure_ascii=False),
                "",
                "WORKING STATE (JSON):",
                json.dumps(policy_obs.get("working_state") if isinstance(policy_obs.get("working_state"), dict) else {}, ensure_ascii=False),
                "",
                *(
                    [
                        f"LLMJudgeEvaluator says score {policy_obs.get('state_score')}",
                        "",
                    ]
                    if policy_obs.get("state_score") is not None
                    else []
                ),
                "SCORE FEEDBACK (JSON):",
                json.dumps(policy_obs.get("score_feedback") if isinstance(policy_obs.get("score_feedback"), dict) else {}, ensure_ascii=False),
                "",
                "LOCAL WORKFLOW CLOSURE (JSON):",
                json.dumps(policy_obs.get("local_workflow_closure") if isinstance(policy_obs.get("local_workflow_closure"), dict) else {}, ensure_ascii=False),
                "",
                "LOCAL HTML CONTEXT (JSON):",
                json.dumps(policy_obs.get("local_html_context") if isinstance(policy_obs.get("local_html_context"), dict) else {}, ensure_ascii=False),
                "",
                "KNOWN SITE MAP (JSON):",
                json.dumps(policy_obs.get("site_knowledge") if isinstance(policy_obs.get("site_knowledge"), dict) else {}, ensure_ascii=False),
                "",
                "AVOID REPEATING (JSON):",
                json.dumps(policy_obs.get("avoid_repeating") if isinstance(policy_obs.get("avoid_repeating"), dict) else {}, ensure_ascii=False),
                "",
                "PREVIOUS REASONING TRACE (JSON):",
                json.dumps(policy_obs.get("reasoning_trace") if isinstance(policy_obs.get("reasoning_trace"), dict) else {}, ensure_ascii=False),
                "",
                "DONE / CONTENT CONTRACT:",
                "- If the task is already answered by the current page, return final now.",
                "- final.content must contain the user-facing answer.",
                "- Do not output a browser action when the answer is already visible.",
                "- Prefer quoting the visible metric / value directly in content.",
                "- Follow ACTIVE OBJECTIVE unless visible evidence already satisfies the task.",
                "- Respect AVOID REPEATING unless the page materially changed.",
                "- If LOCAL WORKFLOW CLOSURE shows ready_to_commit=true and a visible commit control exists, strongly prefer finishing that local workflow before exploring unrelated controls.",
                "- Use LOCAL HTML CONTEXT to understand which inputs and commit controls belong to the same active form or region.",
                "- If multiple actions are returned, they must form one short local workflow and reasoning_trace.plan must say why.",
                "- If SCORE FEEDBACK is present and success=true or score=1.0, prefer finishing now unless the page visibly contradicts that signal.",
                "",
                "BROWSER SNAPSHOT:",
                str(policy_obs.get("browser_state_snapshot") or "")[:3000],
                "",
                "VISIBLE TEXT / PAGE SUMMARY:",
                str(policy_obs.get("page_ir_text") or "")[:14000],
                "",
                "VISIBLE EVIDENCE (JSON):",
                json.dumps(
                    {
                        "likely_answers": (
                            policy_obs.get("page_observations", {}).get("likely_answers")
                            if isinstance(policy_obs.get("page_observations"), dict) and isinstance(policy_obs.get("page_observations", {}).get("likely_answers"), list)
                            else []
                        )[:8],
                        "relevant_lines": (
                            policy_obs.get("page_observations", {}).get("relevant_lines")
                            if isinstance(policy_obs.get("page_observations"), dict) and isinstance(policy_obs.get("page_observations", {}).get("relevant_lines"), list)
                            else []
                        )[:16],
                    },
                    ensure_ascii=False,
                ),
                "",
                "INTERACTIVE ELEMENTS (indexed tree-style):",
                str(policy_obs.get("browser_state_text") or "")[:14000],
                "",
                "INTERACTIVE ELEMENT SHORTLIST (JSON):",
                json.dumps(
                    (policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else [])[:64],
                    ensure_ascii=False,
                ),
                "",
                "RECENT ACTIONS AND RESULTS (JSON):",
                json.dumps(
                    (policy_obs.get("history_recent") if isinstance(policy_obs.get("history_recent"), list) else [])[:10],
                    ensure_ascii=False,
                ),
                "",
                "PAGE OBSERVATIONS (JSON):",
                json.dumps(policy_obs.get("page_observations") if isinstance(policy_obs.get("page_observations"), dict) else {}, ensure_ascii=False),
                "",
                "PAGE GROUPS (FORMS / CONTROL GROUPS / ITEM GROUPS JSON):",
                json.dumps(
                    {
                        "forms": (policy_obs.get("text_ir", {}).get("forms") if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("forms"), list) else [])[
                            :8
                        ],
                        "control_groups": (
                            policy_obs.get("text_ir", {}).get("control_groups")
                            if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("control_groups"), list)
                            else []
                        )[:12],
                        "cards": (policy_obs.get("text_ir", {}).get("cards") if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("cards"), list) else [])[
                            :12
                        ],
                        "visible_lines": (
                            policy_obs.get("text_ir", {}).get("visible_lines")
                            if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("visible_lines"), list)
                            else []
                        )[:32],
                        "page_facts": (
                            policy_obs.get("text_ir", {}).get("page_facts") if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("page_facts"), list) else []
                        )[:16],
                        "value_lines": (
                            policy_obs.get("text_ir", {}).get("value_lines")
                            if isinstance(policy_obs.get("text_ir"), dict) and isinstance(policy_obs.get("text_ir", {}).get("value_lines"), list)
                            else []
                        )[:16],
                    },
                    ensure_ascii=False,
                ),
                "",
                "UNAVAILABLE TOOLS:",
                ", ".join(policy_obs.get("unavailable_browser_tools") if isinstance(policy_obs.get("unavailable_browser_tools"), list) else list(_unavailable_browser_tools())),
                "",
                "PREVIOUS STEP VERDICT:",
                json.dumps(policy_obs.get("previous_step_verdict") if isinstance(policy_obs.get("previous_step_verdict"), dict) else {}, ensure_ascii=False),
            ]
            history_summary = str(policy_obs.get("history_summary") or "").strip()
            if history_summary:
                user_parts.extend(["", "HISTORY SUMMARY:", history_summary[:2000]])
            history_recent = policy_obs.get("history_recent") if isinstance(policy_obs.get("history_recent"), list) else []
            recent_failures = [item for item in history_recent[-8:] if isinstance(item, dict) and (not bool(item.get("exec_ok", True)) or str(item.get("error") or "").strip())]
            if recent_failures:
                user_parts.extend(["", "RECENT FAILURES (JSON):", json.dumps(recent_failures[:4], ensure_ascii=False)])
            loop_nudges = policy_obs.get("loop_nudges") if isinstance(policy_obs.get("loop_nudges"), list) else []
            if loop_nudges:
                user_parts.extend(["", "LOOP NUDGES:"] + [f"- {str(item)[:220]}" for item in loop_nudges[:8]])
            memory = policy_obs.get("memory") if isinstance(policy_obs.get("memory"), dict) else {}
            strategy_summary = str(memory.get("strategy_summary") or "").strip()
            capability_gap = (
                policy_obs.get("page_observations", {}).get("capability_gap")
                if isinstance(policy_obs.get("page_observations"), dict) and isinstance(policy_obs.get("page_observations", {}).get("capability_gap"), dict)
                else {}
            )
            if strategy_summary or capability_gap:
                user_parts.extend(
                    [
                        "",
                        "STRATEGY / CAPABILITY GAP:",
                        json.dumps(
                            {
                                "strategy_summary": strategy_summary,
                                "capability_gap": capability_gap,
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
            plan = policy_obs.get("plan") if isinstance(policy_obs.get("plan"), dict) else {}
            counters = policy_obs.get("counters") if isinstance(policy_obs.get("counters"), dict) else {}
            visual_notes = memory.get("visual_notes") if isinstance(memory.get("visual_notes"), list) else []
            visual_hints = memory.get("visual_element_hints") if isinstance(memory.get("visual_element_hints"), list) else []
            if visual_notes or visual_hints:
                user_parts.extend(
                    [
                        "",
                        "VISUAL OBSERVATIONS:",
                        json.dumps(
                            {"notes": visual_notes[:8], "element_hints": visual_hints[:12]},
                            ensure_ascii=False,
                        ),
                    ]
                )
            user_parts.extend(
                [
                    "",
                    "PLAN / MEMORY (JSON):",
                    json.dumps(
                        {
                            "task_constraints": policy_obs.get("task_constraints") if isinstance(policy_obs.get("task_constraints"), dict) else {},
                            "active_subgoal": policy_obs.get("active_subgoal") if isinstance(policy_obs.get("active_subgoal"), dict) else {},
                            "plan": plan,
                            "memory": memory,
                            "counters": counters,
                            "frontier": policy_obs.get("frontier") if isinstance(policy_obs.get("frontier"), dict) else {},
                        },
                        ensure_ascii=False,
                    ),
                    "ALLOWED BROWSER TOOLS: " + ", ".join(sorted([t for t in list(allowed_tools) if str(t).startswith("browser.")]) if allowed_tools else []),
                    "",
                    "Output schema examples:",
                    '{"type":"browser","tool_call":{"name":"browser.click","arguments":{"index":0}}}',
                    '{"type":"browser","reasoning":"The filter panel is already visible and the next useful step is to finish the local filter sequence before opening any result cards.","reasoning_trace":{"task_interpretation":"Use the current page controls to narrow results.","success_state":"The required filtered result is visible.","current_subgoal":"Apply the current filter set.","next_expected_proof":"An apply/search result change is visible.","drift_risks":"Opening unrelated result cards too early.","where_am_i":"A filter panel with visible controls.","state_assessment":"The target controls are already visible.","plan":"Complete the filter sequence locally before opening any result cards."},"working_state":{"current_page_kind":"filter results page","active_region":"filter panel","active_workflow":"apply the current filter set","completed_fields":["genre"],"pending_fields":["apply filters"],"completion_evidence_missing":["updated result set"],"next_milestone":"Apply the visible filter controls.","completion_state":"awaiting_local_completion"},"tool_calls":[{"name":"browser.select_dropdown","arguments":{"index":1,"text":"Comedy"}},{"name":"browser.click","arguments":{"index":2}}]}',
                    '{"type":"browser","tool_call":{"name":"browser.select_dropdown","arguments":{"index":1,"text":"Comedy"}}}',
                    '{"type":"browser","tool_call":{"name":"browser.done","arguments":{"content":"The total value is 2844."}}}',
                    '{"type":"final","done":true,"content":"The total value is 2844.","reasoning":"The total value is already visible on the page, so the task can finish without another browser action."}',
                    "",
                    "Instructions:",
                    "- Output JSON only.",
                    f"- Return at most {max_actions_per_step} browser actions.",
                    "- Do not burn steps on the same no-op pattern if nothing changed.",
                    "- Prefer arguments.index over selector or element_id when targeting an interactive element.",
                    "- For browser.select_dropdown, include a non-empty arguments.text.",
                    "- If the task is about narrowing results, use current-page controls before opening result items.",
                    "- Preserve placeholders exactly when typing.",
                    "- For informational tasks, prefer answering from what is already visible on the current page before opening more pages.",
                    "- If the current page is sufficient, finish now.",
                    "- Never emit unavailable tools.",
                    "- Do not return content like 'task completed'; return the actual answer.",
                ]
            )
            if meta_enabled:
                user_parts.insert(-5, '{"type":"meta","meta_tool":{"name":"META.FIND_ELEMENTS","arguments":{"role":"input","text":"search","limit":6}}}')
                if "META.VISION_QA" in _obs_meta_tools():
                    user_parts.insert(-5, '{"type":"meta","meta_tool":{"name":"META.VISION_QA","arguments":{"question":"Which visible control best applies the current filters?"}}}')
        user_text = "\n".join([part for part in user_parts if part is not None])[:45000]
        model = plan_model_name if (mode == "PLAN" or mode == "STUCK") else model_name
        self._debug_log(
            str(task_id or "task"),
            {
                "ts": _utc_now(),
                "phase": "llm_request",
                "mode": mode,
                "model": model,
                "system": system,
                "user": user_text,
                "allowed_tools_count": len(allowed_tools),
            },
        )
        try:
            raw = self.llm_call(
                task_id=str(task_id or "local"),
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.1,
                max_tokens=1600,
            )
            content = str((((raw or {}).get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "")
            usage = self._normalize_usage(raw, prompt_text=f"{system}\n{user_text}", response_text=content)
            try:
                obj = self._parse_json(content)
                normalized = self._normalize_decision(obj, allowed_tools, policy_obs=policy_obs)
            except Exception as parse_or_schema_err:
                if not self._repair_enabled():
                    raise parse_or_schema_err
                repaired = self._attempt_repair(
                    task_id=str(task_id or "local"),
                    model=model,
                    mode=mode,
                    raw_content=content,
                    allowed_tools=allowed_tools,
                )
                if repaired is None:
                    raise parse_or_schema_err
                normalized, repair_usage, repair_model, repair_content = repaired
                self._debug_log(
                    str(task_id or "task"),
                    {
                        "ts": _utc_now(),
                        "phase": "llm_repair_response",
                        "mode": mode,
                        "model": str(repair_model or model),
                        "raw_content": repair_content,
                        "normalized": normalized,
                        "usage": repair_usage,
                        "original_error": str(parse_or_schema_err),
                    },
                )
                return normalized, {
                    "source": "llm_repair",
                    "usage": repair_usage,
                    "model": str(repair_model or model),
                }
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "llm_response",
                    "mode": mode,
                    "model": str((raw or {}).get("model") or model),
                    "raw_content": content,
                    "parsed": obj,
                    "normalized": normalized,
                    "usage": usage,
                },
            )
            return normalized, {
                "source": "llm",
                "usage": usage,
                "model": str((raw or {}).get("model") or model),
            }
        except Exception as e:
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "llm_error",
                    "mode": mode,
                    "model": model,
                    "error": str(e),
                },
            )
            fallback = self._fallback(prompt=prompt, mode=mode, policy_obs=policy_obs, allowed_tools=allowed_tools)
            self._debug_log(
                str(task_id or "task"),
                {
                    "ts": _utc_now(),
                    "phase": "fallback_decision",
                    "mode": mode,
                    "decision": fallback,
                },
            )
            return fallback, {"source": "fallback"}

    def _normalize_usage(
        self,
        raw: Dict[str, Any] | None,
        *,
        prompt_text: str = "",
        response_text: str = "",
    ) -> Dict[str, int]:
        usage = (raw or {}).get("usage") if isinstance((raw or {}).get("usage"), dict) else {}
        out = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
        if int(out["total_tokens"] or 0) > 0:
            return out
        prompt_chars = len(str(prompt_text or ""))
        response_chars = len(str(response_text or ""))
        est_prompt = max(1, prompt_chars // 4) if prompt_chars > 0 else 0
        est_completion = max(1, response_chars // 4) if response_chars > 0 else 0
        return {
            "prompt_tokens": int(est_prompt),
            "completion_tokens": int(est_completion),
            "total_tokens": int(est_prompt + est_completion),
        }

    def _extract_first_json_object(self, raw: str) -> str | None:
        if not raw:
            return None
        in_str = False
        esc = False
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
                continue
            if ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return raw[start : i + 1]
        return None

    def _parse_json(self, content: str) -> Dict[str, Any]:
        raw = str(content or "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        first_obj = self._extract_first_json_object(raw)
        if first_obj:
            try:
                obj = json.loads(first_obj)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            # Weak fallback for single-quoted dict-like outputs.
            try:
                lit = ast.literal_eval(first_obj)
                if isinstance(lit, dict):
                    return json.loads(json.dumps(lit, ensure_ascii=False))
            except Exception:
                pass
        raise ValueError("invalid_json_policy_output")

    def _normalize_decision(
        self,
        obj: Dict[str, Any],
        allowed_tools: set[str],
        *,
        policy_obs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        reasoning_trace = _normalize_reasoning_trace(obj.get("reasoning_trace"))
        working_state = _normalize_working_state(obj.get("working_state"))
        reasoning_summary = _candidate_text(_reasoning_trace_summary(reasoning_trace), obj.get("reasoning"))
        working_state_summary = _working_state_summary(working_state)
        if working_state_summary:
            reasoning_summary = _candidate_text(reasoning_summary, working_state_summary)
        max_actions_per_step = max(1, min(_env_int("FSM_MAX_ACTIONS_PER_STEP", 3), 5))
        t = str(obj.get("type") or "").strip().lower()
        if not t:
            if isinstance(obj.get("tool_call"), dict) or isinstance(obj.get("tool_calls"), list):
                t = "browser"
            elif isinstance(obj.get("meta_tool"), dict):
                t = "meta"
            elif str(obj.get("name") or obj.get("tool") or "").strip():
                maybe_name = _canonical_allowed_tool_name(str(obj.get("name") or obj.get("tool") or ""))
                if maybe_name.startswith("browser."):
                    t = "browser"
                elif maybe_name.startswith("meta.") or maybe_name.startswith("META."):
                    t = "meta"
            elif bool(obj.get("done")) or isinstance(obj.get("content"), str):
                t = "final"
        if t == "final":
            return {
                "type": "final",
                "done": True,
                "content": _candidate_text(obj.get("content"), "Task complete."),
                "reasoning": reasoning_summary,
                "reasoning_trace": reasoning_trace,
                "working_state": working_state,
            }
        if t == "meta":
            mt = obj.get("meta_tool") if isinstance(obj.get("meta_tool"), dict) else {}
            name = str(mt.get("name") or obj.get("name") or "").strip().upper()
            if name in _meta_tools() and (not allowed_tools or name in allowed_tools):
                return {
                    "type": "meta",
                    "name": name,
                    "arguments": (mt.get("arguments") if isinstance(mt.get("arguments"), dict) else (obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {})),
                    "reasoning": reasoning_summary,
                    "reasoning_trace": reasoning_trace,
                    "working_state": working_state,
                }
        if t == "browser":
            raw_calls = obj.get("tool_calls") if isinstance(obj.get("tool_calls"), list) else None
            normalized_calls: List[Dict[str, Any]] = []
            if raw_calls:
                for item in raw_calls[:max_actions_per_step]:
                    if isinstance(item, dict):
                        normalized_calls.append(item)
            else:
                tc = obj.get("tool_call") if isinstance(obj.get("tool_call"), dict) else {}
                if not tc:
                    tc = {
                        "name": str(obj.get("name") or obj.get("tool") or "").strip(),
                        "arguments": obj.get("arguments") if isinstance(obj.get("arguments"), dict) else {},
                    }
                normalized_calls = [tc]
            cleaned_calls: List[Dict[str, Any]] = []
            saw_done = False
            for tc in normalized_calls:
                raw_name = str(tc.get("name") or "").strip()
                name = _canonical_allowed_tool_name(raw_name)
                args = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
                if name == "browser.done":
                    saw_done = True
                    continue
                if name == "browser.input" and _is_generic_tool_placeholder(_candidate_text(args.get("text"), args.get("value")), kind="input"):
                    raise ValueError("invalid_browser_input_text")
                if name == "browser.select_dropdown":
                    selected_text = _candidate_text(args.get("text"), args.get("value"))
                    if _is_generic_tool_placeholder(selected_text, kind="select"):
                        if (not allowed_tools) or ("browser.dropdown_options" in allowed_tools):
                            tc = {
                                "name": "browser.dropdown_options",
                                "arguments": {key: value for key, value in args.items() if key in {"index", "element_id", "_element_id", "selector"}},
                            }
                            name = "browser.dropdown_options"
                        else:
                            raise ValueError("invalid_browser_select_text")
                if name.startswith("browser.") and (not allowed_tools or name in allowed_tools):
                    cleaned_calls.append(
                        {
                            "name": name,
                            "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {},
                        }
                    )
            if saw_done and not cleaned_calls:
                first_args = normalized_calls[0].get("arguments") if normalized_calls and isinstance(normalized_calls[0], dict) else {}
                return {
                    "type": "final",
                    "done": True,
                    "content": _candidate_text(
                        (first_args.get("content") if isinstance(first_args, dict) else None),
                        obj.get("content"),
                        "Task complete.",
                    ),
                    "reasoning": reasoning_summary,
                    "reasoning_trace": reasoning_trace,
                    "working_state": working_state,
                }
            if cleaned_calls:
                if isinstance(policy_obs, dict):
                    execution_profile = _effective_execution_profile(policy_obs)
                    if _is_demo_execution_profile(execution_profile):
                        preferred_title_result = _preferred_title_result_action(
                            str(policy_obs.get("prompt") or ""),
                            policy_obs,
                            allowed_tools=allowed_tools,
                        )
                        if preferred_title_result is not None:
                            preferred_title_call = preferred_title_result.get("tool_call")
                            if isinstance(preferred_title_call, dict) and not any(_tool_call_matches(preferred_title_call, call) for call in cleaned_calls):
                                cleaned_calls = [preferred_title_call]
                        preferred_seed_navigation = _preferred_seed_stable_navigation(
                            str(policy_obs.get("prompt") or ""),
                            policy_obs,
                            allowed_tools=allowed_tools,
                        )
                        if preferred_seed_navigation is not None:
                            preferred_seed_call = preferred_seed_navigation.get("tool_call")
                            if isinstance(preferred_seed_call, dict) and not any(_tool_call_matches(preferred_seed_call, call) for call in cleaned_calls):
                                cleaned_calls = [preferred_seed_call]
                        preferred = _preferred_direct_intent_action(
                            str(policy_obs.get("prompt") or ""),
                            policy_obs,
                            allowed_tools=allowed_tools,
                        )
                        if preferred is not None:
                            preferred_call, preferred_candidate = preferred
                            current_task_intents = _autocinema_task_intent_tags(
                                str(policy_obs.get("prompt") or ""),
                                policy_obs,
                            )
                            keeps_direct_intent = False
                            for call in cleaned_calls:
                                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                                if _candidate_matches_tool_args(preferred_candidate, args):
                                    keeps_direct_intent = True
                                    break
                                for item in policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else []:
                                    if not isinstance(item, dict):
                                        continue
                                    if _obs_candidate_primary_intent_tags(item).intersection(current_task_intents) and _candidate_matches_tool_args(item, args):
                                        keeps_direct_intent = True
                                        break
                                if keeps_direct_intent:
                                    break
                            if not keeps_direct_intent:
                                cleaned_calls = [preferred_call]
                        preferred_markup_action = _preferred_direct_intent_action_from_markup(
                            str(policy_obs.get("prompt") or ""),
                            policy_obs,
                            allowed_tools=allowed_tools,
                        )
                        if preferred_markup_action is not None:
                            preferred_markup_call = preferred_markup_action.get("tool_call")
                            if isinstance(preferred_markup_call, dict) and not any(_tool_call_matches(preferred_markup_call, call) for call in cleaned_calls):
                                cleaned_calls = [preferred_markup_call]
                out: Dict[str, Any] = {
                    "type": "browser",
                    "reasoning": reasoning_summary,
                    "reasoning_trace": reasoning_trace,
                    "working_state": working_state,
                }
                if len(cleaned_calls) == 1:
                    out["tool_call"] = cleaned_calls[0]
                out["tool_calls"] = cleaned_calls
                return out
        raise ValueError("invalid_policy_decision")

    def _attempt_repair(
        self,
        *,
        task_id: str,
        model: str,
        mode: str,
        raw_content: str,
        allowed_tools: set[str],
    ) -> tuple[Dict[str, Any], Dict[str, int], str, str] | None:
        repair_system = (
            "You fix malformed agent policy outputs.\n"
            "Return exactly ONE valid JSON object and nothing else.\n"
            "Output must match one of:\n"
            '1) {"type":"browser","tool_call":{"name":"browser.<tool>","arguments":{"index":0}}}\n'
            '2) {"type":"meta","meta_tool":{"name":"META.<TOOL>","arguments":{}}}\n'
            '3) {"type":"final","done":true,"content":"..."}'
        )
        repair_user = {
            "mode": mode,
            "allowed_browser_tools": sorted(list(allowed_tools)) if allowed_tools else [],
            "raw_response": str(raw_content or "")[:12000],
        }
        try:
            raw = self.llm_call(
                task_id=str(task_id or "local"),
                model=str(model),
                messages=[
                    {"role": "system", "content": repair_system},
                    {"role": "user", "content": json.dumps(repair_user, ensure_ascii=False)},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            content = str((((raw or {}).get("choices") or [{}])[0].get("message", {}) or {}).get("content") or "")
            obj = self._parse_json(content)
            normalized = self._normalize_decision(obj, allowed_tools)
            usage = self._normalize_usage(
                raw,
                prompt_text=f"{repair_system}\n{json.dumps(repair_user, ensure_ascii=False)}",
                response_text=content,
            )
            model_name = str((raw or {}).get("model") or model or "")
            return normalized, usage, model_name, content
        except Exception:
            return None

    def _fallback(
        self,
        *,
        prompt: str,
        mode: str,
        policy_obs: Dict[str, Any],
        allowed_tools: set[str],
    ) -> Dict[str, Any]:
        def allow(name: str) -> bool:
            return (not allowed_tools) or (name in allowed_tools)

        execution_profile = _effective_execution_profile(policy_obs)
        candidates = policy_obs.get("candidates") if isinstance(policy_obs.get("candidates"), list) else []
        partitions = policy_obs.get("candidate_partitions") if isinstance(policy_obs.get("candidate_partitions"), dict) else {}
        local_candidates = partitions.get("local") if isinstance(partitions.get("local"), list) else []
        escape_candidates = partitions.get("escape") if isinstance(partitions.get("escape"), list) else []
        global_candidates = partitions.get("global") if isinstance(partitions.get("global"), list) else []
        memory = policy_obs.get("memory") if isinstance(policy_obs.get("memory"), dict) else {}
        visual_hints = {str(x) for x in (memory.get("visual_element_hints") if isinstance(memory.get("visual_element_hints"), list) else []) if str(x).strip()}
        typed_candidate_ids = {str(x) for x in (memory.get("typed_candidate_ids") if isinstance(memory.get("typed_candidate_ids"), list) else []) if str(x).strip()}
        first = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        flags = policy_obs.get("flags") if isinstance(policy_obs.get("flags"), dict) else {}
        counters = policy_obs.get("counters") if isinstance(policy_obs.get("counters"), dict) else {}
        loop_level = str(flags.get("loop_level") or "none")
        stall_count = int(counters.get("stall_count") or 0)
        repeat_count = int(counters.get("repeat_action_count") or 0)
        recovery_attempt_count = int(counters.get("recovery_attempt_count") or 0)
        consecutive_wait_count = int(counters.get("consecutive_wait_count") or 0)
        route_like_stuck = mode in {"STUCK", "PLAN"} or loop_level == "high" or stall_count >= 4 or repeat_count >= 4
        max_consecutive_waits = max(0, min(_env_int("FSM_MAX_CONSECUTIVE_WAITS", 1), 4))
        max_recovery_attempts = max(1, min(_env_int("FSM_MAX_RECOVERY_ATTEMPTS", 3), 8))
        prefer_text_input = _prompt_prefers_text_input(prompt, policy_obs)
        if _is_demo_execution_profile(execution_profile):
            preferred_title_result = _preferred_title_result_action(
                prompt,
                policy_obs,
                allowed_tools=allowed_tools,
            )
            if preferred_title_result is not None:
                return preferred_title_result
            preferred_seed_navigation = _preferred_seed_stable_navigation(
                prompt,
                policy_obs,
                allowed_tools=allowed_tools,
            )
            if preferred_seed_navigation is not None:
                return preferred_seed_navigation
            preferred_prompt_navigation = _preferred_prompt_navigation(
                prompt,
                policy_obs,
                allowed_tools=allowed_tools,
            )
            if preferred_prompt_navigation is not None:
                return preferred_prompt_navigation
            preferred_direct_action = _preferred_direct_intent_action(
                prompt,
                policy_obs,
                allowed_tools=allowed_tools,
            )
            if preferred_direct_action is not None:
                return {"type": "browser", "tool_call": preferred_direct_action[0]}
            preferred_markup_action = _preferred_direct_intent_action_from_markup(
                prompt,
                policy_obs,
                allowed_tools=allowed_tools,
            )
            if preferred_markup_action is not None:
                return preferred_markup_action

        def candidate_id(item: Dict[str, Any]) -> str:
            return str(item.get("id") or item.get("element_id") or item.get("_element_id") or "").strip()

        def candidate_index(item: Dict[str, Any]) -> int | None:
            try:
                return int(item.get("index"))
            except (TypeError, ValueError):
                return None

        def browser_action_for_candidate(item: Dict[str, Any]) -> Dict[str, Any] | None:
            role = str(item.get("role") or "").strip().lower()
            selector = item.get("selector") if isinstance(item.get("selector"), dict) else None
            if not isinstance(selector, dict):
                return None
            element_id = candidate_id(item)
            index = candidate_index(item)

            def target_args(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
                args: Dict[str, Any] = {}
                if index is not None:
                    args["index"] = index
                elif element_id:
                    args["element_id"] = element_id
                if extra:
                    args.update(extra)
                return args

            if role in {"input", "textarea"} and element_id and element_id in typed_candidate_ids:
                return None
            if role in {"button", "link"} and allow("browser.click"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.click",
                        "arguments": target_args(),
                    },
                }
            if role in {"input", "textarea"} and allow("browser.input") and prefer_text_input:
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.input",
                        "arguments": target_args({"text": ""}),
                    },
                }
            if role in {"input", "textarea"} and allow("browser.click"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.click",
                        "arguments": target_args(),
                    },
                }
            if role == "select" and allow("browser.dropdown_options"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.dropdown_options",
                        "arguments": target_args(),
                    },
                }
            if role == "input" and allow("browser.click"):
                return {
                    "type": "browser",
                    "tool_call": {
                        "name": "browser.click",
                        "arguments": target_args(),
                    },
                }
            return None

        if route_like_stuck and escape_candidates:
            for cand in escape_candidates:
                if isinstance(cand, dict):
                    act = browser_action_for_candidate(cand)
                    if act is not None:
                        return act
        if route_like_stuck and allow("browser.go_back"):
            return {"type": "browser", "tool_call": {"name": "browser.go_back", "arguments": {}}}
        ordered = []
        ordered.extend([cand for cand in local_candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in escape_candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in candidates if isinstance(cand, dict)])
        ordered.extend([cand for cand in global_candidates if isinstance(cand, dict)])
        for cand in ordered:
            if not isinstance(cand, dict):
                continue
            if candidate_id(cand) in visual_hints:
                action = browser_action_for_candidate(cand)
                if action is not None:
                    return action
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            action = browser_action_for_candidate(cand)
            if action is not None:
                return action
        if first and allow("browser.click"):
            try:
                first_index = int(first.get("index"))
            except (TypeError, ValueError):
                first_index = None
            if first_index is not None:
                return {"type": "browser", "tool_call": {"name": "browser.click", "arguments": {"index": first_index}}}
            sel = first.get("selector") if isinstance(first.get("selector"), dict) else None
            if sel:
                return {"type": "browser", "tool_call": {"name": "browser.click", "arguments": {"selector": sel}}}
        if allow("browser.wait") and consecutive_wait_count < max_consecutive_waits and recovery_attempt_count < max_recovery_attempts:
            return {"type": "browser", "tool_call": {"name": "browser.wait", "arguments": {"time_seconds": 1.0}}}
        if allow("browser.scroll"):
            return {"type": "browser", "tool_call": {"name": "browser.scroll", "arguments": {"direction": "down", "amount": 600}}}
        return {"type": "final", "done": True, "content": "No safe browser action available from allowed_tools."}


__all__ = [name for name in globals() if not name.startswith("__")]
