from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UseCaseSpec:
    name: str
    harvester_hints: tuple[str, ...]
    harvester_model_ladder: tuple[str, ...] = ("gpt-5.4-mini", "gpt-5.4")
    likely_failure_clusters: tuple[str, ...] = ()
    dagger_base_hints: tuple[str, ...] = ()
    dagger_failure_hints: dict[str, tuple[str, ...]] | None = None
    recommended_gold_target: int = 500
    recommended_holdout_target: int = 50


_USE_CASE_SPECS: dict[str, UseCaseSpec] = {
    "ADD_COMMENT": UseCaseSpec(
        name="ADD_COMMENT",
        harvester_hints=(
            "Open the requested movie detail page first and stay on that movie.",
            "Use the visible comment form only: fill name, fill comment text, then submit the comment.",
            "Do not navigate to registration, profile, or unrelated movies once the detail page is visible.",
        ),
        likely_failure_clusters=("wrong_movie_detail", "comment_form_variant", "no_progress_loop"),
    ),
    "ADD_FILM": UseCaseSpec(
        name="ADD_FILM",
        harvester_hints=(
            "Navigate directly to the add-film or admin new-movie page and stay in the admin film creation workflow.",
            "Fill the visible movie creation form fields before submitting.",
            "Do not drift to search, movie detail pages, or registration once the admin form is visible.",
        ),
        likely_failure_clusters=("admin_page_not_reached", "film_form_variant", "auth_gate_drift", "no_progress_loop"),
    ),
    "ADD_TO_WATCHLIST": UseCaseSpec(
        name="ADD_TO_WATCHLIST",
        harvester_hints=(
            "Open the requested movie detail page first and stay on that movie.",
            "If login is required, complete login, return to the same movie detail page, then click the visible watchlist add control.",
            "Do not register a new user and do not open unrelated movies.",
        ),
        likely_failure_clusters=("detail_not_reached", "auth_gate_drift", "watchlist_control_variant", "no_progress_loop"),
    ),
    "CONTACT": UseCaseSpec(
        name="CONTACT",
        harvester_hints=(
            "Navigate directly to /contact and stay on the contact workflow only.",
            "Use the visible contact form only. The code-backed canonical field ids are contact-name-input, contact-email-input, contact-subject-input, contact-message-textarea, and the submit button is send-message-button.",
            "Fill the contact form exactly to satisfy the task constraints, then submit the form once. Do not leave required fields blank.",
            "Do not visit registration or unrelated pages once the contact page is visible.",
        ),
        likely_failure_clusters=("contact_form_variant", "wrong_page_navigation", "no_progress_loop"),
    ),
    "DELETE_FILM": UseCaseSpec(
        name="DELETE_FILM",
        harvester_hints=(
            "Navigate to the admin edit page for the target movie and stay in that admin workflow.",
            "If login is required, complete login and return to the same admin edit view.",
            "Use the visible delete control for that movie only.",
        ),
        likely_failure_clusters=("admin_edit_not_reached", "auth_gate_drift", "delete_control_variant", "no_progress_loop"),
    ),
    "EDIT_FILM": UseCaseSpec(
        name="EDIT_FILM",
        harvester_hints=(
            "Navigate to the admin edit page for the target movie and stay there.",
            "If login is required, complete login and return to the same admin edit page.",
            "Change the visible movie form fields and save the form.",
        ),
        likely_failure_clusters=("admin_edit_not_reached", "auth_gate_drift", "film_form_variant", "no_progress_loop"),
    ),
    "EDIT_USER": UseCaseSpec(
        name="EDIT_USER",
        harvester_hints=(
            "Navigate to the profile or account settings page and stay in the user profile workflow only.",
            "If login is required, complete login and continue editing the visible profile form.",
            "Save the updated profile and avoid registration or unrelated movie pages.",
        ),
        likely_failure_clusters=("profile_not_reached", "auth_gate_drift", "profile_form_variant", "no_progress_loop"),
    ),
    "FILM_DETAIL": UseCaseSpec(
        name="FILM_DETAIL",
        harvester_hints=(
            "Open the requested movie detail page from the homepage or search results.",
            "Click a matching movie card or detail link and stop on the correct detail page.",
            "Do not wait repeatedly or open an unrelated movie card.",
        ),
        likely_failure_clusters=("wrong_movie_card", "wait_loop", "detail_link_not_found"),
    ),
    "FILTER_FILM": UseCaseSpec(
        name="FILTER_FILM",
        harvester_hints=(
            "Stay on the movies listing page and use the visible filter controls there.",
            "Apply the requested filter and keep the relevant movie visible after filtering.",
            "Do not navigate into registration or unrelated detail pages before the filter is applied.",
        ),
        likely_failure_clusters=("filter_control_variant", "listing_not_reached", "premature_detail_navigation"),
    ),
    "LOGIN": UseCaseSpec(
        name="LOGIN",
        harvester_hints=(
            "Navigate directly to /login first. Stay on the login workflow only.",
            "Fill the username field with user1, then fill the password field with Passw0rd!, then click the visible sign in button.",
            "Do not open movie details, comments, profile edits, search, or registration.",
        ),
        likely_failure_clusters=(
            "never_reaches_login",
            "login_dom_variant",
            "no_progress_loop",
            "action_execution_error",
            "browser_runtime_failure",
        ),
        dagger_base_hints=(
            "The only valid workflow is homepage -> /login -> username -> password -> sign in.",
            "Ignore movie cards, featured movie buttons, comments, trailers, profile links, search, and registration.",
            "If you are not on /login yet, the only acceptable next action is the login link or direct /login navigation.",
            "Once on /login, only interact with inputs whose ids or names refer to login, username, user, email, or password, and the visible sign in button.",
            "Never type into the same username field twice in a row.",
            "After typing username once, move to the password field.",
            "After typing the password once, click the visible sign in button exactly once unless the page content changes.",
            "If username and password are already filled, do not type again; click the sign in control.",
        ),
        dagger_failure_hints={
            "ACTION_EXECUTION_ERROR": (
                "Prefer the visible login form fields with stable ids or names over unrelated inputs.",
                "Do not click decorative controls or navigation once the login form is visible.",
            ),
            "NO_PROGRESS_LOOP": (
                "Break repetition: if the last action was typing username or password, do not repeat it on the next step.",
                "Once both credentials are present, the next useful action is submit, not another type action.",
            ),
        },
    ),
    "LOGOUT": UseCaseSpec(
        name="LOGOUT",
        harvester_hints=(
            "Navigate to the logged-in profile or account page and stay there.",
            "If authentication is required first, complete login, then use the visible log out control.",
            "Do not open unrelated movie pages or registration after login.",
        ),
        likely_failure_clusters=("profile_not_reached", "auth_gate_drift", "logout_control_variant", "no_progress_loop"),
    ),
    "REGISTRATION": UseCaseSpec(
        name="REGISTRATION",
        harvester_hints=(
            "Navigate directly to /register and stay on the registration workflow only.",
            "Fill the visible registration form completely, then submit the registration form.",
            "Do not drift to login, search, or movie detail pages while registering.",
        ),
        likely_failure_clusters=("register_form_variant", "wrong_page_navigation", "no_progress_loop"),
    ),
    "REMOVE_FROM_WATCHLIST": UseCaseSpec(
        name="REMOVE_FROM_WATCHLIST",
        harvester_hints=(
            "Navigate to the watchlist page and stay in the watchlist workflow only.",
            "If login is required, complete login and return to the watchlist.",
            "Use the visible remove or toggle-off watchlist control for the target movie.",
        ),
        likely_failure_clusters=("watchlist_page_not_reached", "auth_gate_drift", "watchlist_control_variant", "no_progress_loop"),
    ),
    "SEARCH_FILM": UseCaseSpec(
        name="SEARCH_FILM",
        harvester_hints=(
            "Stay on the homepage or movies search workflow only.",
            "Use the visible search field to search for the requested movie title, then submit the search.",
            "Do not wait repeatedly, and do not open registration or unrelated movie details before searching.",
        ),
        likely_failure_clusters=("search_field_not_found", "wait_loop", "premature_detail_navigation"),
    ),
    "SHARE_MOVIE": UseCaseSpec(
        name="SHARE_MOVIE",
        harvester_hints=(
            "Open the requested movie detail page first and stay there.",
            "Use the visible share control or share form for that movie only.",
            "Do not open unrelated pages once the detail page is visible.",
        ),
        likely_failure_clusters=("detail_not_reached", "share_control_variant", "no_progress_loop"),
    ),
    "WATCH_TRAILER": UseCaseSpec(
        name="WATCH_TRAILER",
        harvester_hints=(
            "Open the requested movie detail page first and stay there.",
            "Use the visible trailer or play-trailer control for that movie only.",
            "Do not open unrelated pages or wait repeatedly once the detail page is visible.",
        ),
        likely_failure_clusters=("detail_not_reached", "trailer_control_variant", "wait_loop"),
    ),
}


def all_use_case_specs() -> tuple[UseCaseSpec, ...]:
    return tuple(_USE_CASE_SPECS[name] for name in sorted(_USE_CASE_SPECS))


def get_use_case_spec(use_case: str) -> UseCaseSpec:
    normalized = str(use_case or "").strip().upper()
    return _USE_CASE_SPECS.get(normalized, UseCaseSpec(name=normalized or "UNKNOWN", harvester_hints=()))


def dagger_extra_lines(*, use_case: str, failure_category: str) -> list[str]:
    spec = get_use_case_spec(use_case)
    lines = [str(line).strip() for line in spec.dagger_base_hints if str(line).strip()]
    category = str(failure_category or "").strip().upper()
    extra = spec.dagger_failure_hints or {}
    for line in extra.get(category, ()):
        text = str(line).strip()
        if text:
            lines.append(text)
    return lines
