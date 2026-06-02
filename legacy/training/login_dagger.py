from __future__ import annotations

from pathlib import Path

from training.dagger import merge_dagger_episodes, run_dagger_round
from training.use_case_registry import dagger_extra_lines


def login_dagger_extra_lines(*, failure_category: str) -> list[str]:
    return dagger_extra_lines(use_case="LOGIN", failure_category=failure_category)


def run_login_dagger_round(
    *,
    source_summary_path: Path,
    split_name: str,
    teacher_provider: str = "openai",
    teacher_model: str = "gpt-5.4",
    max_steps: int = 12,
) -> dict[str, object]:
    return run_dagger_round(
        use_case="LOGIN",
        source_summary_path=source_summary_path,
        split_name=split_name,
        teacher_provider=teacher_provider,
        teacher_model=teacher_model,
        max_steps=max_steps,
        env_overrides={"DEMO_WEBS_ENDPOINT": "http://84.247.180.192"},
    )


def merge_login_episodes(*, base_episodes_path: Path, dagger_teacher_path: Path, output_path: Path) -> dict[str, object]:
    return merge_dagger_episodes(
        base_episodes_path=base_episodes_path,
        dagger_teacher_path=dagger_teacher_path,
        output_path=output_path,
    )
