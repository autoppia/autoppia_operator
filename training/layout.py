from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UseCaseLayout:
    repo_root: Path
    web_project: str
    use_case: str

    @property
    def slug(self) -> str:
        return self.use_case.strip().lower()

    @property
    def root(self) -> Path:
        return self.repo_root / "data" / self.web_project / self.slug

    @property
    def gold_dir(self) -> Path:
        return self.root / "gold"

    @property
    def runs_dir(self) -> Path:
        return self.gold_dir / "runs"

    @property
    def traces_dir(self) -> Path:
        return self.gold_dir / "traces"

    @property
    def attempts_path(self) -> Path:
        return self.gold_dir / "attempts.jsonl"

    @property
    def episodes_path(self) -> Path:
        return self.gold_dir / "episodes.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.gold_dir / "summary.json"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    @property
    def dagger_dir(self) -> Path:
        return self.root / "dagger"

    @property
    def clusters_dir(self) -> Path:
        return self.dagger_dir / "clusters"

    @property
    def sft_dir(self) -> Path:
        return self.root / "sft"

    @property
    def reward_dir(self) -> Path:
        return self.root / "reward"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def models_dir(self) -> Path:
        return self.repo_root / "models"

    def ensure_dirs(self) -> None:
        for path in (
            self.root,
            self.gold_dir,
            self.runs_dir,
            self.traces_dir,
            self.splits_dir,
            self.dagger_dir,
            self.clusters_dir,
            self.sft_dir,
            self.reward_dir,
            self.eval_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def use_case_layout(*, repo_root: Path, web_project: str, use_case: str) -> UseCaseLayout:
    layout = UseCaseLayout(repo_root=repo_root, web_project=web_project, use_case=use_case)
    layout.ensure_dirs()
    return layout
