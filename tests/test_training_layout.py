from pathlib import Path

from training.focus_pipeline import focus_root
from training.layout import use_case_layout


def test_use_case_layout_builds_canonical_tree(tmp_path: Path) -> None:
    layout = use_case_layout(repo_root=tmp_path, web_project="autocinema", use_case="LOGIN")

    assert layout.root == tmp_path / "data" / "autocinema" / "login"
    assert layout.gold_dir == layout.root / "gold"
    assert layout.sft_dir == layout.root / "sft"
    assert layout.eval_dir == layout.root / "eval"
    assert layout.reward_dir == layout.root / "reward"
    assert layout.traces_dir == layout.gold_dir / "traces"
    assert layout.summary_path == layout.gold_dir / "summary.json"
    assert layout.clusters_dir == layout.dagger_dir / "clusters"
    assert layout.gold_dir.exists()
    assert layout.splits_dir.exists()
    assert layout.clusters_dir.exists()


def test_focus_root_uses_canonical_login_layout() -> None:
    root = focus_root(use_case="LOGIN")
    assert str(root).endswith("data/autocinema/login")
