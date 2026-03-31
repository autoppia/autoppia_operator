#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _branch(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "branch", "--show-current"],
        text=True,
    ).strip()


def main() -> int:
    repo = Path(os.environ["ARBOS_TARGET_REPO"]).resolve()
    iwa_repo = repo.parent / "autoppia_iwa"

    assert repo.name == "autoppia_operator"
    assert iwa_repo.is_dir(), f"Missing sibling autoppia_iwa repo at {iwa_repo}"
    assert _branch(repo) == "arbos", f"autoppia_operator must run on branch arbos, got {_branch(repo)!r}"
    assert _branch(iwa_repo) in {"arbos", "daryxx"}, f"autoppia_iwa must be checked out on a compatible branch for the arbos harvest flow, got {_branch(iwa_repo)!r}"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(iwa_repo),
            str(repo),
            env.get("PYTHONPATH", ""),
        ]
    ).strip(os.pathsep)

    result = subprocess.run(
        [sys.executable, "eval.py", "--help"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"eval.py --help failed under the current daryxx layout.\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    print("PASS: arbos branch alignment and eval CLI smoke are healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
