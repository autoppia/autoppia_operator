"""Canonical Autoppia Operator package.

The Autoppia Operator is the trajectory-discovery agent: a Claude/code process
that can inspect IWA, run browser trajectories, evaluate them, and iterate until
IWA verifies success. Distilled `/act` agents are downstream training artifacts,
not the operator itself.
"""

from training.autoppia_operator.artifacts import (
    OPERATOR_ARTIFACT_VERSION,
    OperatorArtifactPaths,
    build_operator_manifest,
    is_verified_success,
    operator_run_root,
)


def __getattr__(name: str):
    if name in {"OperatorRunConfig", "run_autoppia_operator"}:
        from training.autoppia_operator.claude_operator import OperatorRunConfig, run_autoppia_operator

        return {"OperatorRunConfig": OperatorRunConfig, "run_autoppia_operator": run_autoppia_operator}[name]
    raise AttributeError(name)

__all__ = [
    "OPERATOR_ARTIFACT_VERSION",
    "OperatorArtifactPaths",
    "OperatorRunConfig",
    "build_operator_manifest",
    "is_verified_success",
    "operator_run_root",
    "run_autoppia_operator",
]
