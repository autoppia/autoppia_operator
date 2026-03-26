"""End-to-end training orchestration: collect data -> train -> deploy models.

Coordinates the full pipeline from trajectory data to deployed models.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .export import (
    export_dagger,
    export_pairwise_ranking,
    export_sft,
    export_verifier_classification,
    load_episodes,
)
from .gpu_runner import GPUTrainingRunner
from .runpod_config import RunPodConfig


class TrainingOrchestrator:
    """Orchestrates the full training pipeline.

    Steps:
    1. Load collected trajectory data
    2. Export training datasets (ranking pairs, verifier examples)
    3. Optionally provision GPU via RunPod
    4. Train ranker and verifier models
    5. Validate trained models
    6. Deploy models to models/ directory
    """

    def __init__(
        self,
        data_dir: str = "data/trajectories",
        models_dir: str = "models",
        config: Optional[RunPodConfig] = None,
    ) -> None:
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.gpu_runner: Optional[GPUTrainingRunner] = None
        if config:
            self.gpu_runner = GPUTrainingRunner(config=config)

    def export_datasets(self) -> Dict[str, Any]:
        """Export training datasets from collected episodes.

        Returns counts for each dataset type.
        """
        episodes_path = os.path.join(self.data_dir, "episodes.jsonl")
        episodes = load_episodes(episodes_path)

        if not episodes:
            return {"error": "No episodes found", "path": episodes_path}

        results: Dict[str, Any] = {"total_episodes": len(episodes)}

        # Export pairwise ranking data
        ranking_path = os.path.join(self.data_dir, "ranking_pairs.jsonl")
        results["ranking_pairs"] = export_pairwise_ranking(episodes, ranking_path)

        # Export verifier classification data
        verifier_path = os.path.join(self.data_dir, "verifier_examples.jsonl")
        results["verifier_examples"] = export_verifier_classification(episodes, verifier_path)

        # Export SFT data
        sft_path = os.path.join(self.data_dir, "sft_examples.jsonl")
        results["sft_examples"] = export_sft(episodes, sft_path)

        # Export DAgger data
        dagger_path = os.path.join(self.data_dir, "dagger_examples.jsonl")
        results["dagger_examples"] = export_dagger(episodes, dagger_path)

        return results

    def train_ranker_local(
        self,
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train ranker model locally (CPU or available GPU).

        Args:
            data_path: Path to ranking pairs JSONL. Defaults to data_dir/ranking_pairs.jsonl.
            output_path: Path for model weights. Defaults to models/ranker/model.pt.
            **kwargs: Additional args passed to train().
        """
        from .train_ranker import train

        data = data_path or os.path.join(self.data_dir, "ranking_pairs.jsonl")
        output = output_path or os.path.join(self.models_dir, "ranker", "model.pt")

        return train(data_path=data, output_path=output, **kwargs)

    def train_verifier_local(
        self,
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train verifier model locally.

        Args:
            data_path: Path to verifier examples JSONL. Defaults to data_dir/verifier_examples.jsonl.
            output_path: Path for model weights. Defaults to models/verifier/model.pt.
            **kwargs: Additional args passed to train().
        """
        from .train_verifier import train

        data = data_path or os.path.join(self.data_dir, "verifier_examples.jsonl")
        output = output_path or os.path.join(self.models_dir, "verifier", "model.pt")

        return train(data_path=data, output_path=output, **kwargs)

    def train_on_gpu(self, component: str = "ranker") -> Dict[str, Any]:
        """Train a component on a RunPod GPU.

        Args:
            component: Which component to train ('ranker' or 'verifier').

        Returns:
            Status dict with pod info.
        """
        if self.gpu_runner is None:
            return {"error": "GPU runner not configured"}

        if component == "ranker":
            script = "training.train_ranker"
            data = os.path.join(self.data_dir, "ranking_pairs.jsonl")
            output = os.path.join(self.models_dir, "ranker", "model.pt")
        elif component == "verifier":
            script = "training.train_verifier"
            data = os.path.join(self.data_dir, "verifier_examples.jsonl")
            output = os.path.join(self.models_dir, "verifier", "model.pt")
        else:
            return {"error": f"Unknown component: {component}"}

        return self.gpu_runner.run_training_job(
            script=script,
            data_path=data,
            output_path=output,
        )

    def run_full_pipeline(self, use_gpu: bool = False) -> Dict[str, Any]:
        """Run the complete training pipeline.

        1. Export datasets from episodes
        2. Train ranker
        3. Train verifier
        4. Report results

        Args:
            use_gpu: Whether to use RunPod GPU. If False, trains locally.

        Returns:
            Pipeline results dict.
        """
        results: Dict[str, Any] = {}

        # Step 1: Export
        results["export"] = self.export_datasets()
        if "error" in results["export"]:
            return results

        # Step 2: Train ranker
        if use_gpu:
            results["ranker_training"] = self.train_on_gpu("ranker")
        else:
            results["ranker_training"] = self.train_ranker_local()

        # Step 3: Train verifier
        if use_gpu:
            results["verifier_training"] = self.train_on_gpu("verifier")
        else:
            results["verifier_training"] = self.train_verifier_local()

        # Step 4: Check model files exist
        ranker_path = os.path.join(self.models_dir, "ranker", "model.pt")
        verifier_path = os.path.join(self.models_dir, "verifier", "model.pt")
        results["models"] = {
            "ranker_exists": os.path.exists(ranker_path),
            "verifier_exists": os.path.exists(verifier_path),
        }

        return results
