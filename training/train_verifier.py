"""Training script for the learned progress verifier.

Self-contained — can run on a fresh GPU pod with torch installed.

Usage:
    python -m training.train_verifier --data data/verifier_examples.jsonl --output models/verifier/model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from verifier.features import VERIFIER_FEATURE_DIM
from verifier.learned import LABEL_TO_IDX, NUM_CLASSES, VERIFIER_LABELS, VerifierClassifier


class VerifierClassificationDataset(Dataset):
    """PyTorch dataset for verifier multi-class classification."""

    def __init__(self, path: str) -> None:
        self.features: List[List[float]] = []
        self.labels: List[int] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                label_str = obj.get("label", "no_op")
                label_idx = LABEL_TO_IDX.get(label_str, LABEL_TO_IDX.get("no_op", 2))

                # Extract numeric features from the example
                feat_vec = self._extract_features(obj)
                self.features.append(feat_vec)
                self.labels.append(label_idx)

    def _extract_features(self, obj: Dict[str, Any]) -> List[float]:
        """Build feature vector from exported example fields.

        Matches the feature order from verifier/features.py.
        All features are structural/numeric — no keyword matching.
        """
        feats: List[float] = []

        # URL change features (4 dims)
        feats.append(1.0 if obj.get("url_changed", False) else 0.0)
        url = str(obj.get("url", ""))
        feats.append(min(len(url) / 200.0, 1.0))
        path_parts = [p for p in url.split("/") if p and "://" not in p]
        feats.append(min(len(path_parts) / 10.0, 1.0))
        # URL-task token overlap (computed from available fields)
        task_text = str(obj.get("task_text", "")).lower()
        task_tokens = set(task_text.split()) if task_text else set()
        url_tokens = set(url.lower().replace("/", " ").replace("-", " ").replace("_", " ").split())
        url_overlap = len(task_tokens & url_tokens) if task_tokens else 0
        feats.append(min(url_overlap / max(len(task_tokens), 1), 1.0))

        # Page title features (3 dims)
        feats.append(0.0)  # title_len (not in export)
        feats.append(0.0)  # title_task_overlap (not in export)
        feats.append(0.0)  # title_changed (not in export)

        # DOM diff features (4 dims)
        feats.append(1.0 if obj.get("dom_changed", False) else 0.0)
        feats.append(min(float(obj.get("dom_node_count", 0)) / 500.0, 1.0))
        feats.append(0.0)  # dom_node_delta (not in export)
        feats.append(min(float(obj.get("candidate_count", 0)) / 50.0, 1.0))

        # Action result features (5 dims) — structural encoding
        action_type = str(obj.get("chosen_action_type", "")).lower()
        feats.append(min(len(action_type) / 10.0, 1.0))  # action name length
        prev_action = str(obj.get("previous_action_type", "")).lower()
        feats.append(1.0 if action_type == prev_action and action_type else 0.0)
        feats.append(1.0 if action_type != prev_action and prev_action else 0.0)
        feats.append(0.0)  # action diversity (computed from history, not in single export)
        feats.append(float(obj.get("score_delta", 0.0)))

        # Validation event features (4 dims)
        feats.append(min(float(obj.get("validation_success_count", 0)) / 3.0, 1.0))
        feats.append(min(float(obj.get("validation_fail_count", 0)) / 3.0, 1.0))
        total_val = float(obj.get("validation_success_count", 0)) + float(obj.get("validation_fail_count", 0))
        feats.append(min(total_val / 5.0, 1.0))
        feats.append(float(obj.get("validation_success_count", 0)) / max(total_val, 1))

        # History features (8 dims)
        feats.append(min(float(obj.get("step_index", 0)) / 20.0, 1.0))
        feats.append(min(float(obj.get("unique_urls_visited", 1)) / 10.0, 1.0))
        feats.append(0.0)  # action_type_diversity (not in export)
        feats.append(min(float(obj.get("loop_count", 0)) / 5.0, 1.0))
        feats.append(0.0)  # progress_ratio (not in export)
        feats.append(0.0)  # cumulative_reward (not in export)
        feats.append(0.0)  # error_count (not in export)
        feats.append(0.0)  # consec_no_progress (not in export)

        # Completion signal features (4 dims) — numeric only
        feats.append(0.0)  # page_task_overlap (not in export)
        feats.append(0.0)  # page_text_delta (not in export)
        feats.append(min(abs(float(obj.get("score_delta", 0.0))), 1.0))
        feats.append(1.0 if obj.get("made_progress", False) else 0.0)

        while len(feats) < VERIFIER_FEATURE_DIM:
            feats.append(0.0)
        return feats[:VERIFIER_FEATURE_DIM]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            self.labels[idx],
        )


def compute_class_weights(dataset: VerifierClassificationDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    counts = [0] * NUM_CLASSES
    for label in dataset.labels:
        counts[label] += 1
    total = sum(counts)
    weights = []
    for c in counts:
        weights.append(total / (NUM_CLASSES * max(c, 1)))
    return torch.tensor(weights, dtype=torch.float32)


def train(
    data_path: str,
    output_path: str,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 96,
    val_split: float = 0.15,
    patience: int = 15,
) -> Dict[str, Any]:
    """Train the verifier classifier.

    Returns training metrics.
    """
    dataset = VerifierClassificationDataset(data_path)
    if len(dataset) == 0:
        return {"error": "No training data found", "examples": 0}

    # Split
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VerifierClassifier(
        input_dim=VERIFIER_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_classes=NUM_CLASSES,
    ).to(device)

    class_weights = compute_class_weights(dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    metrics: Dict[str, Any] = {"epochs": 0, "examples": len(dataset)}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.long).to(device) if not isinstance(y_batch, torch.Tensor) else y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y_batch).sum().item()
            train_total += x_batch.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = torch.tensor(y_batch, dtype=torch.long).to(device) if not isinstance(y_batch, torch.Tensor) else y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_batch).sum().item()
                val_total += x_batch.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train loss: {avg_train_loss:.4f} acc: {train_acc:.3f} | "
                f"Val loss: {avg_val_loss:.4f} acc: {val_acc:.3f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        metrics["epochs"] = epoch + 1

    if best_state is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(best_state, output_path)
        print(f"Saved model to {output_path}")

    metrics.update({
        "best_val_loss": best_val_loss,
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "device": device,
        "num_classes": NUM_CLASSES,
        "labels": VERIFIER_LABELS,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned progress verifier")
    parser.add_argument("--data", required=True, help="Path to verifier examples JSONL")
    parser.add_argument("--output", default="models/verifier/model.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    metrics = train(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
    )
    print(f"Training complete: {json.dumps(metrics)}")


# Alias expected by CHECK.sh
train_verifier = train


if __name__ == "__main__":
    main()
