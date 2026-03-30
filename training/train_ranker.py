"""Training script for the learned candidate ranker.

Self-contained — can run on a fresh GPU pod with torch, numpy installed.

Usage:
    python -m training.train_ranker --data data/ranking_pairs.jsonl --output models/ranker/model.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# Add project root to path for standalone execution on GPU pod
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ranker.features import RANKER_FEATURE_DIM
from ranker.learned import PairwiseMarginLoss, RankerMLP


class PairwiseRankingDataset(Dataset):
    """PyTorch dataset for pairwise ranking training."""

    def __init__(self, path: str) -> None:
        self.winners: List[List[float]] = []
        self.losers: List[List[float]] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                w_feats = self._features_to_vec(obj.get("winner_features", {}))
                l_feats = self._features_to_vec(obj.get("loser_features", {}))
                self.winners.append(w_feats)
                self.losers.append(l_feats)

    def _features_to_vec(self, features: Dict[str, Any]) -> List[float]:
        """Convert feature dict to fixed-size vector."""
        if isinstance(features, list):
            vec = [float(x) for x in features]
        elif isinstance(features, dict):
            vec = [float(v) for v in features.values()]
        else:
            vec = []
        # Pad or truncate
        while len(vec) < RANKER_FEATURE_DIM:
            vec.append(0.0)
        return vec[:RANKER_FEATURE_DIM]

    def __len__(self) -> int:
        return len(self.winners)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.winners[idx], dtype=torch.float32),
            torch.tensor(self.losers[idx], dtype=torch.float32),
        )


def train(
    data_path: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    margin: float = 1.0,
    val_split: float = 0.15,
    patience: int = 10,
) -> Dict[str, Any]:
    """Train the ranker MLP on pairwise ranking data.

    Returns training metrics.
    """
    dataset = PairwiseRankingDataset(data_path)
    if len(dataset) == 0:
        return {"error": "No training data found", "pairs": 0}

    # Split train/val
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model, loss, optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RankerMLP(input_dim=RANKER_FEATURE_DIM, hidden_dim=hidden_dim).to(device)
    criterion = PairwiseMarginLoss(margin=margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    metrics: Dict[str, Any] = {"epochs": 0, "pairs": len(dataset)}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for w_batch, l_batch in train_loader:
            w_batch, l_batch = w_batch.to(device), l_batch.to(device)
            w_scores = model(w_batch)
            l_scores = model(l_batch)
            loss = criterion(w_scores, l_scores)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * w_batch.size(0)
            train_correct += (w_scores > l_scores).sum().item()
            train_total += w_batch.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for w_batch, l_batch in val_loader:
                w_batch, l_batch = w_batch.to(device), l_batch.to(device)
                w_scores = model(w_batch)
                l_scores = model(l_batch)
                loss = criterion(w_scores, l_scores)
                val_loss += loss.item() * w_batch.size(0)
                val_correct += (w_scores > l_scores).sum().item()
                val_total += w_batch.size(0)

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

        # Early stopping
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

    # Save best model
    if best_state is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(best_state, output_path)
        print(f"Saved model to {output_path}")

    metrics.update({
        "best_val_loss": best_val_loss,
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "device": device,
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train learned candidate ranker")
    parser.add_argument("--data", required=True, help="Path to ranking pairs JSONL")
    parser.add_argument("--output", default="models/ranker/model.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    metrics = train(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        margin=args.margin,
    )
    print(f"Training complete: {json.dumps(metrics)}")


# Alias expected by CHECK.sh
train_ranker = train


if __name__ == "__main__":
    main()
