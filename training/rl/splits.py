from __future__ import annotations

import random
from dataclasses import dataclass, asdict


@dataclass
class SeedSplits:
    train: list[int]
    dev: list[int]
    test: list[int]

    def to_dict(self) -> dict[str, list[int]]:
        return asdict(self)


def build_seed_splits(
    *,
    start: int = 1,
    end: int = 1000,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    random_seed: int = 42,
) -> SeedSplits:
    if end < start:
        raise ValueError("end must be >= start")
    universe = list(range(int(start), int(end) + 1))
    rng = random.Random(int(random_seed))
    rng.shuffle(universe)
    total = len(universe)
    train_count = max(1, int(total * float(train_ratio)))
    dev_count = max(1, int(total * float(dev_ratio)))
    if train_count + dev_count >= total:
        dev_count = max(1, min(dev_count, total - train_count - 1))
    train = sorted(universe[:train_count])
    dev = sorted(universe[train_count : train_count + dev_count])
    test = sorted(universe[train_count + dev_count :])
    if not test:
        test = sorted(dev[-1:])
        dev = dev[:-1]
    return SeedSplits(train=train, dev=dev, test=test)
