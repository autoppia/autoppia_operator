from training.rl.splits import build_seed_splits


def test_build_seed_splits_is_reproducible() -> None:
    a = build_seed_splits(start=1, end=20, random_seed=7)
    b = build_seed_splits(start=1, end=20, random_seed=7)
    assert a.to_dict() == b.to_dict()
    assert set(a.train).isdisjoint(a.dev)
    assert set(a.train).isdisjoint(a.test)
    assert set(a.dev).isdisjoint(a.test)
