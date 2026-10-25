from collections import Counter

import pytest
from PIL import Image

from train import create_dataloaders


def make_dataset(root):
    for split, counts in {
        "train": {"few": 2, "many": 5},
        "val": {"few": 1, "many": 1},
    }.items():
        for class_name, count in counts.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for index in range(count):
                Image.new("RGB", (2, 2)).save(class_dir / f"{index}.png")


def test_fixed_count_resamples_each_class_to_target_each_epoch(tmp_path):
    """Catches a cap-only implementation that never repeats minority images."""
    make_dataset(tmp_path)
    config = {"dataset": {"resample": {
        "enabled": True, "method": "fixed_count", "target": 3, "per_class": True,
    }}}
    train_loader, val_loader, classes = create_dataloaders(
        str(tmp_path), None, None, batch_size=2, num_workers=0,
        config=config, seed=42,
    )

    first = list(iter(train_loader.sampler))
    second = list(iter(train_loader.sampler))
    labels = [label for _, label in train_loader.dataset.samples]
    first_counts = Counter(labels[index] for index in first)
    second_counts = Counter(labels[index] for index in second)

    assert classes == ["few", "many"]
    assert first_counts == second_counts == {0: 3, 1: 3}
    assert len(first) == len(second) == 6
    assert {0, 1}.issubset(first)
    assert len([index for index in first if labels[index] == 1]) == len(
        {index for index in first if labels[index] == 1}
    )
    assert first != second
    assert len(val_loader.dataset) == 2


def test_fixed_count_global_sampling_fills_target_and_rejects_zero(tmp_path):
    """Catches global fixed_count silently capping at the source size."""
    make_dataset(tmp_path)
    config = {"dataset": {"resample": {
        "enabled": True, "method": "fixed_count", "target": 9, "per_class": False,
    }}}
    train_loader, _, _ = create_dataloaders(
        str(tmp_path), None, None, batch_size=2, num_workers=0,
        config=config, seed=42,
    )
    indices = list(iter(train_loader.sampler))
    assert len(indices) == 9
    assert set(indices) == set(range(7))

    config["dataset"]["resample"]["target"] = 0
    with pytest.raises(ValueError, match="target"):
        create_dataloaders(
            str(tmp_path), None, None, batch_size=2, num_workers=0,
            config=config, seed=42,
        )
