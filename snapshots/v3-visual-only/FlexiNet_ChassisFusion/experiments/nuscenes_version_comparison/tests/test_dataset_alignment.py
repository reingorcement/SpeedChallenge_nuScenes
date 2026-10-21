from __future__ import annotations

import json

import pytest

from dataset.v2_splits import create_split_manifest
from experiments.nuscenes_version_comparison.adapters.dataset_adapter import (
    assert_dataset_alignment,
    dataset_identities,
    select_partition_scenes,
)


class FakeDataset:
    def __init__(self, samples):
        self.samples = samples


def _sample(scene: str, token: str, timestamp: int):
    return {"scene_name": scene, "center_token": token, "center_timestamp": timestamp}


def test_dataset_identity_alignment_is_order_strict() -> None:
    left = FakeDataset([_sample("scene-a", "a", 1), _sample("scene-b", "b", 2)])
    right = FakeDataset([_sample("scene-a", "a", 1), _sample("scene-b", "b", 2)])
    assert_dataset_alignment(left, right)
    assert dataset_identities(left) == [("scene-a", "a", 1), ("scene-b", "b", 2)]

    reversed_dataset = FakeDataset(list(reversed(right.samples)))
    with pytest.raises(ValueError, match="do not align"):
        assert_dataset_alignment(left, reversed_dataset)


def test_domain_selection_and_sealed_partition(tmp_path) -> None:
    train = ["scene-train-a", "scene-train-b", "scene-train-c", "scene-train-d"]
    val = ["scene-val-a", "scene-val-b", "scene-val-c", "scene-val-d"]
    metadata = {
        name: {
            "location": "boston-seaport" if name.endswith(("a", "c")) else "singapore-onenorth",
            "lighting": "day",
            "weather": "dry_or_unknown",
            "motion_regime": "turning",
        }
        for name in train + val
    }
    manifest = create_split_manifest(train, val, metadata, seed=42)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    selected = select_partition_scenes(
        path, "train_core", version="v1.0-trainval", domain="boston"
    )
    assert selected
    assert all(metadata[name]["location"].startswith("boston") for name in selected)
    with pytest.raises(PermissionError, match="sealed"):
        select_partition_scenes(
            path, "final_system_test", version="v1.0-trainval", domain="all"
        )
