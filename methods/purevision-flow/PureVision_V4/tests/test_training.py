import json

import numpy as np
import pandas as pd
import pytest
import torch

from purevision_v4.flow import atomic_torch_save
from purevision_v4.io import ROOT, sha256, target_spec, write_json
from purevision_v4.model import FEATURE_NAMES, load_fusion
from purevision_v4.training import FeatureDataset, train


def make_partition(root, name, cal, contract, seed):
    rng = torch.Generator().manual_seed(seed)
    directory = root / name
    windows = []
    for i in range(5):
        path = directory / "features" / f"{i}.pt"
        atomic_torch_save(dict(descriptor=torch.randn(384, generator=rng), diagnostic=torch.zeros(len(FEATURE_NAMES)),
                              geometry_velocity=torch.tensor([8., .2]), geometry_valid=i % 2 == 0, image_id=i, time_s=i * .05), path)
        windows.append(dict(file=path.name, image_id=i, time_s=i * .05, sha256=sha256(path)))
    feature_file = directory / "features/manifest.json"
    write_json(feature_file, dict(schema="frozen_features_v1", case=name, sequence_hash=name,
                                source_hashes=[name + "image"], target_spec=target_spec(cal), feature_contract=contract, windows=windows))
    label_file = directory / "labels.csv"
    pd.DataFrame(dict(time=np.arange(5) * .05, vx=[8.] * 5, vy=[.2] * 5)).to_csv(label_file, index=False)
    write_json(label_file.with_suffix(".meta.json"), target_spec(cal))
    return dict(features=str(feature_file), labels=str(label_file))


def test_train_save_reload_and_resume_without_geometry_gradients(work, cal):
    contract = dict(visual_weights=sha256(ROOT / "weights/v3_fusion_seed42.pth"), schema="synthetic_test_only")
    a = make_partition(work, "train_case", cal, contract, 1)
    b = make_partition(work, "dev_case", cal, contract, 2)
    manifest = work / "manifest.json"
    write_json(manifest, dict(schema="training_manifest_v1", train=[a], dev=[b]))
    direct = train(manifest, work / "direct", stage="direct", epochs=1, batch_size=3, threads=2)
    best = train(manifest, work / "trained", epochs=2, batch_size=3, threads=2, init=direct)
    head, package = load_fusion(best, contract, target_spec(cal))
    assert package["trained"] and package["stage"] == "fusion"
    assert all(np.isfinite(row["dev"]["epe"]) for row in package["history"])
    data = FeatureDataset([a], work)
    item = data[0]
    with torch.inference_mode():
        output = head(item["descriptor"][None], item["diagnostic"][None], item["geometry_velocity"][None], torch.tensor([True]))
    assert torch.isfinite(output["prediction"]).all()
    resumed = train(manifest, work / "resumed", epochs=3, batch_size=3, threads=2, resume=work / "trained/last.pth")
    assert resumed.is_file()
    last = torch.load(work / "resumed/last.pth", weights_only=True)
    assert last["epoch"] == 2


def test_training_rejects_overlapping_sequences(work, cal):
    contract = dict(visual_weights=sha256(ROOT / "weights/v3_fusion_seed42.pth"))
    a = make_partition(work, "same_case", cal, contract, 1)
    manifest = work / "manifest.json"
    write_json(manifest, dict(schema="training_manifest_v1", train=[a], dev=[a]))
    with pytest.raises(ValueError, match="overlap"):
        train(manifest, work / "bad")


def test_training_rejects_changed_or_legacy_labels(work, cal):
    record = make_partition(work, "case", cal, {}, 1)
    label_file = __import__("pathlib").Path(record["labels"])
    write_json(label_file.with_suffix(".meta.json"), dict(target_spec(cal), version="legacy_yaw"))
    with pytest.raises(ValueError, match="legacy yaw"):
        FeatureDataset([record], work)
