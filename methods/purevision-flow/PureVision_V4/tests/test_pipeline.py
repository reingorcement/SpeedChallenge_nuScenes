import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from purevision_v4 import pipeline
from purevision_v4.flow import atomic_torch_save
from purevision_v4.io import ROOT, read_json, write_json


class NoSupportFlow:
    def __init__(self, *args, **kwargs):
        pass

    def pair(self, source, target, path):
        return dict(flow=torch.zeros(2, 288, 512), confidence=torch.zeros(1, 288, 512),
                    source_id=source["image_id"], target_id=target["image_id"], dt=target["time_s"] - source["time_s"])


def test_pipeline_invalid_outputs_resume_and_content_guard(work, monkeypatch):
    raw = work / "raw"
    raw.mkdir()
    cal = read_json(ROOT / "tests/calibration_legacy.json")
    cal["image_size"] = [80, 60]
    cal["K"] = [[40, 0, 40], [0, 40, 30], [0, 0, 1]]
    write_json(raw / "calibration.json", cal)
    frames = []
    for i in range(7):
        array = np.repeat(np.tile(np.arange(80, dtype=np.uint8), (60, 1))[..., None], 3, axis=2)
        Image.fromarray(array).save(raw / f"{i}.png")
        frames.append(dict(file=f"{i}.png", image_id=i + 100, time_s=i * .05, truth_vx=999))
    write_json(raw / "frames.json", frames)
    monkeypatch.setattr(pipeline, "RaftFlow", NoSupportFlow)
    output = work / "out"
    result = pipeline.run(raw, output, device="cpu", save_features=True, threads=2)
    assert result["coverage"]["F_RC"]["coverage"] == 0
    table = pd.read_csv(output / "speeds.csv")
    assert not table.valid.any() and table.vx_mps.isna().all()
    assert table[table.output_index >= 4].reason.eq("all_pairs_invalid").all()
    assert table.inference_ms.dropna().min() >= 0
    assert result == pipeline.run(raw, output, device="cpu", save_features=True, resume=True, threads=2)
    with pytest.raises(ValueError, match="not empty"):
        pipeline.run(raw, output, device="cpu", save_features=True, threads=2)
    Image.new("RGB", (80, 60), "red").save(raw / "0.png")
    with pytest.raises(ValueError, match="mismatch"):
        pipeline.run(raw, output, device="cpu", save_features=True, resume=True, threads=2)


def test_trained_heads_connect_to_rgb_pipeline(work, monkeypatch):
    from purevision_v4.io import calibration, target_spec
    from purevision_v4.training import train
    monkeypatch.setattr(pipeline, "RaftFlow", NoSupportFlow)
    records = []
    inputs = []
    for name, color in [("train_sequence", "black"), ("dev_sequence", "red")]:
        raw = work / name
        raw.mkdir()
        cal = read_json(ROOT / "tests/calibration_legacy.json")
        cal["image_size"] = [80, 60]
        cal["K"] = [[40, 0, 40], [0, 40, 30], [0, 0, 1]]
        write_json(raw / "calibration.json", cal)
        frames = []
        for i in range(6):
            array = np.repeat(np.tile(np.arange(80, dtype=np.uint8), (60, 1))[..., None], 3, axis=2)
            if color == "red":
                array[..., 0] = 200
            Image.fromarray(array).save(raw / f"{i}.png")
            frames.append(dict(file=f"{i}.png", image_id=i, time_s=i * .05))
        write_json(raw / "frames.json", frames)
        out = work / (name + "_features")
        pipeline.run(raw, out, device="cpu", save_features=True, threads=2)
        labels = work / (name + ".csv")
        # Synthetic functional labels; this test establishes wiring, not accuracy.
        pd.DataFrame(dict(time=np.arange(6) * .05, vx=[1.] * 6, vy=[0.] * 6)).to_csv(labels, index=False)
        write_json(labels.with_suffix(".meta.json"), target_spec(calibration(cal)))
        records.append(dict(features=str(out / "features/manifest.json"), labels=str(labels)))
        inputs.append(raw)
    manifest = work / "training_manifest.json"
    write_json(manifest, dict(schema="training_manifest_v1", train=[records[0]], dev=[records[1]]))
    best = train(manifest, work / "train_result", epochs=1, batch_size=2, threads=2)
    pipeline.run(inputs[1], work / "fusion_result", methods=["fusion", "rgb_only"], fusion_checkpoint=best, device="cpu", threads=2)
    table = pd.read_csv(work / "fusion_result/speeds.csv").query("output_index >= 4")
    assert table.valid.all() and not table.geometry_valid.any()
    fusion, direct = [table[table.method == m][["vx_mps", "vy_mps"]].to_numpy() for m in ["fusion", "rgb_only"]]
    np.testing.assert_array_equal(fusion, direct)
