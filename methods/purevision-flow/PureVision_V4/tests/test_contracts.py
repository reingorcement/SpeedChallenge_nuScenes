import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from purevision_v4.io import ROOT, artifact_path, calibration, intrinsics, load_sequence, read_json, target_spec
from purevision_v4.model import FEATURE_NAMES, FusionHead, load_fusion
from purevision_v4.normals import NormalTracker, image_texture_std, road_mask, select_normal
from purevision_v4.nuscenes_export import body_velocity_labels
from purevision_v4.evaluation import join_labels, metrics, macro_metrics


def test_normal_hold_expiry_and_pair_routing(cal, cfg):
    tracker = NormalTracker(intrinsics(cal), cal["plane_normal_camera"], cfg["normal"])
    tracker.state.ever_valid = True
    tracker.last_update = 0.
    flow, confidence = np.zeros((2, 288, 512)), np.zeros((288, 512))
    first = tracker.step(flow, confidence, 100, 101, 0., .05)
    assert first["status"] == "hold"
    with pytest.raises(ValueError, match="Source-frame"):
        select_normal("F_RC", cal["plane_normal_camera"], first, 99)
    with pytest.raises(ValueError, match="contiguous"):
        tracker.step(flow, confidence, 103, 104, .15, .2)
    record = tracker.step(flow, confidence, 101, 102, .05, .5)
    assert record["status"] == "fixed_fallback" and record["reason"] == "time_gap"


def test_visual_normal_updates_from_planar_correspondences(cal, cfg):
    tracker = NormalTracker(intrinsics(cal), cal["plane_normal_camera"], cfg["normal"])
    k = intrinsics(cal)
    nominal = np.asarray(cal["plane_normal_camera"])
    true_normal = Rotation.from_rotvec([0., 0., .01]).apply(nominal)
    y, x = np.mgrid[:288, :512]
    p = np.stack([x, y, np.ones_like(x)], -1)
    rays = p @ np.linalg.inv(k).T
    den = rays @ true_normal
    depth = -cal["height_m"] / np.where(abs(den) > 1e-8, den, 1e-8)
    points = rays * depth[..., None] - [0., 0., .4]
    projected = points @ k.T
    q = projected[..., :2] / np.where(abs(projected[..., 2:]) > 1e-8, projected[..., 2:], 1e-8)
    flow = (q - p[..., :2]).transpose(2, 0, 1)
    confidence = ((depth > 2) & (depth < 40) & (y > 150) & (q[..., 0] > 0) & (q[..., 0] < 511) & (q[..., 1] > 0) & (q[..., 1] < 287)).astype(float)
    records = [tracker.step(flow, confidence, i, i + 1, i * .05, (i + 1) * .05) for i in range(12)]
    assert any(r["status"] == "update" for r in records)
    assert records[-1]["normal_frame"] == "source_camera"
    assert np.linalg.norm(np.asarray(records[-1]["normal"]) - true_normal) < .002


def test_pose_labels_use_full_rotation_and_vertical_velocity():
    times = np.array([0., .08, .2, .35, .5])
    rotation = Rotation.from_euler("xyz", [20, 15, 30], degrees=True).as_matrix()
    body = np.array([8., .6, 1.2])
    positions = times[:, None] * (rotation @ body)
    actual = body_velocity_labels(times, positions, np.tile(rotation, (len(times), 1, 1)))
    np.testing.assert_allclose(actual, np.tile(body, (len(times), 1)), atol=1e-10)


def test_visual_inputs_are_allowlisted_and_paths_checked(work):
    raw = work / "input"
    raw.mkdir()
    cal = read_json(ROOT / "tests/calibration_legacy.json")
    cal["image_size"] = [80, 60]
    cal["K"] = [[40, 0, 40], [0, 40, 30], [0, 0, 1]]
    cal["dynamic_roll_truth"] = 999
    (raw / "calibration.json").write_text(json.dumps(cal))
    frames = []
    for i in range(5):
        Image.new("RGB", (80, 60)).save(raw / f"{i}.png")
        frames.append(dict(image_id=i, file=f"{i}.png", solver_time=i * .05, truth_vx=999))
    (raw / "frames.json").write_text(json.dumps(frames))
    actual, calibration = load_sequence(raw)
    assert "truth_vx" not in actual[0] and "dynamic_roll_truth" not in calibration
    frames[0]["file"] = "../outside.png"
    (raw / "frames.json").write_text(json.dumps(frames))
    with pytest.raises(ValueError, match="escapes"):
        load_sequence(raw)


def test_writes_cannot_touch_original_projects():
    for path in [ROOT.parent / "PureVision_V3/file", ROOT.parent / "补偿速度估计/file", ROOT / "weights/new.pth", ROOT / "purevision_v4/file"]:
        with pytest.raises(ValueError, match="Outputs"):
            artifact_path(path)


def test_fusion_fallback_and_finite_batch_gradients():
    torch.manual_seed(7)
    model = FusionHead()
    descriptor = torch.randn(4, 384)
    diagnostic = torch.randn(4, len(FEATURE_NAMES))
    geometry = torch.tensor([[3., .2], [float("nan"), float("nan")], [8., -.3], [2., .1]])
    valid = torch.tensor([True, False, True, True])
    output = model(descriptor, diagnostic, geometry, valid)
    torch.testing.assert_close(output["prediction"][1], output["direct"][1], rtol=0, atol=0)
    loss = output["prediction"].square().mean() + output["log_variance"].square().mean()
    loss.backward()
    for group in [model.direct, model.correction, model.gate, model.uncertainty]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters())


def test_legacy_weights_rejected_as_v4_fusion():
    with pytest.raises(ValueError, match="trained V4"):
        load_fusion(ROOT / "weights/v3_fusion_seed42.pth", {}, {})


def test_evaluation_keeps_failed_cases_and_rejects_wrong_coordinates(cal):
    rows = []
    for method in ["F_RC", "F_R"]:
        for i in range(3):
            good = method == "F_R"
            rows.append(dict(method=method, image_id=i, time_s=i * .05, valid=good, reason="ok" if good else "all_pairs_invalid",
                             vx_mps=10.1 if good else np.nan, vy_mps=.3 if good else np.nan,
                             target_version="body_reference_v1", reference_point=cal["reference_point"]))
    labels = pd.DataFrame(dict(time=[0., .05, .1], vx=[10.] * 3, vy=[.2] * 3, roll=[.03] * 3))
    predictions = pd.DataFrame(rows)
    table = metrics(join_labels(predictions, labels, target_spec(cal)))
    macro = macro_metrics(table.assign(case="failure_case"))
    failed = macro.query("method == 'F_RC' and subset == 'all' and population == 'own_valid'")
    assert (failed.min_coverage == 0).all() and (failed.failed_cases == 1).all()
    assert (table.query("population == 'common_valid'").valid == 0).all()
    predictions["target_version"] = "legacy_v3_yaw_level_v0"
    with pytest.raises(ValueError, match="coordinate"):
        join_labels(predictions, labels, target_spec(cal))


def test_package_does_not_import_generic_model_namespace():
    import purevision_v4.pipeline
    assert "model.v3.model" not in sys.modules
    assert "roll_velocity" not in sys.modules


@pytest.mark.parametrize("color", [[0, 0, 0], [255, 255, 255], [255, 0, 0]])
def test_uniform_images_cannot_supply_geometric_motion(cal, cfg, color):
    rgb = np.tile(np.array(color, np.uint8), (288, 512, 1))
    assert image_texture_std(rgb) == 0
    assert not road_mask(rgb, intrinsics(cal), np.array(cal["plane_normal_camera"]), cfg["road"]).any()
