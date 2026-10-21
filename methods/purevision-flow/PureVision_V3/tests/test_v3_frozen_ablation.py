from pathlib import Path
import json
import sys

import pytest
import torch

from model.v3 import V3VisualOnlyVelocityModel, RobustVisualTwistSolver
from scripts import run_v3_ablation as runner
from scripts.summarize_v3_ablation import summarize
from test_v3_visual_only import visual_batch, synthetic_geometry
from utils.sideslip import sideslip_values


def frozen_model() -> V3VisualOnlyVelocityModel:
    model = V3VisualOnlyVelocityModel()
    model.requires_grad_(False)
    return model.eval()


def test_same_checkpoint_predictions_and_buffers_unchanged() -> None:
    model = frozen_model()
    visual = visual_batch()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        expected = model(visual)
        full, _ = runner.predict_mode(model, visual, "fusion")
        rgb, _ = runner.predict_mode(model, {"images": visual["images"]}, "rgb_only")
    torch.testing.assert_close(full, expected.predicted_velocity, atol=0, rtol=0)
    torch.testing.assert_close(rgb, expected.direct_velocity, atol=0, rtol=0)
    runner.predict_mode(model, visual, "flow_geometry_only")
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert not full.requires_grad and not rgb.requires_grad


def test_flow_geometry_never_calls_rgb_or_learned_heads(monkeypatch) -> None:
    model = frozen_model()
    model.geometry_solver = RobustVisualTwistSolver(min_points=32, max_condition_number=1e10)
    expected, flow, confidence, intrinsic, dt, normal, offset, _ = synthetic_geometry()
    visual = {
        "flows": flow[:, None].repeat(1, 3, 1, 1, 1),
        "flow_confidence": confidence[:, None].repeat(1, 3, 1, 1, 1),
        "intrinsics": intrinsic, "pair_dt": dt[:, None].repeat(1, 3),
        "plane_normal_camera": normal, "plane_offset": offset,
        "rotation_ego_from_camera": torch.eye(3)[None],
        "translation_ego_from_camera": torch.zeros(1, 3),
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("RGB or learned head was accessed")

    monkeypatch.setattr(model.backbone, "forward_feature_map", forbidden)
    for module in (model.direct_head, model.reliability_head, model.correction_head, model.gate_head):
        monkeypatch.setattr(module, "forward", forbidden)
    result, valid = runner.predict_mode(model, visual, "flow_geometry_only")
    assert valid.all()
    torch.testing.assert_close(result, expected[:, :2], atol=0.003, rtol=0.003)


def test_unfrozen_model_rejected() -> None:
    with pytest.raises(ValueError, match="frozen"):
        runner.predict_mode(V3VisualOnlyVelocityModel(), visual_batch(), "fusion")


def test_sideslip_wrap_and_zero_speed() -> None:
    angles = torch.deg2rad(torch.tensor([179.0, -179.0]))
    velocity = 5 * torch.stack((angles.cos(), angles.sin()), dim=1)
    pred = torch.cat((velocity.flip(0), torch.zeros(2, 2)))
    target = torch.cat((velocity, torch.tensor([[0.0, 0.0], [5.0, 0.0]])))
    beta = sideslip_values(pred, target)
    torch.testing.assert_close(beta["beta_error_deg"][:2].abs(), torch.tensor([2.0, 2.0]), atol=1e-4, rtol=0)
    assert beta["beta_eval_valid"].tolist() == [True, True, False, True]
    assert torch.isnan(beta["pred_beta_deg"][2:]).all()
    assert beta["beta_error_deg"][3] == 180


def test_test_only_cli_loads_checkpoint_exports_and_preserves_file(tmp_path: Path, monkeypatch) -> None:
    config = runner.load_config(runner.ROOT / "configs/velocity_xy_v3_visual_only.yaml")
    checkpoint = tmp_path / "existing.pth"
    torch.save({"model": frozen_model().state_dict(), "config": config}, checkpoint)
    before = checkpoint.read_bytes()
    for directory in ("precomputed_flow_v2/CAM_FRONT", "precomputed_frames/CAM_FRONT_72x128_rgb"):
        (tmp_path / directory).mkdir(parents=True)
    manifest = tmp_path / "split.json"
    manifest.write_text("{}", encoding="utf-8")
    visual = visual_batch()
    dataset = [{
        "visual": {key: value[0] for key, value in visual.items()},
        "target": {"velocity": torch.tensor([5.0, 0.2])},
        "center_token": "sample", "scene_name": "scene", "center_timestamp": 123,
    }]
    monkeypatch.setattr(runner, "build_dataset", lambda cfg, split: dataset)
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "test", "--checkpoint", str(checkpoint), "--output_dir", str(output),
        "--dataroot", str(tmp_path), "--split_manifest", str(manifest),
        "--device", "cpu", "--max_batches", "1",
    ])
    runner.main()
    assert checkpoint.read_bytes() == before
    assert (output / "comparison.csv").is_file()
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["test_only"] and protocol["partial_test"]
    assert not (output / "paired_velocity_bootstrap.json").exists()
    path = output / "rgb_only/predictions.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("sample,", "different,"), encoding="utf-8")
    with pytest.raises(ValueError, match="token sets differ"):
        summarize(output)


def test_checkpoint_stage_mismatch_rejected(tmp_path: Path) -> None:
    config = runner.load_config(runner.ROOT / "configs/velocity_xy_v3_visual_only.yaml")
    config["model"]["mode"] = "direct_pretrain"
    path = tmp_path / "direct.pth"
    torch.save({"config": config, "model": frozen_model().state_dict()}, path)
    with pytest.raises(ValueError, match="Expected fusion"):
        runner.load_existing_model(path)
