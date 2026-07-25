from __future__ import annotations

from pathlib import Path
import csv

import pytest
import torch
from torch.utils.data import Dataset

from dataset.v3_schema import FORBIDDEN_VISUAL_KEYS, V3NuScenesVisualDataset
from model.v2.ground_geometry import road_plane_depth
from model.v3 import (
    RobustVisualTwistSolver,
    V3VelocityLoss,
    V3VisualOnlyVelocityModel,
    motion_flow_from_twist,
    transform_camera_twist_to_ego,
)
from scripts.evaluate_v3_paired_bootstrap import bootstrap, paired_scene_deltas
from scripts.analyze_v3_results import analyze_results
from train_v3 import clip_gradient_norm, optimizer_state_matches


def synthetic_geometry(
    twist: torch.Tensor | None = None,
    height_m: float = 1.5,
    height: int = 48,
    width: int = 64,
):
    intrinsics = torch.tensor(
        [[[80.0, 0.0, (width - 1) / 2], [0.0, 80.0, (height - 1) / 2], [0.0, 0.0, 1.0]]]
    )
    normal = torch.tensor([[0.0, -1.0, 0.0]])
    offset = torch.tensor([height_m])
    dt = torch.tensor([0.1])
    if twist is None:
        twist = torch.tensor([[2.0, 0.2, 0.1, 0.03, -0.02, 0.05]])
    depth, valid = road_plane_depth(intrinsics, normal, offset, height, width)
    flow = motion_flow_from_twist(twist, intrinsics, depth, dt)
    confidence = valid[:, None].float()
    return twist, flow, confidence, intrinsics, dt, normal, offset, valid[:, None]


def solve_geometry(flow, confidence, intrinsics, dt, normal, offset, mask, **kwargs):
    return RobustVisualTwistSolver(
        min_points=32, max_condition_number=1.0e10, **kwargs
    )(flow, confidence, intrinsics, dt, normal, offset, support_mask=mask)


def test_synthetic_full_twist_recovery_and_rotation_sign() -> None:
    expected, flow, confidence, intrinsics, dt, normal, offset, mask = (
        synthetic_geometry()
    )
    estimate = solve_geometry(flow, confidence, intrinsics, dt, normal, offset, mask)
    assert bool(estimate.valid.all())
    torch.testing.assert_close(
        estimate.twist_camera, expected, atol=2.0e-3, rtol=2.0e-3
    )
    assert torch.sign(estimate.twist_camera[0, 5]) == torch.sign(expected[0, 5])


def test_solver_is_robust_to_sparse_flow_outliers() -> None:
    expected, flow, confidence, intrinsics, dt, normal, offset, mask = (
        synthetic_geometry()
    )
    corrupted = flow.clone()
    corrupted[:, :, 30:34, 20:25] += 30.0
    estimate = solve_geometry(
        corrupted, confidence, intrinsics, dt, normal, offset, mask
    )
    assert bool(estimate.valid.all())
    torch.testing.assert_close(
        estimate.twist_camera, expected, atol=3.0e-2, rtol=3.0e-2
    )


@pytest.mark.parametrize("bad_dt", [0.0, -0.1, float("nan")])
def test_solver_rejects_bad_dt_and_weak_support(bad_dt: float) -> None:
    _, flow, confidence, intrinsics, dt, normal, offset, mask = synthetic_geometry()
    dt.fill_(bad_dt)
    confidence.zero_()
    confidence[..., -2:, -2:] = 1.0
    estimate = solve_geometry(flow, confidence, intrinsics, dt, normal, offset, mask)
    assert not bool(estimate.valid.any())
    assert torch.equal(estimate.twist_camera, torch.zeros_like(estimate.twist_camera))


@pytest.mark.parametrize("scale", [0.9, 1.1])
def test_camera_height_scales_translation_but_not_rotation(scale: float) -> None:
    expected, flow, confidence, intrinsics, dt, normal, offset, mask = (
        synthetic_geometry()
    )
    estimate = solve_geometry(
        flow, confidence, intrinsics, dt, normal, offset * scale, mask
    )
    torch.testing.assert_close(
        estimate.twist_camera[:, :3], expected[:, :3] * scale, atol=2.0e-3, rtol=2.0e-3
    )
    torch.testing.assert_close(
        estimate.twist_camera[:, 3:], expected[:, 3:], atol=2.0e-3, rtol=2.0e-3
    )


def test_reference_transform_includes_lever_arm_and_covariance() -> None:
    twist = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
    covariance = torch.eye(6)[None]
    transformed, transformed_covariance = transform_camera_twist_to_ego(
        twist, covariance, torch.eye(3)[None], torch.tensor([[1.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(transformed[0, :3], torch.tensor([1.0, -2.0, 0.0]))
    assert transformed_covariance.shape == (1, 6, 6)
    assert torch.isfinite(transformed_covariance).all()


def visual_batch(
    batch_size: int = 1, confidence_value: float = 0.0
) -> dict[str, torch.Tensor]:
    height, width = 16, 24
    return {
        "images": torch.randn(batch_size, 5, 3, height, width),
        "flows": torch.zeros(batch_size, 3, 2, height, width),
        "flow_confidence": torch.full(
            (batch_size, 3, 1, height, width), confidence_value
        ),
        "pair_dt": torch.full((batch_size, 3), 0.1),
        "intrinsics": torch.tensor(
            [[[30.0, 0.0, 11.5], [0.0, 30.0, 7.5], [0.0, 0.0, 1.0]]]
        ).repeat(batch_size, 1, 1),
        "plane_normal_camera": torch.tensor([[0.0, -1.0, 0.0]]).repeat(batch_size, 1),
        "plane_offset": torch.full((batch_size,), 1.5),
        "rotation_ego_from_camera": torch.eye(3)[None].repeat(batch_size, 1, 1),
        "translation_ego_from_camera": torch.zeros(batch_size, 3),
    }


def test_all_invalid_geometry_is_exact_direct_fallback() -> None:
    model = V3VisualOnlyVelocityModel().eval()
    with torch.no_grad():
        output = model(visual_batch())
    assert not bool(output.geometry_valid.any())
    assert torch.equal(output.geometry_gate, torch.zeros_like(output.geometry_gate))
    assert torch.equal(output.predicted_velocity, output.direct_velocity)
    assert output.reliability.shape == (1, 3, 1, 16, 24)
    assert output.predicted_log_variance.shape == (1, 2)
    assert torch.isfinite(output.predicted_velocity).all()


def test_outer_can_fields_cannot_change_model_prediction() -> None:
    torch.manual_seed(2)
    model = V3VisualOnlyVelocityModel().eval()
    visual = visual_batch()
    batches = [
        {"visual": visual},
        {
            "visual": visual,
            "base_velocity": torch.randn(1, 2),
            "wheel_speed": torch.randn(1, 4),
        },
        {"visual": visual, "imu": torch.zeros(1, 6), "chassis": torch.randn(1, 10)},
    ]
    with torch.no_grad():
        predictions = [model(batch["visual"]).predicted_velocity for batch in batches]
    for prediction in predictions[1:]:
        torch.testing.assert_close(prediction, predictions[0], atol=1.0e-6, rtol=0.0)


def test_v3_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(3)
    model = V3VisualOnlyVelocityModel(mode="direct_pretrain").train()
    visual = {"images": visual_batch(batch_size=2)["images"]}
    output = model(visual)
    target = {
        "velocity": torch.tensor([[4.0, 0.2], [6.0, -0.1]]),
    }
    criterion = V3VelocityLoss()
    for name in (
        "geometry",
        "rotation",
        "reprojection",
        "residual",
        "support",
        "tv",
    ):
        criterion.weights[name] = 0.0
    terms = criterion(output, target, visual)
    assert torch.isfinite(terms.total)
    assert terms.geometry == 0.0
    assert terms.rotation == 0.0
    assert terms.reprojection == 0.0
    terms.total.backward()
    assert model.direct_head[-1].weight.grad is not None
    assert torch.isfinite(model.direct_head[-1].weight.grad).all()


def test_direct_amp_placeholders_remain_finite_in_fp16() -> None:
    model = V3VisualOnlyVelocityModel(mode="direct_pretrain").half().eval()
    descriptor = torch.randn(2, 384, dtype=torch.float16)
    flows = torch.zeros(2, 3, 2, 8, 12, dtype=torch.float16)
    with torch.no_grad():
        output = model._direct_only_output(descriptor, flows)
    diagnostics = (
        output.geometry_covariance,
        output.geometry_condition_number,
        output.pair_covariance_camera,
        output.pair_condition_number,
    )
    for diagnostic in diagnostics:
        assert diagnostic.dtype == torch.float32
        assert torch.isfinite(diagnostic).all()


def test_fusion_reliability_gate_and_uncertainty_have_finite_gradients() -> None:
    torch.manual_seed(4)
    model = V3VisualOnlyVelocityModel().train()
    visual = visual_batch(batch_size=2, confidence_value=1.0)
    twist = torch.tensor([[1.0, 0.1, 0.05, 0.01, -0.02, 0.03]]).repeat(2, 1)
    depth, _ = road_plane_depth(
        visual["intrinsics"],
        visual["plane_normal_camera"],
        visual["plane_offset"],
        visual["flows"].shape[-2],
        visual["flows"].shape[-1],
    )
    flow = motion_flow_from_twist(
        twist, visual["intrinsics"], depth, visual["pair_dt"][:, 0]
    )
    visual["flows"] = flow[:, None].repeat(1, 3, 1, 1, 1)
    output = model(visual)
    target = {
        "velocity": output.geometry_velocity.detach(),
        "angular_velocity_camera": twist[:, None, 3:].repeat(1, 3, 1),
    }
    terms = V3VelocityLoss()(output, target, visual)
    terms.total.backward()
    for parameter in (
        model.reliability_head.weight,
        model.gate_head[-1].weight,
        model.uncertainty_head[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_fusion_geometry_remains_float32_under_autocast() -> None:
    model = V3VisualOnlyVelocityModel().eval()
    visual = visual_batch(confidence_value=1.0)
    twist = torch.tensor([[1.0, 0.1, 0.05, 0.01, -0.02, 0.03]])
    depth, _ = road_plane_depth(
        visual["intrinsics"],
        visual["plane_normal_camera"],
        visual["plane_offset"],
        visual["flows"].shape[-2],
        visual["flows"].shape[-1],
    )
    flow = motion_flow_from_twist(
        twist, visual["intrinsics"], depth, visual["pair_dt"][:, 0]
    )
    visual["flows"] = flow[:, None].repeat(1, 3, 1, 1, 1)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(visual)

    diagnostics = (
        output.geometry_velocity,
        output.geometry_covariance,
        output.pair_twist_camera,
        output.pair_twist_ego,
        output.pair_covariance_camera,
        output.pair_covariance_ego,
        output.pair_condition_number,
    )
    for diagnostic in diagnostics:
        assert diagnostic.dtype == torch.float32
        assert torch.isfinite(diagnostic).all()


def _write_v3_predictions(path: Path, error: float, seed_offset: float) -> None:
    rows = []
    for scene_index, scene_name in enumerate(("scene-a", "scene-b")):
        for frame_index in range(2):
            target_vx = 3.0 + scene_index + frame_index
            target_vy = 0.2 * scene_index
            rows.append(
                {
                    "center_token": f"{scene_name}-{frame_index}",
                    "scene_name": scene_name,
                    "target_vx_mps": target_vx,
                    "target_vy_mps": target_vy,
                    "pred_vx_mps": target_vx + error + seed_offset,
                    "pred_vy_mps": target_vy + 0.25 * error,
                    "log_variance_vx": 0.0,
                    "log_variance_vy": 0.0,
                    "geometry_valid": 1.0,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_three_seed_analysis_writes_complete_bundle(tmp_path: Path) -> None:
    direct_paths = []
    fusion_paths = []
    for index, seed in enumerate((42, 3407, 2026)):
        direct = tmp_path / f"direct-{seed}.csv"
        fusion = tmp_path / f"fusion-{seed}.csv"
        _write_v3_predictions(direct, error=1.0, seed_offset=index * 0.02)
        _write_v3_predictions(fusion, error=0.5, seed_offset=index * 0.02)
        direct_paths.append(direct)
        fusion_paths.append(fusion)
    output_dir = tmp_path / "analysis"

    report = analyze_results(
        [42, 3407, 2026], direct_paths, fusion_paths, 200, 2026, output_dir
    )

    assert report["paired_scene_bootstrap"]["seed_count"] == 3
    assert report["paired_scene_bootstrap"]["metrics"]["epe_mps"][
        "significant_improvement"
    ]
    for relative_path in (
        "summary.json",
        "per-seed-metrics.csv",
        "analysis-report.md",
        "stats-appendix.md",
        "figure-catalog.md",
        "figures/figure-01-seed-comparison.png",
        "figures/figure-02-scene-epe-deltas.png",
    ):
        assert (output_dir / relative_path).is_file()


def test_optimizer_resume_state_detects_unfreeze_boundary() -> None:
    layer = torch.nn.Linear(2, 1)
    full = torch.optim.AdamW(layer.parameters())
    partial = torch.optim.AdamW([layer.weight])
    assert optimizer_state_matches(full, full.state_dict())
    assert not optimizer_state_matches(full, partial.state_dict())


def test_gradient_clipping_reports_nonfinite_norm_without_raising() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])
    norm = clip_gradient_norm([parameter], max_norm=2.0)
    assert not bool(torch.isfinite(norm))


class FakeNuScenes:
    def __init__(self, records: dict[tuple[str, str], dict]) -> None:
        self.records = records

    def get(self, table: str, token: str) -> dict:
        return self.records[(table, token)]


class FakeBaseDataset(Dataset):
    def __init__(self) -> None:
        self.online_mode = True
        self.can_bus_adapter = None
        self.seq_len = 5
        self.samples = [
            {
                "tokens": [f"frame-{index}" for index in range(5)],
                "center_token": "frame-4",
                "scene_name": "scene-1",
                "center_timestamp": 400_000,
            }
        ]
        records = {}
        for index in range(5):
            records[("sample_data", f"frame-{index}")] = {
                "timestamp": index * 100_000,
                "height": 16,
                "width": 24,
                "calibrated_sensor_token": "calibration",
                "ego_pose_token": f"pose-{index}",
            }
            records[("ego_pose", f"pose-{index}")] = {
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
        records[("calibrated_sensor", "calibration")] = {
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "camera_intrinsic": [[30.0, 0.0, 11.5], [0.0, 30.0, 7.5], [0.0, 0.0, 1.0]],
        }
        self.nusc = FakeNuScenes(records)

    def __len__(self) -> int:
        return 1

    def _frame_cache_enabled(self) -> bool:
        return False

    def _load_frame_tensor(self, token: str) -> torch.Tensor:
        return torch.full((3, 16, 24), float(token[-1]))

    def _compute_velocity_label(self, token: str) -> torch.Tensor:
        return torch.tensor([5.0, 0.2])


def write_flow_packages(root: Path, backend: str = "raft_large") -> None:
    for source in range(1, 4):
        torch.save(
            {
                "flow": torch.zeros(2, 16, 24),
                "confidence": torch.ones(1, 16, 24),
                "source_token": f"frame-{source}",
                "target_token": f"frame-{source + 1}",
                "backend": backend,
                "target_size": (16, 24),
            },
            root / f"frame-{source}.pt",
        )


def test_dataset_is_causal_visual_only_without_can(tmp_path: Path) -> None:
    write_flow_packages(tmp_path)
    dataset = V3NuScenesVisualDataset(
        FakeBaseDataset(), tmp_path, camera_height_m=1.5, geometry_size=(16, 24)
    )
    item = dataset[0]
    assert item["center_token"] == "frame-4"
    assert item["visual"]["images"].shape == (5, 3, 16, 24)
    assert item["visual"]["flows"].shape == (3, 2, 16, 24)
    assert not (
        FORBIDDEN_VISUAL_KEYS
        & (item.keys() | item["visual"].keys() | item["target"].keys())
    )
    assert float(item["visual"]["pair_dt"][-1]) > 0.0


def test_direct_dataset_does_not_require_or_load_flow(tmp_path: Path) -> None:
    dataset = V3NuScenesVisualDataset(
        FakeBaseDataset(),
        tmp_path / "missing-flow-root",
        camera_height_m=1.5,
        geometry_size=(16, 24),
        load_flows=False,
    )
    item = dataset[0]
    assert set(item["visual"]) == {"images"}
    assert set(item["target"]) == {"velocity"}


def test_dataset_rejects_flow_token_or_backend_mismatch(tmp_path: Path) -> None:
    write_flow_packages(tmp_path)
    package_path = tmp_path / "frame-3.pt"
    package = torch.load(package_path, map_location="cpu")
    package["target_token"] = "future-frame"
    torch.save(package, package_path)
    dataset = V3NuScenesVisualDataset(
        FakeBaseDataset(), tmp_path, camera_height_m=1.5, geometry_size=(16, 24)
    )
    with pytest.raises(ValueError, match="flow token mismatch"):
        dataset[0]


def test_offline_scene_bootstrap_pairs_predictions_by_token(tmp_path: Path) -> None:
    header = (
        "scene_name,center_token,target_vx_mps,target_vy_mps,pred_vx_mps,pred_vy_mps\n"
    )
    candidate = tmp_path / "candidate.csv"
    baseline = tmp_path / "baseline.csv"
    candidate.write_text(
        header + "scene-a,t0,5,0,5.1,0.1\nscene-b,t1,4,0,4.1,0.1\n",
        encoding="utf-8",
    )
    baseline.write_text(
        header + "scene-a,t0,5,0,6,1\nscene-b,t1,4,0,5,1\n",
        encoding="utf-8",
    )
    deltas = paired_scene_deltas(candidate, baseline)
    result = bootstrap([deltas, deltas, deltas], samples=200, random_seed=7)
    assert result["seed_count"] == 3
    assert result["metrics"]["epe_mps"]["significant_improvement"] is True


def test_offline_scene_bootstrap_rejects_target_mismatch(tmp_path: Path) -> None:
    header = (
        "scene_name,center_token,target_vx_mps,target_vy_mps,pred_vx_mps,pred_vy_mps\n"
    )
    candidate = tmp_path / "candidate.csv"
    baseline = tmp_path / "baseline.csv"
    candidate.write_text(header + "scene-a,t0,5,0,5.1,0.1\n", encoding="utf-8")
    baseline.write_text(header + "scene-a,t0,6,0,5.1,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Target mismatch"):
        paired_scene_deltas(candidate, baseline)
