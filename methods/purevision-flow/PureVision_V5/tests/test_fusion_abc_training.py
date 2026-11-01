"""Behavioral checks for unequal batches, protocol locks and selection ties."""
from argparse import Namespace
import math

import pytest
import torch

from purevision_v5 import fusion_abc_training as training
from purevision_v5.common import write_json


def test_dev_epe_weights_the_small_final_batch_by_its_sample_count():
    class PredictionsInInput(torch.nn.Module):
        def forward(self, features):
            return features["dt"][:, :2]

    packet = dict(features=dict(dt=torch.zeros(65, 4)), targets=torch.zeros(65, 2),
                  metadata=[dict(scene="one-scene", token=str(i), time_s=float(i)) for i in range(65)])
    packet["features"]["dt"][-1, 0] = 65.
    summary, rows = training.evaluate_packet(PredictionsInInput(), packet, "cpu", batch_size=64)
    # Averaging the two batch means would incorrectly produce 32.5 instead.
    assert summary["overall"]["epe"] == 1.
    assert summary["overall"]["count"] == len(rows) == 65


@pytest.mark.parametrize("field,changed", [
    ("epochs", 21), ("scheduler_epochs", 18), ("amp", True), ("tf32", True),
    ("batch_size", 32), ("drop_last", True), ("learning_rate", .002),
    ("eta_min", .0001), ("loss_scales_m_s", [1., 1.]),
    ("betas", [.9, .99]), ("early_stopping", True),
])
def test_comparison_protocol_changes_are_rejected(field, changed):
    config = dict(training.PROTOCOL, variant="B")
    config[field] = changed
    with pytest.raises(ValueError, match="protocol mismatch"):
        training.validate_config(config)


def test_twenty_complete_epoch_schedule_has_the_expected_learning_rate():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=training.PROTOCOL["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training.PROTOCOL["scheduler_epochs"], eta_min=training.PROTOCOL["eta_min"])
    for epoch_index in range(20):
        # Current epoch uses its pre-step LR; only complete-epoch end advances it.
        expected = .001 * (1 + math.cos(math.pi * epoch_index / 20)) / 2
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected, abs=1e-15)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.


@pytest.mark.parametrize("state", ["interrupted", "preflight_completed", "running", "failed"])
def test_final_evaluation_gate_rejects_every_unfinished_run(tmp_path, state):
    write_json(tmp_path / "run.json", dict(status=state))
    with pytest.raises(ValueError, match="completed formal run"):
        training.require_completed_run(tmp_path, tmp_path / "best.pt", {})


def test_exact_dev_tie_retains_earliest_epoch_and_preserves_full_last_state(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(7)
    features = dict(F_sp=torch.randn(8, 4, 6, 32, generator=generator),
                    F_tm=torch.randn(8, 4, 32, generator=generator),
                    R=torch.randn(8, 4, 6, 256, generator=generator), dt=torch.full((8, 4), 1 / 12))

    def load_partition(_cache, partition, _smoke):
        packet = dict(features=features, targets=torch.ones(8, 2),
                      metadata=[dict(scene=partition, token=f"{partition}-{i}", time_s=float(i)) for i in range(8)])
        return packet, dict(identity="unit-test-cache", scope="preflight")

    def equal_dev(*_args):
        return dict(overall=dict(epe=1.)), [dict(scene="dev", token="dev-0", time_s=0.,
                                                vx=0., vy=0., target_vx=1., target_vy=1.)]

    monkeypatch.setattr(training, "load_partition", load_partition)
    monkeypatch.setattr(training, "evaluate_packet", equal_dev)
    monkeypatch.setattr(training, "code_version", lambda require_pushed=False: dict(commit="test", dirty=False))
    monkeypatch.setattr(training, "environment", lambda: dict(device="CPU unit test"))
    config_path = tmp_path / "config.json"
    write_json(config_path, dict(training.PROTOCOL, variant="A"))
    args = Namespace(config=str(config_path), cache=str(tmp_path / "cache"), output=str(tmp_path / "run"),
                     device="cpu", threads=1, smoke_steps=3, resume=None)
    receipt = training.train(args)
    best = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    last = torch.load(tmp_path / "run" / "last.pt", weights_only=True)
    assert receipt["status"] == "preflight_completed"
    assert best["best_epoch"] == best["completed_epochs"] == 1
    assert last["best_epoch"] == 1 and last["completed_epochs"] == 3
    assert last["processed_batches"] == last["optimizer_updates"] == 3
    assert last["amp_skips"] == 0 and last["amp_enabled"] is False
    assert all(key in last for key in ("model", "optimizer", "scheduler", "rng", "next_batch", "code", "contract"))
    assert len(last["order_records"]) == 3
    assert not torch.equal(best["model"]["head.0.weight"], last["model"]["head.0.weight"])
