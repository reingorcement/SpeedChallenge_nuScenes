"""Task-level evidence for exact recovery, provenance, isolation, and metrics."""
from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from purevision_v5 import training
from purevision_v5.common import object_hash, read_json, sha256, write_json
from purevision_v5.datasets import VelocityDataset
from purevision_v5.evaluation import metrics, summarize
from purevision_v5.labels import LABEL_CONTRACT, LABEL_SHA256


def make_case(root, *, overlap=False, experiment="R00", flows=False):
    root.mkdir(parents=True, exist_ok=True)
    features, flow_root = root / "features", root / "flow"
    features.mkdir()
    records, arrays = [], {}
    generator = np.random.default_rng(17)
    for partition, count in (("train_core", 17), ("dev", 9)):
        scene = f"scene-{partition}"
        prefix = "train_core" if overlap and partition == "dev" else partition
        frames = [dict(token=f"{prefix}-{i:03}", time_s=1. + i / 12,
                       filename=f"{prefix}-{i:03}.jpg", target=[3. + .1*i, .05*(-1)**i])
                  for i in range(count)]
        ends = list(range(4, count - 1))
        sequence = dict(schema="v5_sequence_v1", scene=scene, partition=partition,
                        log_token=f"log-{partition}", labels_available=True, frames=frames, windows=ends)
        path = root / f"{scene}.json"
        write_json(path, sequence)
        records.append(dict(scene=scene, partition=partition, log_token=sequence["log_token"],
                            file=path.name, sha256=sha256(path)))
        arrays[scene] = (generator.normal(size=(len(ends), 384)).astype(np.float32), ends, path)
    manifest = root / "dataset.json"
    write_json(manifest, dict(schema="v5_dataset_v1", seq_len=5, label_contract=LABEL_CONTRACT,
                              label_sha256=LABEL_SHA256, sequences=records, dataroot=str(root)))
    index = dict(contract=dict(schema="v5_rgb_features_v1", dataset_sha256=sha256(manifest)), sequences={})
    for scene, (descriptor, ends, sequence_path) in arrays.items():
        path = features / f"{scene}.npz"
        np.savez(path, descriptor=descriptor, end_indices=np.asarray(ends))
        index["sequences"][scene] = dict(file=path.name, sha256=sha256(path), sequence_sha256=sha256(sequence_path))
    write_json(features / "index.json", index)
    if flows:
        from purevision_v5.flow import PREPROCESSING, RAFT_SHA256, SCHEMA, cache_path
        for record in records:
            sequence = read_json(root / record["file"])
            for a, b in zip(sequence["frames"][:-1], sequence["frames"][1:]):
                provenance = dict(source=dict(token=a["token"], time_s=a["time_s"]),
                                  target=dict(token=b["token"], time_s=b["time_s"]),
                                  dt_s=b["time_s"]-a["time_s"], checkpoint_sha256=RAFT_SHA256,
                                  preprocessing=PREPROCESSING, direction="bidirectional")
                packet = dict(schema=SCHEMA, kind="model_view", provenance=provenance,
                              identity=object_hash(provenance), fb_available=True,
                              native_sha256="0"*64, flow=torch.zeros(2, 72, 128, dtype=torch.float16),
                              valid=torch.ones(1, 72, 128, dtype=torch.bool),
                              fb_mask=torch.ones(1, 72, 128, dtype=torch.bool))
                path = cache_path(flow_root, a["token"], b["token"])
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(packet, path)
    config = root / "config.json"
    write_json(config, dict(experiment=experiment, seed=42, epochs=2, batch_size=4,
                            learning_rate=1e-3, weight_decay=1e-4, amp=False, loss_scales_m_s=[5., 1.],
                            loss_beta=1., grad_clip=2., log_steps=100, checkpoint_steps=1,
                            patience=5, min_delta=0.))
    return Namespace(config=str(config), manifest=str(manifest), output=str(root / "run"),
                     features=str(features), flow=str(flow_root) if flows else None,
                     device="cpu", workers=0, threads=1, resume=None, smoke=True,
                     limit_scenes=0, limit_windows=0, max_batches=0, stop_after_steps=0)


@pytest.fixture(autouse=True)
def deterministic_local_runtime(monkeypatch):
    monkeypatch.setattr(training, "code_version", lambda require_pushed=False: dict(commit="test-fixed-commit", dirty=False))
    monkeypatch.setattr(training, "environment", lambda: dict(device="cpu_test"))


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


def load_state(path):
    return torch.load(Path(path) / "last.pt", map_location="cpu", weights_only=True)


def test_two_step_interruption_matches_uninterrupted_state_and_predictions(tmp_path):
    args = make_case(tmp_path / "case")
    args.output = str(tmp_path / "full")
    full_dir = training.train(args)
    full = load_state(full_dir)
    args.output, args.stop_after_steps = str(tmp_path / "stopped"), 2
    stopped_dir = training.train(args)
    stopped = load_state(stopped_dir)
    assert stopped["global_step"] == 2 and stopped["next_batch"] == 2 and stopped["epoch"] == 0
    assert read_json(stopped_dir / "run.json")["status"] == "interrupted"
    args.output, args.stop_after_steps, args.resume = str(tmp_path / "resumed"), 0, str(stopped_dir / "last.pt")
    resumed_dir = training.train(args)
    resumed = load_state(resumed_dir)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "epoch", "next_batch", "global_step",
                "best_epe", "best_epoch", "bad_epochs", "running_loss", "running_count", "contract_hash"):
        assert_tree_equal(full[key], resumed[key])
    assert full["global_step"] == 6 and resumed["epoch"] == 2
    assert (full_dir / "dev_predictions.csv").read_bytes() == (resumed_dir / "dev_predictions.csv").read_bytes()
    assert read_json(resumed_dir / "run.json")["parent_run"] == "stopped"


def test_resume_preserves_historical_best_when_no_later_improvement(tmp_path):
    args = make_case(tmp_path / "case")
    config = read_json(args.config)
    config["min_delta"] = 1000.
    write_json(args.config, config)
    args.stop_after_steps = 4  # First validation complete, then interrupt epoch 1.
    previous_dir = training.train(args)
    assert (previous_dir / "best.pt").is_file()
    old_best = (previous_dir / "best.pt").read_bytes()
    args.output, args.resume, args.stop_after_steps = str(tmp_path / "resumed"), str(previous_dir / "last.pt"), 0
    output = training.train(args)
    assert (output / "best.pt").read_bytes() == old_best
    for name in ("best_metrics.json", "dev_predictions.csv"):
        assert (output / name).read_bytes() == (previous_dir / name).read_bytes()
    history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [entry["epoch"] for entry in history] == [0, 1]
    assert read_json(output / "status.json")["status"] == "completed"


@pytest.mark.parametrize("mutation", ["loss", "label", "code", "budget_mode"])
def test_resume_rejects_changed_semantics(tmp_path, monkeypatch, mutation):
    args = make_case(tmp_path / "case")
    args.stop_after_steps = 2
    output = training.train(args)
    args.output, args.resume, args.stop_after_steps = str(tmp_path / "resumed"), str(output / "last.pt"), 0
    if mutation == "loss":
        config = read_json(args.config)
        config["loss_scales_m_s"] = [5., .5]
        write_json(args.config, config)
    elif mutation == "label":
        manifest = read_json(args.manifest)
        manifest["label_contract"]["time"] = "some_other_time"
        write_json(args.manifest, manifest)
    elif mutation == "code":
        monkeypatch.setattr(training, "code_version", lambda require_pushed=False: dict(commit="other-commit", dirty=False))
    else:
        args.max_batches = 1
    with pytest.raises(ValueError, match="contract|Resume rejected"):
        training.train(args)
    assert not Path(args.output).exists()


def test_training_rejects_cross_partition_image_overlap(tmp_path):
    args = make_case(tmp_path / "case", overlap=True)
    with pytest.raises(ValueError, match="overlap"):
        training.train(args)


def test_flow_dataset_verifies_sequence_checksum_even_without_rgb(tmp_path):
    args = make_case(tmp_path / "case", experiment="R02", flows=True)
    dataset = VelocityDataset(args.manifest, "train_core", "R02", flow_root=args.flow)
    batch = dataset[0]
    assert set(batch["inputs"]) == {"flows", "valid", "pair_dt"}
    assert "target" not in batch["inputs"]
    sequence_path = Path(args.manifest).parent / "scene-train_core.json"
    sequence = read_json(sequence_path)
    sequence["frames"][4]["target"][1] += 1
    write_json(sequence_path, sequence)
    with pytest.raises(ValueError, match="Sequence checksum"):
        VelocityDataset(args.manifest, "train_core", "R02", flow_root=args.flow)


def test_flow_dataset_binds_packet_times_and_selected_packet_hashes(tmp_path):
    from purevision_v5.flow import cache_path
    args = make_case(tmp_path / "case", experiment="R04", flows=True)
    original = VelocityDataset(args.manifest, "train_core", "R04", args.features, args.flow)
    path = cache_path(args.flow, "train_core-000", "train_core-001")
    packet = torch.load(path, weights_only=True)
    packet["provenance"]["source"]["time_s"] += 10
    packet["provenance"]["target"]["time_s"] += 10
    packet["provenance"]["dt_s"] = packet["provenance"]["target"]["time_s"] - packet["provenance"]["source"]["time_s"]
    packet["identity"] = object_hash(packet["provenance"])
    torch.save(packet, path)
    with pytest.raises(ValueError, match="changed during"):
        original[0]
    changed = VelocityDataset(args.manifest, "train_core", "R04", args.features, args.flow)
    assert changed.selection_hash != original.selection_hash
    with pytest.raises(ValueError, match="time mismatch"):
        changed[0]


def test_heldout_remains_sealed_and_metrics_match_known_errors(tmp_path):
    args = make_case(tmp_path / "case")
    with pytest.raises(ValueError, match="sealed"):
        VelocityDataset(args.manifest, "final_system_test", "R00", args.features)
    prediction, target = [[2., 1.], [0., -1.]], [[1., 0.], [2., 1.]]
    result = metrics(prediction, target)
    assert result["mae_vx"] == result["mae_vy"] == 1.5
    assert result["rmse_vx"] == pytest.approx(np.sqrt(2.5))
    assert result["bias_vy"] == -.5
    assert result["epe"] == pytest.approx(1.5*np.sqrt(2))
    assert result["zero_vy_mae"] == .5
    summary = summarize(prediction, target, ["first", "second"])
    assert summary["scene_count"] == 2 and len(summary["per_scene"]) == 2
    assert summary["nonzero_vy"]["count"] == 1 and not summary["nonzero_vy"]["adequate"]


def test_max_batch_smoke_budget_is_per_epoch_and_failure_status_is_truthful(tmp_path, monkeypatch):
    args = make_case(tmp_path / "case")
    args.max_batches = 1
    output = training.train(args)
    assert load_state(output)["global_step"] == 2
    assert read_json(output / "run.json")["smoke"]
    args.output = str(tmp_path / "failure")
    monkeypatch.setattr(training, "loss_fn", lambda *unused: torch.tensor(float("nan")))
    with pytest.raises(RuntimeError, match="Nonfinite loss"):
        training.train(args)
    assert read_json(Path(args.output) / "run.json")["status"] == "failed"
    assert read_json(Path(args.output) / "status.json")["status"] == "failed"


def test_checkpoint_evaluation_reproduces_best_predictions_with_provenance(tmp_path, monkeypatch):
    import evaluate as evaluation_cli
    monkeypatch.setattr(evaluation_cli, "code_version", lambda: dict(commit="test-fixed-commit", dirty=False))
    args = make_case(tmp_path / "case")
    training_dir = training.train(args)
    evaluation_args = Namespace(checkpoint=str(training_dir / "best.pt"), manifest=args.manifest,
                                features=args.features, flow=None, output=str(tmp_path / "evaluation"),
                                device="cpu", workers=0, threads=1)
    summary = evaluation_cli.evaluate(evaluation_args)
    evaluation_dir = Path(evaluation_args.output)
    assert (evaluation_dir / "predictions.csv").read_bytes() == (training_dir / "dev_predictions.csv").read_bytes()
    assert summary == read_json(training_dir / "best_metrics.json")
    receipt = read_json(evaluation_dir / "evaluation.json")
    assert receipt["status"] == "completed" and receipt["checkpoint_sha256"] == sha256(evaluation_args.checkpoint)
    checkpoint = torch.load(evaluation_args.checkpoint, weights_only=True)
    checkpoint["contract"]["config"]["loss_beta"] = 100.
    corrupt = tmp_path / "corrupt.pt"
    torch.save(checkpoint, corrupt)
    evaluation_args.checkpoint, evaluation_args.output = str(corrupt), str(tmp_path / "rejected")
    with pytest.raises(ValueError, match="intact V5 training checkpoint contract"):
        evaluation_cli.evaluate(evaluation_args)


def test_formal_run_requires_tracked_config_and_external_artifacts(tmp_path, monkeypatch):
    args = make_case(tmp_path / "case")
    args.smoke = False
    repo = tmp_path / "fixed_checkout"
    root = repo / "methods/purevision-flow/PureVision_V5"
    root.mkdir(parents=True)
    monkeypatch.setattr(training, "ROOT", root)
    with pytest.raises(ValueError, match="configuration must be inside"):
        training.train(args)
    config = root / "configs/R00.json"
    write_json(config, read_json(args.config))
    args.config = str(config)
    calls = []
    monkeypatch.setattr(training, "git", lambda *arguments: calls.append(arguments))
    args.output = str(root / "runs/forbidden")
    with pytest.raises(ValueError, match="outside the immutable"):
        training.train(args)
    assert calls == [("ls-files", "--error-unmatch", "methods/purevision-flow/PureVision_V5/configs/R00.json")]
    args.smoke, args.output = True, args.features
    with pytest.raises(ValueError, match="new run directory"):
        training.train(args)
