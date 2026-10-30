from pathlib import Path

import numpy as np
import pytest

from lateral_baselines import baseline_report
from purevision_v5.common import object_hash, read_json, sha256, write_json
from purevision_v5.evaluation import DEFAULT_LATERAL_RULE, lateral_acceptance, summarize, validate_lateral_rule
from purevision_v5.selection import load_selection, make_selection, select_records
from purevision_v5 import training
from test_training import make_case, load_state


@pytest.fixture(autouse=True)
def local_runtime(monkeypatch):
    monkeypatch.setattr(training, "code_version", lambda require_pushed=False: dict(commit="fixed-test", dirty=False))
    monkeypatch.setattr(training, "environment", lambda: dict(device="cpu_test"))


def test_custom_lateral_rule_controls_subset_and_acceptance():
    rule = dict(DEFAULT_LATERAL_RULE, nonzero_threshold_m_s=.05, min_subset_windows=2,
                min_subset_scenes=2, max_absolute_mae_increase_m_s=0., max_relative_mae_increase=0.)
    target = [[1., .1], [1., -.1]]
    baseline = summarize(target, target, ["a", "b"], rule)
    candidate = summarize([[1., .11], [1., -.11]], target, ["a", "b"], rule)
    assert baseline["nonzero_vy"]["adequate"]
    assert baseline["effective_lateral_rule"] == rule
    assert not lateral_acceptance(candidate, baseline, rule)["acceptable"]
    with pytest.raises(ValueError, match="rules differ"):
        lateral_acceptance(candidate, baseline)
    for bad in ({}, dict(rule, min_subset_windows=True), dict(rule, nonzero_threshold_m_s=float("nan"))):
        with pytest.raises(ValueError):
            validate_lateral_rule(bad)


def test_selection_is_fixed_verified_and_cannot_include_holdout(tmp_path):
    args = make_case(tmp_path / "case")
    selection = make_selection(args.manifest, 1, 1, 42)
    path = tmp_path / "selection.json"
    write_json(path, selection)
    assert load_selection(args.manifest, path) == selection
    records, _ = select_records(args.manifest, "dev", path)
    assert len(records) == 1 and records[0]["partition"] == "dev"
    with pytest.raises(ValueError, match="sealed"):
        select_records(args.manifest, "final_system_test", path)
    selection["scenes"]["dev"] = ["scene-not-the-selected-dev"]
    selection["identity"] = object_hash({k: v for k, v in selection.items() if k != "identity"})
    write_json(path, selection)
    with pytest.raises(ValueError, match="deterministic policy"):
        load_selection(args.manifest, path)


def test_train_constant_is_not_fitted_to_dev_and_exact_prediction_coverage(tmp_path):
    args = make_case(tmp_path / "case")
    before = baseline_report(args.manifest)
    manifest = read_json(args.manifest)
    for record in manifest["sequences"]:
        if record["partition"] == "dev":
            path = Path(args.manifest).parent / record["file"]
            sequence = read_json(path)
            for end in sequence["windows"]:
                sequence["frames"][end]["target"][1] += 10
            write_json(path, sequence)
            record["sha256"] = sha256(path)
    write_json(args.manifest, manifest)
    after = baseline_report(args.manifest)
    assert before["fit"] == after["fit"]
    assert before["evaluation"] != after["evaluation"]
    assert before["train_median_vy"]["overall"]["mae_vy"] != after["train_median_vy"]["overall"]["mae_vy"]
    prediction = tmp_path / "predictions.csv"
    prediction.write_text("scene,token,time_s,vx,vy,target_vx,target_vy\nwrong,wrong,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="exact selected dev"):
        baseline_report(args.manifest, predictions=[prediction])


def test_complete_epoch_budget_identical_initialization_order_and_resume(tmp_path):
    args = make_case(tmp_path / "case")
    config = read_json(args.config)
    config.update(epochs=3, patience=1, early_stopping=False, min_delta=1000.)
    write_json(args.config, config)
    full = training.train(args)
    full_state = load_state(full)
    assert full_state["epoch"] == 3 and full_state["bad_epochs"] == 2
    assert read_json(full / "status.json")["stop_reason"] == "epoch_budget"
    run = read_json(full / "run.json")
    args.output, args.stop_after_steps = str(tmp_path / "interrupted"), 7
    interrupted = training.train(args)
    assert load_state(interrupted)["bad_epochs"] >= config["patience"]
    args.output, args.resume, args.stop_after_steps = str(tmp_path / "resumed"), str(interrupted / "last.pt"), 0
    resumed = training.train(args)
    assert load_state(resumed)["epoch"] == 3
    resumed_run = read_json(resumed / "run.json")
    assert run["initialization"] == resumed_run["initialization"]
    assert run["sample_order"] == resumed_run["sample_order"]
    for name, tensor in full_state["model"].items():
        np.testing.assert_array_equal(tensor.numpy(), load_state(resumed)["model"][name].numpy())
    import json
    a = [json.loads(line)["sample_order_sha256"] for line in (full / "metrics.jsonl").read_text().splitlines()]
    b = [json.loads(line)["sample_order_sha256"] for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    assert a == b and len(a) == 3


def test_expected_flow_resolution_rejects_wrong_frontend():
    from purevision_v5.flow import PREPROCESSING, derived_contract, native_contract
    low = dict(expected_flow=dict(raft_size_hw=[288, 512], model_size_hw=[72, 128]))
    training.validate_expected_flow(low, dict(flow=dict(preprocessing=PREPROCESSING)))
    with pytest.raises(ValueError, match="resolution contract differs"):
        training.validate_expected_flow(low, dict(flow=dict(native_contract=native_contract((576, 1024)),
                                                             derived_contract=derived_contract((576, 1024)))))


def test_pilot_is_not_an_ad_hoc_smoke_subset(tmp_path):
    args = make_case(tmp_path / "case")
    path = tmp_path / "selection.json"
    write_json(path, make_selection(args.manifest, 1, 1))
    args.selection = str(path)
    with pytest.raises(ValueError, match="cannot use smoke"):
        training.train(args)


def test_explicit_pilot_training_and_dev_evaluation_keep_selection_identity(tmp_path, monkeypatch):
    import evaluate as evaluation_cli
    from argparse import Namespace

    args = make_case(tmp_path / "case", experiment="R02", flows=True)
    selection_path = tmp_path / "selection.json"
    selection = make_selection(args.manifest, 1, 1)
    write_json(selection_path, selection)
    root = tmp_path / "checkout/methods/purevision-flow/PureVision_V5"
    config_path = root / "configs/pilot.json"
    write_json(config_path, dict(read_json(args.config), early_stopping=False))
    monkeypatch.setattr(training, "ROOT", root)
    monkeypatch.setattr(training, "git", lambda *unused: "")
    monkeypatch.setattr(evaluation_cli, "code_version", lambda: dict(commit="fixed-test", dirty=False))
    args.config, args.selection, args.smoke = str(config_path), str(selection_path), False
    output = training.train(args)
    run = read_json(output / "run.json")
    assert run["scope"] == "resolution_pilot" and run["smoke"] is False
    assert run["contract"]["execution"]["scene_selection_identity"] == selection["identity"]
    evaluate_args = Namespace(checkpoint=str(output / "best.pt"), manifest=args.manifest,
                              features=None, flow=args.flow, selection=str(selection_path),
                              output=str(tmp_path / "evaluation"), device="cpu", workers=0, threads=1)
    evaluation_cli.evaluate(evaluate_args)
    receipt = read_json(Path(evaluate_args.output) / "evaluation.json")
    assert receipt["scope"] == "resolution_pilot"
    assert receipt["data"]["scene_selection"]["identity"] == selection["identity"]
    assert (output / "dev_predictions.csv").read_bytes() == (Path(evaluate_args.output) / "predictions.csv").read_bytes()
    evaluate_args.selection, evaluate_args.output = None, str(tmp_path / "wrong-evaluation")
    with pytest.raises(ValueError, match="differs from the trained contract"):
        evaluation_cli.evaluate(evaluate_args)
