"""Evaluate a frozen downstream checkpoint on dev; heldout access is deliberately sealed."""
import argparse
from pathlib import Path

import torch

from purevision_v5.common import code_version, now, object_hash, sha256, write_json
from purevision_v5.datasets import VelocityDataset
from purevision_v5.model import VelocityModel
from purevision_v5.evaluation import validate_lateral_rule
from purevision_v5.training import evaluate_model, validate_expected_flow, write_predictions


def evaluate(args):
    torch.set_num_threads(args.threads)
    packet = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (packet.get("schema") != "v5_training_state_v1"
            or packet.get("contract_hash") != object_hash(packet.get("contract", {}))
            or packet.get("config") != packet.get("contract", {}).get("config")):
        raise ValueError("Evaluation requires an intact V5 training checkpoint contract")
    model = VelocityModel(experiment=packet["config"]["experiment"]).to(args.device)
    model.load_state_dict(packet["model"], strict=True)
    rule = validate_lateral_rule(packet["config"].get("lateral_rule"))
    dataset = VelocityDataset(args.manifest, "dev", packet["config"]["experiment"], args.features, args.flow,
                              selection_path=getattr(args, "selection", None))
    if dataset.contract != packet["contract"]["data"] or dataset.selection_hash != packet["contract"]["dev_selection"]:
        raise ValueError("Evaluation data differs from the trained contract")
    validate_expected_flow(packet["config"], dataset.contract)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output must be new")
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status="running", started_at=now(), checkpoint=str(Path(args.checkpoint).resolve()),
                   checkpoint_sha256=sha256(args.checkpoint), training_run=packet["run_id"],
                   training_code=packet["code"], evaluation_code=code_version(),
                   contract_hash=packet["contract_hash"], data=dataset.contract,
                   selection_hash=dataset.selection_hash, runtime=vars(args))
    receipt.update(scope="resolution_pilot" if dataset.scene_selection else "full_train_dev",
                   effective_lateral_rule=rule)
    write_json(output / "evaluation.json", receipt)
    try:
        summary, rows = evaluate_model(model, dataset, args.device, packet["config"]["batch_size"], args.workers,
                                       lateral_rule=rule)
        write_json(output / "metrics.json", summary)
        write_predictions(output / "predictions.csv", rows)
        receipt.update(status="completed", finished_at=now(), count=len(rows))
    except BaseException as error:
        receipt.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "evaluation.json", receipt)
    print(summary["overall"])
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--features")
    p.add_argument("--flow")
    p.add_argument("--selection")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
