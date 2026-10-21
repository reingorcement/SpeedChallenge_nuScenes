"""Prepare sequences, optionally evaluate, and create an explicit training manifest."""
import argparse
from pathlib import Path

import pandas as pd

from purevision_v4.evaluation import evaluate, macro_metrics
from purevision_v4.io import artifact_path, object_hash, read_json, sha256, write_json
from purevision_v4.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", default=["F_RC", "F_R"])
    parser.add_argument("--fusion-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save-features", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("schema") != "sequence_manifest_v1":
        raise ValueError("Expected sequence_manifest_v1")
    sequences = manifest["sequences"]
    keys = [item["case"] for item in sequences]
    if len(set(keys)) != len(keys) or any(Path(k).name != k or k in {".", ".."} for k in keys):
        raise ValueError("Case names must be unique directory names")
    out = artifact_path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(manifest_sha256=sha256(args.manifest), methods=args.methods, save_features=args.save_features,
                    evaluate=args.evaluate, fusion_sha256=sha256(args.fusion_checkpoint) if args.fusion_checkpoint else None)
    if (out / "batch.json").exists():
        if not args.resume or read_json(out / "batch.json")["identity"] != identity:
            raise ValueError("Batch identity changed or resume was not requested")
    elif any(out.iterdir()):
        raise ValueError("Batch output must be new")
    write_json(out / "batch.json", dict(status="running", identity=identity))
    reports = []
    training = dict(schema="training_manifest_v1", train=[], dev=[])
    try:
        for seq in sequences:
            destination = out / seq["case"]
            run(seq["input"], destination, methods=args.methods, device=args.device,
                fusion_checkpoint=args.fusion_checkpoint, image_root=seq.get("image_root"),
                resume=args.resume, save_features=args.save_features)
            if args.save_features and seq.get("partition") in {"train_core", "train", "dev"}:
                name = "dev" if seq["partition"] == "dev" else "train"
                training[name].append(dict(features=str(destination / "features/manifest.json"), labels=seq["labels"]))
            if args.evaluate:
                evaluation = destination / "evaluation"
                provenance = dict(predictions_sha256=sha256(destination / "speeds.csv"), labels_sha256=sha256(seq["labels"]),
                                  label_meta_sha256=sha256(Path(seq["labels"]).with_suffix(".meta.json")))
                if args.resume and (evaluation / "evaluation.json").exists():
                    old = read_json(evaluation / "evaluation.json")
                    if any(old[k] != v for k, v in provenance.items()) or sha256(evaluation / "metrics.csv") != old.get("metrics_sha256"):
                        raise ValueError("Evaluation inputs/results changed")
                    report = pd.read_csv(evaluation / "metrics.csv")
                else:
                    report = evaluate(destination / "speeds.csv", seq["labels"], evaluation)
                    old = read_json(evaluation / "evaluation.json")
                    write_json(evaluation / "evaluation.json", dict(old, metrics_sha256=sha256(evaluation / "metrics.csv")))
                reports.append(report.assign(case=seq["case"]))
        if reports:
            table = pd.concat(reports, ignore_index=True)
            table.to_csv(out / "per_case_metrics.csv", index=False)
            macro_metrics(table).to_csv(out / "macro_metrics.csv", index=False)
        if training["train"] and training["dev"]:
            write_json(out / "training_manifest.json", training)
        write_json(out / "batch.json", dict(status="complete", identity=identity, cases=keys))
    except BaseException as error:
        write_json(out / "batch.json", dict(status="failed", identity=identity, error=str(error)))
        raise


if __name__ == "__main__":
    main()
