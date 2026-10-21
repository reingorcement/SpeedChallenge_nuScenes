"""Train RGB/fusion heads on causal frozen features and separately loaded labels."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .flow import atomic_torch_save
from .io import ROOT, TARGET_VERSION, artifact_path, read_json, sha256, write_json
from .model import FEATURE_NAMES, FrozenVisual, FusionHead


def resolve_file(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_labels(path):
    path = Path(path)
    spec_file = path.with_suffix(".meta.json")
    spec = read_json(spec_file)
    expected = dict(version=TARGET_VERSION, axes="forward_left_up", units="m/s", time="output_frame")
    if any(spec.get(k) != v for k, v in expected.items()) or not spec.get("reference_point"):
        raise ValueError("Labels must declare full body-reference velocity at output time; legacy yaw labels are incompatible")
    labels = pd.read_csv(path)
    if not {"time", "vx", "vy"}.issubset(labels.columns):
        raise ValueError("Label CSV must contain time,vx,vy")
    if labels.time.duplicated().any() or not np.isfinite(labels[["time", "vx", "vy"]].to_numpy(float)).all():
        raise ValueError("Label times must be unique and values finite")
    if "roll" in labels and not np.isfinite(labels["roll"].to_numpy(float)).all():
        raise ValueError("Optional roll labels must be finite radians")
    return labels.sort_values("time"), {**expected, "reference_point": spec["reference_point"]}


class FeatureDataset(Dataset):
    def __init__(self, records, base, allow_unlabelled_boundary=False):
        self.items = []
        self.sources = set()
        self.sequence_hashes = set()
        self.case_names = set()
        self.contract = self.target_spec = None
        self.provenance = []
        for record in records:
            feature_file = resolve_file(base, record["features"])
            label_file = resolve_file(base, record["labels"])
            manifest = read_json(feature_file)
            labels, target = load_labels(label_file)
            if manifest.get("schema") != "frozen_features_v1" or manifest["target_spec"] != target:
                raise ValueError("Feature/label target convention mismatch")
            if self.contract is None:
                self.contract, self.target_spec = manifest["feature_contract"], target
            if self.contract != manifest["feature_contract"] or self.target_spec != target:
                raise ValueError("All training features must share preprocessing, weights, code and target conventions")
            if manifest["sequence_hash"] in self.sequence_hashes:
                raise ValueError("Duplicate sequence in a partition")
            self.sequence_hashes.add(manifest["sequence_hash"])
            self.sources.update(manifest["source_hashes"])
            self.case_names.add(manifest["case"])
            self.provenance.append(dict(features=str(feature_file), features_sha256=sha256(feature_file),
                                        labels=str(label_file), labels_sha256=sha256(label_file),
                                        label_meta_sha256=sha256(label_file.with_suffix(".meta.json"))))
            times = labels.time.to_numpy(float)
            for entry in manifest["windows"]:
                if not entry.get("visual_valid", True):
                    continue
                pos = int(np.searchsorted(times, entry["time_s"]))
                candidates = [j for j in [pos - 1, pos] if 0 <= j < len(times)]
                j = min(candidates, key=lambda j: abs(times[j] - entry["time_s"])) if candidates else None
                if j is None or abs(times[j] - entry["time_s"]) > 1e-6:
                    if allow_unlabelled_boundary and len(times) and (entry["time_s"] < times[0] or entry["time_s"] > times[-1]):
                        continue
                    raise ValueError(f"Missing timestamp-matched training label: {manifest['case']} {entry['time_s']}")
                file = (feature_file.parent / entry["file"]).resolve()
                if not file.is_relative_to(feature_file.parent.resolve()) or sha256(file) != entry["sha256"]:
                    raise ValueError("Feature file escaped its directory or changed")
                self.items.append((file, labels.iloc[j][["vx", "vy"]].to_numpy(np.float32), entry))
        if not self.items:
            raise ValueError("No labeled feature windows")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        file, label, entry = self.items[index]
        item = torch.load(file, map_location="cpu", weights_only=True)
        if item["image_id"] != entry["image_id"] or abs(item["time_s"] - entry["time_s"]) > 1e-9:
            raise ValueError("Feature timestamp/ID mismatch")
        for key, shape in [("descriptor", (384,)), ("diagnostic", (len(FEATURE_NAMES),)), ("geometry_velocity", (2,))]:
            if item[key].shape != shape or not torch.isfinite(item[key]).all():
                raise ValueError(f"Invalid {key} training feature")
        return {key: item[key] for key in ["descriptor", "diagnostic", "geometry_velocity", "geometry_valid"]} | {"target": torch.from_numpy(label.copy())}


def training_loss(output, target, valid, stage):
    sigma = target.new_tensor([5., 1.])
    error = (output["prediction"] - target) / sigma
    nll = .5 * (torch.exp(-output["log_variance"]) * error.square() + output["log_variance"]).mean()
    direct = nn.functional.smooth_l1_loss(output["direct"] / sigma, target / sigma)
    speed = nn.functional.smooth_l1_loss(torch.linalg.vector_norm(output["prediction"], dim=-1), torch.linalg.vector_norm(target, dim=-1))
    loss = nll + .25 * direct + .1 * speed
    if stage == "fusion":
        loss = loss + .01 * (output["correction"].square().mean(-1) * valid).sum() / valid.sum().clamp_min(1)
    return loss


def _epoch(model, loader, device, stage, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, errors, count = 0., [], 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch["descriptor"], batch["diagnostic"], batch["geometry_velocity"], batch["geometry_valid"], mode=stage)
            loss = training_loss(output, batch["target"], batch["geometry_valid"], stage)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
                optimizer.step()
            errors.append((output["prediction"] - batch["target"]).detach().cpu())
            loss_sum += float(loss.detach()) * len(batch["target"])
            count += len(batch["target"])
    error = torch.cat(errors)
    return dict(loss=loss_sum / count, epe=float(torch.linalg.vector_norm(error, dim=-1).mean()),
                mae_vx=float(error[:, 0].abs().mean()), mae_vy=float(error[:, 1].abs().mean()), count=count)


def train(manifest_path, output, *, stage="fusion", epochs=20, batch_size=32, learning_rate=1e-4,
          device="cpu", seed=42, patience=5, init=None, resume=None, threads=4):
    if stage not in {"direct", "fusion"} or min(epochs, batch_size, patience, threads) < 1 or learning_rate <= 0:
        raise ValueError("Invalid training options")
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "training_manifest_v1" or not manifest.get("train") or not manifest.get("dev"):
        raise ValueError("Training requires explicit independent train/dev partitions")
    datasets = {name: FeatureDataset(manifest[name], manifest_path.parent, manifest.get("allow_unlabelled_boundary", False)) for name in ["train", "dev"]}
    training, validation = datasets["train"], datasets["dev"]
    if training.sources & validation.sources or training.sequence_hashes & validation.sequence_hashes or training.case_names & validation.case_names:
        raise ValueError("Train/dev overlap: split by independent sequences, not neighboring windows")
    if training.contract != validation.contract or training.target_spec != validation.target_spec:
        raise ValueError("Train/dev contracts differ")
    protected = [manifest_path]
    for data in datasets.values():
        for row in data.provenance:
            protected.extend([Path(row["features"]), Path(row["labels"])])
        protected.extend(file for file, _, _ in data.items)
    output = artifact_path(output)
    if any(path.is_relative_to(output) for path in protected):
        raise ValueError("Training output must not contain its input data")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be new; resume into a new V4 directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device(device)
    frozen = FrozenVisual(ROOT / "weights/v3_fusion_seed42.pth")
    if training.contract["visual_weights"] != frozen.digest:
        raise ValueError("Frozen visual initialization differs from feature cache")
    model = FusionHead(frozen.legacy.direct_head).to(device)
    del frozen
    for p in list(model.gate.parameters()) + list(model.correction.parameters()):
        p.requires_grad_(stage == "fusion")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-4)
    history, first_epoch, best, stale = [], 0, float("inf"), 0
    contract = dict(schema="purevision_v4_fusion_v1", feature_contract=training.contract, target_spec=training.target_spec,
                    provenance={name: data.provenance for name, data in datasets.items()}, seed=seed, stage=stage)
    if init is not None and resume is not None:
        raise ValueError("Use init or resume, not both")
    if init is not None or resume is not None:
        saved = torch.load(init or resume, map_location="cpu", weights_only=True)
        if saved.get("schema") != contract["schema"] or saved["feature_contract"] != training.contract or saved["target_spec"] != training.target_spec:
            raise ValueError("Checkpoint training contract mismatch")
        model.load_state_dict(saved["model"], strict=True)
        if resume is not None:
            if saved["stage"] != stage or saved["provenance"] != contract["provenance"] or saved["seed"] != seed:
                raise ValueError("Resume stage/data/seed mismatch")
            optimizer.load_state_dict(saved["optimizer"])
            first_epoch, best, stale, history = saved["epoch"] + 1, saved["best_epe"], saved["stale"], saved["history"]
            if first_epoch >= epochs:
                raise ValueError("Resume epochs must exceed the completed epoch")
            torch.set_rng_state(saved["torch_rng"])
            if device.type == "cuda" and saved.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            # Carry the historical best into the new isolated output directory.
            best_source = Path(resume).parent / "best.pth"
            best_saved = torch.load(best_source, map_location="cpu", weights_only=True)
            if best_saved["feature_contract"] != training.contract or best_saved["provenance"] != contract["provenance"]:
                raise ValueError("Resume best checkpoint mismatch")
            atomic_torch_save(best_saved, output / "best.pth")
    loaders = {name: DataLoader(data, batch_size=batch_size, shuffle=name == "train", num_workers=0) for name, data in datasets.items()}
    write_json(output / "training.json", dict(status="running", **contract, epochs=epochs, learning_rate=learning_rate,
               batch_size=batch_size, frozen_modules=["backbone", "region_projection", "reliability", "finite_geometry"]))
    try:
        for epoch in range(first_epoch, epochs):
            train_metrics = _epoch(model, loaders["train"], device, stage, optimizer)
            dev_metrics = _epoch(model, loaders["dev"], device, stage)
            improved = dev_metrics["epe"] < best
            best = min(best, dev_metrics["epe"])
            stale = 0 if improved else stale + 1
            history.append(dict(epoch=epoch, train=train_metrics, dev=dev_metrics))
            checkpoint = dict(**contract, trained=True, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                              best_epe=best, stale=stale, history=history, torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
            atomic_torch_save(checkpoint, output / "last.pth")
            if improved:
                atomic_torch_save(checkpoint, output / "best.pth")
            write_json(output / "history.json", history)
            print(f"epoch={epoch} train={train_metrics['loss']:.5f} dev_epe={dev_metrics['epe']:.5f}", flush=True)
            if stale >= patience:
                break
        write_json(output / "training.json", dict(status="complete", **contract, epochs_completed=len(history),
                   best_epe=best, checkpoint=str(output / "best.pth"), frozen_geometry=True))
    except BaseException as error:
        write_json(output / "training.json", dict(status="failed", **contract, error=str(error)))
        raise
    return output / "best.pth"
