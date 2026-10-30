"""Explicit, immutable scene subsets; a resolution pilot never becomes a full run."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import object_hash, read_json, sha256
from .labels import LABEL_CONTRACT, LABEL_SHA256

PARTITIONS = ("train_core", "dev")
SCHEMA = "v5_scene_selection_v1"


def verified_sequence(manifest_path, record):
    root = Path(manifest_path).resolve().parent
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != record["sha256"]:
        raise ValueError("Selection sequence checksum/path mismatch")
    sequence = read_json(path)
    if sequence.get("schema") != "v5_sequence_v1" or any(sequence.get(k) != record[k] for k in ("scene", "partition", "log_token")):
        raise ValueError("Selection sequence identity mismatch")
    return sequence


def make_selection(manifest_path, train_scenes=32, dev_scenes=8, seed=42):
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "v5_dataset_v1" or manifest.get("seq_len") != 5:
        raise ValueError("Selection requires a V5 five-frame manifest")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Selection seed must be an integer")
    result = dict(schema=SCHEMA, scope="resolution_pilot", dataset_sha256=sha256(manifest_path),
                  seed=seed, policy="ascending_sha256_of_canonical_json_seed_partition_scene_v1",
                  scenes={}, counts={})
    for partition, count in zip(PARTITIONS, (train_scenes, dev_scenes)):
        records = [r for r in manifest["sequences"] if r["partition"] == partition]
        if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= len(records):
            raise ValueError("Requested selection scene count is invalid")
        ordered = sorted(records, key=lambda r: (object_hash([seed, partition, r["scene"]]), r["scene"]))[:count]
        result["scenes"][partition] = sorted(r["scene"] for r in ordered)
        sequences = [verified_sequence(manifest_path, r) for r in ordered]
        result["counts"][partition] = dict(scenes=count, frames=sum(len(s["frames"]) for s in sequences),
                                           windows=sum(len(s["windows"]) for s in sequences))
    result["identity"] = object_hash(result)
    return result


def load_selection(manifest_path, selection_path=None):
    if selection_path is None:
        return None
    value = read_json(selection_path)
    if (value.get("schema") != SCHEMA or value.get("scope") != "resolution_pilot"
            or value.get("dataset_sha256") != sha256(manifest_path)
            or value.get("identity") != object_hash({k: v for k, v in value.items() if k != "identity"})):
        raise ValueError("Selection schema/scope/dataset/identity mismatch")
    if set(value.get("scenes", {})) != set(PARTITIONS) or set(value.get("counts", {})) != set(PARTITIONS):
        raise ValueError("Selection may only name train_core and dev")
    # Reproduce the declared policy, counts and scene lists from immutable source.
    expected = make_selection(manifest_path, value["counts"]["train_core"]["scenes"],
                              value["counts"]["dev"]["scenes"], value["seed"])
    if value != expected:
        raise ValueError("Selection differs from its declared deterministic policy")
    return value


def select_records(manifest_path, partition, selection_path=None):
    if partition not in PARTITIONS:
        raise ValueError("Held-out partitions are sealed")
    manifest = read_json(manifest_path)
    selection = load_selection(manifest_path, selection_path)
    selected = set(selection["scenes"][partition]) if selection else None
    return [r for r in manifest["sequences"] if r["partition"] == partition
            and (selected is None or r["scene"] in selected)], selection


def label_rows(manifest_path, partition, selection_path=None):
    """Read only verified window labels; no RGB/flow cache and no held-out labels."""
    manifest = read_json(manifest_path)
    if manifest.get("label_contract") != LABEL_CONTRACT or manifest.get("label_sha256") != LABEL_SHA256:
        raise ValueError("Unsupported label contract")
    records, selection = select_records(manifest_path, partition, selection_path)
    rows = []
    for record in records:
        sequence = verified_sequence(manifest_path, record)
        ends, frames = sequence["windows"], sequence["frames"]
        if ends != sorted(set(ends)):
            raise ValueError("Duplicate or unordered window ends")
        for end in ends:
            if not isinstance(end, int) or isinstance(end, bool) or not 4 <= end < len(frames):
                raise ValueError("Invalid causal window end")
            frame = frames[end]
            target = np.asarray(frame["target"], dtype=np.float32)
            if target.shape != (2,) or not np.isfinite(target).all():
                raise ValueError("Invalid label")
            rows.append(dict(scene=record["scene"], token=frame["token"], time_s=frame["time_s"],
                             target_vx=float(target[0]), target_vy=float(target[1])))
    if not rows:
        raise ValueError("Empty label selection")
    return rows, selection
