"""Build small auditable visual manifests without loading all nuScenes tables."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .labels import LABEL_CONTRACT, LABEL_SHA256, legacy_chord_labels, planar_velocity_labels, sensitivity_audit, synthetic_audit, yaw_from_quaternions


PARTITIONS = ("train_core", "dev", "stage1_test", "final_system_test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_json_array(path, chunk_size=1024 * 1024):
    """Stream the large JSON array; retain only one chunk plus one record."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer, pos, eof, opened, closed = "", 0, False, False, False
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if not opened and pos < len(buffer):
                if buffer[pos] != "[":
                    raise ValueError(f"Expected JSON array: {path}")
                opened = True
                pos += 1
            while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ","):
                pos += 1
            if opened and pos < len(buffer) and buffer[pos] == "]":
                closed = True
                break
            try:
                item, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"Truncated/invalid JSON array: {path}")
                chunk = handle.read(chunk_size)
                buffer, pos, eof = buffer[pos:] + chunk, 0, not chunk
                continue
            if not isinstance(item, dict):
                raise ValueError("nuScenes table entries must be objects")
            pos = end
            yield item
        if not closed:
            raise ValueError(f"Unclosed JSON array: {path}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_split(split):
    payload = {k: v for k, v in split.items() if k != "checksum_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    if digest != split.get("checksum_sha256") or set(split.get("partitions", {})) != set(PARTITIONS):
        raise ValueError("Split checksum/partitions invalid")
    seen = set()
    for names in split["partitions"].values():
        if len(names) != len(set(names)) or seen.intersection(names):
            raise ValueError("Scene split overlap or duplicate")
        seen.update(names)


def quantiles(values):
    array = np.asarray(values)
    return {"count": len(array), "min": np.min(array, axis=0).tolist(),
        "p50": np.quantile(array, .5, axis=0).tolist(), "p95": np.quantile(array, .95, axis=0).tolist(),
        "p99": np.quantile(array, .99, axis=0).tolist(), "max": np.max(array, axis=0).tolist()}


def prepare_dataset(dataroot, split_manifest, output, *, label_partitions=("train_core", "dev"), seq_len=5):
    """Index all retained scenes; only materialize labels for explicitly allowed partitions."""
    root, output = Path(dataroot).resolve(), Path(output).resolve()
    if seq_len != 5 or set(label_partitions) - {"train_core", "dev"}:
        raise ValueError("First-round export requires T=5; holdout labels remain sealed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    split = json.loads(Path(split_manifest).read_text())
    validate_split(split)
    metadata = root / split["version"]
    scenes = {s["name"]: s for s in iter_json_array(metadata / "scene.json")}
    partition_for = {name: partition for partition, names in split["partitions"].items() for name in names}
    missing_scenes = set(partition_for) - scenes.keys()
    if missing_scenes:
        raise ValueError(f"Missing metadata scenes: {sorted(missing_scenes)}")
    scene_tokens = {scenes[name]["token"]: name for name in partition_for}
    samples = {s["token"]: scene_tokens[s["scene_token"]] for s in iter_json_array(metadata / "sample.json") if s["scene_token"] in scene_tokens}
    sensor_tokens = {s["token"] for s in iter_json_array(metadata / "sensor.json") if s["channel"] == "CAM_FRONT"}
    calibrated = {s["token"] for s in iter_json_array(metadata / "calibrated_sensor.json") if s["sensor_token"] in sensor_tokens}
    print("Reading CAM_FRONT sample_data (streamed)", flush=True)
    camera = {}
    starts = {}
    for record in iter_json_array(metadata / "sample_data.json"):
        if record["calibrated_sensor_token"] not in calibrated or record["sample_token"] not in samples:
            continue
        token = record["token"]
        if token in camera:
            raise ValueError(f"Duplicate camera token: {token}")
        camera[token] = record
        scene = samples[record["sample_token"]]
        if record["is_key_frame"] and record["sample_token"] == scenes[scene]["first_sample_token"]:
            if scene in starts:
                raise ValueError(f"Multiple first camera keyframes: {scene}")
            starts[scene] = token
    if set(starts) != set(partition_for):
        raise ValueError("Missing first camera keyframe in one or more scenes")
    needed_poses = {r["ego_pose_token"] for r in camera.values() if partition_for[samples[r["sample_token"]]] in label_partitions}
    print(f"Reading {len(needed_poses)} selected camera poses (streamed)", flush=True)
    poses = {p["token"]: p for p in iter_json_array(metadata / "ego_pose.json") if p["token"] in needed_poses}
    if poses.keys() != needed_poses:
        raise ValueError("Missing camera pose records")
    entries, counts, intervals, differences, targets, asymmetry = [], defaultdict(lambda: {"scenes": 0, "frames": 0, "windows": 0}), [], [], defaultdict(list), []
    image_count, timestamp_max_error_us, zero_z_count, pose_count = 0, 0, 0, 0
    sensitivity = None
    frame_tokens_seen = set()
    logs = defaultdict(set)
    for scene in sorted(partition_for):
        partition = partition_for[scene]
        sequence, token = [], starts[scene]
        while token:
            if token not in camera or token in frame_tokens_seen:
                raise ValueError(f"Missing/duplicate/cyclic/cross-scene camera token: {token}")
            record = camera[token]
            if samples[record["sample_token"]] != scene:
                raise ValueError(f"Camera chain crosses scenes: {scene}")
            if sequence and record["prev"] != sequence[-1]["token"]:
                raise ValueError(f"Inconsistent prev/next camera links: {token}")
            filename = Path(record["filename"])
            if filename.is_absolute() or ".." in filename.parts or not (root / filename).is_file():
                raise FileNotFoundError(f"Missing/invalid image: {root / filename}")
            if (int(record["height"]), int(record["width"])) != (900, 1600):
                raise ValueError("Unexpected CAM_FRONT image dimensions")
            frame_tokens_seen.add(token)
            sequence.append(record)
            image_count += 1
            token = record.get("next", "")
        times_us = np.array([r["timestamp"] for r in sequence], dtype=np.int64)
        times = (times_us - times_us[0]).astype(np.float64) / 1e6
        dt = np.diff(times)
        if len(sequence) < seq_len + 1 or np.any(dt <= 0) or np.any(dt > 1.):
            raise ValueError(f"Invalid/abnormally gapped sequence timestamps: {scene}")
        labels = None
        if partition in label_partitions:
            matched = [poses[r["ego_pose_token"]] for r in sequence]
            error = max(abs(int(p["timestamp"]) - int(r["timestamp"])) for p, r in zip(matched, sequence))
            timestamp_max_error_us = max(timestamp_max_error_us, error)
            if error != 0:
                raise ValueError(f"Pose/camera timestamp mismatch requires explicit interpolation: {scene}")
            positions3d = np.asarray([p["translation"] for p in matched], dtype=np.float64)
            if not np.isfinite(positions3d).all():
                raise ValueError(f"Nonfinite position: {scene}")
            yaw = yaw_from_quaternions([p["rotation"] for p in matched])
            labels = planar_velocity_labels(times, positions3d[:, :2], yaw)
            old = legacy_chord_labels(times, positions3d[:, :2], yaw)
            differences.extend(abs(labels[1:-1] - old[1:-1]))
            intervals.extend(dt.tolist())
            asymmetry.extend(abs(dt[1:] - dt[:-1]).tolist())
            targets[partition].extend(labels[seq_len-1:-1])
            zero_z_count += int(np.count_nonzero(positions3d[:, 2] == 0))
            pose_count += len(sequence)
            if partition == "train_core" and sensitivity is None:
                sensitivity = {"scene": scene, **sensitivity_audit(times, positions3d[:, :2], yaw)}
        windows = list(range(seq_len - 1, len(sequence) - 1))
        frames = [{"token": r["token"], "time_s": int(r["timestamp"]) / 1e6, "filename": r["filename"],
            "target": labels[i].tolist() if labels is not None and np.isfinite(labels[i]).all() else None}
            for i, r in enumerate(sequence)]
        relative_file = f"sequences/{scene}.json"
        write_json(output / relative_file, {"schema": "v5_sequence_v1", "scene": scene, "partition": partition,
            "log_token": scenes[scene]["log_token"], "labels_available": labels is not None, "frames": frames, "windows": windows})
        entries.append({"scene": scene, "partition": partition, "log_token": scenes[scene]["log_token"],
            "file": relative_file, "sha256": file_sha256(output / relative_file)})
        counts[partition]["scenes"] += 1
        counts[partition]["frames"] += len(sequence)
        counts[partition]["windows"] += len(windows)
        logs[partition].add(scenes[scene]["log_token"])
    manifest = {"schema": "v5_dataset_v1", "dataroot": str(root), "version": split["version"], "seq_len": seq_len,
        "camera": "CAM_FRONT", "label_contract": LABEL_CONTRACT, "label_sha256": LABEL_SHA256,
        "label_implementation_sha256": file_sha256(Path(__file__).with_name("labels.py")),
        "split_sha256": file_sha256(split_manifest), "split_checksum_sha256": split["checksum_sha256"],
        "label_partitions": list(label_partitions), "sequences": entries, "counts": dict(counts),
        "field_roles": {"inference_input": ["images", "pair_dt"], "training_supervision": ["target"],
            "split_metadata": ["scene", "partition", "log_token", "token", "filename"], "evaluation_only": []}}
    write_json(output / "dataset.json", manifest)
    stats = {}
    for partition, values in targets.items():
        values = np.asarray(values)
        stats[partition] = {"velocity": quantiles(values), "abs_vy": quantiles(abs(values[:, 1])),
            "zero_vy_mae_m_s": float(np.mean(abs(values[:, 1]))), "zero_vy_rmse_m_s": float(np.sqrt(np.mean(values[:, 1]**2)))}
    audit = {"schema": "v5_data_audit_v1", "dataset_sha256": file_sha256(output / "dataset.json"),
        "label_sha256": LABEL_SHA256, "counts": dict(counts), "checked_images": image_count,
        "checked_camera_pose_timestamps": pose_count, "max_camera_pose_time_error_us": timestamp_max_error_us,
        "pose_z_zero_count": zero_z_count, "labelled_pair_dt_s": quantiles(intervals), "neighbor_interval_asymmetry_s": quantiles(asymmetry),
        "absolute_new_minus_old_vx_vy_m_s": quantiles(differences), "label_statistics": stats,
        "synthetic": synthetic_audit(), "bounded_train_sensitivity": sensitivity,
        "excluded_scenes": split.get("excluded_scenes", {}),
        "scene_overlap_count": 0, "shared_logs": {f"{a}__{b}": len(logs[a] & logs[b]) for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i+1:]},
        "holdouts": "indexed only; targets null, no holdout label stats, no model evaluation",
        "metadata_sha256": {name: file_sha256(metadata / name) for name in ("scene.json", "sample.json", "sample_data.json", "ego_pose.json", "calibrated_sensor.json", "sensor.json")}}
    write_json(output / "audit.json", audit)
    print(json.dumps({"dataset": str(output / "dataset.json"), "audit": str(output / "audit.json"), "counts": dict(counts)}, indent=2), flush=True)
    return output / "dataset.json"
