"""Label-free nuScenes camera chains, separate from any supervised dataset."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

from .common import code_version, object_hash, read_json, sha256, write_json
from .data import iter_json_array


SCHEMA = "v5_nuscenes_flow_index_v1"
GROUP_SCHEMA = "v5_nuscenes_flow_group_v1"
VERSIONS = {"v1.0-trainval", "v1.0-test"}
CAMERAS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
TABLES = ("scene.json", "sample.json", "sample_data.json", "sensor.json", "calibrated_sensor.json")
PARTITIONS = {"train_core", "dev", "stage1_test", "final_system_test", "unassigned_cache_only", "official_test_cache_only"}
GROUP_KEYS = ("version", "camera", "scene", "partition", "cache_only")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _choices(values, allowed, name):
    require(isinstance(values, (list, tuple)) and values and len(values) == len(set(values))
            and set(values) <= allowed, f"Explicit nonempty unique {name} are required")
    return sorted(values)


def _table(path):
    rows = {}
    for row in iter_json_array(path):
        token = row.get("token")
        require(isinstance(token, str) and token and token not in rows, f"Missing/duplicate token in {path}")
        rows[token] = row
    return rows


def _image(root, filename):
    require(isinstance(filename, str), "Image filename must be a string")
    relative = Path(filename)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe image path")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), f"Missing/unsafe image: {filename}")
    return str(path)


def _partition_map(manifest):
    require(manifest.get("schema") == "v5_dataset_v1" and manifest.get("version") == "v1.0-trainval",
            "Partition source must be the original trainval V5 manifest")
    result = {}
    for row in manifest["sequences"]:
        scene, partition = row["scene"], row["partition"]
        require(scene not in result and partition in PARTITIONS - {"unassigned_cache_only", "official_test_cache_only"},
                "Original partition mapping contains duplicate scenes or unknown partitions")
        result[scene] = partition
    require(result, "Original partition mapping is empty")
    return result


def _partition(version, scene, mapping):
    return "official_test_cache_only" if version == "v1.0-test" else mapping.get(scene, "unassigned_cache_only")


def _counts(groups):
    return dict(groups=len(groups), frames=sum(g["frame_count"] for g in groups), pairs=sum(g["pair_count"] for g in groups))


def build_index(dataroot, original_manifest, output, *, versions, cameras):
    """Write a new index directory; never read poses, targets or window labels."""
    versions = _choices(versions, VERSIONS, "versions")
    cameras = _choices(cameras, CAMERAS, "cameras")
    root, original, output = Path(dataroot).resolve(strict=True), Path(original_manifest).resolve(strict=True), Path(output).resolve()
    manifest = read_json(original)
    require(Path(manifest["dataroot"]).resolve(strict=True) == root, "Partition source has another dataroot")
    mapping = _partition_map(manifest)
    if output.exists():
        raise FileExistsError(f"Use a new full-flow index directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    groups, metadata, global_tokens = [], {}, set()
    for version in versions:
        base = root / version
        metadata[version] = {table: sha256(base / table) for table in TABLES}
        scenes = _table(base / "scene.json")
        require(len({s["name"] for s in scenes.values()}) == len(scenes), "Duplicate scene names")
        if version == manifest["version"]:
            require(set(mapping) <= {s["name"] for s in scenes.values()}, "Original partition scenes absent from metadata")
        samples = _table(base / "sample.json")
        require(all(s["scene_token"] in scenes for s in samples.values()), "Sample references missing scene")
        sensors = _table(base / "sensor.json")
        calibrated = _table(base / "calibrated_sensor.json")
        require(all(c["sensor_token"] in sensors for c in calibrated.values()), "Calibration references missing sensor")
        channel_for = {token: sensors[row["sensor_token"]]["channel"] for token, row in calibrated.items()
                       if sensors[row["sensor_token"]]["channel"] in cameras}
        require(set(channel_for.values()) == set(cameras), "A requested camera is absent from sensor calibration")
        records, buckets = {}, defaultdict(list)
        for row in iter_json_array(base / "sample_data.json"):
            channel = channel_for.get(row["calibrated_sensor_token"])
            if channel is None:
                continue
            token = row["token"]
            require(isinstance(token, str) and token and token not in global_tokens, "Duplicate selected sample_data token")
            require(row["sample_token"] in samples, "Camera sample_data references missing sample")
            scene = scenes[samples[row["sample_token"]]["scene_token"]]["name"]
            require((row["height"], row["width"]) == (900, 1600), "Unexpected camera metadata dimensions")
            require(isinstance(row["timestamp"], int) and not isinstance(row["timestamp"], bool), "Invalid camera timestamp")
            _image(root, row["filename"])
            global_tokens.add(token)
            # Deliberately discard ego_pose_token and all other unused fields.
            records[token] = dict(token=token, time_s=row["timestamp"] / 1e6, filename=row["filename"],
                                  prev=row["prev"], next=row["next"], scene=scene, camera=channel)
            buckets[(channel, scene)].append(token)
        require(set(buckets) == {(camera, s["name"]) for camera in cameras for s in scenes.values()},
                "Requested scene/camera coverage is incomplete")
        for (camera, scene), tokens in sorted(buckets.items()):
            token_set = set(tokens)
            starts = [token for token in tokens if not records[token]["prev"]]
            require(len(starts) == 1, "Camera scene must contain one connected chain start")
            for token in tokens:
                row = records[token]
                for direction, reverse in (("next", "prev"), ("prev", "next")):
                    other = row[direction]
                    if other:
                        require(other in token_set and records[other][reverse] == token,
                                "Missing, cross-scene/camera or nonreciprocal link")
            ordered, seen, token = [], set(), starts[0]
            while token:
                require(token not in seen, "Cyclic camera chain")
                row = records[token]
                require(not ordered or row["time_s"] > ordered[-1]["time_s"], "Nonpositive camera pair time interval")
                seen.add(token)
                ordered.append({key: row[key] for key in ("token", "time_s", "filename")})
                token = row["next"]
            require(seen == token_set and len(ordered) >= 2, "Disconnected or too-short camera chain")
            partition = _partition(version, scene, mapping)
            group = dict(version=version, camera=camera, scene=scene, partition=partition,
                         cache_only=partition not in {"train_core", "dev"})
            # Scene names originate in metadata, but still must not escape output.
            require(Path(scene).name == scene and scene not in {"", ".", ".."}, "Unsafe scene name")
            filename = f"groups/{version}/{camera}/{scene}.json"
            write_json(output / filename, dict(schema=GROUP_SCHEMA, **group, frames=ordered))
            groups.append(dict(group, frame_count=len(ordered), pair_count=len(ordered)-1,
                               file=filename, sha256=sha256(output / filename)))
    groups.sort(key=lambda g: (g["version"], g["camera"], g["scene"]))
    value = dict(schema=SCHEMA, purpose="label_free_all_adjacent_camera_pairs", dataroot=str(root),
                 versions=versions, cameras=cameras, metadata_sha256=metadata,
                 generation=dict(code=code_version(require_pushed=False), implementation_sha256=sha256(__file__),
                                 kind="index_only_no_raft_or_labels"),
                 partition_source=dict(manifest=str(original), manifest_sha256=sha256(original),
                                       split_sha256=manifest["split_sha256"], role="partition_mapping_only"),
                 boundary=dict(labels="not_read_or_generated", poses="not_read", training_dataset="unchanged",
                               cache_only_partitions="never_training_or_model_selection"),
                 groups=groups, counts=_counts(groups))
    value["identity"] = object_hash(value)
    write_json(output / "index.json", value)
    return output / "index.json"


def load_index(path, *, verify_sources=True):
    path = Path(path).resolve(strict=True)
    value = read_json(path)
    require(value.get("schema") == SCHEMA and value.get("purpose") == "label_free_all_adjacent_camera_pairs"
            and value.get("identity") == object_hash({k: v for k, v in value.items() if k != "identity"}),
            "Full-flow index schema/purpose/identity mismatch")
    require(value["versions"] == _choices(value["versions"], VERSIONS, "versions")
            and value["cameras"] == _choices(value["cameras"], CAMERAS, "cameras"), "Unsorted index scope")
    root = Path(value["dataroot"]).resolve(strict=True)
    groups = value["groups"]
    keys = [(g["version"], g["camera"], g["scene"]) for g in groups]
    require(keys == sorted(set(keys)) and bool(keys), "Unsorted/duplicate/empty index groups")
    require(value["counts"] == _counts(groups), "Index aggregate counts mismatch")
    require(set(value["metadata_sha256"]) == set(value["versions"]), "Index metadata versions differ")
    for group in groups:
        require(group["version"] in value["versions"] and group["camera"] in value["cameras"]
                and group["partition"] in PARTITIONS and isinstance(group["cache_only"], bool)
                and group["cache_only"] == (group["partition"] not in {"train_core", "dev"}), "Invalid group scope/partition")
        require(isinstance(group["frame_count"], int) and not isinstance(group["frame_count"], bool)
                and group["frame_count"] >= 2 and group["pair_count"] == group["frame_count"]-1, "Invalid group pair/frame counts")
        filename = Path(group["file"])
        require(not filename.is_absolute() and ".." not in filename.parts, "Unsafe group path")
    if verify_sources:
        source = value["partition_source"]
        require(source["role"] == "partition_mapping_only" and sha256(source["manifest"]) == source["manifest_sha256"],
                "Original partition manifest changed")
        manifest = read_json(source["manifest"])
        require(manifest["split_sha256"] == source["split_sha256"] and Path(manifest["dataroot"]).resolve(strict=True) == root,
                "Original split/dataroot differs")
        mapping = _partition_map(manifest)
        for group in groups:
            require(group["partition"] == _partition(group["version"], group["scene"], mapping), "Group changed original partition")
        for version, tables in value["metadata_sha256"].items():
            require(set(tables) == set(TABLES), "Unexpected source metadata tables")
            for name, digest in tables.items():
                require(sha256(root / version / name) == digest, "Source metadata changed")
    return value


def iter_groups(index, *, shard_index=0, shard_count=1):
    require(isinstance(shard_count, int) and not isinstance(shard_count, bool) and shard_count > 0
            and isinstance(shard_index, int) and not isinstance(shard_index, bool) and 0 <= shard_index < shard_count,
            "Invalid full-flow shard index/count")
    yield from index["groups"][shard_index::shard_count]


def iter_pairs(index_path, *, shard_index=0, shard_count=1, verify_sources=True):
    """Yield absolute-path frame pairs and group metadata, including final edges."""
    path = Path(index_path).resolve(strict=True)
    index = load_index(path, verify_sources=verify_sources)
    root, seen = Path(index["dataroot"]).resolve(strict=True), set()
    for group in iter_groups(index, shard_index=shard_index, shard_count=shard_count):
        group_path = (path.parent / group["file"]).resolve(strict=True)
        require(group_path.is_relative_to(path.parent) and sha256(group_path) == group["sha256"], "Group path/checksum mismatch")
        payload = read_json(group_path)
        require(set(payload) == {"schema", "frames", *GROUP_KEYS} and payload["schema"] == GROUP_SCHEMA
                and all(payload[k] == group[k] for k in GROUP_KEYS), "Group identity or field boundary mismatch")
        frames = payload["frames"]
        require(len(frames) == group["frame_count"], "Group frame count mismatch")
        previous = None
        for frame in frames:
            require(set(frame) == {"token", "time_s", "filename"}, "Frame contains nonvisual or unknown fields")
            require(isinstance(frame["token"], str) and frame["token"] and frame["token"] not in seen, "Duplicate frame token")
            require(isinstance(frame["time_s"], (int, float)) and not isinstance(frame["time_s"], bool)
                    and math.isfinite(frame["time_s"]), "Nonfinite/invalid frame time")
            seen.add(frame["token"])
            current = dict(frame, filename=_image(root, frame["filename"]), scene=group["scene"], camera=group["camera"],
                           dataset="nuScenes", version=group["version"])
            if previous is not None:
                require(current["time_s"] > previous["time_s"], "Nonpositive pair interval")
                yield previous, current, group
            previous = current
