import json
from pathlib import Path

import pytest

from purevision_v5.common import object_hash, read_json, sha256, write_json
from purevision_v5.full_flow_index import build_index, iter_groups, iter_pairs, load_index


@pytest.fixture
def raw_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr("purevision_v5.full_flow_index.code_version", lambda **kw: dict(commit="fixture", dirty=True, verified_remote_commit=None))
    root = tmp_path / "raw"
    cameras = ["CAM_FRONT", "CAM_BACK"]
    for version, names in (("v1.0-trainval", ["scene-a", "scene-b", "scene-c"]), ("v1.0-test", ["scene-test"])):
        scenes, samples, frames = [], [], []
        for scene in names:
            scenes.append(dict(token=scene, name=scene))
            samples.append(dict(token=scene + "-sample", scene_token=scene))
            for camera in cameras:
                for index in range(3):
                    token = f"{scene}-{camera}-{index}"
                    filename = f"samples/{camera}/{token}.jpg"
                    path = root / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"header_decoding_is_generator_responsibility")
                    frames.append(dict(token=token, sample_token=scene + "-sample", calibrated_sensor_token=camera,
                                       timestamp=1_000_000 + 100_000 * index, filename=filename, height=900, width=1600,
                                       prev=f"{scene}-{camera}-{index-1}" if index else "",
                                       next=f"{scene}-{camera}-{index+1}" if index < 2 else "",
                                       ego_pose_token="deliberately_missing_pose"))
        tables = dict(scene=scenes, sample=samples, sample_data=frames,
                      sensor=[dict(token=c, channel=c) for c in cameras],
                      calibrated_sensor=[dict(token=c, sensor_token=c) for c in cameras])
        for name, records in tables.items():
            write_json(root / version / f"{name}.json", records)
    manifest = tmp_path / "original.json"
    write_json(manifest, dict(schema="v5_dataset_v1", version="v1.0-trainval", dataroot=str(root), split_sha256="a" * 64,
                              sequences=[dict(scene="scene-a", partition="train_core"), dict(scene="scene-b", partition="final_system_test")]))
    return root, manifest, tmp_path / "output"


def build(fixture, **kwargs):
    return build_index(*fixture, versions=kwargs.pop("versions", ["v1.0-trainval", "v1.0-test"]),
                       cameras=kwargs.pop("cameras", ["CAM_FRONT", "CAM_BACK"]), **kwargs)


def mutate_table(root, name, edit):
    path = root / "v1.0-trainval" / f"{name}.json"
    value = read_json(path)
    edit(value)
    write_json(path, value)


def test_all_adjacent_edges_versions_cameras_and_label_isolation(raw_dataset):
    root, original, output = raw_dataset
    original_sha = sha256(original)
    path = build(raw_dataset)
    index = load_index(path)
    assert index["counts"] == dict(groups=8, frames=24, pairs=16)
    assert index["generation"]["code"]["dirty"] is True
    assert sha256(original) == original_sha
    assert index["partition_source"]["manifest_sha256"] == original_sha
    assert all("ego_pose.json" not in tables for tables in index["metadata_sha256"].values())
    assert not list(root.glob("*/ego_pose.json"))
    groups = index["groups"]
    assert [(g["version"], g["camera"], g["scene"]) for g in groups] == sorted((g["version"], g["camera"], g["scene"]) for g in groups)
    expected = {"scene-a": ("train_core", False), "scene-b": ("final_system_test", True),
                "scene-c": ("unassigned_cache_only", True), "scene-test": ("official_test_cache_only", True)}
    for group in groups:
        assert (group["partition"], group["cache_only"]) == expected[group["scene"]]
        payload = read_json(output / group["file"])
        assert all(set(frame) == {"token", "time_s", "filename"} for frame in payload["frames"])
    pairs = list(iter_pairs(path))
    assert len(pairs) == 16
    for source, target, group in pairs:
        assert source["scene"] == target["scene"] == group["scene"]
        assert source["camera"] == target["camera"] == group["camera"]
        assert source["version"] == target["version"] == group["version"]
        assert Path(source["filename"]).is_absolute() and Path(target["filename"]).is_file()
        assert target["time_s"] > source["time_s"]
    # Each final frame is included, although no future-neighbor label can exist.
    assert sum(target["token"].endswith("-2") for _, target, _ in pairs) == 8
    shards = [list(iter_pairs(path, shard_index=i, shard_count=3)) for i in range(3)]
    assert sum(map(len, shards)) == len(pairs)
    assert len({(a["token"], b["token"]) for shard in shards for a, b, _ in shard}) == len(pairs)
    for i, shard in enumerate(shards):
        assert len(shard) == sum(g["pair_count"] for g in iter_groups(index, shard_index=i, shard_count=3))
    with pytest.raises(FileExistsError):
        build(raw_dataset)


def test_explicit_front_only_scope(raw_dataset):
    path = build(raw_dataset, cameras=["CAM_FRONT"])
    index = load_index(path)
    assert index["cameras"] == ["CAM_FRONT"] and index["counts"]["pairs"] == 8
    assert {g["camera"] for g in index["groups"]} == {"CAM_FRONT"}


@pytest.mark.parametrize("kwargs", [dict(cameras=[]), dict(cameras=["CAM_FRONT", "CAM_FRONT"]),
                                    dict(cameras=["LIDAR_TOP"]), dict(versions=[]), dict(versions=["v1.0-mini"])])
def test_explicit_valid_scope_required(raw_dataset, kwargs):
    with pytest.raises(ValueError):
        build(raw_dataset, **kwargs)


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0].update(next="scene-b-CAM_FRONT-1"),
    lambda rows: rows[1].update(prev=""),
    lambda rows: rows[0].update(next="scene-a-CAM_BACK-1"),
    lambda rows: rows[1].update(timestamp=1_000_000),
    lambda rows: rows[0].update(width=800),
    lambda rows: rows.append(dict(rows[0])),
    lambda rows: rows.pop(2),
])
def test_reject_incomplete_invalid_or_cross_boundary_chain(raw_dataset, mutation):
    mutate_table(raw_dataset[0], "sample_data", mutation)
    with pytest.raises(ValueError):
        build(raw_dataset)
    assert not (raw_dataset[2] / "index.json").exists()


def test_missing_image_fails_explicitly(raw_dataset):
    next(raw_dataset[0].glob("samples/CAM_FRONT/*.jpg")).unlink()
    with pytest.raises(FileNotFoundError):
        build(raw_dataset)


def test_changed_source_manifest_or_metadata_rejected(raw_dataset):
    path = build(raw_dataset)
    original = raw_dataset[1]
    old = original.read_bytes()
    original.write_bytes(old + b" ")
    with pytest.raises(ValueError, match="manifest changed"):
        load_index(path)
    original.write_bytes(old)
    table = raw_dataset[0] / "v1.0-test/sample_data.json"
    table.write_bytes(table.read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata changed"):
        load_index(path)


def test_changed_group_and_frame_field_boundary_rejected(raw_dataset):
    path = build(raw_dataset)
    index = load_index(path)
    group = index["groups"][0]
    gp = path.parent / group["file"]
    value = read_json(gp)
    value["frames"][0]["target"] = [0, 0]
    write_json(gp, value)
    with pytest.raises(ValueError, match="checksum"):
        list(iter_pairs(path))
    # Even a recomputed checksum cannot authorize label fields in a visual index.
    group["sha256"] = sha256(gp)
    index["identity"] = object_hash({k: v for k, v in index.items() if k != "identity"})
    write_json(path, index)
    with pytest.raises(ValueError, match="nonvisual"):
        list(iter_pairs(path))


def test_original_partition_cannot_be_relabelled_cache_train(raw_dataset):
    path = build(raw_dataset)
    index = load_index(path)
    group = next(g for g in index["groups"] if g["partition"] == "final_system_test")
    group.update(partition="train_core", cache_only=False)
    index["identity"] = object_hash({k: v for k, v in index.items() if k != "identity"})
    write_json(path, index)
    with pytest.raises(ValueError, match="original partition"):
        load_index(path)


@pytest.mark.parametrize("shard_index,shard_count", [(0, 0), (3, 3), (-1, 2), (True, 2), (0, False)])
def test_invalid_shards_rejected(shard_index, shard_count):
    with pytest.raises(ValueError):
        list(iter_groups({"groups": []}, shard_index=shard_index, shard_count=shard_count))
