import hashlib
import json

import numpy as np
import pytest

from purevision_v5.data import iter_json_array, prepare_dataset
from purevision_v5.labels import LABEL_CONTRACT, planar_velocity_labels, synthetic_audit, yaw_from_quaternions


def test_known_motion_and_reference_point():
    cases = synthetic_audit()
    for name in ("straight", "leftward", "rightward"):
        np.testing.assert_allclose(cases[name]["new"], cases[name]["expected"], atol=1e-12)
    assert abs(cases["no_slip_circle"]["legacy"][1]) > .14
    assert abs(cases["no_slip_circle"]["new"][1]) < 1e-4
    np.testing.assert_allclose(cases["lever_arm"]["new_camera_minus_reference"], [0., .8], atol=2e-4)
    assert LABEL_CONTRACT["reference_point"] == "nuscenes_rear_axle_midpoint"
    assert LABEL_CONTRACT["mode"] == "PLANAR_PROXY"


def test_nonuniform_derivative_exact_quadratic_and_invalid_endpoints():
    t = np.array([0., .05, .15, .19, .30])
    p = np.stack((3*t + 2*t*t, -t + 4*t*t), axis=1)
    v = planar_velocity_labels(t, p, np.zeros_like(t))
    np.testing.assert_allclose(v[1:-1], np.stack((3+4*t, -1+8*t), axis=1)[1:-1], atol=1e-12)
    assert np.isnan(v[[0, -1]]).all()


@pytest.mark.parametrize("times", [[0., 0., 1.], [0., 2., 1.], [0., np.nan, 1.]])
def test_invalid_time_rejected(times):
    with pytest.raises(ValueError):
        planar_velocity_labels(times, np.zeros((3, 2)), np.zeros(3))


def test_yaw_axes():
    yaw = yaw_from_quaternions([[np.sqrt(.5), 0, 0, np.sqrt(.5)]] * 3)
    t = np.array([0., .05, .15])
    p = np.stack((np.zeros(3), 2*t), axis=1)
    np.testing.assert_allclose(planar_velocity_labels(t, p, yaw)[1], [2., 0.], atol=1e-12)


def test_streamed_json_across_small_chunks(tmp_path):
    records = [{"token": "a" * 1000, "n": i} for i in range(4)]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(records))
    assert list(iter_json_array(path, chunk_size=17)) == records
    path.write_text('[{"token":"incomplete"}')
    with pytest.raises(ValueError):
        list(iter_json_array(path, chunk_size=17))


def test_export_causal_windows_field_isolation_and_sealed_holdouts(tmp_path):
    root = tmp_path / "raw"
    table_root = root / "v1.0-trainval"
    table_root.mkdir(parents=True)
    parts = {name: [f"scene-{i}"] for i, name in enumerate(("train_core", "dev", "stage1_test", "final_system_test"))}
    split = {"version": "v1.0-trainval", "partitions": parts}
    split["checksum_sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    scenes, samples, sd, poses = [], [], [], []
    for i, names in enumerate(parts.values()):
        scene = names[0]
        scenes.append({"name": scene, "token": scene, "log_token": f"log-{i}", "first_sample_token": f"sample-{i}-0"})
        for j in range(7):
            token, sample = f"frame-{i}-{j}", f"sample-{i}-{j}"
            time_us = 1_000_000 + j * 100_000
            filename = f"{token}.jpg"
            (root / filename).write_bytes(b"fixture_not_decoded")
            samples.append({"token": sample, "scene_token": scene})
            sd.append({"token": token, "sample_token": sample, "ego_pose_token": token,
                "calibrated_sensor_token": "cal", "timestamp": time_us, "filename": filename,
                "height": 900, "width": 1600, "is_key_frame": True,
                "prev": f"frame-{i}-{j-1}" if j else "", "next": f"frame-{i}-{j+1}" if j < 6 else ""})
            poses.append({"token": token, "timestamp": time_us, "translation": [j, 0., 0.], "rotation": [1., 0., 0., 0.]})
    tables = {"scene": scenes, "sample": samples, "sample_data": sd, "ego_pose": poses,
        "sensor": [{"token": "sensor", "channel": "CAM_FRONT"}],
        "calibrated_sensor": [{"token": "cal", "sensor_token": "sensor"}]}
    for name, records in tables.items():
        (table_root / f"{name}.json").write_text(json.dumps(records))
    output = tmp_path / "prepared"
    manifest = json.loads(prepare_dataset(root, split_path, output).read_text())
    assert manifest["seq_len"] == 5
    assert manifest["counts"]["train_core"]["windows"] == 2
    for entry in manifest["sequences"]:
        sequence = json.loads((output / entry["file"]).read_text())
        assert sequence["windows"] == [4, 5]
        for frame in sequence["frames"]:
            assert set(frame) == {"token", "time_s", "filename", "target"}
        if entry["partition"] in ("stage1_test", "final_system_test"):
            assert not sequence["labels_available"]
            assert all(f["target"] is None for f in sequence["frames"])
        else:
            np.testing.assert_allclose(sequence["frames"][4]["target"], [10., 0.])
            assert sequence["frames"][0]["target"] is None
            assert sequence["frames"][-1]["target"] is None
        for end in sequence["windows"]:
            current_time = sequence["frames"][end]["time_s"]
            assert all(f["time_s"] <= current_time for f in sequence["frames"][end-4:end+1])
    with pytest.raises(FileExistsError):
        prepare_dataset(root, split_path, output)
