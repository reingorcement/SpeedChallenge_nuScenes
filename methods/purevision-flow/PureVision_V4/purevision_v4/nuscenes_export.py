"""Offline nuScenes conversion. Pose data creates labels, never visual inputs."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .io import ROOT, artifact_path, object_hash, read_json, sha256, target_spec, write_json


def quaternion_matrix(wxyz):
    q = np.asarray(wxyz, float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def body_velocity_labels(times, positions, rotations):
    """Full 3D derivative, expressed in the reference body axes at each time.

    Unequal sampling is supported. Labels may use neighboring poses offline;
    no pose or future image is fed into inference.
    """
    times, positions, rotations = (np.asarray(x, float) for x in (times, positions, rotations))
    if len(times) < 3 or positions.shape != (len(times), 3) or rotations.shape != (len(times), 3, 3):
        raise ValueError("At least three 3D poses are required")
    if not np.isfinite(times).all() or not np.isfinite(positions).all() or not np.isfinite(rotations).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Pose times and values must be finite and ordered")
    global_velocity = np.gradient(positions, times, axis=0, edge_order=2)
    return np.einsum("nji,nj->ni", rotations, global_velocity)


def export(dataroot, output, *, version="v1.0-trainval", split_manifest=None,
           partitions=("train_core", "dev", "stage1_test"), camera="CAM_FRONT", camera_height=1.51, limit=0):
    from nuscenes.nuscenes import NuScenes
    if camera_height <= 0 or not np.isfinite(camera_height) or limit < 0:
        raise ValueError("Invalid camera height or scene limit")
    output = artifact_path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export output must be new")
    split_path = Path(split_manifest or ROOT / "configs/nuscenes_splits.json")
    split = read_json(split_path)
    # Verify the original split checksum and prevent accidental partition overlap.
    import hashlib
    import json
    payload = {k: v for k, v in split.items() if k != "checksum_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    if digest != split.get("checksum_sha256") or split.get("version") != version:
        raise ValueError("nuScenes split checksum/version mismatch")
    if not partitions or set(partitions) - {"train_core", "dev", "stage1_test"}:
        raise ValueError("Use train_core/dev/stage1_test; the final system holdout stays sealed")
    seen = set()
    for values in split["partitions"].values():
        if seen & set(values) or len(values) != len(set(values)):
            raise ValueError("Overlapping scene partitions")
        seen.update(values)
    nusc = NuScenes(version=version, dataroot=str(Path(dataroot).resolve()), verbose=False)
    records = []
    for partition in partitions:
        names = set(split["partitions"][partition])
        scenes = [s for s in nusc.scene if s["name"] in names]
        if limit:
            scenes = scenes[:limit]
        for scene in scenes:
            sample = nusc.get("sample", scene["first_sample_token"])
            token = sample["data"][camera]
            frames, positions, rotations, times = [], [], [], []
            fixed = None
            while token:
                sd = nusc.get("sample_data", token)
                sensor = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                c = quaternion_matrix(sensor["rotation"])
                cal = dict(image_size=[sd["width"], sd["height"]], K=sensor["camera_intrinsic"], height_m=camera_height,
                           plane_normal_camera=(c.T @ [0., 0., 1.]).tolist(), rotation_body_from_camera=c.tolist(),
                           camera_lever_body_m=sensor["translation"], reference_point="ego_origin")
                if fixed is not None and object_hash(fixed) != object_hash(cal):
                    raise ValueError("Camera calibration changes within a sequence")
                fixed = cal
                time = sd["timestamp"] / 1e6
                if abs(pose["timestamp"] / 1e6 - time) > 1e-3:
                    raise ValueError("Pose/camera time mismatch requires explicit interpolation before export")
                frames.append(dict(image_id=len(frames), file=sd["filename"], time_s=time))
                times.append(time)
                positions.append(pose["translation"])
                rotations.append(quaternion_matrix(pose["rotation"]))
                token = sd.get("next", "")
            if len(frames) < 5:
                continue
            velocity = body_velocity_labels(times, positions, rotations)
            input_path = output / "input" / scene["name"]
            labels_path = output / "labels" / f"{scene['name']}.csv"
            write_json(input_path / "frames.json", frames)
            write_json(input_path / "calibration.json", fixed)
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dict(time=times, vx=velocity[:, 0], vy=velocity[:, 1], vz=velocity[:, 2],
                              roll=Rotation.from_matrix(np.array(rotations)).as_euler("xyz")[:, 0])).to_csv(labels_path, index=False)
            spec = target_spec(fixed)
            write_json(labels_path.with_suffix(".meta.json"), dict(**spec, derivation="full 3D nonuniform quadratic derivative; second-order one-sided endpoints",
                       source="nuScenes ego pose, offline labels only", frame_cache_version="v4_body_reference_v1"))
            records.append(dict(case=scene["name"], partition=partition, input=str(input_path),
                                image_root=str(Path(dataroot).resolve()), labels=str(labels_path)))
            print(f"Exported {partition}/{scene['name']} ({len(frames)} frames)", flush=True)
    if not records:
        raise ValueError("No scenes exported")
    manifest = dict(schema="sequence_manifest_v1", sequences=records, split_sha256=sha256(split_path),
                    dataset=version, target_version="body_reference_v1", input_images_copied=False)
    write_json(output / "dataset.json", manifest)
    return output / "dataset.json"
