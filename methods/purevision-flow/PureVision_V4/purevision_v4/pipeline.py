"""Causal image -> flow -> normal -> finite geometry -> optional trained fusion."""
from __future__ import annotations

from collections import deque
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import random
import time

import cv2
import numpy as np
import scipy
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as TF

from .flow import RaftFlow, atomic_torch_save
from .geometry import aggregate, invalid, solve_pair, to_body
from .io import ROOT, artifact_path, code_hashes, intrinsics, load_config, load_sequence, object_hash, read_json, sha256, target_spec, write_json
from .model import FEATURE_NAMES, FrozenVisual, geometry_features, load_fusion
from .normals import NormalTracker, image_texture_std, road_mask, select_normal, supported_flow

METHODS = ("F_RC", "F_R", "F_B", "fusion", "rgb_only", "legacy_v3", "rgb_legacy")
SPEED_COLUMNS = ["case", "method", "image_id", "output_index", "time_s", "effective_time_s", "measurement_age_s",
                 "valid", "valid_pairs", "reason", "vx_mps", "vy_mps", "vz_mps", "speed_mps", "normal_status",
                 "normal_reason", "geometry_valid", "target_version", "reference_point", "time_basis",
                 "inference_ms", "shared_ms", "frame_total_ms"]
PAIR_COLUMNS = ["case", "method", "output_index", "source_id", "target_id", "source_time_s", "target_time_s",
                "valid", "reason", "count", "condition", "residual", "dt", "support_fraction",
                "nx", "ny", "nz", "vx", "vy", "vz", "wx", "wy", "wz"]


def feature_contract(config, visual_digest, raft_digest):
    hashes = code_hashes()
    hashes = {k: v for k, v in hashes.items() if k.split("/")[-1] not in {"training.py", "evaluation.py", "nuscenes_export.py"}}
    return dict(schema="frozen_features_v1", features=list(FEATURE_NAMES), visual_weights=visual_digest,
                raft_weights=raft_digest, algorithm=config, code=object_hash(hashes), geometry_method="F_RC")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _legacy_input(images, packets, cal, size, device):
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)[None]
    return dict(images=images, flows=torch.stack([p["flow"] for p in packets])[None].float().to(device),
                flow_confidence=torch.stack([p["confidence"] for p in packets])[None].float().to(device),
                pair_dt=tensor([p["dt"] for p in packets]), intrinsics=tensor(intrinsics(cal, size, legacy=True)),
                plane_normal_camera=tensor(cal["plane_normal_camera"]), plane_offset=tensor(cal["height_m"]),
                rotation_ego_from_camera=tensor(cal.get("legacy_rotation_ego_from_camera", cal["rotation_body_from_camera"])),
                translation_ego_from_camera=tensor(cal.get("legacy_translation_ego_from_camera", cal["camera_lever_body_m"])))


def run(input_path, output, *, methods=("F_RC", "F_R"), device="cpu", config_path=None,
        fusion_checkpoint=None, start=0, count=0, image_root=None, resume=False, save_features=False, threads=4):
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError("Methods must be unique supported names")
    if {"fusion", "rgb_only"} & set(methods) and fusion_checkpoint is None:
        raise ValueError("fusion/rgb_only require a trained V4 checkpoint; use rgb_legacy for the old RGB baseline")
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    cv2.setRNGSeed(42)
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Unavailable device")
    config = load_config(config_path)
    out = artifact_path(output)
    raw = Path(input_path).resolve()
    if out == raw or out.is_relative_to(raw) or raw.is_relative_to(out):
        raise ValueError("Input and output must not contain one another")
    frames, cal = load_sequence(raw, start, count, image_root)
    if any(Path(f["file"]).is_relative_to(out) for f in frames):
        raise ValueError("Output contains input images")
    weights = {"visual": ROOT / "weights/v3_fusion_seed42.pth", "raft": ROOT / "weights/raft_large_C_T_SKHT_V2.pth"}
    digests = {key: sha256(path) for key, path in weights.items()}
    contract = feature_contract(config, digests["visual"], digests["raft"])
    target = target_spec(cal)
    fusion = None
    if fusion_checkpoint is not None:
        fusion, package = load_fusion(fusion_checkpoint, contract, target)
        if "fusion" in methods and package["stage"] != "fusion":
            raise ValueError("A direct-only checkpoint cannot be used for fusion")
        fusion.to(device)
        digests["fusion"] = sha256(fusion_checkpoint)
    environment = dict(python=platform.python_version(), torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                       numpy=np.__version__, scipy=scipy.__version__, opencv=cv2.__version__, device=str(device),
                       gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None, threads=threads)
    identity = dict(frames=frames, calibration=cal, config=config, methods=list(methods), weights=digests,
                    environment=environment, code=code_hashes(), save_features=save_features)
    signature = object_hash(identity)
    metadata_path = out / "run.json"
    if out.exists() and any(out.iterdir()):
        if not resume or not metadata_path.is_file():
            raise ValueError("Output is not empty; select a new output or resume the identical run")
        old = read_json(metadata_path)
        if old["signature"] != signature:
            raise ValueError("Resume input/config/code/weights/environment mismatch")
        if old["status"] == "complete":
            if not all((out / p).is_file() and sha256(out / p) == digest for p, digest in old["result_hashes"].items()):
                raise ValueError("Completed artifacts changed")
            return old
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".running"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    metadata = dict(status="running", signature=signature, identity=identity, target_spec=target,
                    started_utc=datetime.now(timezone.utc).isoformat(), allowed_inputs=["RGB", "timestamps", "fixed calibration"],
                    geometry_time="mean of valid pair midpoints; output frame time retained", feature_contract=contract)
    try:
        write_json(metadata_path, metadata)
        result = _infer(frames, cal, config, out, raw.name, methods, weights, device, fusion, save_features, contract, target)
        files = [out / "speeds.csv", out / "pairs.csv", out / "normals.json"]
        if save_features:
            files.extend(sorted((out / "features").glob("*.pt")))
            files.append(out / "features/manifest.json")
        metadata.update(status="complete", elapsed_s=time.perf_counter() - started, **result,
                        result_hashes={p.relative_to(out).as_posix(): sha256(p) for p in files})
        write_json(metadata_path, metadata)
        return metadata
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_s=time.perf_counter() - started)
        write_json(metadata_path, metadata)
        raise
    finally:
        lock.unlink(missing_ok=True)


def _infer(frames, cal, config, out, case, methods, weights, device, fusion, save_features, contract, target):
    visual = FrozenVisual(weights["visual"]).to(device)
    flow_engine = RaftFlow(weights["raft"], device, config["preprocessing"], cal["image_size"])
    size = tuple(config["preprocessing"]["geometry_size"])
    k = intrinsics(cal, size)
    tracker = NormalTracker(k, cal["plane_normal_camera"], config["normal"])
    images, packets, normal_window = deque(maxlen=5), deque(maxlen=3), deque(maxlen=3)
    image_validity = deque(maxlen=5)
    records, feature_index = [], []
    counts = {m: 0 for m in methods}
    previous_mask = None
    speed_tmp, pair_tmp = out / "speeds.csv.partial", out / "pairs.csv.partial"
    with speed_tmp.open("w", encoding="utf-8", newline="") as sf, pair_tmp.open("w", encoding="utf-8", newline="") as pf, torch.inference_mode():
        sw, pw = csv.DictWriter(sf, SPEED_COLUMNS), csv.DictWriter(pf, PAIR_COLUMNS)
        sw.writeheader()
        pw.writeheader()
        for i, frame in enumerate(frames):
            _sync(device)
            frame_started = time.perf_counter()
            with Image.open(frame["file"]) as image:
                rgb = image.convert("RGB")
                image_validity.append(image_texture_std(np.asarray(rgb)) >= config["road"].get("min_texture_std", 1.))
                images.append(TF.pil_to_tensor(TF.resize(rgb, [72, 128])).float() / 127.5 - 1)
                current_mask = road_mask(cv2.resize(np.asarray(rgb), (size[1], size[0])), k, np.asarray(cal["plane_normal_camera"]), config["road"])
            if i:
                packet = flow_engine.pair(frames[i - 1], frame, out / "cache/flow" / f"{i - 1:06d}.pt")
                flow = packet["flow"].float().numpy()
                improved, support = supported_flow(flow, packet["confidence"][0].float().numpy(), previous_mask, current_mask, config["road"])
                normal = tracker.step(flow, improved, frames[i - 1]["image_id"], frame["image_id"], frames[i - 1]["time_s"], frame["time_s"])
                normal.update(support)
                records.append(normal)
                normal_window.append(normal)
                # Keep the historical float16 support cache precision for frozen reliability compatibility.
                packets.append(dict(packet, supported_confidence=improved.astype(np.float16).astype(np.float32)))
            previous_mask = current_mask
            base = dict(case=case, output_index=i, image_id=frame["image_id"], time_s=frame["time_s"],
                        target_version=target["version"], reference_point=cal["reference_point"], time_basis="output_frame")
            if i < 4:
                for m in methods:
                    sw.writerow(dict(base, method=m, valid=False, valid_pairs=0, geometry_valid=False, reason="warmup"))
                continue
            image_tensor = torch.stack(list(images))[None].to(device)
            descriptor, reliability = visual(image_tensor, size)
            visual_valid = bool(image_validity[-1] and image_validity[-2])
            reliability_np = reliability[0, :, 0].cpu().numpy()
            _sync(device)
            shared_ms = (time.perf_counter() - frame_started) * 1000
            required = set(methods) & {"F_RC", "F_R", "F_B"}
            if save_features or {"fusion", "rgb_only"} & set(methods):
                required.add("F_RC")
            geometry = {}
            timings = {}
            ps, ns = list(packets), list(normal_window)
            for method in sorted(required):
                tick = time.perf_counter()
                pairs = []
                for slot, packet in enumerate(ps):
                    n = select_normal(method, cal["plane_normal_camera"], ns[slot], packet["source_id"])
                    if packet["dt"] > config["aggregation"]["max_pair_dt_seconds"]:
                        pair = invalid("time_gap")
                        pair["twist_body"], pair["covariance_body"] = to_body(pair["twist"], pair["covariance"], cal)
                        pair.update(normal=n.tolist(), dt=packet["dt"], support_fraction=0.)
                    else:
                        pair = solve_pair(packet["flow"].float().numpy(), packet["supported_confidence"], reliability_np[slot],
                                          k, n, cal, packet["dt"], config["finite_solver"])
                    pairs.append(pair)
                    pair_row = {key: pair[key] for key in PAIR_COLUMNS if key in pair}
                    pair_row.update(case=case, method=method, output_index=i, source_id=packet["source_id"], target_id=packet["target_id"],
                                    source_time_s=ns[slot]["source_time_s"], target_time_s=ns[slot]["target_time_s"],
                                    nx=n[0], ny=n[1], nz=n[2])
                    pair_row.update({axis: float(pair["twist_body"][j]) if pair["valid"] else "" for j, axis in enumerate(["vx", "vy", "vz", "wx", "wy", "wz"])})
                    pw.writerow(pair_row)
                value = aggregate(pairs, [n["source_time_s"] for n in ns], [n["target_time_s"] for n in ns], frame["time_s"])
                geometry[method] = (value, pairs)
                timings[method] = (time.perf_counter() - tick) * 1000
            diagnostic = None
            if "F_RC" in geometry:
                value, pairs = geometry["F_RC"]
                diagnostic = geometry_features(value, pairs, ns)
                if save_features:
                    path = out / "features" / f"{i:06d}.pt"
                    atomic_torch_save(dict(descriptor=descriptor[0].cpu().clone(), diagnostic=torch.from_numpy(diagnostic),
                                          geometry_velocity=torch.tensor(value["velocity"][:2], dtype=torch.float32),
                                          geometry_valid=bool(value["valid"]), visual_valid=visual_valid,
                                          image_id=frame["image_id"], time_s=frame["time_s"]), path)
                    feature_index.append(dict(file=path.name, sha256=sha256(path), image_id=frame["image_id"],
                                              time_s=frame["time_s"], visual_valid=visual_valid))
            rows = []
            for method in methods:
                _sync(device)
                tick = time.perf_counter()
                row = dict(base, method=method, normal_status=ns[-1]["status"], normal_reason=ns[-1]["reason"], shared_ms=shared_ms)
                if method in geometry:
                    value, _ = geometry[method]
                    velocity = value["velocity"]
                    row.update(valid=value["valid"], geometry_valid=value["valid"], valid_pairs=value["valid_pairs"], reason=value["reason"],
                               effective_time_s=value["effective_time_s"], measurement_age_s=value["age_s"], time_basis="pair_midpoint_mean")
                    if method == "F_R":
                        row.update(normal_status="fixed", normal_reason="fixed_calibration")
                    elapsed = timings[method]
                elif method in {"fusion", "rgb_only"}:
                    value, _ = geometry["F_RC"]
                    output = fusion(descriptor, torch.from_numpy(diagnostic)[None].to(device),
                                    torch.tensor(value["velocity"][:2], device=device, dtype=torch.float32)[None],
                                    torch.tensor([value["valid"]], device=device), mode="fusion" if method == "fusion" else "direct")
                    velocity = output["prediction"][0].cpu().numpy()
                    row.update(valid=bool(np.isfinite(velocity).all()), geometry_valid=value["valid"], valid_pairs=value["valid_pairs"],
                               reason="rgb_fallback" if not value["valid"] and method == "fusion" else "ok",
                               effective_time_s=frame["time_s"], measurement_age_s=value["age_s"] if method == "fusion" else 0.)
                    _sync(device)
                    elapsed = (time.perf_counter() - tick) * 1000 + (timings["F_RC"] if method == "fusion" else 0)
                else:
                    if method == "rgb_legacy":
                        velocity = visual.legacy.direct_head(descriptor)[0].cpu().numpy()
                        good = False
                    else:
                        old = visual.legacy(_legacy_input(image_tensor, ps, cal, size, device))
                        velocity = old.predicted_velocity[0].cpu().numpy()
                        good = bool(old.geometry_valid[0])
                    row.update(valid=bool(np.isfinite(velocity).all()), geometry_valid=good, valid_pairs="", reason="legacy_target_baseline",
                               target_version="legacy_v3_yaw_level_v0", effective_time_s=frame["time_s"], measurement_age_s=0.)
                    _sync(device)
                    elapsed = (time.perf_counter() - tick) * 1000
                row["inference_ms"] = elapsed
                if not visual_valid:
                    row.update(valid=False, reason="textureless_input")
                for j, axis in enumerate(["vx", "vy", "vz"]):
                    row[axis + "_mps"] = float(velocity[j]) if row["valid"] and j < len(velocity) else ""
                row["speed_mps"] = float(np.linalg.norm(velocity[:2])) if row["valid"] else ""
                counts[method] += int(row["valid"])
                rows.append(row)
            _sync(device)
            total_ms = (time.perf_counter() - frame_started) * 1000
            for row in rows:
                sw.writerow(dict(row, frame_total_ms=total_ms))
            if i % 20 == 0 or i == len(frames) - 1:
                sf.flush()
                pf.flush()
                print(f"V4 {i + 1}/{len(frames)} {counts}", flush=True)
    speed_tmp.replace(out / "speeds.csv")
    pair_tmp.replace(out / "pairs.csv")
    write_json(out / "normals.json", records)
    if save_features:
        write_json(out / "features/manifest.json", dict(schema="frozen_features_v1", case=case,
                   sequence_hash=object_hash(frames), source_hashes=[f["sha256"] for f in frames],
                   feature_contract=contract, target_spec=target, windows=feature_index))
    return dict(coverage={m: dict(eligible=len(frames) - 4, valid=counts[m], coverage=counts[m] / (len(frames) - 4)) for m in methods},
                normal_status={s: sum(n["status"] == s for n in records) for s in ["update", "hold", "fixed_fallback"]})
