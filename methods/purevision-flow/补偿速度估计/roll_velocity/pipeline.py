"""Bounded-memory server runner reproducing corrected v2 F_RC geometry."""
from collections import deque
import csv
import inspect
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from model.v3 import V3VisualOnlyVelocityModel
from .flow import load_packet, save_packet
from .io import intrinsics, write_json
from .normal_policy import pair_planes
from .reliable_normal import ReliableNormal
from .solver_adapter import Adapter
from .standalone_geometry import aggregate
from .visual_support import road_mask, supported_flow


def prepare_region(raw, frames, cal, config, flow_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "flow").mkdir(exist_ok=True)
    cv2.setRNGSeed(42)
    k = intrinsics(cal)
    nominal = np.asarray(cal["plane_normal_camera"], float)
    state = ReliableNormal(k, nominal, config["normal"])

    def mask(index):
        with Image.open(raw / frames[index]["file"]) as image:
            rgb = cv2.resize(np.array(image.convert("RGB")), (512, 288))
        return road_mask(rgb, k, nominal, config["road"])

    previous = mask(0)
    records = []
    # Always replay the whole normal state on resume; no stale per-pair state reuse.
    for i in range(len(frames) - 1):
        following = mask(i + 1)
        packet = load_packet(flow_dir / f"{i:06d}.pt", i, frames)
        flow = packet["flow"].float().numpy()
        confidence = packet["confidence"].float().numpy()[0]
        improved, region = supported_flow(flow, confidence, previous, following, config["road"])
        packet["confidence"] = torch.from_numpy(improved[None]).half()
        save_packet(out / "flow" / f"{i:06d}.pt", packet)
        result = state.step(flow, improved)
        result.update(source_id=i, target_id=i + 1, **region)
        records.append(result)
        previous = following
        if i % 50 == 0:
            print(f"Road/normal {i + 1}/{len(frames) - 1}", flush=True)
    write_json(out / "normals.json", records)
    write_json(out / "summary.json", {
        "normal_status": {s: sum(r["status"] == s for r in records)
                          for s in ("update", "hold", "fixed_fallback")},
        "normal_reason": {s: sum(r["reason"] == s for r in records) for s in sorted({r["reason"] for r in records})},
    })
    return records


def infer(raw, frames, cal, config, region, normals, checkpoint, out, groups, device):
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    kwargs = {k: v for k, v in checkpoint["config"]["model"].items()
              if k in inspect.signature(V3VisualOnlyVelocityModel).parameters}
    kwargs.update(mode="fusion", aggregation="precision")
    if kwargs.get("pair_count", 3) != 3 or kwargs.get("support_mode", "learned") != "learned":
        raise ValueError("This release requires the frozen 3-pair learned-reliability checkpoint")
    model = V3VisualOnlyVelocityModel(**kwargs).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    adapter = Adapter(config["finite_solver"])
    model.geometry_solver = adapter
    static = {k: torch.tensor(cal[k], dtype=torch.float32, device=device)[None]
              for k in ("plane_normal_camera", "rotation_ego_from_camera", "translation_ego_from_camera")}
    static.update(intrinsics=torch.tensor(intrinsics(cal), dtype=torch.float32, device=device)[None],
                  plane_offset=torch.tensor([cal["height_m"]], dtype=torch.float32, device=device))
    columns = ["case", "method", "output_index", "image_id", "time_s", "effective_time_s",
               "valid", "valid_pairs", "reason", "vx_mps", "vy_mps", "vz_mps", "speed_mps",
               "normal_status", "normal_reason", "normal_update_pairs", "normal_hold_pairs",
               "normal_fallback_pairs", "inference_ms"]
    pair_columns = ["case", "method", "output_index", "source_id", "target_id", "source_image_id",
                    "target_image_id", "source_time_s", "target_time_s", "pair_slot", "valid", "reason",
                    "count", "condition", "residual", "plane_normal_camera_x",
                    "plane_normal_camera_y", "plane_normal_camera_z", "vx", "vy", "vz", "wx", "wy", "wz"]
    speeds_temp, pairs_temp = out / "speeds.csv.partial", out / "pairs.csv.partial"
    images = deque(maxlen=5)
    packets = deque(maxlen=3)
    counts = {g: 0 for g in groups}
    with speeds_temp.open("w", encoding="utf-8", newline="") as sf, pairs_temp.open("w", encoding="utf-8", newline="") as pf, torch.inference_mode():
        sw, pw = csv.DictWriter(sf, columns), csv.DictWriter(pf, pair_columns)
        sw.writeheader()
        pw.writeheader()
        for i, frame in enumerate(frames):
            with Image.open(raw / frame["file"]) as image:
                images.append(TF.pil_to_tensor(TF.resize(image.convert("RGB"), [72, 128])).float() / 127.5 - 1)
            if i:
                packets.append(load_packet(region / "flow" / f"{i - 1:06d}.pt", i - 1, frames))
            base = dict(case=raw.name, output_index=i, image_id=frame["image_id"], time_s=frame["solver_time"])
            if i < 4:
                for g in groups:
                    group_started = time.perf_counter()
                    sw.writerow(dict(base, method=g, valid=False, valid_pairs=0, reason="warmup"))
                continue
            ps = list(packets)
            visual = dict(static, images=torch.stack(list(images))[None].to(device),
                          flows=torch.stack([p["flow"] for p in ps])[None].float().to(device),
                          flow_confidence=torch.stack([p["confidence"] for p in ps])[None].float().to(device),
                          pair_dt=torch.tensor([[p["dt"] for p in ps]], device=device))
            started = time.perf_counter()
            forward = model.backbone.forward_feature_map
            features = forward(visual["images"])
            model.backbone.forward_feature_map = lambda images: features
            try:
                for g in groups:
                    adapter.backend = "V3_matched" if g == "V3_matched_R" else "finite"
                    adapter.support_variant = "original"
                    adapter.index, adapter.records = 0, []
                    adapter.planes = pair_planes(g, cal["plane_normal_camera"], normals, i, cal["height_m"])
                    result = model(visual)
                    valid = result.pair_valid[0].cpu().numpy()
                    twists = result.pair_twist_ego[0].cpu().numpy()
                    value = aggregate(twists.astype(float), valid, cal)
                    counts[g] += int(value["valid"])
                    pair_normals = normals[i - 3:i]
                    status_counts = {s: sum(r["status"] == s for r in pair_normals)
                                     for s in ("update", "hold", "fixed_fallback")}
                    effective = np.mean([(frames[j]["solver_time"] + frames[j + 1]["solver_time"]) / 2
                                         for j, good in zip(range(i - 3, i), valid) if good]) if valid.any() else ""
                    is_visual = g in ("F_RC", "F_B")
                    row = dict(base, method=g, effective_time_s=effective, valid=value["valid"],
                               valid_pairs=int(valid.sum()), reason=value["reason"],
                               normal_status=pair_normals[-1]["status"] if is_visual else "fixed",
                               normal_reason=pair_normals[-1]["reason"] if is_visual else "fixed_calibration",
                               normal_update_pairs=status_counts["update"] if is_visual else 0,
                               normal_hold_pairs=status_counts["hold"] if is_visual else 0,
                               normal_fallback_pairs=status_counts["fixed_fallback"] if is_visual else 0,
                               inference_ms=(time.perf_counter() - group_started) * 1000)
                    for axis in ("vx", "vy", "vz"):
                        row[axis + "_mps"] = value[axis] if value["valid"] else ""
                    row["speed_mps"] = float(np.hypot(value["vx"], value["vy"])) if value["valid"] else ""
                    sw.writerow(row)
                    for slot, record in enumerate(adapter.records):
                        source = i - 3 + slot
                        pair = {key: record[key] for key in pair_columns if key in record}
                        pair.update(case=raw.name, method=g, output_index=i, source_id=source, target_id=source + 1,
                                    source_image_id=frames[source]["image_id"], target_image_id=frames[source + 1]["image_id"],
                                    source_time_s=frames[source]["solver_time"], target_time_s=frames[source + 1]["solver_time"],
                                    pair_slot=slot)
                        pair.update({axis: float(twists[slot, j]) if valid[slot] else ""
                                     for j, axis in enumerate(("vx", "vy", "vz", "wx", "wy", "wz"))})
                        pw.writerow(pair)
            finally:
                model.backbone.forward_feature_map = forward
            if i % 20 == 0 or i == len(frames) - 1:
                sf.flush()
                pf.flush()
                print(f"Velocity {i + 1}/{len(frames)} {counts}", flush=True)
    speeds_temp.replace(out / "speeds.csv")
    pairs_temp.replace(out / "pairs.csv")
    return {g: dict(eligible=len(frames) - 4, valid=counts[g], coverage=counts[g] / (len(frames) - 4)) for g in groups}
