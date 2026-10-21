"""Frozen RAFT preprocessing, local weights only; no runtime downloads."""
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


def load_rgb(path):
    with Image.open(path) as image:
        tensor = TF.pil_to_tensor(image.convert("RGB")).float().div_(255.)
    return TF.resize(tensor, [384, 512], antialias=True)


def consistency_confidence(forward, backward):
    _, _, height, width = forward.shape
    y, x = torch.meshgrid(torch.arange(height, device=forward.device, dtype=forward.dtype),
                          torch.arange(width, device=forward.device, dtype=forward.dtype), indexing="ij")
    gx = 2 * (x[None] + forward[:, 0]) / max(width - 1, 1) - 1
    gy = 2 * (y[None] + forward[:, 1]) / max(height - 1, 1) - 1
    warped = F.grid_sample(backward, torch.stack((gx, gy), -1), mode="bilinear",
                          padding_mode="zeros", align_corners=True)
    valid = ((gx.abs() <= 1) & (gy.abs() <= 1))[:, None]
    error = torch.linalg.vector_norm(forward + warped, dim=1, keepdim=True)
    scale = .5 + .01 * torch.linalg.vector_norm(forward, dim=1, keepdim=True)
    return (torch.exp(-error / scale) * valid).clamp(0, 1)


def save_packet(path, packet):
    temporary = path.with_suffix(".partial")
    torch.save(packet, temporary)
    temporary.replace(path)


def load_packet(path, index, frames):
    packet = torch.load(path, map_location="cpu", weights_only=True)
    dt = frames[index + 1]["solver_time"] - frames[index]["solver_time"]
    if packet.get("source_id") != index or packet.get("target_id") != index + 1:
        raise ValueError(f"Flow source/target mismatch: {path}")
    if abs(float(packet["dt"]) - dt) > 1e-8 or dt <= 1e-4:
        raise ValueError(f"Invalid flow interval: {path}")
    for key, shape in (("flow", (2, 288, 512)), ("confidence", (1, 288, 512))):
        if tuple(packet[key].shape) != shape or not torch.isfinite(packet[key]).all():
            raise ValueError(f"Invalid {key}: {path}")
    if packet["confidence"].min() < 0 or packet["confidence"].max() > 1:
        raise ValueError(f"Confidence outside [0, 1]: {path}")
    return packet


def prepare_flow(raw, frames, out, checkpoint, device):
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    for i in range(len(frames) - 1):
        path = out / f"{i:06d}.pt"
        if path.exists():
            load_packet(path, i, frames)
        else:
            pending.append(i)
    if not pending:
        print("RAFT cache verified", flush=True)
        return
    # Explicit variant and checkpoint prevent a future DEFAULT alias change.
    transform = Raft_Large_Weights.C_T_SKHT_V2.transforms()
    model = raft_large(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        for i in pending:
            source, target = [load_rgb(raw / frames[j]["file"])[None].to(device) for j in (i, i + 1)]
            source, target = transform(source, target)
            context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with context:
                forward = model(source, target)[-1].float()
                backward = model(target, source)[-1].float()
                confidence = consistency_confidence(forward, backward)
            flow = F.interpolate(forward, size=(288, 512), mode="bilinear", align_corners=False)
            flow[:, 1] *= 288 / 384
            conf = F.interpolate(confidence, size=(288, 512), mode="bilinear", align_corners=False)
            packet = dict(flow=flow[0].cpu().half(), confidence=conf[0].cpu().half(),
                          dt=frames[i + 1]["solver_time"] - frames[i]["solver_time"],
                          source_id=i, target_id=i + 1)
            save_packet(out / f"{i:06d}.pt", packet)
            load_packet(out / f"{i:06d}.pt", i, frames)
            if i % 20 == 0 or i == pending[-1]:
                print(f"RAFT {i + 1}/{len(frames) - 1}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
