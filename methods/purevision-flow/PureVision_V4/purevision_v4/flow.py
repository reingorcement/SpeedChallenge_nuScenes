"""Local frozen RAFT with versioned, pair-specific resumable caches."""
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms import functional as TF

from .io import artifact_path, object_hash, sha256


def atomic_torch_save(value, path):
    path = artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(value, tmp)
    tmp.replace(path)


def consistency_confidence(forward, backward):
    _, _, h, w = forward.shape
    y, x = torch.meshgrid(torch.arange(h, device=forward.device, dtype=forward.dtype),
                          torch.arange(w, device=forward.device, dtype=forward.dtype), indexing="ij")
    gx, gy = 2 * (x[None] + forward[:, 0]) / (w - 1) - 1, 2 * (y[None] + forward[:, 1]) / (h - 1) - 1
    warped = F.grid_sample(backward, torch.stack((gx, gy), -1), align_corners=True)
    error = torch.linalg.vector_norm(forward + warped, dim=1, keepdim=True)
    scale = .5 + .01 * torch.linalg.vector_norm(forward, dim=1, keepdim=True)
    return (torch.exp(-error / scale) * ((gx.abs() <= 1) & (gy.abs() <= 1))[:, None]).clamp(0, 1)


def resize_flow(flow, confidence, size):
    sh, sw = flow.shape[-2:]
    h, w = size
    out = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    out[:, 0] *= w / sw
    out[:, 1] *= h / sh
    conf = F.interpolate(confidence, size=size, mode="bilinear", align_corners=False).clamp(0, 1)
    return out, conf


class RaftFlow:
    def __init__(self, checkpoint, device, preprocessing, image_size):
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.config = dict(preprocessing)
        self.digest = sha256(checkpoint)
        self.model = None
        h, w = self.config["raft_size"]
        if self.config.get("raft_resize", "preserve_aspect") == "preserve_aspect":
            h = max(128, round(w * image_size[1] / image_size[0] / 8) * 8)
        self.raft_size = (h, w)

    def pair(self, source, target, path):
        identity = object_hash(dict(source=source, target=target, preprocessing=self.config,
                                    actual_raft_size=self.raft_size, weights=self.digest,
                                    torch=str(torch.__version__), device=str(self.device)))
        path = artifact_path(path)
        if path.exists():
            packet = torch.load(path, map_location="cpu", weights_only=True)
            if packet.get("identity") != identity:
                raise ValueError(f"Flow cache identity mismatch: {path}")
        else:
            if self.model is None:
                self.model = raft_large(weights=None)
                self.model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True), strict=True)
                self.model.to(self.device).eval().requires_grad_(False)
            images = []
            for frame in (source, target):
                with Image.open(frame["file"]) as image:
                    value = TF.pil_to_tensor(image.convert("RGB")).float() / 255
                images.append(TF.resize(value, list(self.raft_size), antialias=True)[None].to(self.device))
            images = Raft_Large_Weights.C_T_SKHT_V2.transforms()(*images)
            context = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" else nullcontext()
            with torch.inference_mode(), context:
                forward = self.model(*images)[-1].float()
                backward = self.model(*images[::-1])[-1].float()
                confidence = consistency_confidence(forward, backward)
                forward, confidence = resize_flow(forward, confidence, self.config["geometry_size"])
            packet = dict(identity=identity, flow=forward[0].cpu().half(), confidence=confidence[0].cpu().half(),
                          source_id=source["image_id"], target_id=target["image_id"], dt=target["time_s"] - source["time_s"])
            atomic_torch_save(packet, path)
        h, w = self.config["geometry_size"]
        for key, shape in [("flow", (2, h, w)), ("confidence", (1, h, w))]:
            if tuple(packet[key].shape) != shape or not torch.isfinite(packet[key]).all():
                raise ValueError(f"Malformed flow cache {key}: {path}")
        if packet["confidence"].min() < 0 or packet["confidence"].max() > 1:
            raise ValueError("Confidence must be in [0,1]")
        return packet
