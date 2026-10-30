"""Frozen, selectively loaded V3 RGB descriptor without geometry dependencies."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from ._rgb_reference.flexinet_backbone import FlexiNetBackbone


RGB_SHAPE = (5, 3, 72, 128)
DESCRIPTOR_DIM = 384
RGB_PREPROCESS_VERSION = "v3_pil_bilinear_rgb_72x128_minus1_plus1_v1"


def preprocess_rgb(image: Image.Image) -> torch.Tensor:
    """Match V3's PIL Resize -> RGB -> ToTensor -> Normalize(.5,.5)."""
    resized = image.resize((128, 72), resample=Image.Resampling.BILINEAR).convert("RGB")
    array = np.asarray(resized, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255).sub_(.5).div_(.5)


class FrozenRGB(nn.Module):
    """Five causal normalized RGB frames -> six ordered 64D region descriptors.

    Only backbone and region_projection tensors are accepted from the source
    checkpoint. Its old velocity, reliability and geometry modules are absent.
    """

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        checkpoint = Path(checkpoint)
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = package.get("config", {})
        data, model = config.get("data", {}), config.get("model", {})
        if (data.get("seq_len") != 5 or data.get("online_mode") is not True
                or data.get("visual_height") != 72 or data.get("visual_width") != 128
                or model.get("pooling") != "regions"
                or model.get("mode") != "direct_pretrain"):
            raise ValueError("FrozenRGB requires the causal T5 72x128 regional V3 direct checkpoint")
        state = package.get("model")
        if not isinstance(state, dict):
            raise ValueError("Source checkpoint lacks model state")
        self.backbone = FlexiNetBackbone(input_channels=3)
        self.region_projection = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True))
        selected = {key: value for key, value in state.items()
                    if key.startswith(("backbone.", "region_projection."))}
        self.load_state_dict(selected, strict=True)
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.preprocess_version = RGB_PREPROCESS_VERSION
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True):
        """Keep frozen BatchNorm statistics fixed under a parent's train()."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or tuple(images.shape[1:]) != RGB_SHAPE:
            raise ValueError(f"images must have shape [B,{','.join(map(str, RGB_SHAPE))}]")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("RGB images must be finite floating-point normalized values")
        if bool((images.abs() > 1.00001).any()):
            raise ValueError("RGB images must be normalized to [-1,1]")
        feature_map = self.backbone.forward_feature_map(images)
        # Exact V3 regional descriptor: near-to-far, then left-to-right.
        pooled = F.adaptive_avg_pool3d(feature_map, (1, 2, 3)).flip(dims=(3,))
        tokens = pooled.squeeze(2).permute(0, 2, 3, 1).reshape(images.shape[0], 6, 256)
        descriptor = self.region_projection(tokens).flatten(1)
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("Non-finite frozen RGB descriptor")
        return descriptor
