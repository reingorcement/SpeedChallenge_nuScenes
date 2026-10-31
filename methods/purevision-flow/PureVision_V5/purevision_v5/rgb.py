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


class AppearanceFlexiNetBackbone(FlexiNetBackbone):
    """One raw-image bypass; shared pretrained names and motion attention stay intact."""
    def __init__(self):
        super().__init__(input_channels=3)
        # Isolate new-module initialization from the common model RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.appearance_stem = nn.Sequential(
                nn.Conv3d(3, 64, (1, 3, 3), padding=(0, 1, 1)), nn.ReLU())
        self.appearance_alpha = nn.Parameter(torch.zeros(()))

    def forward_feature_map(self, images):
        images = images.permute(0, 2, 1, 3, 4)
        motion = self.cma_block(images)
        appearance = self.appearance_stem(images)
        x = motion + self.appearance_alpha * appearance
        aft = self.aft_block(x, motion)
        return self.dig(self.sfe_module(aft), self.mfe_module(aft))


def preprocess_rgb(image: Image.Image) -> torch.Tensor:
    """Match V3's PIL Resize -> RGB -> ToTensor -> Normalize(.5,.5)."""
    resized = image.resize((128, 72), resample=Image.Resampling.BILINEAR).convert("RGB")
    array = np.asarray(resized, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255).sub_(.5).div_(.5)


class RGBDescriptor(nn.Module):
    """Shared V3 descriptor with a genuine trainable RGB path."""
    def __init__(self, *, trainable=True, variant="difference"):
        super().__init__()
        if variant not in {"difference", "appearance_difference"}:
            raise ValueError("Unknown RGB information path")
        self.variant = variant
        self.backbone = (AppearanceFlexiNetBackbone() if variant == "appearance_difference"
                         else FlexiNetBackbone(input_channels=3))
        self.region_projection = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True))
        self.trainable_rgb = bool(trainable)
        self.preprocess_version = RGB_PREPROCESS_VERSION
        self.requires_grad_(self.trainable_rgb)
        self.train(self.trainable_rgb)

    def load_pretrained(self, checkpoint):
        checkpoint = Path(checkpoint)
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = package.get("config", {})
        data, model = config.get("data", {}), config.get("model", {})
        if (data.get("seq_len") != 5 or data.get("online_mode") is not True
                or data.get("visual_height") != 72 or data.get("visual_width") != 128
                or model.get("pooling") != "regions" or model.get("mode") != "direct_pretrain"):
            raise ValueError("RGB requires the causal T5 72x128 regional V3 direct checkpoint")
        state = package.get("model")
        if not isinstance(state, dict):raise ValueError("Source checkpoint lacks model state")
        shared = {k:v for k,v in state.items() if k.startswith(("backbone.", "region_projection."))}
        if self.variant == "difference":
            self.load_state_dict(shared, strict=True)
        else:
            allowed = {"backbone.appearance_alpha", "backbone.appearance_stem.0.weight",
                       "backbone.appearance_stem.0.bias"}
            result = self.load_state_dict(shared, strict=False)
            if set(result.missing_keys) != allowed or result.unexpected_keys:
                raise ValueError("Appearance initialization changed shared RGB keys")
            if not all(torch.equal(self.state_dict()[k], v) for k, v in shared.items()):
                raise ValueError("Common RGB initialization is not exact")
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return self

    def train(self, mode=True):
        return super().train(mode if self.trainable_rgb else False)

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


class FrozenRGB(RGBDescriptor):
    """Backward-compatible frozen wrapper; FT does not inherit this no_grad path."""
    def __init__(self, checkpoint):
        super().__init__(trainable=False)
        self.load_pretrained(checkpoint)

    @torch.no_grad()
    def forward(self, images):
        return super().forward(images)
