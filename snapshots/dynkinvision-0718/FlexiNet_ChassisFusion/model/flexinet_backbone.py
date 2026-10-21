from __future__ import annotations

import torch
import torch.nn as nn

from .modules import (
    AdaptiveAvgPool3dStatic,
    AdaptiveFeatureTransformer,
    ContextualMotionAnalysis,
    DynamicIntegrationGate,
    MotionFeatureExtraction,
    SpatialFeatureExtraction,
)


class FlexiNetBackbone(nn.Module):
    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.cma_block = ContextualMotionAnalysis(input_channels, 64)
        self.aft_block = AdaptiveFeatureTransformer(64, 128)
        self.sfe_module = SpatialFeatureExtraction(128, 256)
        self.mfe_module = MotionFeatureExtraction(128, 192, 256)
        self.dig = DynamicIntegrationGate(output_channels=256)
        self.global_pool = AdaptiveAvgPool3dStatic(mode="1")

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)
        cma_out = self.cma_block(x)
        aft_out = self.aft_block(cma_out, cma_out)
        sfe_out = self.sfe_module(aft_out)
        mfe_out = self.mfe_module(aft_out)
        return self.dig(sfe_out, mfe_out)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        integrated = self.forward_feature_map(x)
        pooled = self.global_pool(integrated).view(integrated.size(0), -1)
        return pooled

