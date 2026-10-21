from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAvgPool3dStatic(nn.Module):
    def __init__(self, mode: str = "none_1_1") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "1":
            return x.mean(dim=(-3, -2, -1), keepdim=True)
        if self.mode == "none_1_1":
            return x.mean(dim=(-1, -2), keepdim=True)
        return x


class ContextualMotionAnalysis(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        motion = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        motion = F.pad(motion, (0, 0, 0, 0, 1, 0), "constant", 0)
        motion = self.relu(self.bn1(self.conv1(motion)))
        motion = self.relu(self.bn2(self.conv2(motion)))
        return motion


class AdaptiveFeatureTransformer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.attention_bn = nn.BatchNorm3d(out_channels)
        self.attention_sigmoid = nn.Sigmoid()
        self.feature_transform = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.feature_bn = nn.BatchNorm3d(out_channels)
        self.feature_relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention_sigmoid(self.attention_bn(self.attention_conv(motion)))
        transformed_features = self.feature_relu(self.feature_bn(self.feature_transform(x)))
        residual = self.residual(x)
        return attention_weights * transformed_features + residual


class SpatialFeatureExtraction(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.dilated_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), dilation=2, padding=(0, 2, 2))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.res_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_bn(self.residual(x))
        x = self.depthwise(x)
        x = self.bn1(self.pointwise(x))
        x = self.relu(x)
        x = self.dilated_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = x + residual
        return self.relu(x)


class MotionFeatureExtraction(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_conv1 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_pool = AdaptiveAvgPool3dStatic(mode="none_1_1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.temporal_conv1(x)
        x = self.temporal_conv2(x)
        return self.temporal_pool(x)


class DynamicIntegrationGate(nn.Module):
    def __init__(self, output_channels: int) -> None:
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Conv3d(output_channels * 2, output_channels, kernel_size=1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
            AdaptiveAvgPool3dStatic(mode="1"),
            nn.Conv3d(output_channels, output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        target_size = spatial_features.size()[2:]
        temporal_features_upsampled = F.interpolate(
            temporal_features, size=target_size, mode="trilinear", align_corners=False
        )
        combined_features = torch.cat([spatial_features, temporal_features_upsampled], dim=1)
        gating_coefficients = self.gate_network(combined_features)
        return gating_coefficients * spatial_features + (1 - gating_coefficients) * temporal_features_upsampled

