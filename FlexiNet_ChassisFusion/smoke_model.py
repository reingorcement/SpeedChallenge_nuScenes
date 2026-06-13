from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model.fusion_model import FlexiNetChassisFusion
from dataset.nuscenes_chassis import FAIR_CHASSIS_DIM


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the fusion model forward pass")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=13)
    parser.add_argument("--with_flow", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexiNetChassisFusion(
        chassis_input_dim=FAIR_CHASSIS_DIM,
        chassis_feature_dim=16,
        fusion_mode="residual",
        output_dim=2,
        chassis_sequence_model="gru",
        flow_input_channels=2 if args.with_flow else 0,
        flow_feature_dim=8 if args.with_flow else 0,
        geometry_input_dim=4,
        geometry_feature_dim=8,
    )
    if args.pretrained:
        model.load_flexinet_backbone(Path(args.pretrained), strict=False)
    model.to(device).eval()

    images = torch.randn(args.batch_size, args.seq_len, 1, 64, 64, device=device)
    chassis = torch.randn(args.batch_size, 5, FAIR_CHASSIS_DIM, device=device)
    base_velocity = torch.zeros(args.batch_size, 2, device=device)
    base_velocity[:, 0] = torch.rand(args.batch_size, device=device) * 15.0
    flow = torch.randn(args.batch_size, args.seq_len - 1, 2, 64, 64, device=device) if args.with_flow else None
    geometry = torch.randn(args.batch_size, 4, device=device)

    with torch.no_grad():
        output = model(images, chassis, base_velocity=base_velocity, flow=flow, geometry=geometry)
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output sample: {output[: min(args.batch_size, 3)].detach().cpu().tolist()}")


if __name__ == "__main__":
    main()

