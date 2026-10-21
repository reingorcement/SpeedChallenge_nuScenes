from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model.fusion_model import FlexiNetChassisFusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the fusion model forward pass")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=13)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexiNetChassisFusion(chassis_input_dim=16, fusion_mode="residual")
    if args.pretrained:
        model.load_flexinet_backbone(Path(args.pretrained), strict=False)
    model.to(device).eval()

    images = torch.randn(args.batch_size, args.seq_len, 1, 64, 64, device=device)
    chassis = torch.randn(args.batch_size, 16, device=device)
    base_speed = torch.rand(args.batch_size, 1, device=device) * 15.0

    with torch.no_grad():
        output = model(images, chassis, base_speed=base_speed)
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output sample: {output[: min(args.batch_size, 3)].squeeze(1).detach().cpu().tolist()}")


if __name__ == "__main__":
    main()

