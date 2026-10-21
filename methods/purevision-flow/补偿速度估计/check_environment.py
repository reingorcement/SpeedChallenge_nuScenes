import argparse
import platform

from roll_velocity.io import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description="Check the inference runtime and local checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()
    import torch
    import torchvision
    import cv2
    import numpy
    import scipy
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the pinned torch/torchvision CUDA wheels")
    # Actually execute a CUDA kernel; availability alone does not check the GPU architecture.
    a = torch.eye(3, device=device)
    assert float((a @ a).sum()) == 3
    data = dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                cv2=cv2.__version__, numpy=numpy.__version__, scipy=scipy.__version__, device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                checkpoints={p.name: sha256(p) for p in (ROOT / "weights").glob("*.pth")})
    if len(data["checkpoints"]) != 2:
        raise RuntimeError("Both bundled weight files are required")
    for key, value in data.items():
        print(key, value)
    if args.output:
        write_json(args.output, data)


if __name__ == "__main__":
    main()
