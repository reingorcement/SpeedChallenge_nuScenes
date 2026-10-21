"""Check V4 dependencies, local weights and device availability without training."""
import json

import cv2
import numpy as np
import scipy
import torch
import torchvision
from torchvision.models.optical_flow import raft_large

from purevision_v4.io import ROOT, load_config, sha256
from purevision_v4.model import FrozenVisual


def main():
    load_config()
    visual = FrozenVisual(ROOT / "weights/v3_fusion_seed42.pth")
    raft = raft_large(weights=None)
    raft.load_state_dict(torch.load(ROOT / "weights/raft_large_C_T_SKHT_V2.pth", map_location="cpu", weights_only=True), strict=True)
    print(json.dumps(dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__), numpy=np.__version__,
                         scipy=scipy.__version__, opencv=cv2.__version__, cuda=torch.cuda.is_available(),
                         devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                         visual_weights=visual.digest, raft_weights=sha256(ROOT / "weights/raft_large_C_T_SKHT_V2.pth")), indent=2))


if __name__ == "__main__":
    main()
