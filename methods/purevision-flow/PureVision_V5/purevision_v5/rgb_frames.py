"""Lossless uint8 views of real causal images; no learned RGB features."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import object_hash, read_json, sha256
from .rgb import RGB_PREPROCESS_VERSION


def uint8_view(path):
    with Image.open(path) as image:
        return np.asarray(image.resize((128,72),Image.Resampling.BILINEAR).convert("RGB"),dtype=np.uint8).copy()


class RGBFrames:
    def __init__(self, root, manifest_sha):
        self.root=Path(root).resolve()
        index=read_json(self.root/"index.json")
        self.identity=index.pop("identity")
        if object_hash(index)!=self.identity or index["dataset_sha256"]!=manifest_sha:
            raise ValueError("RGB image view identity/manifest mismatch")
        if index["preprocess"]!=RGB_PREPROCESS_VERSION or index["schema"]!="v5_rgb_uint8_views_v1":
            raise ValueError("Unsupported image view contract")
        self.scenes=index["scenes"]
        self.frames={}
        for scene,info in self.scenes.items():
            path=(self.root/info["file"]).resolve()
            if not path.is_relative_to(self.root) or sha256(path)!=info["sha256"]:
                raise ValueError("RGB view file changed or escaped root")
            for i,frame in enumerate(info["frames"]):
                if frame["token"] in self.frames:raise ValueError("Duplicate image token")
                self.frames[frame["token"]]=(scene,i,frame["filename"])

    @lru_cache(maxsize=16)
    def array(self,scene):
        info=self.scenes[scene]
        value=np.load(self.root/info["file"],mmap_mode="r",allow_pickle=False)
        if value.shape!=(len(info["frames"]),72,128,3) or value.dtype!=np.uint8:
            raise ValueError("Invalid image view shape/dtype")
        return value

    def load(self,token,filename):
        scene,row,expected=self.frames[token]
        if filename!=expected:raise ValueError("RGB filename/token mismatch")
        return torch.from_numpy(self.array(scene)[row].copy()).permute(2,0,1).float().div_(255).sub_(.5).div_(.5)
