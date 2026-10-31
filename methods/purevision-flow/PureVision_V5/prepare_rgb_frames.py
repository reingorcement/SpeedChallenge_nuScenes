"""Prepare only original train/dev pixels for true online RGB fine-tuning."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from purevision_v5.common import now, object_hash, read_json, sha256, write_json
from purevision_v5.rgb import RGB_PREPROCESS_VERSION
from purevision_v5.rgb_frames import uint8_view


def prepare(manifest_path,output,workers=8):
    manifest_path,output=Path(manifest_path).resolve(),Path(output).resolve()
    manifest=read_json(manifest_path);dataroot=Path(manifest["dataroot"]).resolve()
    output.mkdir(parents=True,exist_ok=False)
    records=[r for r in manifest["sequences"] if r["partition"] in {"train_core","dev"}]
    def scene(record):
        source=manifest_path.parent/record["file"]
        if sha256(source)!=record["sha256"]:raise ValueError("Sequence changed")
        sequence=read_json(source)
        used=sorted({i for end in sequence["windows"] for i in range(end-4,end+1)})
        frames=[];pixels=[]
        for i in used:
            f=sequence["frames"][i];path=(dataroot/f["filename"]).resolve()
            if not path.is_relative_to(dataroot):raise ValueError("Image outside dataset")
            frames.append(dict(token=f["token"],filename=f["filename"],source_sha256=sha256(path)))
            pixels.append(uint8_view(path))
        filename=record["scene"]+".npy"
        np.save(output/filename,np.stack(pixels),allow_pickle=False)
        return record["scene"],dict(file=filename,sha256=sha256(output/filename),
            sequence_sha256=record["sha256"],partition=record["partition"],frames=frames)
    scenes={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name,info in pool.map(scene,records):
            scenes[name]=info
            if len(scenes)%25==0:print(dict(time=now(),completed_scenes=len(scenes),total=len(records)),flush=True)
    index=dict(schema="v5_rgb_uint8_views_v1",dataset_sha256=sha256(manifest_path),
        preprocess=RGB_PREPROCESS_VERSION,shape_hwc=[72,128,3],dtype="uint8",scenes=scenes)
    index["identity"]=object_hash(index);write_json(output/"index.json",index)
    receipt=dict(status="completed",finished_at=now(),root=str(output),identity=index["identity"],
        index_sha256=sha256(output/"index.json"),scenes=len(scenes),frames=sum(len(s["frames"]) for s in scenes.values()),
        bytes=sum((output/s["file"]).stat().st_size for s in scenes.values()),
        purpose="Exact resized uint8 real images; FT backbone still executes and receives gradients")
    write_json(output/"receipt.json",receipt);print(receipt,flush=True)
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",required=True)
    p.add_argument("--workers",type=int,default=8);a=p.parse_args();prepare(a.manifest,a.output,a.workers)
