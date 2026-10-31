"""Bounded real-packed-data implementation check; never a formal experiment."""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from purevision_v5.common import ROOT, code_version, now, read_json, sha256, write_json
from purevision_v5.datasets import VelocityDataset
from purevision_v5.model import build_model
from purevision_v5.training import (train, seed_everything, model_state_sha256, move_inputs,
                                    loss_fn, validate_expected_flow, evaluate_model)


def verify(manifest, flow, output, device):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ds = {p: VelocityDataset(manifest, p, "R02", flow_root=flow) for p in ("train_core", "dev")}
    assert [len(ds[p].scene_names) for p in ds] == [582, 103]
    assert not ds["train_core"].scene_names & ds["dev"].scene_names
    assert not ds["train_core"].source_tokens & ds["dev"].source_tokens
    indices = np.linspace(0, len(ds["train_core"])-1, 64).astype(int).tolist()
    batch = next(iter(DataLoader(Subset(ds["train_core"], indices), batch_size=64)))
    assert set(batch["inputs"]) == {"flows", "valid", "pair_dt"}
    inputs, target = move_inputs(batch, device), batch["target"].to(device)
    dt = np.concatenate([i["pair_dt"] for d in ds.values() for i in d.items])
    report = dict(schema="v5_encoder_preflight_v1", scope="smoke_not_formal_metrics", started_at=now(),
                  code=code_version(), device=str(device), gpu=torch.cuda.get_device_name(device),
                  manifest_sha256=sha256(manifest), packed_index_sha256=sha256(Path(flow)/"packed_index.json"),
                  data_contract=ds["train_core"].contract,
                  counts={p:dict(scenes=len(d.scene_names),windows=len(d)) for p,d in ds.items()},
                  dt_quantiles_seconds=np.quantile(dt,[0,.01,.5,.99,1]).tolist(),
                  input_shapes={k:list(v.shape) for k,v in inputs.items()},
                  sampled_flow_min_max=[float(inputs["flows"].min()),float(inputs["flows"].max())],
                  partition_scene_and_image_tokens_disjoint=True, rows={})
    for name in ("BASE", "FLEX"):
        config_path = ROOT / f"configs/next/R02_{name}.json"
        conf = read_json(config_path)
        validate_expected_flow(conf, ds["train_core"].contract)
        seed_everything(conf["seed"])
        model = build_model(conf).to(device)
        row = dict(parameters=sum(p.numel() for p in model.parameters()),
                   initial_head_sha256=model_state_sha256(model.head), components={})
        modules = dict(spatial=model.flow_encoder.spatial, temporal=model.flow_encoder.temporal,
                       projection=model.flow_encoder.projection, head=model.head)
        if name == "FLEX":
            modules["sfe"] = model.flow_encoder.sfe
        before = {k:{n:p.detach().clone() for n,p in m.named_parameters()} for k,m in modules.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"])
        model.train()
        prediction = model(inputs)
        loss = loss_fn(prediction, target, conf)
        loss.backward()
        for k,m in modules.items():
            grads = [p.grad for p in m.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            row["components"][k] = dict(gradient_l1=float(sum(g.abs().sum() for g in grads)))
            assert row["components"][k]["gradient_l1"] > 0
        optimizer.step()
        for k,m in modules.items():
            changed = sum(not torch.equal(before[k][n],p) for n,p in m.named_parameters())
            assert changed > 0
            row["components"][k]["changed_parameter_tensors"] = changed
        row.update(output_shape=list(prediction.shape),finite_loss=float(loss.detach()))
        # Measure real batch=64 on the SAME otherwise idle GPU, warmup excluded.
        torch.cuda.reset_peak_memory_stats(device)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                value = loss_fn(model(inputs).float(), target, conf)
            scaler.scale(value).backward()
            scaler.step(optimizer); scaler.update()
        for _ in range(10): step()
        torch.cuda.synchronize(device); start=time.perf_counter()
        for _ in range(100): step()
        torch.cuda.synchronize(device); elapsed=time.perf_counter()-start
        row["compute_benchmark"] = dict(batch=64,warmup=10,measured_steps=100,seconds=elapsed,
            samples_per_second=6400/elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            scope="resident_real_batch AMP forward/backward/AdamW; excludes data loading, validation and checkpoint IO")
        del model, optimizer, before, modules
        # Exercise actual trainer's interrupt/resume/epoch-dev/checkpoint path.
        args = Namespace(config=str(config_path),manifest=str(manifest),flow=str(flow),features=None,selection=None,
            output=str(output/name/"interrupted"),device=str(device),workers=0,threads=4,resume=None,smoke=True,
            limit_scenes=1,limit_windows=128,max_batches=2,stop_after_steps=3)
        first=train(args)
        args.output=str(output/name/"resumed");args.resume=str(first/"last.pt");args.stop_after_steps=5
        resumed=train(args)
        packet=torch.load(resumed/"last.pt",map_location="cpu",weights_only=True)
        assert packet["processed_batches"] == 5 == packet["optimizer_updates"] + packet["amp_skips"]
        assert packet["optimizer_updates"] > 0
        restored=build_model(conf).to(device);restored.load_state_dict(packet["model"],strict=True)
        summary, rows=evaluate_model(restored,ds["dev"],device,batch_size=64,workers=0,max_batches=2,lateral_rule=conf["lateral_rule"])
        assert len(rows)==128 and np.isfinite(summary["overall"]["epe"])
        row["resume"] = dict(processed_batches=packet["processed_batches"],optimizer_updates=packet["optimizer_updates"],
                             amp_skips=packet["amp_skips"],checkpoint_sha256=sha256(resumed/"last.pt"),
                             independent_evaluation_count=len(rows),status="passed")
        report["rows"][name]=row
        write_json(output/"summary.json",report)
    assert report["rows"]["BASE"]["initial_head_sha256"] == report["rows"]["FLEX"]["initial_head_sha256"]
    report.update(status="passed",finished_at=now(),identical_head_initialization=True)
    write_json(output/"summary.json",report)
    return report


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    for k in ("manifest","flow","output"): p.add_argument("--"+k,required=True)
    p.add_argument("--device",default="cuda:1")
    a=p.parse_args(); print(verify(a.manifest,a.flow,a.output,a.device))
