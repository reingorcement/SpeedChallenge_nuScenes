"""Bounded real-data preflight for the two S23 continuation stages; no formal run."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, default_collate

from purevision_v5.common import atomic_save, now, object_hash, read_json, sha256, write_json
from purevision_v5.fusion_abc_cache import fp32_policy, row_metadata
from purevision_v5.fusion_abc_runtime import peak_memory, runtime_gpu
from purevision_v5.fusion_full_cache import (COUNTS, SCENE_COUNTS, full_datasets, identity_map,
                                           load_full_cache, sample_key, semantic_metadata)
from purevision_v5.fusion_tail import GROUPS, READOUT_SHA, SOURCE_SHA, TAIL_MODULES, S23TailModel
from purevision_v5.training import epoch_order, model_state_sha256, restore_rng, rng_state, seed_everything

ATOL, RTOL = 1e-6, 1e-5


def close(left, right):
    torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)
    return float((left.detach() - right.detach()).abs().max())


def loss(pred, target):
    scale = target.new_tensor([5., 1.])
    return F.smooth_l1_loss(pred / scale, target / scale, beta=1., reduction="mean")


def move(inputs, device):
    return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


def independent(left, right):
    a, b = left.state_dict(), right.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) and a[k].data_ptr() != b[k].data_ptr() for k in a)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def unchanged(model, before):
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    frozen.update(name for name, _ in model.named_buffers())
    current = model.state_dict()
    assert all(torch.equal(before[k], current[k].detach().cpu()) for k in frozen)
    assert all(not module.training for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm))
    return dict(unchanged_tensor_count=len(frozen), frozen_parameters_and_all_buffers_exact=True,
                BN_affine_and_buffers_unchanged=True, BN_eval_under_outer_train=True)


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: nested_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): nested_equal(x, y)
    else:
        assert a == b


def nested_close(a, b):
    if isinstance(a, torch.Tensor):
        return close(a.detach().cpu(), b.detach().cpu()) if a.is_floating_point() else (0. if torch.equal(a.cpu(),b.cpu()) else math.inf)
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((nested_close(a[k],b[k]) for k in a), default=0.)
    if isinstance(a, (list,tuple)):
        assert len(a)==len(b)
        return max((nested_close(x,y) for x,y in zip(a,b)),default=0.)
    assert a==b
    return 0.


def verify(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model_preflight.json").exists():
        raise ValueError("Completed model preflight evidence must not be overwritten")
    started = time.monotonic()
    fp32_policy(); torch.set_num_threads(args.threads); seed_everything(42)
    hardware = runtime_gpu(args.device)
    if hardware.get("uuid") != "GPU-cedbf5f4-647b-640c-1244-babecaef88e7":
        raise ValueError("This bounded online preflight is assigned to physical GPU0")
    print("validating fixed parents and full source identities", flush=True)
    tail = S23TailModel(args.encoder_checkpoint, args.readout_checkpoint, "S23_TAIL_FT")
    frozen = S23TailModel(args.encoder_checkpoint, args.readout_checkpoint, "S23_FROZEN_CONT")
    independent(tail, frozen)
    initial_state = cpu_state(tail)
    manifest = dict(status="passed", groups={m.group: m.parameter_manifest() for m in (tail, frozen)},
                    provenance=tail.provenance)
    write_json(output / "parameter_manifest.json", manifest)
    datasets = full_datasets(args.manifest, args.flow, args.rgb_frames)
    packets, indexes, maps, scope = {}, {}, {}, {}
    for part in COUNTS:
        packets[part], indexes[part] = load_full_cache(args.cache, part)
        packet = packets[part]
        maps[part] = identity_map(packet["metadata"])
        assert len(maps[part]) == COUNTS[part]
        for i, item in enumerate(datasets[part].items):
            metadata = row_metadata(datasets[part], i)
            j = maps[part][sample_key(metadata)]
            assert semantic_metadata(metadata) == semantic_metadata(packet["metadata"][j])
            assert torch.equal(torch.as_tensor(item["target"]), packet["targets"][j])
            assert torch.equal(torch.as_tensor(item["pair_dt"]), packet["features"]["dt"][j])
        scope[part] = dict(count=COUNTS[part], scenes=SCENE_COUNTS[part],
            identity_mapping="scene,token", all_metadata_frames_pairs_timestamps_labels_dt_equal=True)
    index = indexes["dev"]
    assert indexes["train_core"]["identity"] == index["identity"] == tail.provenance["readout"]["cache_identity"]
    evaluation_root = Path(args.readout_checkpoint).parent / "eval_dev"
    evaluation = read_json(evaluation_root / "evaluation.json")
    assert evaluation["checkpoint_sha256"] == READOUT_SHA and evaluation["cache_identity"] == index["identity"]
    prediction_path = evaluation_root / "predictions.csv"
    assert sha256(prediction_path) == evaluation["predictions_sha256"]
    with prediction_path.open(newline="") as stream:
        prediction_rows = list(csv.DictReader(stream))
    predictions = {(r["scene"], r["token"]): r for r in prediction_rows}
    assert len(predictions) == COUNTS["dev"]
    tail.to(args.device); frozen.to(args.device)
    feature_evidence = {}
    fixed_batches = {}
    for part in COUNTS:
        ids = [len(datasets[part])-1, 0, len(datasets[part])//2, 19, len(datasets[part])//3, 103]
        raw = default_collate([datasets[part][i] for i in ids])
        inputs = move(raw["inputs"], args.device)
        metadata = [row_metadata(datasets[part], i) for i in ids]
        matched = [maps[part][sample_key(m)] for m in metadata]
        cached = {k: v[matched].to(args.device) for k, v in packets[part]["features"].items()}
        tail.eval(); frozen.eval()
        with torch.no_grad():
            features = tail.extract_features(inputs)
            online = tail.readout(features)
            cache_pred = frozen.forward_cached(cached)
            frozen_online = frozen(inputs)
        feature_evidence[part] = dict(sample_metadata=metadata,
            feature_max_abs_error={k: close(features[k], cached[k]) for k in features},
            online_vs_frozen_cached_prediction_max_abs_error=close(online, cache_pred),
            online_both_groups_prediction_max_abs_error=close(online, frozen_online))
        if part == "dev":
            matched_predictions = [predictions[sample_key(m)] for m in metadata]
            assert all(float(p["time_s"]) == m["time_s"] for p, m in zip(matched_predictions, metadata))
            historical_targets = torch.tensor([[float(p["target_vx"]), float(p["target_vy"])] for p in matched_predictions])
            assert torch.equal(historical_targets, raw["target"])
            historical = torch.tensor([[float(p["vx"]), float(p["vy"])] for p in matched_predictions], device=args.device)
            feature_evidence[part]["parent_published_prediction_max_abs_error"] = close(online, historical)
        fixed_batches[part] = dict(inputs=inputs, target=raw["target"].to(args.device),
                                   initial_features={k:v.detach().clone() for k,v in features.items()})
    print("initial online/cache and parent prediction checks passed", flush=True)
    write_json(output/"initial_path_evidence.json",dict(status="passed",full_scope=scope,
        cache_identity=index["identity"],initial_online_comparison=feature_evidence,
        comparison_tolerance=dict(atol=ATOL,rtol=RTOL),source_sha256=sha256(Path(__file__).parent/"purevision_v5/fusion_tail.py")))
    # Discard the frozen GPU copy before bounded online backward checks.
    frozen.cpu(); del inputs, features, online, cache_pred, frozen_online, cached; gc.collect(); torch.cuda.empty_cache()
    nn.Sequential(tail).train()
    unchanged(tail, initial_state)
    optimizer = torch.optim.AdamW(tail.optimizer_groups(), betas=(.9,.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.)
    small = {k:v[:2] for k,v in fixed_batches["train_core"]["inputs"].items()}
    target = fixed_batches["train_core"]["target"][:2]
    optimizer.zero_grad(set_to_none=True)
    prediction = tail(small)
    assert prediction.shape == (2, 2) and torch.isfinite(prediction).all()
    first_loss = loss(prediction, target); assert torch.isfinite(first_loss)
    first_loss.backward()
    gradients = {}
    for prefix in (*TAIL_MODULES, "readout"):
        parameters = [(n,p) for n,p in tail.named_parameters() if n.startswith(prefix + ".") and p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in parameters)
        norm = float(torch.sqrt(sum(p.grad.detach().double().square().sum() for _,p in parameters)))
        assert norm > 0
        gradients[prefix] = dict(gradient_l2=norm, parameter_tensors=len(parameters), finite=True, nonzero=True)
    global_norm = nn.utils.clip_grad_norm_([p for p in tail.parameters() if p.requires_grad], 2.)
    assert torch.isfinite(global_norm)
    optimizer.step()
    for prefix in gradients:
        changes = [float((p.detach().cpu()-initial_state[n]).abs().max())
                   for n,p in tail.named_parameters() if p.requires_grad and n.startswith(prefix + ".")]
        assert max(changes) > 0
        gradients[prefix]["short_update_parameter_max_abs_change"] = max(changes)
    frozen_evidence = unchanged(tail, initial_state)
    tail.eval()
    with torch.no_grad():
        updated_features = tail.extract_features(fixed_batches["dev"]["inputs"])
        updated_prediction = tail.readout(updated_features)
    changed = {k:float((updated_features[k]-fixed_batches["dev"]["initial_features"][k]).abs().max())
               for k in ("F_sp", "F_tm", "R")}
    assert all(v > 0 for v in changed.values())
    calls = {name:0 for name in TAIL_MODULES}
    handles = []
    for name in TAIL_MODULES:
        def hook(module, inputs, output, key=name): calls[key] += 1
        handles.append(tail.get_submodule(name).register_forward_hook(hook))
    with torch.no_grad():
        direct = tail(fixed_batches["dev"]["inputs"])
    close(direct, updated_prediction)
    assert all(count == 1 for count in calls.values())
    for handle in handles: handle.remove()
    try:
        tail.forward_cached(fixed_batches["dev"]["initial_features"])
    except ValueError:
        cache_rejected = True
    else: raise AssertionError("Tail model accepted stale cached features")
    # Complete model and optimizer/scheduler/RNG roundtrip, then same next update.
    scheduler.step()
    boundary = dict(schema="v5_s23_tail_preflight_state_v1", model=cpu_state(tail),
        optimizer=copy.deepcopy(optimizer.state_dict()), scheduler=scheduler.state_dict(), rng=rng_state(),
        group=tail.group, provenance=tail.provenance, freeze_manifest=tail.parameter_manifest(),
        progress=dict(completed_epochs=1, optimizer_updates=1, formal_training=False))
    checkpoint = output / "preflight_complete_state.pt"
    attempt=1
    while checkpoint.exists():
        attempt+=1;checkpoint=output/f"preflight_complete_state_attempt{attempt}.pt"
    atomic_save(checkpoint, boundary)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with patch("purevision_v5.fusion_tail.file_sha256", side_effect=AssertionError("parent read forbidden")):
        restored = S23TailModel.from_state(loaded["model"], loaded["group"], loaded["provenance"]).to(args.device)
    restored_optimizer = torch.optim.AdamW(restored.optimizer_groups(), betas=(.9,.999), eps=1e-8, weight_decay=1e-4)
    restored_optimizer.load_state_dict(loaded["optimizer"])
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=20, eta_min=0.)
    restored_scheduler.load_state_dict(loaded["scheduler"])
    restore_rng(loaded["rng"]); nested_equal(rng_state(), loaded["rng"])
    nested_equal(tail.state_dict(),restored.state_dict())
    nested_equal(optimizer.state_dict(),restored_optimizer.state_dict())
    nested_equal(scheduler.state_dict(),restored_scheduler.state_dict())
    for candidate, opt, sch in ((tail,optimizer,scheduler),(restored,restored_optimizer,restored_scheduler)):
        candidate.train(); opt.zero_grad(set_to_none=True)
        loss(candidate(small), target).backward()
        nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad], 2.)
        opt.step(); sch.step()
    model_next_error=nested_close(tail.state_dict(),restored.state_dict())
    optimizer_next_error=nested_close(optimizer.state_dict(),restored_optimizer.state_dict())
    nested_equal(scheduler.state_dict(),restored_scheduler.state_dict())
    restored.cpu(); del restored, restored_optimizer, optimizer; gc.collect(); torch.cuda.empty_cache()
    print("tail gradients, immutable BN/frontends and independent full-state resume passed", flush=True)
    # True final-batch normalization on real S23 features; compare complete
    # 50-sample backward with weighted 16+16+16+2 forward/backward accumulation.
    a = frozen.readout.to(args.device).train(); b = copy.deepcopy(a)
    feat = {k:v[:50].to(args.device) for k,v in packets["train_core"]["features"].items()}
    target50 = packets["train_core"]["targets"][:50].to(args.device)
    a.zero_grad(set_to_none=True); reference_loss=loss(a(feat), target50);reference_loss.backward()
    b.zero_grad(set_to_none=True); weighted=0.; sizes=[]
    for begin in range(0,50,16):
        end=min(begin+16,50);sizes.append(end-begin)
        value=loss(b({k:v[begin:end] for k,v in feat.items()}),target50[begin:end]) * ((end-begin)/50)
        value.backward();weighted+=float(value.detach())
    assert math.isclose(weighted,float(reference_loss),rel_tol=RTOL,abs_tol=ATOL)
    accumulation_error=max(close(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters()))
    a.cpu();del b,feat,target50;gc.collect();torch.cuda.empty_cache()
    # Bound the benchmark to three microbatch candidates and three measured
    # training batches each. Includes Dataset loading, pinned transfer and AdamW.
    tail.train(); bench_optimizer=torch.optim.AdamW(tail.optimizer_groups(),betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    trials=[]
    ids=torch.linspace(0,len(datasets["train_core"])-1,128).long().tolist()
    for microbatch in (8,16,32):
        loader=DataLoader(Subset(datasets["train_core"],ids[:microbatch*4]),batch_size=microbatch,
            num_workers=2,pin_memory=True,persistent_workers=False,shuffle=False)
        elapsed=[];torch.cuda.reset_peak_memory_stats(args.device)
        for step,raw in enumerate(loader):
            tick=time.monotonic(); batch_inputs=move(raw["inputs"],args.device);batch_target=raw["target"].to(args.device,non_blocking=True)
            bench_optimizer.zero_grad(set_to_none=True);value=loss(tail(batch_inputs),batch_target);value.backward()
            nn.utils.clip_grad_norm_([p for p in tail.parameters() if p.requires_grad],2.);bench_optimizer.step()
            torch.cuda.synchronize(args.device)
            if step:elapsed.append(time.monotonic()-tick)
        memory=peak_memory(args.device)
        trials.append(dict(microbatch=microbatch,measured_steps=3,warmup_steps=1,
            train_samples_per_second=3*microbatch/sum(elapsed),measured_seconds=elapsed,
            timing="pinned H2D, online forward/backward, clip and AdamW; DataLoader prefetch outside step timer",
            **memory))
        print(f"bounded microbatch {microbatch}: {trials[-1]['train_samples_per_second']:.2f} samples/s peak {memory['cuda_peak_allocated_bytes']/1e9:.2f} GB",flush=True)
        del loader,raw,batch_inputs,batch_target,value;gc.collect();torch.cuda.empty_cache()
    limit=torch.cuda.get_device_properties(args.device).total_memory*.8
    safe=[trial for trial in trials if trial["cuda_peak_reserved_bytes"]<limit]
    assert safe
    selected=max(safe,key=lambda row:row["train_samples_per_second"])
    microbatch=selected["microbatch"]
    loader=DataLoader(Subset(datasets["dev"],list(range(microbatch*4))),batch_size=microbatch,
        num_workers=2,pin_memory=True,shuffle=False)
    tail.eval();eval_elapsed=[]
    with torch.no_grad():
        for step,raw in enumerate(loader):
            tick=time.monotonic();pred=tail(move(raw["inputs"],args.device));assert torch.isfinite(pred).all()
            torch.cuda.synchronize(args.device)
            if step:eval_elapsed.append(time.monotonic()-tick)
    eval_rate=3*microbatch/sum(eval_elapsed)
    estimate=COUNTS["train_core"]/selected["train_samples_per_second"]+COUNTS["dev"]/eval_rate
    unchanged(tail,initial_state)
    # Formal processes must rebuild both parents; no updated weights are outputs
    # consumed by training. Restore initial state here as an additional check.
    tail.load_state_dict(initial_state,strict=True);tail.eval()
    assert model_state_sha256(tail)==model_state_sha256(S23TailModel.from_state(initial_state,tail.group,tail.provenance))
    orders=[]
    for epoch in range(20):
        order,digest=epoch_order(COUNTS["train_core"],42,epoch)
        assert len(order)==COUNTS["train_core"] and len(order)%64==50
        orders.append(dict(zero_based_epoch=epoch,order_sha256=digest,effective_updates=2078,last_batch=50))
    receipt=dict(schema="v5_s23_tail_model_preflight_v1",status="passed",observed_at=now(),formal_training=False,
        runtime_gpu=hardware,parent_encoder_sha256=SOURCE_SHA,parent_readout_sha256=READOUT_SHA,
        cache_identity=index["identity"],cache_index_sha256=sha256(Path(args.cache)/"index.json"),
        full_scope=scope,parameter_manifest_sha256=sha256(output/"parameter_manifest.json"),
        independent_initial_parameter_and_buffer_storage=True,initial_online_comparison=feature_evidence,
        comparison_tolerance=dict(atol=ATOL,rtol=RTOL),module_gradients_and_updates=gradients,
        frozen_state=frozen_evidence,online_eval=dict(current_encoder_tail_calls=calls,
            feature_max_abs_change_after_update=changed,stale_feature_cache_rejected=cache_rejected),
        final_batch_accumulation=dict(actual_batch=50,microbatches=sizes,weights=[n/50 for n in sizes],
            loss=float(reference_loss.detach()),weighted_loss=weighted,gradient_max_abs_error=accumulation_error,
            clip_and_step="once per effective batch; verified by trainer preflight separately"),
        complete_state_restore=dict(path=str(checkpoint),sha256=sha256(checkpoint),parent_files_not_read=True,
            complete_model_optimizer_scheduler_rng_restored=True,roundtrip_state_exact_equal=True,
            next_update_model_max_abs_error=model_next_error,next_update_optimizer_max_abs_error=optimizer_next_error,
            next_update_within_existing_tolerance=True,bitwise_trajectory_guaranteed=False,
            numerical_policy="unchanged FP32 and existing cuDNN nondeterministic backward policy; no tolerance relaxed",
            scheduler_epochs=20,formal_resume_boundary="complete epoch only; synthetic preflight progress is not formal"),
        epoch0_reference=dict(reuse_parent_published_metrics=True,source=str(evaluation_root),
            evaluation_sha256=sha256(evaluation_root/"evaluation.json"),predictions_sha256=evaluation["predictions_sha256"],
            dev_epe=tail.provenance["readout"]["dev_epe"]),
        selected_microbatch=microbatch,throughput=dict(trials=trials,selected_train_samples_per_second=selected["train_samples_per_second"],
            online_eval_samples_per_second=eval_rate,estimated_epoch_seconds=estimate,
            stage_epochs=20,stage_updates=41560,estimated_20_epoch_seconds=20*estimate,
            estimated_20_epoch_range_seconds=[20*estimate*.8,20*estimate*1.4],
            final_best_full_train_eval_seconds=COUNTS["train_core"]/eval_rate,
            caveat="bounded warm measurements; full filesystem loading/checkpoint/publication overhead adds time"),
        epoch_orders=orders,initial_state_restored_after_preflight=True,
        formal_initialization="independent parent reload plus new optimizer, new20 cosine and reset RNG required",
        source_sha256={"fusion_tail.py":sha256(Path(__file__).parent/"purevision_v5/fusion_tail.py"),
                       "verify_fusion_tail.py":sha256(__file__)},elapsed_seconds=time.monotonic()-started)
    write_json(output/"model_preflight.json",receipt)
    print(f"PASSED {output/'model_preflight.json'} selected_microbatch={microbatch}",flush=True)
    return receipt


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("encoder-checkpoint","readout-checkpoint","cache","manifest","flow","rgb-frames","output"):
        parser.add_argument("--"+flag,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--threads",type=int,default=4)
    verify(parser.parse_args())
