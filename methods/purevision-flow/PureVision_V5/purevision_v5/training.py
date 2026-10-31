"""One trainer for R00--R05 with immutable provenance and complete resumable state."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .common import ROOT, atomic_save, code_version, environment, git, now, object_hash, read_json, write_json
from .datasets import VelocityDataset
from .evaluation import summarize, validate_lateral_rule
from .model import VelocityModel, build_model
from .optimization import make_optimizer, make_scheduler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def epoch_order(count, seed, epoch):
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch))
    identity = hashlib.sha256(order.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
    return order.tolist(), identity


def validate_expected_flow(config, data_contract):
    expected = config.get("expected_flow")
    if expected is None:
        return
    flow = data_contract.get("flow") or {}
    native = flow.get("native_contract", flow.get("preprocessing", {}))
    view = flow.get("derived_contract", flow.get("preprocessing", {}))
    actual = dict(raft_size_hw=native.get("raft_size"), model_size_hw=view.get("model_size"))
    if expected != actual:
        raise ValueError(f"Expected optical-flow resolution contract differs: {expected} != {actual}")

def rng_state():
    value = np.random.get_state()
    return dict(python=random.getstate(), numpy=[value[0], value[1].tolist(), value[2], value[3], value[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def loss_fn(prediction, target, config):
    scale = target.new_tensor(config["loss_scales_m_s"])
    return torch.nn.functional.smooth_l1_loss(prediction / scale, target / scale, beta=config["loss_beta"])


def optimizer_update(model, optimizer, scaler, grad_clip):
    """Unscale before clipping; let GradScaler skip overflowed gradients safely.

    GradScaler.unscale_ records inf/NaN per optimizer. Its public step() skips
    that update and update() reduces the scale. Clipping a nonfinite gradient
    before this branch would either abort normal AMP recovery or mutate it.
    """
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("No gradients for the velocity model")
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters], error_if_nonfinite=False)
    if not torch.isfinite(norm):
        finite_gradients = all(bool(torch.isfinite(p.grad).all()) for p in parameters)
        if finite_gradients:
            raise RuntimeError("Nonfinite gradient norm with finite gradients; not an AMP overflow skip")
        if not scaler.is_enabled():
            raise RuntimeError("Nonfinite gradients with AMP disabled")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)  # Recorded inf/NaN guarantees no parameter/Adam update.
        scaler.update()
        scale_after = scaler.get_scale()
        if not np.isfinite(scale_after) or not 0 < scale_after < scale_before:
            raise RuntimeError("AMP overflow did not reduce its finite positive loss scale")
        return dict(updated=False, scale_before=scale_before, scale_after=scale_after)
    torch.nn.utils.clip_grads_with_norm_(parameters, grad_clip, norm)
    scaler.step(optimizer)
    scaler.update()
    return dict(updated=True)


def move_inputs(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch["inputs"].items()}


def evaluate_model(model, dataset, device, batch_size=64, workers=4, max_batches=0, lateral_rule=None):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                        pin_memory=str(device).startswith("cuda"), generator=torch.Generator().manual_seed(0))
    predictions, targets, scenes, tokens, times = [], [], [], [], []
    valid_sum = fb_sum = pixel_count = 0
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            prediction = model(move_inputs(batch, device)).float().cpu().numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError("Nonfinite evaluation prediction")
            predictions.extend(prediction)
            targets.extend(batch["target"].numpy())
            scenes.extend(batch["scene"])
            tokens.extend(batch["token"])
            times.extend(batch["time_s"].tolist())
            if "valid" in batch["inputs"]:
                valid = batch["inputs"]["valid"]
                valid_sum += int(valid.sum())
                pixel_count += valid.numel()
                if "fb_mask" in batch["inputs"]:
                    fb_sum += int(batch["inputs"]["fb_mask"].sum())
    summary = summarize(predictions, targets, scenes, lateral_rule=lateral_rule)
    if pixel_count:
        summary["flow_coverage"] = dict(valid_fraction=valid_sum/pixel_count,
                                        fb_fraction=fb_sum/pixel_count if dataset.experiment in {"R04", "R05"} else None)
    rows = [dict(scene=s, token=t, time_s=ts, vx=float(p[0]), vy=float(p[1]),
                 target_vx=float(y[0]), target_vy=float(y[1]))
            for s, t, ts, p, y in zip(scenes, tokens, times, predictions, targets)]
    return summary, rows


def write_predictions(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    config = read_json(args.config)
    lateral_rule = validate_lateral_rule(config.get("lateral_rule"))
    early_stopping = config.get("early_stopping", True)
    if not isinstance(early_stopping, bool):
        raise ValueError("early_stopping must be explicitly boolean")
    for key in ("epochs", "batch_size", "log_steps", "checkpoint_steps", "patience"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    scales = np.asarray(config["loss_scales_m_s"], dtype=float)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Loss requires two positive finite m/s scales")
    for key in ("learning_rate", "loss_beta", "grad_clip"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    experiment = config["experiment"]
    micro_batch=config.get("micro_batch_size",config["batch_size"])
    if not isinstance(micro_batch,int) or not 1<=micro_batch<=config["batch_size"] or config["batch_size"]%micro_batch:
        raise ValueError("micro_batch_size must divide the effective batch_size")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new run directory, including when resuming")
    config_path = Path(args.config).resolve()
    repo = ROOT.parents[2]
    if not args.smoke:
        if not config_path.is_relative_to(repo):
            raise ValueError("Formal configuration must be inside the fixed training checkout")
        git("ls-files", "--error-unmatch", str(config_path.relative_to(repo)))
        if output.is_relative_to(repo):
            raise ValueError("Formal outputs must live outside the immutable training checkout")
    protected = [Path(args.manifest).resolve(), config_path]
    protected.extend(Path(value).resolve() for value in (args.features, args.flow) if value)
    if getattr(args,"rgb_frames",None):protected.append(Path(args.rgb_frames).resolve())
    if getattr(args, "selection", None):
        protected.append(Path(args.selection).resolve())
    if any(path.is_relative_to(output) or output.is_relative_to(path) for path in protected):
        raise ValueError("Output must not contain or overwrite any input/configuration path")
    version = code_version(require_pushed=not args.smoke)
    torch.set_num_threads(args.threads)
    seed_everything(config["seed"])
    device = torch.device(args.device)
    limits = dict(limit_scenes=args.limit_scenes, limit_windows=args.limit_windows)
    selection_path = getattr(args, "selection", None)
    if selection_path and (args.smoke or args.limit_scenes or args.limit_windows or args.max_batches):
        raise ValueError("An explicit resolution pilot cannot use smoke or ad hoc subset limits")
    if not args.smoke and (args.limit_scenes or args.limit_windows or args.max_batches):
        raise ValueError("Subset/batch limits are smoke-only, not formal experiment results")
    rgb_args=dict(rgb_policy=config.get("rgb_policy"),frame_root=getattr(args,"rgb_frames",None))
    train_data = VelocityDataset(args.manifest, "train_core", experiment, args.features, args.flow, selection_path=selection_path, **rgb_args, **limits)
    dev_data = VelocityDataset(args.manifest, "dev", experiment, args.features, args.flow, selection_path=selection_path, **rgb_args, **limits)
    if train_data.scene_names & dev_data.scene_names or train_data.source_tokens & dev_data.source_tokens:
        raise ValueError("Training/validation overlap")
    if train_data.contract != dev_data.contract:
        raise ValueError("Training and validation data contracts differ")
    validate_expected_flow(config, train_data.contract)
    if config.get("rgb_policy")=="frozen" and config.get("flow_initialization"):
        features=train_data.contract["features"]
        if (features.get("weights_sha256")!=config["rgb_initialization"]["sha256"]
                or features.get("input_size")!=[5,3,72,128]
                or features.get("normalization")!="PIL RGB bilinear; x/127.5-1"
                or features.get("dtype")!="float32"
                or features.get("compute_precision")!="float32"):
            raise ValueError("Frozen descriptor source/precision differs from RGB initialization")
    contract = dict(config=config, data=train_data.contract, train_selection=train_data.selection_hash,
                    dev_selection=dev_data.selection_hash,
                    execution=dict(smoke=bool(args.smoke), max_batches=args.max_batches))
    if train_data.scene_selection:
        contract["execution"].update(scope="resolution_pilot", scene_selection_identity=train_data.scene_selection["identity"])
    contract_hash = object_hash(contract)
    model = build_model(config)
    continuation=None
    if config.get("flow_initialization") and not args.resume:
        from .initialization import initialize_continuation
        continuation=initialize_continuation(model,config)
    model=model.to(device)
    initialization = dict(policy="seeded_fresh_model_v1", seed=config["seed"], model_state_sha256=model_state_sha256(model))
    if config.get("head_seed") is not None:
        initialization.update(policy="encoder_torch_default_and_isolated_head_rng_v1",
                              head_seed=config["head_seed"], head_state_sha256=model_state_sha256(model.head),
                              encoder_policy="PyTorch Conv/Linear reset_parameters; BN weight=1 bias=0",
                              encoder_state_sha256=model_state_sha256(model.flow_encoder))
    if continuation is not None:initialization=continuation
    sample_order = dict(policy="torch_randperm_seed_plus_zero_based_epoch_v1", seed=config["seed"],
                        training_samples_sha256=object_hash([[i["scene"], i["token"], i["time_s"], i["target"].tolist()]
                                                             for i in train_data.items]),
                        epoch_index_dtype="little_endian_int64")
    optimizer = make_optimizer(model,config)
    scheduler = make_scheduler(optimizer,config)
    amp = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epoch, next_batch, global_step, bad_epochs = 0, 0, 0, 0
    optimizer_updates, amp_skips = 0, 0
    processed_microbatches = 0
    best_epe, best_epoch = float("inf"), None
    running_loss, running_count = 0., 0
    parent_run = None
    inherited_best = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("schema") != "v5_training_state_v1" or checkpoint["contract_hash"] != contract_hash:
            raise ValueError("Resume rejected: model/label/loss/data/config semantics changed")
        if checkpoint["code"]["commit"] != version["commit"]:
            raise ValueError("Resume rejected: code commit changed; start a new experiment")
        counters = [checkpoint.get(k) for k in ("processed_batches", "optimizer_updates", "amp_skips")]
        if (any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counters)
                or counters[0] != checkpoint["global_step"] or counters[0] != counters[1] + counters[2]):
            raise ValueError("Resume rejected: update/overflow counters missing or inconsistent")
        _, optimizer_updates, amp_skips = counters
        processed_microbatches=checkpoint.get("processed_microbatches",checkpoint["processed_batches"])
        initialization=checkpoint["initialization"]
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch, next_batch, global_step = (checkpoint[key] for key in ("epoch", "next_batch", "global_step"))
        best_epe, best_epoch, bad_epochs = checkpoint["best_epe"], checkpoint["best_epoch"], checkpoint["bad_epochs"]
        running_loss, running_count = checkpoint["running_loss"], checkpoint["running_count"]
        parent_run = checkpoint["run_id"]
        restore_rng(checkpoint["rng"])
        if epoch >= config["epochs"] or (early_stopping and bad_epochs >= config["patience"]):
            raise ValueError("Resume rejected: the saved run already exhausted its stopping condition")
        if best_epoch is not None:
            inherited_best = Path(args.resume).resolve().parent / "best.pt"
            previous = torch.load(inherited_best, map_location="cpu", weights_only=True)
            if (previous.get("contract_hash") != contract_hash or previous.get("best_epoch") != best_epoch
                    or previous.get("best_epe") != best_epe):
                raise ValueError("Resume rejected: historical best checkpoint is missing or inconsistent")
    output.mkdir(parents=True, exist_ok=True)
    if inherited_best is not None:
        for name in ("best.pt", "best_metrics.json", "dev_predictions.csv"):
            source = inherited_best.parent / name
            if not source.is_file():
                raise ValueError(f"Resume requires historical best artifact: {source}")
            shutil.copy2(source, output / name)
    if args.resume:
        history = Path(args.resume).resolve().parent / "metrics.jsonl"
        if history.is_file():
            retained = [line for line in history.read_text().splitlines()
                        if json.loads(line)["epoch"] < epoch and json.loads(line)["global_step"] <= global_step]
            if retained:
                (output / "metrics.jsonl").write_text("\n".join(retained) + "\n")
    run = dict(run_id=output.name, experiment=experiment, status="running", started_at=now(), code=version,
               config=config, contract=contract, contract_hash=contract_hash, smoke=args.smoke,
               parent_run=parent_run, resume=str(Path(args.resume).resolve()) if args.resume else None,
               runtime=vars(args), environment=environment(), train_windows=len(train_data), dev_windows=len(dev_data),
               train_scenes=len(train_data.scene_names), dev_scenes=len(dev_data.scene_names),
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               total_parameters=sum(p.numel() for p in model.parameters()),
               step_semantics="global_step = processed_batches = optimizer_updates + amp_skips",
               checkpoint=str(output / "last.pt"))
    run.update(scope="resolution_pilot" if train_data.scene_selection else ("smoke" if args.smoke else "full_train_dev"),
               initialization=initialization, sample_order=sample_order, effective_lateral_rule=lateral_rule,
               stopping_policy=dict(early_stopping=early_stopping, complete_epoch_budget=config["epochs"]))
    run.update(micro_batch_size=micro_batch,effective_batch_size=config["batch_size"],
        accumulation_policy="dataloader batch is effective batch; micro losses weighted by actual sample count; update/checkpoint at group boundary",
        optimizer_groups=[dict(name=g["name"],initial_lr=g["initial_lr"],parameters=sum(p.numel() for p in g["params"]),
            tensor_count=len(g["params"])) for g in optimizer.param_groups])
    write_json(output / "run.json", run)
    stop_requested = [False]
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))
    started = last_save = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def state(status, **extra):
        write_json(output / "status.json", dict(run_id=output.name, experiment=experiment, status=status,
                   updated_at=now(), epoch=epoch, global_step=global_step,
                   processed_batches=global_step, optimizer_updates=optimizer_updates, amp_skips=amp_skips,
                   processed_microbatches=processed_microbatches,
                   amp_scale=scaler.get_scale(),
                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   best_epe=best_epe if np.isfinite(best_epe) else None, best_epoch=best_epoch,
                   elapsed_seconds=time.monotonic()-started, **extra))

    def save(path, epoch_value, batch_value):
        atomic_save(path, dict(schema="v5_training_state_v1", run_id=output.name, code=version,
                    contract_hash=contract_hash, contract=contract, config=config,
                    model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(), rng=rng_state(), epoch=epoch_value, next_batch=batch_value,
                    initialization=initialization, sample_order=sample_order, effective_lateral_rule=lateral_rule,
                    global_step=global_step, best_epe=best_epe, best_epoch=best_epoch, bad_epochs=bad_epochs,
                    processed_batches=global_step, optimizer_updates=optimizer_updates, amp_skips=amp_skips,
                    processed_microbatches=processed_microbatches,
                    running_loss=running_loss, running_count=running_count))

    state("running")
    try:
        while epoch < config["epochs"]:
            model.train()
            order, order_sha256 = epoch_order(len(train_data), config["seed"], epoch)
            offset = next_batch * config["batch_size"]
            loader = DataLoader(Subset(train_data, order[offset:]), batch_size=config["batch_size"],
                                shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda",
                                generator=torch.Generator().manual_seed(config["seed"] + epoch))
            start_batch = next_batch
            for local_batch, batch in enumerate(loader):
                batch_number = start_batch + local_batch
                if args.max_batches and batch_number >= args.max_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                target = batch["target"].to(device)
                loss=target.new_zeros(())
                for lo in range(0,len(target),micro_batch):
                    hi=min(lo+micro_batch,len(target))
                    micro={k:v[lo:hi].to(device,non_blocking=True) for k,v in batch["inputs"].items()}
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                        prediction=model(micro)
                        value=loss_fn(prediction.float(),target[lo:hi],config)
                    if not torch.isfinite(value):raise RuntimeError("Nonfinite loss")
                    weight=(hi-lo)/len(target)
                    scaler.scale(value*weight).backward()
                    loss+=value.detach()*weight
                    processed_microbatches+=1
                update = optimizer_update(model, optimizer, scaler, config["grad_clip"])
                global_step += 1
                optimizer_updates += int(update["updated"])
                amp_skips += int(not update["updated"])
                next_batch = batch_number + 1
                running_loss += float(loss.detach()) * len(target)
                running_count += len(target)
                if not update["updated"]:
                    print(json.dumps(dict(event="amp_overflow", epoch=epoch, step=global_step,
                          processed_batches=global_step, optimizer_updates=optimizer_updates, amp_skips=amp_skips,
                          loss=float(loss.detach()), **update, time=now())), flush=True)
                if global_step <= 3 or global_step % config["log_steps"] == 0:
                    state("running", loss=float(loss.detach()))
                    print(json.dumps(dict(event="step", epoch=epoch, step=global_step, processed_batches=global_step,
                          optimizer_updates=optimizer_updates, amp_skips=amp_skips, amp_scale=scaler.get_scale(),
                          processed_microbatches=processed_microbatches,
                          loss=float(loss.detach()), time=now())), flush=True)
                checkpoint_due = global_step % config["checkpoint_steps"] == 0 or time.monotonic()-last_save >= 120
                interrupted = stop_requested[0] or (args.stop_after_steps and global_step >= args.stop_after_steps)
                if checkpoint_due or interrupted:
                    save(output / "last.pt", epoch, next_batch)
                    last_save = time.monotonic()
                if interrupted:
                    state("interrupted", next_batch=next_batch)
                    run.update(status="interrupted", finished_at=now())
                    write_json(output / "run.json", run)
                    return output
            summary, rows = evaluate_model(model, dev_data, device, config.get("evaluation_batch_size",config["batch_size"]), args.workers, args.max_batches,
                                           lateral_rule=lateral_rule)
            epe = summary["overall"]["epe"]
            improved = epe < best_epe - config["min_delta"]
            if improved:
                best_epe, best_epoch, bad_epochs = epe, epoch, 0
            else:
                bad_epochs += 1
            entry = dict(epoch=epoch, global_step=global_step, train_loss=running_loss/max(running_count, 1),
                         sample_order_sha256=order_sha256,
                         processed_batches=global_step, optimizer_updates=optimizer_updates, amp_skips=amp_skips,
                         processed_microbatches=processed_microbatches,
                         amp_scale=scaler.get_scale(),
                         learning_rate=optimizer.param_groups[0]["lr"], validation=summary, time=now())
            entry["learning_rates"]={g["name"]:g["lr"] for g in optimizer.param_groups}
            with open(output / "metrics.jsonl", "a") as stream:
                stream.write(json.dumps(entry, allow_nan=False)+"\n")
            print(json.dumps(dict(event="validation", epoch=epoch, **summary["overall"])), flush=True)
            scheduler.step()
            epoch += 1
            next_batch, running_loss, running_count = 0, 0., 0
            if improved:
                save(output / "best.pt", epoch, 0)
                write_json(output / "best_metrics.json", summary)
                write_predictions(output / "dev_predictions.csv", rows)
            # Commit the resumable epoch state only after all best artifacts exist.
            save(output / "last.pt", epoch, 0)
            state("running", validation=summary["overall"])
            if early_stopping and bad_epochs >= config["patience"]:
                break
        state("completed", stop_reason="early_stopping" if early_stopping and bad_epochs >= config["patience"] else "epoch_budget")
        run["status"], run["finished_at"] = "completed", now()
        write_json(output / "run.json", run)
        return output
    except BaseException as error:
        state("failed", error=f"{type(error).__name__}: {error}")
        run.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", run)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "manifest", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--features")
    parser.add_argument("--rgb-frames",help="Verified uint8 RGB image views for actual RGB fine-tuning")
    parser.add_argument("--flow")
    parser.add_argument("--selection", help="Immutable explicit resolution-pilot scene selection JSON")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--stop-after-steps", type=int, default=0)
    train(parser.parse_args())
