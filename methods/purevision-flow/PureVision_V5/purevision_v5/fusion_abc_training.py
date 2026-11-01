"""FP32 training/readout evaluation for the frozen-feature A/B/C pilot.

The only inputs to these models are a single validated common feature cache.
Encoder construction, extraction, and any earlier velocity prediction are
deliberately absent from this training path.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import torch

from .common import ROOT, atomic_save, code_version, environment, git, now, object_hash, read_json, sha256, write_json
from .evaluation import summarize, validate_lateral_rule
from .training import epoch_order, model_state_sha256, restore_rng, rng_state, seed_everything, write_predictions


PROTOCOL = dict(schema="v5_fusion_abc_config_v1", seed=42, epochs=20, batch_size=64,
                drop_last=False, amp=False, tf32=False, learning_rate=1e-3,
                weight_decay=1e-4, betas=[.9, .999], optimizer_eps=1e-8,
                grad_clip=2., scheduler_epochs=20, eta_min=0.,
                loss_scales_m_s=[5., 1.], loss_beta=1., early_stopping=False)
SCHEMA = "v5_fusion_abc_training_state_v1"
EXPECTED_COUNTS = dict(train_core=7324, dev=1826)


def validate_config(config):
    for key, expected in PROTOCOL.items():
        if key not in config or config[key] != expected:
            raise ValueError(f"ABC common protocol mismatch: {key}={config.get(key)!r}, expected {expected!r}")
        if isinstance(expected, bool) and not isinstance(config[key], bool):
            raise ValueError(f"{key} must be explicitly boolean")
    if config.get("variant") not in {"A", "B", "C"}:
        raise ValueError("variant must be A, B or C")
    validate_lateral_rule(config.get("lateral_rule"))


def configure_fp32(threads):
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def new_model(config):
    from .fusion_abc import build_triplet
    # Construct the triplet each time: B/C shared values are explicitly copied
    # by the model factory, independent of variant-specific RNG consumption.
    return build_triplet(config["seed"])[config["variant"]].float()


def load_partition(cache, partition, smoke=False):
    from .fusion_abc_cache import load_cache
    packet, index = load_cache(cache, partition, allow_preflight=smoke)
    if index.get("scope") not in ({"preflight"} if smoke else {"pilot"}):
        raise ValueError("Formal training/evaluation requires a pilot cache; smoke requires a preflight cache")
    count = len(packet["targets"])
    if count < 1:
        raise ValueError("Empty cache partition")
    if not smoke and count != EXPECTED_COUNTS[partition]:
        raise ValueError(f"Formal {partition} cache has {count} windows; expected {EXPECTED_COUNTS[partition]}")
    if len(packet["metadata"]) != count:
        raise ValueError("Features, targets and sample identities must remain one-to-one")
    return packet, index


def _batch(packet, indices, device):
    values = {key: value[indices].to(device=device, dtype=torch.float32)
              for key, value in packet["features"].items()}
    target = packet["targets"][indices].to(device=device, dtype=torch.float32)
    return values, target


def evaluate_packet(model, packet, device, batch_size=64, lateral_rule=None):
    """Accumulate per-sample predictions; the last small batch is not overweighted."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(packet["targets"]), batch_size):
            values, _ = _batch(packet, slice(start, start + batch_size), device)
            prediction = model(values)
            if prediction.shape != (len(values["dt"]), 2) or not torch.isfinite(prediction).all():
                raise RuntimeError("Prediction must be finite physical [vx, vy], shape [N,2]")
            outputs.append(prediction.detach().float().cpu())
    predictions = torch.cat(outputs).numpy()
    targets = packet["targets"].numpy()
    rows = [dict(scene=m["scene"], token=m["token"], time_s=m["time_s"],
                 vx=float(p[0]), vy=float(p[1]), target_vx=float(y[0]), target_vy=float(y[1]))
            for m, p, y in zip(packet["metadata"], predictions, targets)]
    summary = summarize(predictions, targets, [m["scene"] for m in packet["metadata"]], lateral_rule)
    summary["nonzero_vy"]["status"] = "sufficient_data" if summary["nonzero_vy"]["adequate"] else "insufficient_data"
    summary["task"] = "PLANAR_PROXY"
    return summary, rows


def train(args):
    smoke = args.smoke_steps > 0
    if args.smoke_steps < 0:
        raise ValueError("smoke_steps cannot be negative")
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    validate_config(config)
    cache = Path(args.cache).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new run directory, also when resuming")
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("Run output must not overlap the immutable common cache")
    repo = ROOT.parents[2]
    if not smoke:
        if not config_path.is_relative_to(repo) or output.is_relative_to(repo):
            raise ValueError("Formal config must be tracked in the fixed checkout; output must be external")
        git("ls-files", "--error-unmatch", str(config_path.relative_to(repo)))
    version = code_version(require_pushed=not smoke)
    configure_fp32(args.threads)
    seed_everything(config["seed"])
    train_packet, index = load_partition(cache, "train_core", smoke)
    dev_packet, dev_index = load_partition(cache, "dev", smoke)
    if index["identity"] != dev_index["identity"]:
        raise ValueError("Train/dev must use the same cache identity")
    train_scenes = {m["scene"] for m in train_packet["metadata"]}
    dev_scenes = {m["scene"] for m in dev_packet["metadata"]}
    train_tokens = {m["token"] for m in train_packet["metadata"]}
    dev_tokens = {m["token"] for m in dev_packet["metadata"]}
    if train_scenes & dev_scenes or train_tokens & dev_tokens:
        raise ValueError("Training and validation overlap")
    contract = dict(config=config, cache_identity=index["identity"],
                    scope="preflight" if smoke else "frozen_fusion_pilot",
                    smoke_steps=args.smoke_steps, output_units="m/s", task="PLANAR_PROXY")
    contract_hash = object_hash(contract)
    device = torch.device(args.device)
    from .fusion_abc_runtime import runtime_gpu, peak_memory
    hardware=runtime_gpu(device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    model = new_model(config).to(device)
    initialization = dict(policy="build_triplet_explicit_BC_shared_copy_v1",
                          state_sha256=model_state_sha256(model), seed=config["seed"])
    if config["variant"] in {"B", "C"}:
        initialization["common_modules_sha256"] = {
            name: model_state_sha256(getattr(model, name))
            for name in ("input_projection", "input_norm", "head")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"], betas=tuple(config["betas"]),
                                 eps=config["optimizer_eps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["scheduler_epochs"],
                                                         eta_min=config["eta_min"])
    epoch_index, next_batch, processed, updates, completed_epochs = 0, 0, 0, 0, 0
    running_loss_sum, running_samples = 0., 0
    best_epe, best_epoch = math.inf, None
    epoch_lrs, order_records = None, []
    parent_run = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, index)
        if checkpoint["contract_hash"] != contract_hash or checkpoint["code"]["commit"] != version["commit"]:
            raise ValueError("Resume requires the same code, config, cache and execution scope")
        if checkpoint["completed_epochs"] >= config["epochs"]:
            raise ValueError("This run already completed its full training budget")
        if smoke and checkpoint["processed_batches"] >= args.smoke_steps:
            raise ValueError("This disposable preflight already consumed its requested smoke-step budget")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch_index, next_batch = checkpoint["epoch_index"], checkpoint["next_batch"]
        processed, updates = checkpoint["processed_batches"], checkpoint["optimizer_updates"]
        completed_epochs = checkpoint["completed_epochs"]
        running_loss_sum, running_samples = checkpoint["running_loss_sum"], checkpoint["running_samples"]
        best_epe, best_epoch = checkpoint["best_epe"], checkpoint["best_epoch"]
        epoch_lrs, order_records = checkpoint["epoch_lrs"], checkpoint["order_records"]
        initialization = checkpoint["initialization"]
        parent_run = dict(run_id=checkpoint["run_id"], checkpoint=str(Path(args.resume).resolve()),
                          checkpoint_sha256=sha256(args.resume))
        restore_rng(checkpoint["rng"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "cache_identity.json", index)
    receipt = dict(schema="v5_fusion_abc_run_v1", run_id=output.name, status="running",
                   started_at=now(), code=version, contract=contract, contract_hash=contract_hash,
                   initialization=initialization, train_count=len(train_packet["targets"]),
                   dev_count=len(dev_packet["targets"]), parent_run=parent_run,
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   environment=environment(), command=vars(args),
                   arithmetic=dict(dtype="float32", amp=False, tf32=False, gradient_accumulation=1),
                   encoder_training=False)
    receipt['runtime_gpu']=hardware
    write_json(output / "run.json", receipt)
    write_json(output / "epoch_orders.json", order_records)
    # A resumed run keeps the earlier best available without rewriting its
    # source file. Its checkpoint run_id continues to identify the parent.
    if args.resume and best_epoch is not None:
        original_best = Path(args.resume).resolve().with_name("best.pt")
        if not original_best.exists():
            raise ValueError("Resume requires the original best.pt alongside last.pt")
        inherited = torch.load(original_best, map_location="cpu", weights_only=True)
        validate_checkpoint(inherited, index)
        if inherited["best_epe"] != best_epe or inherited["contract_hash"] != contract_hash:
            raise ValueError("Original best checkpoint does not match resume history")
        atomic_save(output / "best.pt", inherited)
    stopping = {"signal": None}

    def request_stop(signum, _frame):
        stopping["signal"] = signum

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    start = time.monotonic()

    def save(path):
        atomic_save(path, dict(schema=SCHEMA, run_id=output.name, saved_at=now(), model=model.state_dict(),
                              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                              scaler={}, amp_enabled=False, rng=rng_state(), config=config, code=version,
                              contract=contract, contract_hash=contract_hash, cache_identity=index["identity"],
                              initialization=initialization, epoch_index=epoch_index, next_batch=next_batch,
                              completed_epochs=completed_epochs, processed_batches=processed,
                              optimizer_updates=updates, amp_skips=0, running_loss_sum=running_loss_sum,
                              running_samples=running_samples, best_epe=best_epe, best_epoch=best_epoch,
                              epoch_lrs=epoch_lrs, order_records=order_records, parent_run=parent_run))

    def status(state):
        value = dict(status=state, updated_at=now(), run_id=output.name, scope=contract["scope"],
                     completed_epochs=completed_epochs,
                     epoch=None if completed_epochs == config["epochs"] else epoch_index + 1,
                     next_batch=next_batch, processed_batches=processed, optimizer_updates=updates,
                     amp_skips=0, best_dev_epe=None if best_epoch is None else best_epe,
                     best_epoch=best_epoch, elapsed_seconds=time.monotonic() - start,
                     runtime_gpu=hardware,**peak_memory(device))
        write_json(output / "status.json", value)
        return value

    try:
        for e in range(epoch_index, config["epochs"]):
            epoch_index = e
            order, order_hash = epoch_order(len(train_packet["targets"]), config["seed"], e)
            record = dict(epoch=e + 1, zero_based_epoch=e, sample_count=len(order), order_sha256=order_hash)
            if len(order_records) <= e:
                order_records.append(record)
            elif order_records[e] != record:
                raise ValueError("Resumed epoch order differs")
            write_json(output / "epoch_orders.json", order_records)
            if next_batch == 0:
                epoch_lrs = [group["lr"] for group in optimizer.param_groups]
            model.train()
            batches = math.ceil(len(order) / config["batch_size"])
            for b in range(next_batch, batches):
                chosen = order[b * config["batch_size"]:(b + 1) * config["batch_size"]]
                values, target = _batch(train_packet, chosen, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(values)
                scale = target.new_tensor(config["loss_scales_m_s"])
                loss = torch.nn.functional.smooth_l1_loss(prediction / scale, target / scale,
                                                          beta=config["loss_beta"], reduction="mean")
                if prediction.shape != target.shape or not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss or invalid prediction shape")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                processed += 1
                updates += 1
                next_batch = b + 1
                running_loss_sum += float(loss.detach()) * len(chosen)
                running_samples += len(chosen)
                print(json.dumps(dict(event="update", epoch=e + 1, batch=b + 1,
                                      processed_batches=processed, optimizer_updates=updates, amp_skips=0,
                                      loss=float(loss.detach()), grad_norm=float(norm))), flush=True)
                status("running")
                if processed % 100 == 0 or stopping["signal"] or (smoke and processed >= args.smoke_steps):
                    save(output / "last.pt")
                # At a complete epoch boundary always validate and scheduler.step
                # before exiting. A partial epoch never advances the scheduler.
                if b + 1 < batches and (stopping["signal"] or (smoke and processed >= args.smoke_steps)):
                    break
            if next_batch == batches:
                summary, _ = evaluate_packet(model, dev_packet, device, config["batch_size"], config.get("lateral_rule"))
                epe = summary["overall"]["epe"]
                improved = epe < best_epe  # Strict tie rule preserves the earlier epoch.
                if improved:
                    best_epe, best_epoch = epe, e + 1
                mean_loss = running_loss_sum / running_samples
                scheduler.step()
                completed_epochs, epoch_index, next_batch = e + 1, e + 1, 0
                row = dict(epoch=e + 1, training_loss=mean_loss, dev=summary,
                           learning_rates=epoch_lrs, next_learning_rates=[g["lr"] for g in optimizer.param_groups],
                           sample_order_sha256=order_hash, processed_batches=processed,
                           optimizer_updates=updates, amp_skips=0, best_epoch=best_epoch,
                           best_dev_epe=best_epe, updated_at=now())
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                running_loss_sum, running_samples, epoch_lrs = 0., 0, None
                save(output / "last.pt")
                if improved:
                    save(output / "best.pt")
                print(json.dumps(dict(event="validation", **row)), flush=True)
                status("running")
            if stopping["signal"] or (smoke and processed >= args.smoke_steps):
                break
        save(output / "last.pt")
        final_state = "interrupted" if stopping["signal"] else ("preflight_completed" if smoke else "completed")
        receipt.update(status=final_state, finished_at=now(), stop_signal=stopping["signal"], **{
            k: v for k, v in status(final_state).items() if k not in {"status", "run_id", "scope"}})
        receipt["last_checkpoint_sha256"] = sha256(output / "last.pt")
        if (output / "best.pt").exists():
            receipt["best_checkpoint_sha256"] = sha256(output / "best.pt")
        if smoke:
            # Prove the independent reload path using the saved, full state.
            restored = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            validate_checkpoint(restored, index)
            reloaded = new_model(config).to(device)
            reloaded.load_state_dict(restored["model"], strict=True)
            if model_state_sha256(reloaded) != model_state_sha256(model):
                raise RuntimeError("Saved model changed on independent reload")
            smoke_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=config["learning_rate"])
            smoke_optimizer.load_state_dict(restored["optimizer"])
            smoke_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(smoke_optimizer, T_max=20, eta_min=0)
            smoke_scheduler.load_state_dict(restored["scheduler"])
            summary, rows = evaluate_packet(reloaded, dev_packet, device, config["batch_size"], config.get("lateral_rule"))
            write_json(output / "preflight_dev_metrics.json", summary)
            write_predictions(output / "preflight_dev_predictions.csv", rows)
            receipt["preflight_reload"] = dict(model_exact=True, optimizer_loaded=True, scheduler_loaded=True,
                                               evaluated_dev_count=len(rows), formal_training=False)
        write_json(output / "run.json", receipt)
        return receipt
    except BaseException as error:
        receipt.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", receipt)
        status("failed")
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def validate_checkpoint(checkpoint, index):
    if (checkpoint.get("schema") != SCHEMA
            or checkpoint.get("contract_hash") != object_hash(checkpoint.get("contract", {}))
            or checkpoint.get("config") != checkpoint.get("contract", {}).get("config")
            or checkpoint.get("cache_identity") != index["identity"]
            or checkpoint.get("cache_identity") != checkpoint.get("contract", {}).get("cache_identity")):
        raise ValueError("Checkpoint/cache contract mismatch")
    validate_config(checkpoint["config"])
    if (checkpoint.get("amp_enabled") is not False or checkpoint.get("amp_skips") != 0
            or checkpoint.get("processed_batches") != checkpoint.get("optimizer_updates")):
        raise ValueError("FP32 checkpoint update counters are inconsistent")


def require_completed_run(run_path, checkpoint_path, checkpoint):
    """Optional lifecycle gate: a safe pause must not trigger final comparison."""
    run_path = Path(run_path).resolve()
    receipt = read_json(run_path / "run.json")
    if receipt.get("status") != "completed":
        raise ValueError("Final evaluation requires a completed formal run, not a pause or preflight")
    if (Path(checkpoint_path).resolve().parent != run_path
            or receipt.get("run_id") != run_path.name
            or receipt.get("code", {}).get("commit") != checkpoint.get("code", {}).get("commit")
            or receipt.get("contract_hash") != checkpoint.get("contract_hash")
            or receipt.get("contract", {}).get("cache_identity") != checkpoint.get("cache_identity")
            or receipt.get("completed_epochs") != checkpoint["config"]["epochs"]):
        raise ValueError("Completed run receipt and evaluated checkpoint do not match")
    return receipt


def evaluate(args):
    configure_fp32(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    smoke = checkpoint.get("contract", {}).get("scope") == "preflight"
    packet, index = load_partition(args.cache, args.partition, smoke)
    validate_checkpoint(checkpoint, index)
    completed_run = getattr(args, "require_completed_run", None)
    if completed_run:
        require_completed_run(completed_run, args.checkpoint, checkpoint)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output must be new")
    if output.is_relative_to(Path(args.cache).resolve()) or Path(args.cache).resolve().is_relative_to(output):
        raise ValueError("Evaluation output must not overlap the cache")
    model = new_model(checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(status="running", started_at=now(), checkpoint=str(Path(args.checkpoint).resolve()),
                   checkpoint_sha256=sha256(args.checkpoint), training_run=checkpoint["run_id"],
                   training_code=checkpoint["code"], evaluation_code=code_version(),
                   cache_identity=index["identity"], contract_hash=checkpoint["contract_hash"],
                   scope=checkpoint["contract"]["scope"], partition=args.partition,
                   config=checkpoint["config"], environment=environment(), command=vars(args))
    write_json(output / "evaluation.json", receipt)
    try:
        summary, rows = evaluate_packet(model, packet, args.device, checkpoint["config"]["batch_size"],
                                       checkpoint["config"].get("lateral_rule"))
        write_json(output / "metrics.json", summary)
        write_predictions(output / "predictions.csv", rows)
        receipt.update(status="completed", finished_at=now(), count=len(rows))
    except BaseException as error:
        receipt.update(status="failed", finished_at=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "evaluation.json", receipt)
    print(json.dumps(summary["overall"]), flush=True)
    return summary


def train_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="Positive count: disposable preflight-cache updates only, never a formal run")
    parser.add_argument("--resume", help="Full state from same fixed code/config/cache; output must be new")
    train(parser.parse_args())


def evaluate_main():
    parser = argparse.ArgumentParser(description="Independently reload an A/B/C checkpoint and export physical predictions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partition", choices=("train_core", "dev"), default="dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--require-completed-run",
                        help="Optional final-report gate: require this associated run.json to show 20 completed epochs")
    evaluate(parser.parse_args())
