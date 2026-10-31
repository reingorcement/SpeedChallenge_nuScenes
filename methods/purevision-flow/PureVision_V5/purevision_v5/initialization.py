"""Explicit weight-only continuation; no optimizer state crosses model semantics."""
from pathlib import Path
import torch

from .common import ROOT, sha256

FLOW_SOURCE_SHA = "4d24688a71935d135a9b153ce568f31b4f2fd9845197c89bdf5accbec43ec823"
RGB_SOURCE_SHA = "1d24368adb86643e68e28b0f69ea4ec658e008ff52b5fb7fc09d7bad06ac0e82"


def resolve_source(path):
    return (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)


def initialize_continuation(model, config):
    from .training import model_state_sha256
    source=resolve_source(config["flow_initialization"]["path"])
    if sha256(source) != config["flow_initialization"]["sha256"]:
        raise ValueError("Flow initialization checksum changed")
    if config["flow_initialization"]["sha256"] != FLOW_SOURCE_SHA:
        raise ValueError("This stage is locked to seed42 FLEX best, not best-of-seeds")
    packet=torch.load(source,map_location="cpu",weights_only=True)
    if packet["run_id"]!="20261031_r02_base_flex_s42_R02_FLEX" or packet["config"]["encoder_variant"]!="flex":
        raise ValueError("Wrong flow initialization run")
    state=packet["model"]
    model.flow_encoder.load_state_dict({k[len("flow_encoder."):]:v for k,v in state.items() if k.startswith("flow_encoder.")},strict=True)
    with torch.no_grad():
        model.head[0].weight.zero_()
        offset=384 if model.uses_rgb else 0
        model.head[0].weight[:,offset:].copy_(state["head.0.weight"])
        model.head[0].bias.copy_(state["head.0.bias"])
        model.head[2].weight.copy_(state["head.2.weight"])
        model.head[2].bias.copy_(state["head.2.bias"])
    mapped={"0.weight":model.head[0].weight[:,offset:],"0.bias":model.head[0].bias,
            "2.weight":model.head[2].weight,"2.bias":model.head[2].bias}
    if not all(torch.equal(tensor,state["head."+name]) for name,tensor in mapped.items()):
        raise ValueError("Copied source head columns/bias/final layer are not exact")
    if offset and torch.count_nonzero(model.head[0].weight[:,:offset]):
        raise ValueError("New RGB input columns must be exactly zero")
    record=dict(policy="seed42_flex_weight_only_continuation_v1",flow_checkpoint_sha256=sha256(source),
        source_run=packet["run_id"],source_training_commit=packet["code"]["commit"],
        source_best_epoch_one_based=packet["best_epoch"]+1,optimizer_inherited=False,head_mapping_exact=True,
        encoder_state_sha256=model_state_sha256(model.flow_encoder),head_state_sha256=model_state_sha256(model.head),
        mapping="RGB columns 0:384 zero; source flow/dt columns -> 384:452" if model.uses_rgb else "source 68-column head unchanged")
    if model.rgb_encoder is not None:
        rgb=resolve_source(config["rgb_initialization"]["path"])
        if sha256(rgb)!=config["rgb_initialization"]["sha256"] or sha256(rgb)!=RGB_SOURCE_SHA:
            raise ValueError("RGB initialization checksum changed")
        model.rgb_encoder.load_pretrained(rgb)
        record.update(rgb_checkpoint_sha256=RGB_SOURCE_SHA,rgb_state_sha256=model_state_sha256(model.rgb_encoder))
        if config.get("rgb_variant")=="appearance_difference":
            from types import SimpleNamespace
            common={k:v for k,v in model.rgb_encoder.state_dict().items() if not k.startswith(("backbone.appearance_stem.","backbone.appearance_alpha"))}
            record["common_rgb_state_sha256"]=model_state_sha256(SimpleNamespace(state_dict=lambda:common))
            record["appearance_initialization"]=dict(policy="isolated_seed42_torch_conv_default",
                stem="Conv3d 3->64 kernel(1,3,3) padding(0,1,1), ReLU",seed=42,
                alpha=float(model.rgb_encoder.backbone.appearance_alpha.detach()),
                stem_state_sha256=model_state_sha256(model.rgb_encoder.backbone.appearance_stem),
                shared_rgb_load="exact V3 keys; only stem weight/bias and alpha are new")
    record["model_state_sha256"]=model_state_sha256(model)
    return record
