"""S23 continuation with strictly scoped differentiable encoder tails.

The frozen extractor is used only to validate and load its trained source.
Online forward below is independent of its permanent no_grad forward.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path

import torch
from torch import nn

from .common import object_hash
from .fusion_abc import FrozenFusionExtractor, SOURCE_SHA, file_sha256, spatial_tokens
from .fusion_grid import FusionGrid, validate_grid_features
from .model import FLOW_SHAPE, MASK_SHAPE, _binary_mask, _finite_tensor, build_model as build_encoder
from .rgb import RGB_SHAPE

READOUT_SHA = "39d517001824414da1a49b6660a4cf5fdb9545c484cc87da0203032b068a5171"
PARENT_EPE = 0.17847165237392404
GROUPS = ("S23_FROZEN_CONT", "S23_TAIL_FT")
TAIL_MODULES = ("flow_encoder.sfe", "flow_encoder.temporal", "rgb_encoder.backbone.sfe_module",
                "rgb_encoder.backbone.mfe_module", "rgb_encoder.backbone.dig")
ARCHITECTURE = dict(experiment="R03", encoder_variant="flex", rgb_policy="finetune",
                    rgb_variant="appearance_difference")


class S23TailModel(nn.Module):
    """Complete encoders + historical S23 readout, with independent storage."""

    def __init__(self, encoder_checkpoint, readout_checkpoint, group="S23_TAIL_FT"):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(f"Unknown continuation group: {group}")
        readout_checkpoint = Path(readout_checkpoint).resolve()
        readout_sha = file_sha256(readout_checkpoint)
        if readout_sha != READOUT_SHA:
            raise ValueError("S23 continuation requires the fixed complete-data seed42 epoch19 best")
        parent = torch.load(readout_checkpoint, map_location="cpu", weights_only=True)
        if (parent.get("schema") != "v5_fusion_grid_epoch_state_v1"
                or parent.get("contract_hash") != object_hash(parent.get("contract", {}))
                or parent.get("config") != parent.get("contract", {}).get("config")
                or parent["config"].get("group") != "S23" or parent["config"].get("scope") != "full"
                or parent["config"].get("seed") != 42 or parent.get("completed_epochs") != 19
                or parent.get("best_epoch") != 19 or parent.get("best_epe") != PARENT_EPE):
            raise ValueError("S23 parent protocol, selection epoch or checkpoint contract differs")
        with torch.random.fork_rng(devices=[]):
            frozen = FrozenFusionExtractor(encoder_checkpoint)
            self.flow_encoder = frozen.source.flow_encoder
            self.rgb_encoder = frozen.source.rgb_encoder
            # Retain the exact old head only as frozen source state; never execute it.
            self.legacy_head = frozen.source.head
            self.readout = FusionGrid("S23")
        self.readout.load_state_dict(parent["model"], strict=True)
        self.provenance = dict(schema="v5_s23_tail_parent_provenance_v1",
            encoder=copy.deepcopy(frozen.provenance),
            readout=dict(checkpoint=str(readout_checkpoint), checkpoint_sha256=readout_sha,
                source_run_id=parent["run_id"], source_code=copy.deepcopy(parent["code"]),
                source_contract_hash=parent["contract_hash"], cache_identity=parent["cache_identity"],
                source_config=copy.deepcopy(parent["config"]), selected_epoch=19,
                parent_budget_epochs=20, parent_budget_optimizer_updates=41560,
                selected_checkpoint_optimizer_updates=parent["optimizer_updates"], dev_epe=PARENT_EPE),
            architecture=copy.deepcopy(ARCHITECTURE), stage_budget_epochs=20,
            stage_optimizer_updates=41560, stage_initialization="new stage from best; reset optimizer/scheduler/RNG",
            unused_source_head="legacy_head is checkpointed and frozen, never called")
        self._configure(group)

    def _configure(self, group):
        if group not in GROUPS:
            raise ValueError("Unknown continuation group")
        self.group = group
        self.requires_grad_(False)
        self.readout.requires_grad_(True)
        if group == "S23_TAIL_FT":
            for name in TAIL_MODULES:
                self.get_submodule(name).requires_grad_(True)
        for encoder in (self.flow_encoder, self.rgb_encoder):
            for module in encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        # No caller, including an enclosing model.train(), may reactivate BN.
        self.flow_encoder.eval()
        self.rgb_encoder.eval()
        self.legacy_head.eval()
        if mode and self.group == "S23_TAIL_FT":
            for name in TAIL_MODULES:
                self.get_submodule(name).train(True)
        for encoder in (self.flow_encoder, self.rgb_encoder):
            for module in encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        self.readout.train(mode)
        return self

    @classmethod
    def from_state(cls, state_dict, group, provenance):
        """Restore complete model without reading either parent checkpoint."""
        if (provenance.get("schema") != "v5_s23_tail_parent_provenance_v1"
                or provenance["encoder"]["checkpoint_sha256"] != SOURCE_SHA
                or provenance["readout"]["checkpoint_sha256"] != READOUT_SHA
                or provenance.get("architecture") != ARCHITECTURE):
            raise ValueError("Full-state checkpoint has incompatible parent provenance")
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        with torch.random.fork_rng(devices=[]):
            source = build_encoder(ARCHITECTURE)
            self.flow_encoder = source.flow_encoder
            self.rgb_encoder = source.rgb_encoder
            self.legacy_head = source.head
            self.readout = FusionGrid("S23")
        self.provenance = copy.deepcopy(provenance)
        self.load_state_dict(state_dict, strict=True)
        self._configure(group)
        return self

    def _inputs(self, inputs):
        required = {"images", "flows", "valid", "pair_dt"}
        if not isinstance(inputs, Mapping) or set(inputs) != required:
            raise ValueError(f"Online S23 accepts exactly {sorted(required)}")
        dt = inputs["pair_dt"]
        if not isinstance(dt, torch.Tensor) or dt.ndim != 2 or len(dt) < 1:
            raise ValueError("pair_dt must be nonempty [B,4]")
        n = len(dt)
        _finite_tensor(dt, (n, 4), "pair_dt")
        images = _finite_tensor(inputs["images"], (n, *RGB_SHAPE), "images")
        flow = inputs["flows"]
        if not isinstance(flow, torch.Tensor) or tuple(flow.shape) != (n, *FLOW_SHAPE):
            raise ValueError("Invalid flow geometry")
        valid = _binary_mask(inputs["valid"], (n, *MASK_SHAPE), "valid")
        if (any(x.dtype != torch.float32 for x in (images, flow, dt))
                or any(x.device != dt.device for x in (images, flow, valid))
                or bool((dt <= 0).any()) or bool((images.abs() > 1.00001).any())):
            raise ValueError("Online inputs must preserve FP32, positive seconds and existing RGB normalization")
        return n, images, flow, valid, dt

    def extract_features(self, inputs):
        n, images, flow, valid, dt = self._inputs(inputs)
        # Only frozen frontends are outside autograd. SFE/MFE receive the same
        # stem/AFT tensors in parallel; no detach occurs after their trainable ops.
        with torch.no_grad():
            valid = valid & torch.isfinite(flow).all(dim=2, keepdim=True)
            safe = torch.where(valid.expand_as(flow), flow, torch.zeros_like(flow))
            encoded = torch.cat((safe, valid.to(flow.dtype), valid.to(flow.dtype)), dim=2)
            stem = self.flow_encoder.spatial(encoded.reshape(n * 4, 4, 72, 128))
            stem = stem.reshape(n, 4, 32, *stem.shape[-2:]).transpose(1, 2)
            backbone = self.rgb_encoder.backbone
            rgb = images.permute(0, 2, 1, 3, 4)
            motion = backbone.cma_block(rgb)
            appearance = backbone.appearance_stem(rgb)
            combined = motion + backbone.appearance_alpha * appearance
            aft = backbone.aft_block(combined, motion)
        spatial = self.flow_encoder.sfe(stem)
        temporal = self.flow_encoder.temporal(stem)
        rgb = backbone.dig(backbone.sfe_module(aft), backbone.mfe_module(aft))
        if (tuple(spatial.shape[1:]) != (32, 4, 9, 16)
                or tuple(temporal.shape[1:]) != (32, 4, 1, 1)
                or tuple(rgb.shape[1:]) != (256, 5, 72, 128)):
            raise RuntimeError("Online tap geometry changed")
        features = dict(F_sp=spatial_tokens(spatial),
            F_tm=temporal.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous(),
            R=spatial_tokens(rgb)[:, :4].contiguous(), dt=dt.clone())
        validate_grid_features(features, (2, 3), check_finite=False)
        return features

    def forward(self, inputs):
        return self.readout(self.extract_features(inputs))

    def forward_cached(self, features):
        if self.group != "S23_FROZEN_CONT":
            raise ValueError("Updated encoder tails must use current online features in train and dev")
        return self.readout(features)

    def optimizer_groups(self, readout_lr=1e-4, encoder_lr=1e-5):
        groups = [dict(params=list(self.readout.parameters()), lr=readout_lr, name="readout")]
        tail = [p for name, p in self.named_parameters() if p.requires_grad and not name.startswith("readout.")]
        if tail:
            groups.append(dict(params=tail, lr=encoder_lr, name="encoder_tail"))
        assert sum(len(group["params"]) for group in groups) == sum(p.requires_grad for p in self.parameters())
        return groups

    def parameter_manifest(self):
        bn = {name for name, module in self.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)}
        def role(name):
            if any(name.startswith(prefix + ".") for prefix in bn):
                return "encoder_batchnorm_frozen"
            if name.startswith("readout."):
                return "S23_readout"
            if any(name.startswith(prefix + ".") for prefix in TAIL_MODULES):
                return "encoder_tail_non_BN"
            if name.startswith(("flow_encoder.projection.", "rgb_encoder.region_projection.", "legacy_head.")):
                return "unused_source_descriptor_or_head"
            return "frozen_encoder_frontend_or_appearance"
        parameters = [dict(name=name, shape=list(p.shape), count=p.numel(), trainable=p.requires_grad,
                           role=role(name)) for name, p in self.named_parameters()]
        buffers = [dict(name=name, shape=list(value.shape), count=value.numel(), mutable=False,
                        role=role(name)) for name, value in self.named_buffers()]
        modules = {}
        for name in (*TAIL_MODULES, "readout"):
            entries = [p for p in parameters if p["name"].startswith(name + ".")]
            modules[name] = dict(total=sum(p["count"] for p in entries),
                trainable=sum(p["count"] for p in entries if p["trainable"]),
                frozen=sum(p["count"] for p in entries if not p["trainable"]))
        result = dict(schema="v5_s23_tail_freeze_manifest_v1", group=self.group,
            requested_tail_modules=list(TAIL_MODULES), actual_tail_modules=list(TAIL_MODULES),
            module_name_mapping="exact; no expanded scope", modules=modules,
            parameters=parameters, buffers=buffers, batchnorm_modules=sorted(bn),
            parameter_count=sum(p["count"] for p in parameters),
            trainable_parameter_count=sum(p["count"] for p in parameters if p["trainable"]),
            frozen_parameter_count=sum(p["count"] for p in parameters if not p["trainable"]),
            encoder_BN="affine and buffers frozen; eval persists under outer train()",
            LayerNorm="all existing S23 LayerNorm affine parameters trainable")
        result["identity"] = object_hash(result)
        return result


def build_model(group, encoder_checkpoint, readout_checkpoint):
    return S23TailModel(encoder_checkpoint, readout_checkpoint, group)
