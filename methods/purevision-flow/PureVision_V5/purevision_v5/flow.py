"""Frozen RAFT pixels and independently auditable forward/backward reliability.

Training reads only a compact view. Native flows remain beside it for audit and
for adding backward flow without changing any already-computed forward values.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import tempfile

import PIL
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms import functional as TF

from .common import object_hash, sha256


SCHEMA = "v5_flow_v1"
CONFIGURABLE_SCHEMA = "v5_flow_v2"
RAFT_SIZE = (288, 512)
MODEL_SIZE = (72, 128)
RAFT_SHA256 = "ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "PureVision_V4/weights/raft_large_C_T_SKHT_V2.pth"
PREPROCESSING = {
    "raft_size": list(RAFT_SIZE), "model_size": list(MODEL_SIZE),
    "input": "RGB", "resize": "bilinear", "antialias": False,
    "range": [-1.0, 1.0], "transforms": "Raft_Large_Weights.C_T_SKHT_V2.transforms",
    "num_flow_updates": 12, "compute_dtype": "float32", "tf32": False,
    "flow_units": "pixels_at_raft_resolution", "model_flow_units": "pixels_at_model_resolution",
    "flow_resize": "bilinear_align_corners_false_vector_scaled", "mask_resize": "nearest",
    "fb_rule": "squared_relative", "fb_alpha": 0.01, "fb_beta_px2": 0.5,
    "warp_align_corners": True, "invalid_model_flow": "zero",
}


def native_contract(raft_size=RAFT_SIZE) -> dict:
    """Prediction contract, independent of any derived training resolution.

    Sizes are always (height, width). This stage supports the two approved
    nuScenes 16:9 frontends; a different aspect/padding policy is a new contract.
    """
    if tuple(raft_size) not in (RAFT_SIZE, (576, 1024)):
        raise ValueError("Supported RAFT sizes are HxW 288x512 and 576x1024")
    return dict(version="v5_raft_native_v2", prediction="frozen_raft_not_ground_truth",
                raft_size=list(raft_size), input="RGB", resize="bilinear", antialias=False,
                range=[-1.0, 1.0], transforms=PREPROCESSING["transforms"],
                num_flow_updates=12, compute_dtype="float32", tf32=False,
                flow_units="pixels_at_raft_resolution", component_order=["u_right", "v_down"],
                valid="finite_vector_and_endpoint_inside_target",
                fb_rule="squared_relative", fb_alpha=0.01, fb_beta_px2=0.5,
                warp_align_corners=True)


def derived_contract(raft_size=RAFT_SIZE) -> dict:
    """The fixed 72x128 model view; vector scaling is explicitly dimensional."""
    native_contract(raft_size)
    return dict(version="v5_flow_view_v2", model_size=list(MODEL_SIZE),
                native_size=list(raft_size), model_flow_units="pixels_at_model_resolution",
                flow_resize="bilinear_align_corners_false_vector_scaled",
                vector_scale_uv=[MODEL_SIZE[1] / raft_size[1], MODEL_SIZE[0] / raft_size[0]],
                mask_resize="nearest", invalid_model_flow="zero", flow_dtype="float16",
                mask_dtype="bool", dt="positive_seconds_separate_not_divided_into_flow")


def validate_flow_contract(contract: dict) -> None:
    """Accept the exact legacy contract or the independently versioned v2 parts."""
    if contract.get("raft_sha256") != RAFT_SHA256:
        raise ValueError("Flow weight/preprocessing contract mismatch")
    if contract.get("schema") == SCHEMA:
        expected = dict(schema=SCHEMA, raft_sha256=RAFT_SHA256, preprocessing=PREPROCESSING)
    elif contract.get("schema") == CONFIGURABLE_SCHEMA:
        size = contract.get("native_contract", {}).get("raft_size", ())
        expected = dict(schema=CONFIGURABLE_SCHEMA, raft_sha256=RAFT_SHA256,
                        native_contract=native_contract(size), derived_contract=derived_contract(size))
    else:
        raise ValueError("Unknown flow cache schema")
    if contract != expected:
        raise ValueError("Flow weight/preprocessing contract mismatch")


def packet_contract(packet: dict) -> dict:
    """Actual input contract for loaders; never infer resolution from a default."""
    provenance = _validate_metadata(packet, "model_view")
    if packet["schema"] == SCHEMA:
        return dict(schema=SCHEMA, raft_sha256=RAFT_SHA256, preprocessing=provenance["preprocessing"])
    return dict(schema=CONFIGURABLE_SCHEMA, raft_sha256=RAFT_SHA256,
                native_contract=provenance["native_contract"], derived_contract=packet["derived_contract"])


def atomic_save(value: dict, path: str | Path) -> None:
    """Commit one packet by rename; a compact file is written last."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False) as stream:
            tmp = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def validate_pair_tokens(source_token, target_token):
    for token in (source_token, target_token):
        if not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", token) is None:
            raise ValueError("Unsafe or missing image token")
    if source_token == target_token:
        raise ValueError("A flow pair requires distinct frame tokens")


def cache_path(root: str | Path, source_token: str, target_token: str, *, schema: str | None = None) -> Path:
    validate_pair_tokens(source_token,target_token)
    root = Path(root)
    present = [s for s in (SCHEMA, CONFIGURABLE_SCHEMA) if (root / s).exists()]
    if len(present) > 1 or (schema is not None and present and schema not in present):
        raise ValueError("A cache root cannot mix legacy and configurable schemas")
    schema = schema or (present[0] if present else SCHEMA)
    if schema not in (SCHEMA, CONFIGURABLE_SCHEMA):
        raise ValueError("Unknown flow cache schema")
    return root / schema / source_token[:2] / f"{source_token}_{target_token}.pt"


def native_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".native.pt")


def _check_flow(value: torch.Tensor) -> None:
    if value.ndim != 3 or value.shape[0] != 2 or min(value.shape[-2:]) < 2:
        raise ValueError("Expected flow [2,H,W] with H,W >= 2")


def legal_mask(flow: torch.Tensor) -> torch.Tensor:
    """Finite forward displacement and a destination inside the image extent."""
    _check_flow(flow)
    _, h, w = flow.shape
    y, x = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                          torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij")
    qx, qy = x + flow[0], y + flow[1]
    return (torch.isfinite(flow).all(0) & (qx >= 0) & (qx <= w - 1) & (qy >= 0) & (qy <= h - 1))[None]


def fb_mask(forward: torch.Tensor, backward: torch.Tensor, *, backward_finite: torch.Tensor | None = None) -> torch.Tensor:
    """|F + warp(B,F)|² <= .01 (|F|² + |warp(B,F)|²) + .5 px².

    Nonfinite backward interpolation neighbours invalidate the FB test, without
    altering the forward-only legal mask used by every experiment.
    """
    _check_flow(forward)
    if backward.shape != forward.shape:
        raise ValueError("Forward and backward shapes differ")
    forward, backward = forward.float(), backward.float()
    _, h, w = forward.shape
    valid = legal_mask(forward)
    clean_f = torch.nan_to_num(forward, nan=0.0, posinf=0.0, neginf=0.0)
    clean_b = torch.nan_to_num(backward, nan=0.0, posinf=0.0, neginf=0.0)
    y, x = torch.meshgrid(torch.arange(h, device=forward.device, dtype=forward.dtype),
                          torch.arange(w, device=forward.device, dtype=forward.dtype), indexing="ij")
    grid = torch.stack((2 * (x + clean_f[0]) / (w - 1) - 1,
                        2 * (y + clean_f[1]) / (h - 1) - 1), -1)[None]
    warped_b = F.grid_sample(clean_b[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    finite_b = torch.isfinite(backward).all(0, keepdim=True)
    if backward_finite is not None:
        if backward_finite.shape != finite_b.shape or backward_finite.dtype != torch.bool:
            raise ValueError("Malformed original finite mask")
        finite_b &= backward_finite.to(backward.device)
    finite_b = finite_b.float()[None]
    finite_neighbours = F.grid_sample(finite_b, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0] >= 1 - 1e-6
    error_sq = ((clean_f + warped_b) ** 2).sum(0, keepdim=True)
    threshold = 0.01 * ((clean_f ** 2).sum(0, keepdim=True) + (warped_b ** 2).sum(0, keepdim=True)) + 0.5
    return valid & finite_neighbours & (error_sq <= threshold)


def resize_flow(flow: torch.Tensor, size: tuple[int, int] = MODEL_SIZE) -> torch.Tensor:
    _check_flow(flow)
    old_h, old_w = flow.shape[-2:]
    result = F.interpolate(flow[None].float(), size=size, mode="bilinear", align_corners=False)[0]
    result[0] *= size[1] / old_w
    result[1] *= size[0] / old_h
    return result


def resize_mask(mask: torch.Tensor, size: tuple[int, int] = MODEL_SIZE) -> torch.Tensor:
    return F.interpolate(mask[None].float(), size=size, mode="nearest")[0].bool()


def _validate_metadata(packet: dict, kind: str) -> dict:
    schema = packet.get("schema")
    if schema not in (SCHEMA, CONFIGURABLE_SCHEMA) or packet.get("kind") != kind:
        raise ValueError("Unexpected flow cache schema or kind")
    provenance = packet.get("provenance", {})
    expected_identity = object_hash(provenance)
    if schema == CONFIGURABLE_SCHEMA and kind == "model_view":
        if packet.get("native_identity") != expected_identity:
            raise ValueError("Derived flow native identity mismatch")
        expected_identity = object_hash(dict(native_identity=expected_identity,
                                            native_sha256=packet.get("native_sha256"),
                                            derived_contract=packet.get("derived_contract")))
    if packet.get("identity") != expected_identity:
        raise ValueError("Flow provenance identity mismatch")
    if provenance.get("checkpoint_sha256") != RAFT_SHA256:
        raise ValueError("Flow weight/preprocessing contract mismatch")
    if schema == SCHEMA:
        if provenance.get("preprocessing") != PREPROCESSING:
            raise ValueError("Flow weight/preprocessing contract mismatch")
    else:
        size = provenance.get("native_contract", {}).get("raft_size", ())
        if provenance.get("native_contract") != native_contract(size) or "preprocessing" in provenance:
            raise ValueError("Flow native generation contract mismatch")
        if kind == "model_view" and packet.get("derived_contract") != derived_contract(size):
            raise ValueError("Flow derived view contract mismatch")
    if provenance.get("direction") not in ("forward", "bidirectional"):
        raise ValueError("Unknown flow direction contract")
    if packet.get("fb_available") != (provenance["direction"] == "bidirectional"):
        raise ValueError("FB availability does not match provenance")
    source, target = provenance.get("source", {}), provenance.get("target", {})
    cache_path(".", source.get("token"), target.get("token"))
    dt = target.get("time_s", math.nan) - source.get("time_s", math.nan)
    if not math.isfinite(dt) or dt <= 0 or provenance.get("dt_s") != dt:
        raise ValueError("Flow pair timestamps/dt are invalid")
    return provenance


def _tensor(packet: dict, key: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    value = packet.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
        raise ValueError(f"Malformed {key} tensor in flow cache")
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite {key} in flow cache")
    return value


def load_model_pair(path: str | Path, *, require_fb: bool = False,
                    expected_source: str | None = None, expected_target: str | None = None,
                    expected_times: tuple[float, float] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read only compact tensors: float32 [2,72,128], bool [1,72,128] × 2."""
    packet = torch.load(path, map_location="cpu", weights_only=True)
    provenance = _validate_metadata(packet, "model_view")
    if require_fb and not packet["fb_available"]:
        raise ValueError("FB reliability was not computed for this pair")
    if expected_source is not None and provenance["source"]["token"] != expected_source:
        raise ValueError("Flow source token mismatch")
    if expected_target is not None and provenance["target"]["token"] != expected_target:
        raise ValueError("Flow target token mismatch")
    if expected_times is not None and tuple(expected_times) != (provenance["source"]["time_s"], provenance["target"]["time_s"]):
        raise ValueError("Flow pair time mismatch")
    if re.fullmatch(r"[0-9a-f]{64}", packet.get("native_sha256", "")) is None:
        raise ValueError("Missing native packet checksum")
    flow = _tensor(packet, "flow", (2, *MODEL_SIZE), torch.float16)
    valid = _tensor(packet, "valid", (1, *MODEL_SIZE), torch.bool)
    reliability = _tensor(packet, "fb_mask", (1, *MODEL_SIZE), torch.bool)
    if (reliability & ~valid).any() or (not packet["fb_available"] and reliability.any()):
        raise ValueError("Invalid FB mask semantics")
    if (flow.masked_select((~valid).expand_as(flow)) != 0).any():
        raise ValueError("Illegal model flow must be zero")
    return flow.float(), valid, reliability


def _load_native(path: str | Path, expected: dict | None = None) -> dict:
    packet = torch.load(path, map_location="cpu", weights_only=True)
    provenance = _validate_metadata(packet, "native")
    comparable = lambda item: {k:v for k,v in item.items() if k != "requested_batch_size"}
    if expected is not None and comparable(provenance) != comparable(expected):
        raise ValueError(f"Native flow identity mismatch: {path}")
    directions = ("forward", "backward") if packet["fb_available"] else ("forward",)
    size = tuple(provenance["native_contract"]["raft_size"]) if packet["schema"] == CONFIGURABLE_SCHEMA else RAFT_SIZE
    for direction in directions:
        _tensor(packet, direction, (2, *size), torch.float32)
        valid = _tensor(packet, f"{direction}_valid", (1, *size), torch.bool)
        reliability = _tensor(packet, f"{direction}_fb", (1, *size), torch.bool)
        if (reliability & ~valid).any() or (not packet["fb_available"] and reliability.any()):
            raise ValueError("Malformed native reliability mask")
    return packet


def _write_model_view(native: dict, path: str | Path, *, native_file: str | Path | None = None) -> None:
    valid = resize_mask(native["forward_valid"])
    reliability = resize_mask(native["forward_fb"]) & valid
    # All legal flow is retained for R01/R03/R04, irrespective of FB reliability.
    flow = resize_flow(native["forward"]).masked_fill(~valid.expand(2, -1, -1), 0).half()
    packet = {key: native[key] for key in ("schema", "identity", "provenance", "fb_available")}
    packet.update(kind="model_view", flow=flow, valid=valid, fb_mask=reliability,
                  native_sha256=sha256(native_file or native_path(path)))
    if native["schema"] == CONFIGURABLE_SCHEMA:
        packet["native_identity"] = native["identity"]
        packet["derived_contract"] = derived_contract(native["provenance"]["native_contract"]["raft_size"])
        packet["identity"] = object_hash({k: packet[k] for k in ("native_identity", "native_sha256", "derived_contract")})
    _validate_metadata(packet, "model_view")
    atomic_save(packet, path)


def derive_model_view(native_file: str | Path, output: str | Path) -> None:
    """Rebuild the current model view from immutable native predictions, no RAFT.

    Deliberately does not offer a new downstream size until the model/dataset
    contracts for such an experiment are approved and implemented together.
    """
    if Path(output).exists():
        raise FileExistsError("A derived view must use a new output path")
    native = _load_native(native_file)
    _write_model_view(native, output, native_file=native_file)


def save_pair(path: str | Path, forward: torch.Tensor, backward: torch.Tensor | None, provenance: dict,
              *, forward_valid: torch.Tensor | None = None, execution: dict | None = None) -> None:
    """Save native flow first, then the compact commit marker. No overwrites."""
    path = Path(path)
    if path.exists() or native_path(path).exists():
        raise FileExistsError(f"Flow cache already exists: {path}")
    available = backward is not None
    if available != (provenance.get("direction") == "bidirectional"):
        raise ValueError("Missing or unexpected backward flow")
    schema = CONFIGURABLE_SCHEMA if "native_contract" in provenance else SCHEMA
    size = tuple(provenance["native_contract"]["raft_size"]) if schema == CONFIGURABLE_SCHEMA else RAFT_SIZE
    packet = dict(schema=schema, kind="native", identity=object_hash(provenance),
                  provenance=provenance, fb_available=available, execution=execution or {})
    # Reused native forward replaces original NaN/Inf with zero. Recover their
    # original finite indicator so the inverse FB mask cannot treat them as data.
    original_forward_finite = None
    if forward_valid is not None:
        original_forward_finite = forward_valid | ~legal_mask(forward).cpu()
    for direction, value, opposite in (("forward", forward, backward), ("backward", backward, forward)):
        if value is None:
            continue
        if tuple(value.shape) != (2, *size):
            raise ValueError("Native RAFT output shape mismatch")
        value = value.detach().float()
        valid = legal_mask(value).cpu()
        if direction == "forward" and forward_valid is not None:
            if forward_valid.shape != valid.shape or forward_valid.dtype != torch.bool or (forward_valid & ~valid).any():
                raise ValueError("Invalid reused forward legality mask")
            valid = forward_valid.cpu()
        packet[f"{direction}_valid"] = valid
        opposite_finite = original_forward_finite if direction == "backward" else None
        packet[f"{direction}_fb"] = (fb_mask(value, opposite, backward_finite=opposite_finite).cpu()
                                      if opposite is not None else torch.zeros_like(valid)) & valid
        packet[direction] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).cpu()
    _validate_metadata(packet, "native")
    atomic_save(packet, native_path(path))
    _write_model_view(packet, path)


def _frame_metadata(frame: dict, raft_size=RAFT_SIZE) -> dict:
    path = Path(frame["filename"]).resolve(strict=True)
    with Image.open(path) as image:
        width, height = image.size
    if width * raft_size[0] != height * raft_size[1]:
        raise ValueError(f"Input aspect differs from 16:9: {path} ({width}x{height})")
    return dict(token=frame["token"], time_s=float(frame["time_s"]), filename=str(path),
                sha256=sha256(path), size_bytes=path.stat().st_size, size=[height, width])


class FrozenRaft:
    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str = "cuda:0", *,
                 direction: str = "forward", batch_size: int = 16,
                 raft_size: tuple[int, int] | None = None, generation_code: dict | None = None):
        if direction not in ("forward", "bidirectional"):
            raise ValueError("direction must be forward or bidirectional")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.checkpoint = Path(checkpoint)
        if sha256(self.checkpoint) != RAFT_SHA256:
            raise ValueError("Expected the audited RAFT C_T_SKHT_V2 weights")
        self.device = torch.device(device)
        self.direction = direction
        self.batch_size = batch_size
        self.schema = SCHEMA if raft_size is None else CONFIGURABLE_SCHEMA
        self.raft_size = RAFT_SIZE if raft_size is None else tuple(raft_size)
        native_contract(self.raft_size)
        self.generation_code = generation_code
        self.model = None
        self.transforms = Raft_Large_Weights.C_T_SKHT_V2.transforms()

    def _model(self):
        if self.model is None:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model = raft_large(weights=None, progress=False)
            model.load_state_dict(torch.load(self.checkpoint, map_location="cpu", weights_only=True), strict=True)
            self.model = model.to(self.device).eval().requires_grad_(False)
        return self.model

    def _image(self, frame: dict) -> torch.Tensor:
        with Image.open(frame["filename"]) as image:
            tensor = TF.pil_to_tensor(image.convert("RGB")).float() / 255.0
        return TF.resize(tensor, list(self.raft_size), antialias=False)[None]

    def provenance(self, source: dict, target: dict) -> dict:
        context = {key:source[key] for key in ("dataset", "camera", "scene") if key in source}
        if any(target.get(key) != value for key,value in context.items()):
            raise ValueError("Source and target dataset/camera/scene differ")
        source, target = _frame_metadata(source, self.raft_size), _frame_metadata(target, self.raft_size)
        if source["size"] != target["size"]:
            raise ValueError("Source and target image dimensions differ")
        result = dict(source=source, target=target, dt_s=target["time_s"] - source["time_s"],
                      checkpoint_sha256=RAFT_SHA256,
                      direction=self.direction, requested_batch_size=self.batch_size,
                      versions={"torch": str(torch.__version__),
                      "torchvision": str(torchvision.__version__), "pillow": PIL.__version__, "cuda": torch.version.cuda})
        result.update({"preprocessing": PREPROCESSING} if self.schema == SCHEMA
                      else {"native_contract": native_contract(self.raft_size)})
        if self.schema == CONFIGURABLE_SCHEMA:
            result.update(context=context, generation_code=self.generation_code)
        _validate_metadata(dict(schema=self.schema, kind="native", identity=object_hash(result), provenance=result,
                                fb_available=self.direction == "bidirectional"), "native")
        return result

    def pair(self, source: dict, target: dict, path: str | Path, *, reuse_forward: str | Path | None = None) -> str:
        """Return created/reused/recovered; cached pairs never invoke RAFT."""
        return self.pairs([(source, target, path, reuse_forward)])[0]

    def _prepare_pair(self, source, target, path, reuse_forward):
        path = Path(path)
        provenance = self.provenance(source, target)
        forward = None
        forward_valid = None
        old_identity = None
        old_sha256 = None
        if reuse_forward is not None:
            if self.direction != "bidirectional":
                raise ValueError("reuse_forward is only valid for bidirectional generation")
            expected = dict(provenance, direction="forward")
            old = _load_native(native_path(reuse_forward))
            # The frozen forward values may have been prepared at another batch
            # size; retain them exactly and record the original identity.
            comparable = lambda item: {k:v for k,v in item.items() if k != "requested_batch_size"}
            if comparable(old["provenance"]) != comparable(expected):
                raise ValueError("Reused forward input/weight/preprocessing identity mismatch")
            old_compact = torch.load(reuse_forward, map_location="cpu", weights_only=True)
            if (old_compact.get("native_sha256") != sha256(native_path(reuse_forward))
                    or old_compact.get("native_identity", old_compact.get("identity")) != old["identity"]):
                raise ValueError("Reused forward packet checksum/identity mismatch")
            load_model_pair(reuse_forward)
            forward = old["forward"]
            forward_valid = old["forward_valid"]
            old_identity = old["identity"]
            old_sha256 = old_compact["native_sha256"]
        if native_path(path).exists():
            native = _load_native(native_path(path), provenance)
            if reuse_forward is not None:
                execution = native.get("execution", {})
                if (execution.get("forward_origin_identity") != old_identity
                        or execution.get("forward_origin_sha256") != old_sha256
                        or not torch.equal(native["forward"], forward)
                        or not torch.equal(native["forward_valid"], forward_valid)):
                    raise ValueError("Cached bidirectional flow uses a different frozen forward source")
            if path.exists():
                compact = torch.load(path, map_location="cpu", weights_only=True)
                if (compact.get("native_identity", compact.get("identity")) != native["identity"]
                        or compact.get("native_sha256") != sha256(native_path(path))):
                    raise ValueError(f"Native/compact cache mismatch: {path}")
                load_model_pair(path, require_fb=self.direction == "bidirectional")
                return "reused", None
            _write_model_view(native, path)
            return "recovered", None
        if path.exists():
            raise ValueError(f"Compact flow has no native audit packet: {path}")
        return "created", dict(source=source, target=target, path=path, reuse_forward=reuse_forward,
                               provenance=provenance, forward=forward, forward_valid=forward_valid,
                               forward_origin_identity=old_identity, forward_origin_sha256=old_sha256)

    def pairs(self, pairs: list[tuple[dict, dict, Path, Path | None]]) -> list[str]:
        """Batch GPU inference while keeping pair files and identities separate."""
        if not pairs or len(pairs) > getattr(self, "batch_size", 1):
            raise ValueError("Pair batch is empty or exceeds configured batch_size")
        prepared = [self._prepare_pair(*pair) for pair in pairs]
        pending = [row for _,row in prepared if row is not None]
        if not pending:
            return [outcome for outcome,_ in prepared]
        images = {}
        for row in pending:
            for frame in (row["source"], row["target"]):
                if frame["filename"] not in images:
                    images[frame["filename"]] = self._image(frame)
        source_image = torch.cat([images[row["source"]["filename"]] for row in pending]).to(self.device)
        target_image = torch.cat([images[row["target"]["filename"]] for row in pending]).to(self.device)
        source_image, target_image = self.transforms(source_image, target_image)
        need_forward = [index for index,row in enumerate(pending) if row["forward"] is None]
        with torch.inference_mode():
            model = self._model()
            if need_forward:
                result = model(source_image[need_forward], target_image[need_forward], num_flow_updates=12)[-1].float().cpu()
                for index,value in zip(need_forward, result):
                    pending[index]["forward"] = value
            backwards = model(target_image, source_image, num_flow_updates=12)[-1].float().cpu() if self.direction == "bidirectional" else None
        for index,row in enumerate(pending):
            execution = {"device":str(self.device),
                         "device_name":torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
                         "forward_batch_size":len(need_forward) if row["forward_origin_identity"] is None else None,
                         "backward_batch_size":len(pending) if backwards is not None else None,
                         "forward_origin_identity":row["forward_origin_identity"],
                         "forward_origin_sha256":row["forward_origin_sha256"]}
            save_pair(row["path"], row["forward"], backwards[index] if backwards is not None else None,
                      row["provenance"], forward_valid=row["forward_valid"], execution=execution)
            if row["reuse_forward"] is not None:
                previous_flow, previous_valid, _ = load_model_pair(row["reuse_forward"])
                new_flow, new_valid, _ = load_model_pair(row["path"], require_fb=True)
                if not torch.equal(previous_flow, new_flow) or not torch.equal(previous_valid, new_valid):
                    raise RuntimeError("Backward preparation changed frozen forward inputs")
        return [outcome for outcome,_ in prepared]
