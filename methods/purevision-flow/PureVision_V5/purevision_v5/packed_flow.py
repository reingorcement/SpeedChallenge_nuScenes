"""Immutable, checksum-bound mmap views of already computed numerical flows.

Packing copies existing fp16/bool values; it never recomputes optical flow,
resizes a field, changes a mask, or normalizes motion. The index is committed
last, so an incomplete pack is never accepted as a training input.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import re
import time

import numpy as np
from numpy.lib.format import open_memmap
import torch

from .common import code_version, now, object_hash, read_json, sha256, write_json
from .flow import (MODEL_SIZE, _tensor, _validate_metadata, cache_path,
                   packet_contract, validate_flow_contract)


PACK_SCHEMA = "v5_packed_flow_v1"
INDEX_NAME = "packed_index.json"
ARRAY_SPECS = {"flow": (np.dtype("float16"), (2, *MODEL_SIZE)),
               "valid": (np.dtype("bool"), (1, *MODEL_SIZE)),
               "fb_mask": (np.dtype("bool"), (1, *MODEL_SIZE))}


def _stat(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns


@lru_cache(maxsize=16)
def _checksum_unchanged_file(path_string, size, mtime_ns):
    """Train/dev readers in one process share the initial multi-GB hash pass."""
    path = Path(path_string)
    if _stat(path) != (size, mtime_ns):
        raise ValueError("Packed array changed while being validated")
    digest = sha256(path)
    if _stat(path) != (size, mtime_ns):
        raise ValueError("Packed array changed while being validated")
    return digest


def _manifest_pairs(manifest_path, partitions, selection_path=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "v5_dataset_v1" or manifest.get("seq_len") != 5:
        raise ValueError("Packing requires the five-frame V5 dataset manifest")
    if not partitions or not set(partitions) <= {"train_core", "dev"}:
        raise ValueError("Only the declared train_core/dev preparation partitions are allowed")
    pairs = {}
    selection = None
    if selection_path is not None:
        from .selection import load_selection
        selection = load_selection(manifest_path, selection_path)
    for record in manifest["sequences"]:
        if record["partition"] not in partitions:
            continue
        if selection is not None and record["scene"] not in selection["scenes"].get(record["partition"], []):
            continue
        path = (manifest_path.parent / record["file"]).resolve()
        if not path.is_relative_to(manifest_path.parent) or record.get("sha256") != sha256(path):
            raise ValueError("Sequence checksum/path changed before packing")
        sequence = read_json(path)
        if sequence.get("schema") != "v5_sequence_v1" or any(sequence.get(k) != record.get(k) for k in ("scene", "partition", "log_token")):
            raise ValueError("Sequence identity mismatch")
        frames = sequence["frames"]
        for end in sequence["windows"]:
            if not isinstance(end, int) or isinstance(end, bool) or not 4 <= end < len(frames):
                raise ValueError("Invalid causal window during packing")
            for source, target in zip(frames[end-4:end], frames[end-3:end+1]):
                key = (source["token"], target["token"])
                times = (float(source["time_s"]), float(target["time_s"]))
                cache_path(".", *key)
                if not all(math.isfinite(value) for value in times) or times[1] <= times[0]:
                    raise ValueError("Invalid pair timestamps during packing")
                if key in pairs and pairs[key] != times:
                    raise ValueError("The same frame pair has conflicting timestamps")
                pairs[key] = times
    if not pairs:
        raise ValueError("No declared pairs to pack")
    return [(source, target, *times) for (source, target), times in sorted(pairs.items())]


def _verified_compact(path, pair, require_fb):
    """Use the flow module's metadata/tensor validators without a second load."""
    packet = torch.load(path, map_location="cpu", weights_only=True)
    provenance = _validate_metadata(packet, "model_view")
    observed = (provenance["source"]["token"], provenance["target"]["token"],
                provenance["source"]["time_s"], provenance["target"]["time_s"])
    if observed != pair:
        raise ValueError("Compact flow pair token/time mismatch")
    if require_fb and not packet["fb_available"]:
        raise ValueError("FB reliability is missing from a requested bidirectional pack")
    if re.fullmatch(r"[0-9a-f]{64}", packet.get("native_sha256", "")) is None:
        raise ValueError("Missing native packet checksum")
    values = {name: _tensor(packet, name, shape, torch.float16 if name == "flow" else torch.bool)
              for name, (_, shape) in ARRAY_SPECS.items()}
    if (values["fb_mask"] & ~values["valid"]).any() or (not packet["fb_available"] and values["fb_mask"].any()):
        raise ValueError("Invalid compact FB semantics")
    if (values["flow"].masked_select(~values["valid"].expand_as(values["flow"])) != 0).any():
        raise ValueError("Invalid compact flow must already be zero")
    return values, bool(packet["fb_available"]), packet_contract(packet)


class PackedFlow:
    """Verified mmap reader, with small pickled state and shared kernel page cache."""

    def __init__(self, root, *, dataset_sha256=None, require_fb=False):
        self.root = Path(root).resolve()
        index_path = self.root / INDEX_NAME
        self.index = read_json(index_path)
        identity = self.index.get("identity")
        if self.index.get("schema") != PACK_SCHEMA or identity != object_hash({k: v for k, v in self.index.items() if k != "identity"}):
            raise ValueError("Packed index schema/identity mismatch")
        contract = self.index["contract"]
        self.flow_contract = {key:contract[key] for key in ("raft_sha256", "preprocessing", "native_contract", "derived_contract")
                              if key in contract}
        self.flow_contract["schema"] = contract.get("flow_schema")
        validate_flow_contract(self.flow_contract)
        if dataset_sha256 is not None and contract.get("dataset_sha256") != dataset_sha256:
            raise ValueError("Packed flow belongs to another dataset")
        if require_fb and not contract.get("fb_available"):
            raise ValueError("Packed cache has no FB reliability")
        self.index_sha256 = sha256(index_path)
        self.identity = identity
        self._files = {"index": index_path}
        self._file_stats = {"index": _stat(index_path)}
        self._rows = {}
        pairs = self.index["pairs"]
        if not pairs:
            raise ValueError("Packed cache is empty")
        for row_number, row in enumerate(pairs):
            pair = (row["source"], row["target"])
            cache_path(".", *pair)
            times = row["source_time"], row["target_time"]
            if (pair in self._rows or not all(math.isfinite(t) for t in times) or times[1] <= times[0]
                    or re.fullmatch(r"[0-9a-f]{64}", row.get("source_sha256", "")) is None):
                raise ValueError("Invalid or duplicate packed pair index")
            self._rows[pair] = row_number
        self._arrays = {}
        if set(self.index["arrays"]) != set(ARRAY_SPECS):
            raise ValueError("Packed cache must contain flow, valid and fb_mask arrays")
        for name, (dtype, shape) in ARRAY_SPECS.items():
            info = self.index["arrays"][name]
            path = (self.root / info["file"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Packed array path escaped its directory")
            signature = _stat(path)
            if _checksum_unchanged_file(str(path), *signature) != info["sha256"]:
                raise ValueError("Packed array checksum changed")
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if (value.shape != (len(pairs), *shape) or value.dtype != dtype
                    or info["shape"] != list(value.shape) or info["dtype"] != value.dtype.name):
                raise ValueError("Packed array shape/dtype mismatch")
            self._files[name], self._file_stats[name], self._arrays[name] = path, signature, value

    def __getstate__(self):
        state = dict(self.__dict__)
        # numpy.memmap's default pickle can serialize its entire payload.
        state["_arrays"] = None
        return state

    def _unchanged(self):
        if any(_stat(path) != self._file_stats[name] for name, path in self._files.items()):
            raise ValueError("Packed cache changed during a running experiment")
        if self._arrays is None:
            self._arrays = {name: np.load(self._files[name], mmap_mode="r", allow_pickle=False)
                            for name in ARRAY_SPECS}

    def pair_row(self, source, target, source_time, target_time):
        row_number = self._rows.get((source, target))
        if row_number is None:
            raise ValueError("Requested pair is absent from the packed cache")
        row = self.index["pairs"][row_number]
        if (source_time, target_time) != (row["source_time"], row["target_time"]):
            raise ValueError("Packed flow pair time mismatch")
        return row_number

    def pair_checksum(self, *pair):
        return self.index["pairs"][self.pair_row(*pair)]["source_sha256"]

    def load_pairs(self, pairs):
        """Return the same float32 flow/bool masks as legacy load_model_pair."""
        self._unchanged()
        rows = [self.pair_row(*pair) for pair in pairs]
        # Advanced indexing already creates an owned writable ndarray; avoid a
        # second full-window copy before the required float32 flow conversion.
        values = [torch.from_numpy(self._arrays[name][rows]) for name in ARRAY_SPECS]
        return values[0].float(), values[1], values[2]


def pack(manifest_path, source_root, output, *, partitions=("train_core", "dev"),
         require_fb=False, reference_pack=None, selection_path=None):
    manifest_path, source_root, output = (Path(p).resolve() for p in (manifest_path, source_root, output))
    if output.exists():
        raise FileExistsError("Packed output must be a new immutable directory")
    if (output.is_relative_to(source_root) or source_root.is_relative_to(output)
            or manifest_path.is_relative_to(output)):
        raise ValueError("Packed output must be separate from source caches and dataset inputs")
    pairs = _manifest_pairs(manifest_path, partitions, selection_path=selection_path)
    dataset_hash = sha256(manifest_path)
    reference = PackedFlow(reference_pack, dataset_sha256=dataset_hash) if reference_pack else None
    if reference is not None and not require_fb:
        raise ValueError("Forward equality proof is intended for a bidirectional pack")
    output.mkdir(parents=True)
    arrays = {name: open_memmap(output / f"{name}.npy.partial", mode="w+", dtype=dtype,
                                shape=(len(pairs), *shape)) for name, (dtype, shape) in ARRAY_SPECS.items()}
    rows, fb_available, flow_contract = [], None, None
    start = time.monotonic()
    for row_number, pair in enumerate(pairs):
        path = cache_path(source_root, *pair[:2])
        original_stat = _stat(path)
        values, available, actual_contract = _verified_compact(path, pair, require_fb)
        source_digest = sha256(path)
        if _stat(path) != original_stat:
            raise ValueError("Compact cache changed during packing")
        if fb_available is not None and available != fb_available:
            raise ValueError("Forward-only and bidirectional packets cannot share one pack")
        fb_available = available
        if flow_contract is not None and flow_contract != actual_contract:
            raise ValueError("A packed cache cannot mix native/derived flow contracts")
        flow_contract = actual_contract
        if reference is not None:
            if reference.flow_contract != flow_contract:
                raise ValueError("Forward equality requires the same native/derived flow contract")
            old_flow, old_valid, _ = reference.load_pairs([pair])
            if not torch.equal(old_flow[0], values["flow"].float()) or not torch.equal(old_valid[0], values["valid"]):
                raise ValueError("Bidirectional preparation changed forward flow/valid values")
        for name in arrays:
            arrays[name][row_number] = values[name].numpy()
        rows.append(dict(source=pair[0], target=pair[1], source_time=pair[2], target_time=pair[3], source_sha256=source_digest))
        if row_number == 0 or (row_number + 1) % 1000 == 0:
            print(f"PACK pairs={row_number+1}/{len(pairs)} pairs_s={(row_number+1)/max(time.monotonic()-start, .001):.2f}", flush=True)
    array_info = {}
    for name, (dtype, shape) in ARRAY_SPECS.items():
        arrays[name].flush()
        path = output / f"{name}.npy"
        (output / f"{name}.npy.partial").replace(path)
        array_info[name] = dict(file=path.name, sha256=sha256(path), shape=[len(pairs), *shape], dtype=dtype.name)
    arrays.clear()
    contract = {key:value for key,value in flow_contract.items() if key != "schema"}
    contract.update(flow_schema=flow_contract["schema"], dataset_sha256=dataset_hash,
                    partitions=list(partitions), source_root=str(source_root), fb_available=fb_available)
    if selection_path is not None:
        from .selection import load_selection
        contract.update(selection=load_selection(manifest_path, selection_path), selection_sha256=sha256(selection_path))
    index = dict(schema=PACK_SCHEMA, created_at=now(), code=code_version(), contract=contract,
                 arrays=array_info, pairs=rows)
    if reference is not None:
        index["forward_equality"] = dict(reference_index_sha256=reference.index_sha256,
                                         compared_pairs=len(pairs), flow_and_valid_exact=True)
    index["identity"] = object_hash(index)
    write_json(output / INDEX_NAME, index)
    print(f"PACK complete pairs={len(pairs)} seconds={time.monotonic()-start:.2f} index={output / INDEX_NAME}", flush=True)
    return output / INDEX_NAME
