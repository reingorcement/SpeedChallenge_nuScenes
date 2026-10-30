"""Full-dataset flow storage with explicit, byte-preserving cache adoption.

This module changes orchestration only. Numerical prediction and the fixed
72x128 training view remain the implementation in :mod:`flow`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import torch

from .common import object_hash, sha256
from .flow import (CONFIGURABLE_SCHEMA, FrozenRaft, _load_native, cache_path,
                   derive_model_view, legal_mask, load_model_pair, native_path,
                   packet_contract, resize_flow, resize_mask)


ADOPTION_SCHEMA = "v5_full_flow_adoption_v1"
FLOW_SOURCE = Path(__file__).with_name("flow.py")


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def load_adoption_plan(path, current_flow_sha256=None):
    """Validate declared source code and the compact-file checksum allowlist.

    Packed arrays are not read: each allowlisted compact packet subsequently
    binds its native bytes. Origins without a native_root allow only explicit
    resume of an already-present destination from that compatible commit.
    """
    if path is None:
        return {"schema": ADOPTION_SCHEMA, "origins": []}
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if value.get("schema") != ADOPTION_SCHEMA or not isinstance(value.get("origins"), list):
        raise ValueError("Invalid full-flow adoption plan")
    current_flow_sha256 = current_flow_sha256 or sha256(FLOW_SOURCE)
    origins, seen = [], set()
    for raw in value["origins"]:
        origin = dict(raw)
        commit = origin.get("commit")
        if not _hex(commit, 40) or commit in seen:
            raise ValueError("Invalid or duplicate adoption generation commit")
        seen.add(commit)
        source = Path(origin["flow_py"]).resolve(strict=True)
        if (not _hex(origin.get("flow_py_sha256"), 64)
                or sha256(source) != origin["flow_py_sha256"]
                or origin["flow_py_sha256"] != current_flow_sha256):
            raise ValueError("Adoption requires identical audited numerical flow.py")
        origin["flow_py"] = str(source)
        origin["pairs"] = {}
        keys = {"native_root", "packed_index", "packed_index_sha256"}
        present = keys & origin.keys()
        if present and present != keys:
            raise ValueError("Adoption source requires native_root and a pinned packed_index")
        if present:
            root = Path(origin["native_root"]).resolve(strict=True)
            index_path = Path(origin["packed_index"]).resolve(strict=True)
            if (not _hex(origin["packed_index_sha256"], 64)
                    or sha256(index_path) != origin["packed_index_sha256"]):
                raise ValueError("Adoption packed index checksum mismatch")
            index = json.loads(index_path.read_text())
            if (index.get("schema") != "v5_packed_flow_v1"
                    or index.get("identity") != object_hash({k:v for k,v in index.items() if k != "identity"})):
                raise ValueError("Adoption packed index identity mismatch")
            code = index.get("code", {})
            # Packing is a CPU derivation; the original packer records a clean
            # commit but does not query GitHub. The native packet itself must
            # still carry its verified generation commit (checked in _verify).
            if code not in (dict(commit=commit, dirty=False, verified_remote_commit=commit),
                            dict(commit=commit, dirty=False, verified_remote_commit=None)):
                raise ValueError("Adoption packed index generation code mismatch")
            origin["packed_index_code"] = copy.deepcopy(code)
            for pair in index["pairs"]:
                key = (pair["source"], pair["target"])
                cache_path(root, *key, schema=CONFIGURABLE_SCHEMA)
                if key in origin["pairs"] or not _hex(pair.get("source_sha256"), 64):
                    raise ValueError("Invalid duplicate or unpinned adoption pair")
                if pair["target_time"] <= pair["source_time"]:
                    raise ValueError("Invalid adoption pair timestamps")
                origin["pairs"][key] = dict(pair)
            origin.update(native_root=str(root), packed_index=str(index_path))
        origins.append(origin)
    return {"schema": ADOPTION_SCHEMA, "path": str(path), "sha256": sha256(path), "origins": origins}


def atomic_copy_verified(source, destination, expected_sha256):
    """Copy immutable bytes, fsync, then publish without replacing any file.

    The temporary file and destination share a filesystem. A hard link is an
    atomic no-replace publication; it does not link the old source across disks.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        with source.open("rb") as reader, tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=destination.name + ".", suffix=".partial", delete=False) as writer:
            temporary = Path(writer.name)
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Adoption source changed during copy")
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class FullFlowCache:
    """Verify/restore/adopt existing pairs before handing missing pairs to RAFT."""

    def __init__(self, engine: FrozenRaft, output_root, adoption=None):
        if engine.schema != CONFIGURABLE_SCHEMA or engine.direction != "forward" or tuple(engine.raft_size) != (576, 1024):
            raise ValueError("Full generation requires forward v2 RAFT 576x1024")
        self.engine = engine
        self.output_root = Path(output_root).resolve()
        self.current_code = copy.deepcopy(engine.generation_code)
        if not isinstance(self.current_code, dict) or not _hex(self.current_code.get("commit"), 40):
            raise ValueError("Full generation requires an explicit code version")
        self.adoption = adoption or {"schema": ADOPTION_SCHEMA, "origins": []}
        self.origins = {origin["commit"]: origin for origin in self.adoption["origins"]}

    def path(self, source, target, group):
        for key in ("version", "camera"):
            value = group.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None or value in (".", ".."):
                raise ValueError("Unsafe full-flow version/camera")
            if source.get(key) != value or target.get(key) != value:
                raise ValueError("Full-flow frame group identity mismatch")
        if source.get("scene") != group.get("scene") or target.get("scene") != group.get("scene"):
            raise ValueError("Full-flow frame scene differs from group")
        return cache_path(self.output_root / group["version"] / group["camera"],
                          source["token"], target["token"], schema=CONFIGURABLE_SCHEMA)

    def _validate_code(self, code):
        if code == self.current_code:
            return
        if not isinstance(code, dict):
            raise ValueError("Missing generation code in full-flow cache")
        commit = code.get("commit")
        if (commit not in self.origins
                or code != dict(commit=commit, dirty=False, verified_remote_commit=commit)):
            raise ValueError("Unapproved full-flow generation commit")

    def _verify(self, path, source, target, *, expected_compact_sha=None, recover=False):
        """Validate image identity, contract, native bytes, and exact derivation."""
        path = Path(path)
        npath = native_path(path)
        if not npath.exists():
            raise ValueError(f"Compact/adoption flow has no native packet: {path}")
        native = _load_native(npath)
        self._validate_code(native["provenance"].get("generation_code"))
        expected = self.engine.provenance(source, target)
        # Only the explicitly approved original code is substituted. All image,
        # environment, direction, checkpoint, context and native units must match.
        expected["generation_code"] = native["provenance"]["generation_code"]
        observed = native["provenance"]
        comparable = lambda p: {k:v for k,v in p.items() if k != "requested_batch_size"}
        if comparable(observed) != comparable(expected):
            raise ValueError("Full-flow input/weight/environment contract mismatch")
        if native["schema"] != CONFIGURABLE_SCHEMA or native["fb_available"]:
            raise ValueError("Full-flow source must be forward-only v2")
        if (native["forward_valid"] & ~legal_mask(native["forward"])).any():
            raise ValueError("Native valid mask marks illegal flow as valid")
        native_sha = sha256(npath)
        origin = self.origins.get(observed["generation_code"]["commit"])
        pinned_compact = None
        if origin is not None and "native_root" in origin:
            key = (source["token"], target["token"])
            pin = origin["pairs"].get(key)
            if pin is None or (pin["source_time"], pin["target_time"]) != (source["time_s"], target["time_s"]):
                raise ValueError("Existing adopted pair is absent from the approved source index")
            original = cache_path(origin["native_root"], *key, schema=CONFIGURABLE_SCHEMA)
            if sha256(original) != pin["source_sha256"]:
                raise ValueError("Approved adoption compact checksum changed")
            original_packet = torch.load(original, map_location="cpu", weights_only=True)
            if (original_packet.get("native_sha256") != native_sha
                    or original_packet.get("native_identity") != native["identity"]):
                raise ValueError("Adopted native packet differs from the pinned original")
            pinned_compact = original
            expected_compact_sha = pin["source_sha256"]
        recovered = False
        if not path.exists():
            if not recover:
                raise ValueError("Adoption requires a committed compact packet")
            if pinned_compact is not None:
                # A copied native packet keeps the exact original compact
                # bytes too, even after an interrupted cross-disk adoption.
                atomic_copy_verified(pinned_compact, path, expected_compact_sha)
            else:
                derive_model_view(npath, path)
            recovered = True
        compact_sha = sha256(path)
        if expected_compact_sha is not None and compact_sha != expected_compact_sha:
            raise ValueError("Adoption compact file checksum mismatch")
        compact = torch.load(path, map_location="cpu", weights_only=True)
        if (compact.get("native_sha256") != native_sha
                or compact.get("native_identity") != native["identity"]
                or compact.get("provenance") != native["provenance"]):
            raise ValueError("Full-flow native/compact checksum or identity mismatch")
        load_model_pair(path, expected_source=source["token"], expected_target=target["token"],
                        expected_times=(source["time_s"], target["time_s"]))
        packet_contract(compact)
        valid = resize_mask(native["forward_valid"])
        expected_flow = resize_flow(native["forward"]).masked_fill(~valid.expand(2, -1, -1), 0).half()
        if (not torch.equal(compact["flow"], expected_flow)
                or not torch.equal(compact["valid"], valid)
                or compact["fb_mask"].any()):
            raise ValueError("Full-flow compact values differ from native derivation")
        return {"native_sha256": native_sha, "compact_sha256": compact_sha,
                "identity": native["identity"], "generation_code": native["provenance"]["generation_code"],
                "bytes": npath.stat().st_size + path.stat().st_size,
                "bytes_written": path.stat().st_size if recovered else 0, "recovered": recovered}

    def prepare_batch(self, rows, on_pair=None):
        """Return one outcome/receipt per row; only missing pairs invoke RAFT."""
        if not rows or len(rows) > self.engine.batch_size:
            raise ValueError("Invalid full-flow batch length")
        results, pending, positions = [None] * len(rows), [], []
        next_emit = 0
        def finish(index, result):
            nonlocal next_emit
            row = rows[index]
            source, target, group = row
            result.update(source=source["token"], target=target["token"],
                          version=group["version"], camera=group["camera"], scene=group["scene"])
            results[index] = result
            # Cached later rows must not overtake earlier rows still in RAFT.
            # This makes the reported prefix digest independently reproducible.
            while next_emit < len(results) and results[next_emit] is not None:
                if on_pair is not None:
                    on_pair(results[next_emit])
                next_emit += 1
        seen = set()
        for row_index, (source, target, group) in enumerate(rows):
            path = self.path(source, target, group)
            if path in seen:
                raise ValueError("Duplicate pair in full-flow batch")
            seen.add(path)
            key = (source["token"], target["token"])
            if path.exists() or native_path(path).exists():
                result = self._verify(path, source, target, recover=True)
                result["outcome"] = "recovered" if result.pop("recovered") else "reused"
            else:
                candidates = [origin for origin in self.origins.values() if key in origin["pairs"]]
                if len(candidates) > 1:
                    raise ValueError("Ambiguous full-flow adoption origins")
                if candidates:
                    origin = candidates[0]
                    row = origin["pairs"][key]
                    if (row["source_time"], row["target_time"]) != (source["time_s"], target["time_s"]):
                        raise ValueError("Adoption pair timestamps differ from generation index")
                    old_path = cache_path(origin["native_root"], *key, schema=CONFIGURABLE_SCHEMA)
                    result = self._verify(old_path, source, target, expected_compact_sha=row["source_sha256"])
                    if result["generation_code"]["commit"] != origin["commit"]:
                        raise ValueError("Adoption packet is from a different approved origin")
                    atomic_copy_verified(native_path(old_path), native_path(path), result["native_sha256"])
                    atomic_copy_verified(old_path, path, result["compact_sha256"])
                    result.update(outcome="adopted", bytes_written=result["bytes"], adopted_from=str(old_path))
                    result.pop("recovered")
                else:
                    positions.append(row_index)
                    pending.append((source, target, path, None))
                    result = None
            if result is not None:
                finish(row_index, result)
        if pending:
            try:
                outcomes = self.engine.pairs(pending)
            except BaseException:
                # RAFT may have committed an earlier pair before a later save
                # failed. Record only complete, fully revalidated packets.
                for index, job in zip(positions, pending):
                    if job[2].exists() and native_path(job[2]).exists():
                        result = self._verify(job[2], job[0], job[1])
                        result.pop("recovered")
                        result.update(outcome="created", bytes_written=result["bytes"])
                        finish(index, result)
                raise
            for index, job, outcome in zip(positions, pending, outcomes):
                path = job[2]
                if outcome != "created":
                    raise RuntimeError("Full-flow destination changed after exclusive shard preparation")
                packet = torch.load(path, map_location="cpu", weights_only=True)
                size = path.stat().st_size + native_path(path).stat().st_size
                finish(index, dict(outcome="created", bytes=size, bytes_written=size,
                                   native_sha256=packet["native_sha256"], compact_sha256=sha256(path),
                                   identity=packet["native_identity"], generation_code=self.current_code))
        return results
