"""Training data separates model inputs, offline labels, and reporting metadata."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import object_hash, read_json, sha256
from .labels import LABEL_CONTRACT, LABEL_SHA256
from .model import EXPERIMENTS
from .selection import select_records


class VelocityDataset(Dataset):
    def __init__(self, manifest_path, partition, experiment, feature_root=None, flow_root=None,
                 limit_scenes=0, limit_windows=0, selection_path=None):
        self.manifest_path = Path(manifest_path).resolve()
        manifest = read_json(self.manifest_path)
        if manifest.get("schema") != "v5_dataset_v1" or manifest.get("seq_len") != 5:
            raise ValueError("Wrong dataset schema")
        if experiment not in EXPERIMENTS:
            raise ValueError("Unknown experiment")
        if manifest.get("label_contract") != LABEL_CONTRACT or manifest.get("label_sha256") != LABEL_SHA256:
            raise ValueError("Unsupported or changed label contract")
        if partition not in {"train_core", "dev"}:
            raise ValueError("Held-out partitions are sealed until final model lock")
        self.experiment, self.flow_root = experiment, Path(flow_root).resolve() if flow_root else None
        self.items, self.scene_names, self.source_tokens = [], set(), set()
        self.feature_contract = None
        self.packed_flow = None
        self.flow_checksums, self.flow_file_stats = {}, {}
        self.manifest_hash = sha256(self.manifest_path)
        self.label_contract = manifest["label_contract"]
        feature_index = None
        if experiment != "R02":
            if not feature_root:
                raise ValueError("RGB experiments require a verified frozen feature cache")
            feature_root = Path(feature_root)
            feature_index = read_json(feature_root / "index.json")
            if feature_index["contract"]["dataset_sha256"] != self.manifest_hash:
                raise ValueError("Descriptor cache belongs to another dataset")
            self.feature_contract = feature_index["contract"]
        records, self.scene_selection = select_records(self.manifest_path, partition, selection_path)
        if self.scene_selection and (limit_scenes or limit_windows):
            raise ValueError("An explicit pilot selection cannot be combined with smoke subset limits")
        if limit_scenes:
            records = records[:limit_scenes]
        for record in records:
            path = (self.manifest_path.parent / record["file"]).resolve()
            if not path.is_relative_to(self.manifest_path.parent) or record.get("sha256") != sha256(path):
                raise ValueError("Sequence checksum/path changed")
            sequence = read_json(path)
            if sequence.get("schema") != "v5_sequence_v1" or any(sequence.get(k) != record.get(k) for k in ("scene", "partition", "log_token")):
                raise ValueError("Sequence identity differs from manifest")
            if record["scene"] in self.scene_names:
                raise ValueError("Duplicate scene in partition")
            frames = sequence["frames"]
            ends = sequence["windows"]
            if any(not isinstance(end, int) or isinstance(end, bool) or end < 4 or end >= len(frames) for end in ends) or ends != sorted(set(ends)):
                raise ValueError("Window indices must be unique ordered causal five-frame ends")
            frame_tokens = [frame["token"] for frame in frames]
            if len(frame_tokens) != len(set(frame_tokens)) or self.source_tokens.intersection(frame_tokens):
                raise ValueError("Duplicate image tokens across sequences")
            descriptors = None
            if feature_index is not None:
                info = feature_index["sequences"].get(record["scene"])
                if info is None or info["sequence_sha256"] != sha256(path):
                    raise ValueError(f"Missing/stale frozen features: {record['scene']}")
                feature_path = feature_root / info["file"]
                if sha256(feature_path) != info["sha256"]:
                    raise ValueError(f"Feature checksum changed: {record['scene']}")
                with np.load(feature_path, allow_pickle=False) as values:
                    if not np.array_equal(values["end_indices"], ends):
                        raise ValueError("Feature window indices differ")
                    descriptors = values["descriptor"].copy()
                if descriptors.shape != (len(ends), 384) or descriptors.dtype != np.float32 or not np.isfinite(descriptors).all():
                    raise ValueError("Invalid frozen descriptor values/shape/dtype")
            for row, end in enumerate(ends):
                end = int(end)
                window = frames[end-4:end+1]
                target = frames[end]["target"]
                if len(window) != 5 or target is None:
                    raise ValueError("Missing causal window or valid offline label")
                times = np.asarray([f["time_s"] for f in window], dtype=np.float64)
                dt = np.diff(times).astype(np.float32)
                target = np.asarray(target, dtype=np.float32)
                if dt.shape != (4,) or np.any(dt <= 0) or not np.isfinite(dt).all() or target.shape != (2,) or not np.isfinite(target).all():
                    raise ValueError("Invalid window time/target")
                pairs = [(a["token"], b["token"], float(a["time_s"]), float(b["time_s"]))
                         for a, b in zip(window[:-1], window[1:])]
                self.items.append(dict(descriptor=descriptors[row] if descriptors is not None else None,
                                       pairs=pairs, pair_dt=dt, target=target, scene=record["scene"],
                                       token=frames[end]["token"], time_s=float(times[-1])))
                self.source_tokens.update(f["token"] for f in window)
                self.scene_names.add(record["scene"])
                if limit_windows and len(self.items) >= limit_windows:
                    break
            if limit_windows and len(self.items) >= limit_windows:
                break
        if not self.items:
            raise ValueError("Empty training/validation dataset")
        if experiment != "R00" and self.flow_root is None:
            raise ValueError("Flow experiments require a cache root")
        flow_contract = None
        if experiment != "R00":
            from .flow import cache_path, packet_contract
            from .packed_flow import INDEX_NAME, PackedFlow
            if (self.flow_root / INDEX_NAME).exists():
                self.packed_flow = PackedFlow(self.flow_root, dataset_sha256=self.manifest_hash,
                                              require_fb=experiment in {"R04", "R05"})
                flow_contract = dict(self.packed_flow.flow_contract)
                flow_contract["packed_index_sha256"] = self.packed_flow.index_sha256
            for pair in sorted({pair for item in self.items for pair in item["pairs"]}):
                if self.packed_flow is not None:
                    self.flow_checksums[pair] = self.packed_flow.pair_checksum(*pair)
                    continue
                path = cache_path(self.flow_root, *pair[:2])
                self.flow_checksums[pair] = sha256(path)
                current_contract = packet_contract(torch.load(path, map_location="cpu", weights_only=True))
                if flow_contract is None:
                    flow_contract = current_contract
                elif flow_contract != current_contract:
                    raise ValueError("Mixed optical-flow frontend/view contracts in one dataset")
                stat = path.stat()
                self.flow_file_stats[pair] = (stat.st_size, stat.st_mtime_ns)
        self.contract = dict(dataset_sha256=self.manifest_hash, label=self.label_contract,
                             features=self.feature_contract, flow_root=str(self.flow_root) if self.flow_root else None,
                             flow=flow_contract)
        if self.scene_selection:
            self.contract["scene_selection"] = self.scene_selection
        self.selection_hash = object_hash(dict(tokens=[item["token"] for item in self.items],
                                               flow_packets=[[*pair, digest] for pair, digest in self.flow_checksums.items()]))

    def __len__(self):
        return len(self.items)

    @lru_cache(maxsize=64)
    def _pair(self, source, target, source_time, target_time):
        from .flow import cache_path, load_model_pair
        path = cache_path(self.flow_root, source, target)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != self.flow_file_stats[(source, target, source_time, target_time)]:
            raise ValueError("Flow cache changed during a running experiment")
        return load_model_pair(path, expected_source=source, expected_target=target,
                               expected_times=(source_time, target_time),
                               require_fb=self.experiment in {"R04", "R05"})

    def __getitem__(self, index):
        item = self.items[index]
        visual = {"pair_dt": torch.from_numpy(item["pair_dt"].copy())}
        if self.experiment != "R02":
            visual["descriptor"] = torch.from_numpy(item["descriptor"].copy())
        if self.experiment != "R00":
            if self.packed_flow is not None:
                flows, valid, reliability = self.packed_flow.load_pairs(item["pairs"])
            else:
                packets = [self._pair(*pair) for pair in item["pairs"]]
                flows = torch.stack([packet[0] for packet in packets])
                valid = torch.stack([packet[1] for packet in packets])
                reliability = torch.stack([packet[2] for packet in packets])
            visual["flows"], visual["valid"] = flows, valid
            if self.experiment in {"R04", "R05"}:
                visual["fb_mask"] = reliability
        return dict(inputs=visual, target=torch.from_numpy(item["target"].copy()),
                    scene=item["scene"], token=item["token"], time_s=item["time_s"])
