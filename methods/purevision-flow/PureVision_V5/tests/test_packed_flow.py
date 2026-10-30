"""Bounded evidence that mmap packing preserves verified model inputs exactly."""
from pathlib import Path
import pickle
import shutil

import numpy as np
import pytest
import torch

from purevision_v5.common import object_hash, read_json
from purevision_v5.datasets import VelocityDataset
from purevision_v5.flow import cache_path
from purevision_v5.packed_flow import INDEX_NAME, PackedFlow, pack
from test_training import make_case


def diverse_packets(root):
    for path in Path(root).rglob("*.pt"):
        packet = torch.load(path, weights_only=True)
        packet["flow"][0, 20, 20] = -3.25
        packet["flow"][1, 30, 30] = 2.125
        packet["flow"][0, 15, 15] = -0.0
        packet["valid"][:, 0] = False
        packet["fb_mask"][:, :8] = False
        torch.save(packet, path)


def make_forward_copy(source, output):
    shutil.copytree(source, output)
    for path in Path(output).rglob("*.pt"):
        packet = torch.load(path, weights_only=True)
        packet["provenance"]["direction"] = "forward"
        packet["identity"] = object_hash(packet["provenance"])
        packet["fb_available"] = False
        packet["fb_mask"].zero_()
        torch.save(packet, path)


def test_packed_dataset_is_bit_exact_and_pickle_reopens_shared_mmap(tmp_path):
    args = make_case(tmp_path / "case", experiment="R04", flows=True)
    diverse_packets(args.flow)
    output = tmp_path / "packed"
    index_path = pack(args.manifest, args.flow, output, require_fb=True)
    raw = VelocityDataset(args.manifest, "train_core", "R04", args.features, args.flow)
    packed = VelocityDataset(args.manifest, "train_core", "R04", args.features, output)
    assert raw.selection_hash == packed.selection_hash
    assert len(raw) == len(packed)
    for row in (0, 3, len(raw)-1):
        original, result = raw[row], packed[row]
        for key in original["inputs"]:
            torch.testing.assert_close(original["inputs"][key], result["inputs"][key], rtol=0, atol=0)
        torch.testing.assert_close(original["target"], result["target"], rtol=0, atol=0)
    reader = packed.packed_flow
    assert isinstance(reader._arrays["flow"], np.memmap)
    # Conversion does not even change the sign bit of a zero flow component.
    pair = reader.index["pairs"][0]
    source = torch.load(cache_path(args.flow, pair["source"], pair["target"]), weights_only=True)
    np.testing.assert_array_equal(reader._arrays["flow"][0].view(np.uint16), source["flow"].numpy().view(np.uint16))
    serialized = pickle.dumps(reader)
    assert len(serialized) < 100_000
    restored = pickle.loads(serialized)
    assert restored._arrays is None
    restored.load_pairs(raw.items[0]["pairs"])
    assert isinstance(restored._arrays["flow"], np.memmap)
    assert read_json(index_path)["contract"]["fb_available"]
    with pytest.raises(FileExistsError):
        pack(args.manifest, args.flow, output, require_fb=True)


def test_forward_and_bidirectional_packs_prove_same_forward_values(tmp_path):
    args = make_case(tmp_path / "case", experiment="R04", flows=True)
    diverse_packets(args.flow)
    forward_source = tmp_path / "forward-source"
    make_forward_copy(args.flow, forward_source)
    forward_pack = tmp_path / "packed-forward"
    pack(args.manifest, forward_source, forward_pack)
    with pytest.raises(ValueError, match="no FB reliability"):
        PackedFlow(forward_pack, require_fb=True)
    bidirectional_pack = tmp_path / "packed-bidirectional"
    index = read_json(pack(args.manifest, args.flow, bidirectional_pack,
                           require_fb=True, reference_pack=forward_pack))
    assert index["forward_equality"]["flow_and_valid_exact"]
    assert index["forward_equality"]["compared_pairs"] == len(index["pairs"]) == 22
    first, second = PackedFlow(forward_pack), PackedFlow(bidirectional_pack)
    selected = [(row["source"], row["target"], row["source_time"], row["target_time"])
                for row in index["pairs"][:4]]
    old, new = first.load_pairs(selected), second.load_pairs(selected)
    torch.testing.assert_close(old[0], new[0], rtol=0, atol=0)
    torch.testing.assert_close(old[1], new[1], rtol=0, atol=0)
    assert not old[2].any() and new[2].any()
    # A different forward field is rejected even if masks/provenance look valid.
    path = cache_path(args.flow, "train_core-000", "train_core-001")
    packet = torch.load(path, weights_only=True)
    packet["flow"][0, 20, 20] += 1
    torch.save(packet, path)
    failed = tmp_path / "different-forward"
    with pytest.raises(ValueError, match="changed forward flow/valid"):
        pack(args.manifest, args.flow, failed, require_fb=True, reference_pack=forward_pack)
    assert not (failed / INDEX_NAME).exists()


def test_pack_rejects_wrong_pair_times_and_never_commits_partial_index(tmp_path):
    args = make_case(tmp_path / "case", flows=True)
    path = cache_path(args.flow, "train_core-000", "train_core-001")
    packet = torch.load(path, weights_only=True)
    for name in ("source", "target"):
        packet["provenance"][name]["time_s"] += 1
    packet["provenance"]["dt_s"] = packet["provenance"]["target"]["time_s"] - packet["provenance"]["source"]["time_s"]
    packet["identity"] = object_hash(packet["provenance"])
    torch.save(packet, path)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="token/time mismatch"):
        pack(args.manifest, args.flow, output)
    assert not (output / INDEX_NAME).exists()


def test_packed_reader_rejects_array_mutation_and_wrong_requested_time(tmp_path):
    args = make_case(tmp_path / "case", flows=True)
    output = tmp_path / "packed"
    pack(args.manifest, args.flow, output, require_fb=True)
    reader = PackedFlow(output)
    row = reader.index["pairs"][0]
    pair = (row["source"], row["target"], row["source_time"], row["target_time"])
    with pytest.raises(ValueError, match="time mismatch"):
        reader.load_pairs([(*pair[:2], pair[2] + 1, pair[3] + 1)])
    array = np.load(output / "flow.npy", mmap_mode="r+")
    array[0, 0, 20, 20] = 5
    array.flush()
    with pytest.raises(ValueError, match="changed during"):
        reader.load_pairs([pair])
    with pytest.raises(ValueError, match="checksum changed"):
        PackedFlow(output)
