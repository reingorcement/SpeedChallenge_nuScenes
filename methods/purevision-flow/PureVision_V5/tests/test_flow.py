"""Numerical flow, cache provenance, and FB ablation boundaries."""
import copy
import json

import pytest
import torch

from purevision_v5.flow import (MODEL_SIZE, PREPROCESSING, RAFT_SHA256, RAFT_SIZE,
                               FrozenRaft, _load_native, _write_model_view, atomic_save,
                               cache_path, fb_mask, legal_mask, load_model_pair,
                               native_path, resize_flow, resize_mask, save_pair)
from prepare_flow import batched_pairs, enumerate_pairs


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def provenance(direction="bidirectional"):
    return {"source":{"token":"source", "time_s":1.0, "sha256":"a" * 64},
            "target":{"token":"target", "time_s":1.1, "sha256":"b" * 64},
            "dt_s":1.1 - 1.0, "direction":direction,
            "checkpoint_sha256":RAFT_SHA256, "preprocessing":copy.deepcopy(PREPROCESSING),
            "versions":{"torch":str(torch.__version__)}}


def test_signed_translation_warp_direction_and_border():
    forward = torch.zeros(2, 12, 16)
    forward[0] = 3
    backward = -forward
    valid = legal_mask(forward)
    assert valid[:, :, :13].all() and not valid[:, :, 13:].any()
    assert torch.equal(fb_mask(forward, backward), valid)
    reverse = fb_mask(backward, forward)
    assert not reverse[:, :, :3].any() and reverse[:, :, 3:].all()
    assert not fb_mask(forward, forward).any()


def test_squared_relative_threshold_and_nonfinite_neighbours():
    forward = torch.zeros(2, 12, 32)
    backward = torch.zeros_like(forward)
    backward[0] = .7
    assert fb_mask(forward, backward).all()  # .49 <= .0049 + .5
    backward[0] = .8
    assert not fb_mask(forward, backward).any()  # .64 > .0064 + .5
    forward[0], backward[0] = 10, -9
    assert fb_mask(forward, backward)[:, :, :22].all()  # 1 <= .01 * 181 + .5
    backward[0, 4, 12] = torch.nan
    assert legal_mask(forward)[0, 4, 2]
    assert not fb_mask(forward, backward)[0, 4, 2]
    forward[0, 5, 2] = torch.inf
    assert not legal_mask(forward)[0, 5, 2]
    assert not fb_mask(forward, backward)[0, 5, 2]


def test_resize_scales_signed_vector_units_and_binary_mask():
    flow = torch.empty(2, 32, 48)
    flow[0], flow[1] = 4, -8
    resized = resize_flow(flow, (8, 12))
    assert torch.equal(resized[0], torch.ones(8, 12))
    assert torch.equal(resized[1], torch.full((8, 12), -2.0))
    mask = torch.zeros(1, 32, 48, dtype=torch.bool)
    mask[:, ::2, ::2] = True
    assert resize_mask(mask, (8, 12)).dtype == torch.bool
    assert resize_mask(mask, (8, 12)).all()


def test_forward_cache_cannot_claim_fb_and_keeps_unreliable_flow(tmp_path):
    forward = torch.zeros(2, *RAFT_SIZE)
    forward[0] = 4
    path = cache_path(tmp_path / "forward", "source", "target")
    save_pair(path, forward, None, provenance("forward"))
    model_flow, valid, reliability = load_model_pair(path, expected_source="source", expected_target="target", expected_times=(1.0,1.1))
    assert model_flow.shape == (2, *MODEL_SIZE)
    assert model_flow.dtype == torch.float32
    assert model_flow[0, 10, 10] == 1
    assert not reliability.any()
    with pytest.raises(ValueError, match="not computed"):
        load_model_pair(path, require_fb=True)
    with pytest.raises(ValueError, match="source token"):
        load_model_pair(path, expected_source="different")
    bidir_path = cache_path(tmp_path / "bidirectional", "source", "target")
    # A zero backward flow fails FB at magnitude four, but valid numeric F stays.
    save_pair(bidir_path, forward, torch.zeros_like(forward), provenance())
    bidir_flow, bidir_valid, bidir_fb = load_model_pair(bidir_path, require_fb=True)
    assert torch.equal(bidir_flow, model_flow) and torch.equal(valid, bidir_valid)
    assert not bidir_fb.any()
    assert path.stat().st_size < 100_000
    assert native_path(path).stat().st_size > 1_000_000


def test_compact_reader_never_opens_native_and_rejects_corruption(tmp_path):
    path = cache_path(tmp_path, "source", "target")
    flow = torch.zeros(2, *RAFT_SIZE)
    save_pair(path, flow, flow, provenance())
    native_path(path).unlink()
    assert load_model_pair(path, require_fb=True)[2].all()
    packet = torch.load(path, weights_only=True)
    packet["provenance"]["source"]["token"] = "wrong"
    atomic_save(packet, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_model_pair(path)


def test_nonfinite_native_values_invalid_zero_and_recovery(tmp_path):
    path = cache_path(tmp_path, "source", "target")
    forward = torch.zeros(2, *RAFT_SIZE)
    forward[0, 40:44, 40:44] = torch.nan
    save_pair(path, forward, None, provenance("forward"))
    native = _load_native(native_path(path), provenance("forward"))
    assert not native["forward_valid"][0, 40, 40]
    assert torch.isfinite(native["forward"]).all()
    path.unlink()  # Simulate interruption between native and compact commits.
    _write_model_view(native, path)
    model_flow, valid, _ = load_model_pair(path)
    assert not valid[0, 10, 10]
    assert model_flow[:, 10, 10].eq(0).all()
    bidir_path = cache_path(tmp_path / "bidirectional", "source", "target")
    save_pair(bidir_path, native["forward"], torch.zeros_like(forward), provenance(), forward_valid=native["forward_valid"])
    assert torch.equal(load_model_pair(path)[1], load_model_pair(bidir_path)[1])
    assert not _load_native(native_path(bidir_path))["backward_fb"][0, 40, 40]


def test_resume_validates_native_checksum_without_invoking_raft(tmp_path):
    path = cache_path(tmp_path, "source", "target")
    flow = torch.zeros(2, *RAFT_SIZE)
    contract = provenance("forward")
    save_pair(path, flow, None, contract)
    engine = FrozenRaft.__new__(FrozenRaft)
    engine.direction = "forward"
    engine.model = None
    engine.provenance = lambda source, target: contract
    assert engine.pair({}, {}, path) == "reused"
    assert engine.model is None
    packet = torch.load(native_path(path), weights_only=True)
    packet["forward"][0, 50, 50] = 0.25
    atomic_save(packet, native_path(path))
    with pytest.raises(ValueError, match="Native/compact cache mismatch"):
        engine.pair({}, {}, path)


def test_reject_overwrite_and_bad_mask(tmp_path):
    path = cache_path(tmp_path, "source", "target")
    flow = torch.zeros(2, *RAFT_SIZE)
    save_pair(path, flow, None, provenance("forward"))
    with pytest.raises(FileExistsError):
        save_pair(path, flow, None, provenance("forward"))
    packet = torch.load(path, weights_only=True)
    packet["fb_mask"][0, 0, 0] = True
    atomic_save(packet, path)
    with pytest.raises(ValueError, match="FB mask"):
        load_model_pair(path)
    with pytest.raises(ValueError, match="Unsafe"):
        cache_path(tmp_path, "../source", "target")


def test_manifest_pairing_is_causal_unique_and_partition_filtered(tmp_path):
    frames = []
    for index in range(7):
        image = tmp_path / f"{index}.jpg"
        image.touch()
        frames.append({"token":f"frame{index}", "time_s":float(index), "filename":image.name, "target":[999,999]})
    entry = {"scene":"scene", "partition":"train_core", "log_token":"log", "file":"sequence.json"}
    (tmp_path / "sequence.json").write_text(json.dumps({"schema":"v5_sequence_v1", **entry, "frames":frames, "windows":[4,5]}))
    manifest = {"schema":"v5_dataset_v1", "seq_len":5, "dataroot":str(tmp_path), "sequences":[entry]}
    (tmp_path / "dataset.json").write_text(json.dumps(manifest))
    pairs = list(enumerate_pairs(tmp_path / "dataset.json", ["train_core"]))
    assert [(a["token"],b["token"]) for a,b,_,_ in pairs] == [(f"frame{i}",f"frame{i+1}") for i in range(5)]
    assert all("target" not in source for source,_,_,_ in pairs)
    with pytest.raises(ValueError, match="Unknown partitions"):
        list(enumerate_pairs(tmp_path / "dataset.json", ["test"]))


def test_local_checkpoint_integrity_is_required(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not audited RAFT weights")
    with pytest.raises(ValueError, match="audited"):
        FrozenRaft(bad, "cpu")


def test_batched_inference_keeps_pair_order_and_skips_cached_inputs(tmp_path):
    engine = FrozenRaft.__new__(FrozenRaft)
    engine.batch_size, engine.direction, engine.device = 3, "forward", torch.device("cpu")
    engine.transforms = lambda source,target: (source,target)
    engine._image = lambda frame: torch.full((1,3,*RAFT_SIZE),float(frame["brightness"]))
    calls = []
    def model(source, target, num_flow_updates):
        calls.append((len(source),num_flow_updates))
        output = torch.zeros(len(source),2,*RAFT_SIZE)
        output[:,0] = (target[:,0,0,0] - source[:,0,0,0])[:,None,None]
        return [output]
    engine._model = lambda: model
    def make_provenance(source,target):
        p=provenance("forward")
        p["source"]["token"],p["target"]["token"]=source["token"],target["token"]
        return p
    engine.provenance=make_provenance
    jobs=[]
    for i in range(3):
        source={"token":f"s{i}","filename":f"s{i}.jpg","brightness":0}
        target={"token":f"t{i}","filename":f"t{i}.jpg","brightness":4*(i+1)}
        jobs.append((source,target,cache_path(tmp_path,source["token"],target["token"]),None))
    assert engine.pairs(jobs) == ["created"]*3
    assert calls == [(3,12)]
    for i,job in enumerate(jobs):
        assert load_model_pair(job[2])[0][0,10,10] == i+1
        native=_load_native(native_path(job[2]))
        assert native["execution"]["forward_batch_size"] == 3
    assert engine.pairs(jobs) == ["reused"]*3
    assert calls == [(3,12)]


def test_batch_limit_and_scene_shards_are_disjoint(tmp_path):
    assert list(batched_pairs(iter(range(9)),4,5)) == [[0,1,2,3],[4]]
    assert list(batched_pairs(iter(range(9)),4,0)) == [[0,1,2,3],[4,5,6,7],[8]]
    entries=[]
    for scene_index in range(3):
        entry={"scene":str(scene_index),"partition":"train_core","log_token":f"log{scene_index}","file":f"s{scene_index}.json"}
        frames=[]
        for frame_index in range(5):
            name=f"s{scene_index}f{frame_index}"
            (tmp_path/name).touch()
            frames.append({"token":name,"time_s":float(frame_index),"filename":name})
        (tmp_path/entry["file"]).write_text(json.dumps({"schema":"v5_sequence_v1",**entry,"frames":frames,"windows":[4]}))
        entries.append(entry)
    manifest=tmp_path/"dataset.json"
    manifest.write_text(json.dumps({"schema":"v5_dataset_v1","seq_len":5,"dataroot":str(tmp_path),"sequences":entries}))
    parts=[]
    for shard_index in range(2):
        parts.append({(source["token"],target["token"]) for source,target,_,_ in enumerate_pairs(manifest,["train_core"],shard_index=shard_index,shard_count=2)})
    assert len(parts[0]) == 8 and len(parts[1]) == 4
    assert parts[0].isdisjoint(parts[1])


def test_cached_fb_rejects_different_frozen_forward_source(tmp_path):
    from purevision_v5.flow import sha256
    fpath=cache_path(tmp_path/"forward","source","target")
    bpath=cache_path(tmp_path/"bidirectional","source","target")
    zeros=torch.zeros(2,*RAFT_SIZE)
    save_pair(fpath,zeros,None,provenance("forward"))
    original=_load_native(native_path(fpath))
    execution={"forward_origin_identity":original["identity"],"forward_origin_sha256":sha256(native_path(fpath))}
    save_pair(bpath,zeros,zeros,provenance(),execution=execution)
    engine=FrozenRaft.__new__(FrozenRaft)
    engine.batch_size,engine.direction=1,"bidirectional"
    engine.provenance=lambda source,target: provenance()
    assert engine.pair({}, {}, bpath, reuse_forward=fpath) == "reused"
    alternative=cache_path(tmp_path/"different_forward","source","target")
    changed=zeros.clone();changed[0]=.1
    save_pair(alternative,changed,None,provenance("forward"))
    with pytest.raises(ValueError,match="different frozen forward source"):
        engine.pair({}, {}, bpath, reuse_forward=alternative)
