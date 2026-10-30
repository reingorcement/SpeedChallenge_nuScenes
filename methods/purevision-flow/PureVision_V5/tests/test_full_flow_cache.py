"""Full-cache adoption is explicit, preserves bytes, and resumes without RAFT."""
import copy
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest
import torch

from purevision_v5.common import object_hash, sha256
from purevision_v5.flow import (CONFIGURABLE_SCHEMA, FrozenRaft, atomic_save, cache_path,
                               native_path, save_pair)
from purevision_v5 import full_flow_cache as full


CURRENT = dict(commit="a"*40, dirty=False, verified_remote_commit="a"*40)
OLD = dict(commit="b"*40, dirty=False, verified_remote_commit="b"*40)


@pytest.fixture(autouse=True)
def few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def engine(code=CURRENT):
    value = FrozenRaft.__new__(FrozenRaft)
    value.schema, value.direction = CONFIGURABLE_SCHEMA, "forward"
    value.raft_size, value.batch_size, value.device = (576,1024), 8, torch.device("cpu")
    value.generation_code = copy.deepcopy(code)
    value.calls = []
    def create(jobs):
        value.calls.append(len(jobs))
        for source,target,path,_ in jobs:
            save_pair(path, torch.zeros(2,576,1024), None, value.provenance(source,target))
        return ["created"] * len(jobs)
    value.pairs = create
    return value


def pair(tmp_path, n=0):
    group = dict(version="v1.0-trainval", camera="CAM_FRONT", scene=f"scene-{n:04d}")
    frames=[]
    for suffix,timestamp in (("s",1.0),("t",1.1)):
        file=tmp_path/f"{n}{suffix}.jpg"
        Image.new("RGB", (16,9), (30+n,40,50)).save(file)
        frames.append(dict(token=f"{suffix}{n}", filename=str(file), time_s=timestamp,
                           dataset="nuScenes", **group))
    return (*frames,group)


def stored(root, row, code=OLD):
    source,target,_=row
    path=cache_path(root,source["token"],target["token"],schema=CONFIGURABLE_SCHEMA)
    flow=torch.zeros(2,576,1024);flow[0]=8;flow[1]=-8
    save_pair(path,flow,None,engine(code).provenance(source,target))
    return path


def adoption(tmp_path, row, old_path):
    source,target,_=row
    index=dict(schema="v5_packed_flow_v1",code=OLD,contract={},arrays={},
               pairs=[dict(source=source["token"],target=target["token"],
                           source_time=source["time_s"],target_time=target["time_s"],source_sha256=sha256(old_path))])
    index["identity"]=object_hash(index)
    ipath=tmp_path/"packed_index.json";ipath.write_text(json.dumps(index))
    plan=dict(schema=full.ADOPTION_SCHEMA,origins=[dict(commit=OLD["commit"],flow_py=str(full.FLOW_SOURCE),
               flow_py_sha256=sha256(full.FLOW_SOURCE),native_root=str(old_path.parents[2]),
               packed_index=str(ipath),packed_index_sha256=sha256(ipath))])
    ppath=tmp_path/"adoption.json";ppath.write_text(json.dumps(plan))
    return ppath, full.load_adoption_plan(ppath)


def test_adoption_preserves_both_packets_original_code_and_skips_raft(tmp_path):
    row=pair(tmp_path);old=stored(tmp_path/"pilot",row)
    original=(native_path(old).read_bytes(),old.read_bytes())
    _,plan=adoption(tmp_path,row,old)
    predictor=engine();cache=full.FullFlowCache(predictor,tmp_path/"full",plan)
    result=cache.prepare_batch([row])[0];new=cache.path(*row)
    assert result["outcome"]=="adopted" and not predictor.calls
    assert (native_path(new).read_bytes(),new.read_bytes())==original
    assert (native_path(old).read_bytes(),old.read_bytes())==original
    assert torch.load(new,weights_only=True)["provenance"]["generation_code"]==OLD
    assert cache.prepare_batch([row])[0]["outcome"]=="reused"
    assert not predictor.calls


def test_clean_cpu_packer_may_have_no_remote_check_but_native_origin_stays_verified(tmp_path):
    row=pair(tmp_path);old=stored(tmp_path/"pilot",row);ppath,_=adoption(tmp_path,row,old)
    plan=json.loads(ppath.read_text());origin=plan["origins"][0]
    ipath=Path(origin["packed_index"]);index=json.loads(ipath.read_text())
    index["code"]["verified_remote_commit"]=None
    index["identity"]=object_hash({k:v for k,v in index.items() if k!="identity"})
    ipath.write_text(json.dumps(index));origin["packed_index_sha256"]=sha256(ipath);ppath.write_text(json.dumps(plan))
    loaded=full.load_adoption_plan(ppath)
    assert loaded["origins"][0]["packed_index_code"]["verified_remote_commit"] is None
    cache=full.FullFlowCache(engine(),tmp_path/"full",loaded)
    assert cache.prepare_batch([row])[0]["generation_code"]==OLD


def test_adoption_requires_explicit_commit_and_numerical_source(tmp_path):
    row=pair(tmp_path);old=stored(tmp_path/"pilot",row)
    ppath,plan=adoption(tmp_path,row,old)
    cache=full.FullFlowCache(engine(),tmp_path/"full",plan);cache.prepare_batch([row])
    with pytest.raises(ValueError,match="Unapproved"):
        full.FullFlowCache(engine(),tmp_path/"full").prepare_batch([row])
    wrong=tmp_path/"changed_flow.py";wrong.write_text(full.FLOW_SOURCE.read_text()+"\n# changed\n")
    raw=json.loads(ppath.read_text());raw["origins"][0].update(flow_py=str(wrong),flow_py_sha256=sha256(wrong))
    ppath.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="identical audited"):
        full.load_adoption_plan(ppath)


@pytest.mark.parametrize("mutation",["image","time","environment","context"])
def test_reuse_rejects_current_input_or_environment_change(tmp_path,mutation):
    row=pair(tmp_path);old=stored(tmp_path/"pilot",row);_,plan=adoption(tmp_path,row,old)
    predictor=engine();cache=full.FullFlowCache(predictor,tmp_path/"full",plan)
    cache.prepare_batch([row])
    if mutation=="image": Image.new("RGB",(16,9),(2,3,4)).save(row[0]["filename"])
    elif mutation=="time":row[0]["time_s"]-=.01
    elif mutation=="context":row[0]["dataset"]=row[1]["dataset"]="other_dataset"
    else:
        original=predictor.provenance
        def altered(source,target):
            p=original(source,target);p["versions"]["torch"]="other-version";return p
        predictor.provenance=altered
    with pytest.raises(ValueError,match="contract mismatch"):
        cache.prepare_batch([row])
    assert not predictor.calls


def test_partial_copy_recovers_native_without_overwriting_or_recomputing(tmp_path,monkeypatch):
    row=pair(tmp_path);old=stored(tmp_path/"pilot",row);_,plan=adoption(tmp_path,row,old)
    predictor=engine();cache=full.FullFlowCache(predictor,tmp_path/"full",plan)
    actual=full.atomic_copy_verified;calls=[]
    def interrupt(source,destination,digest):
        calls.append(str(destination))
        if len(calls)==2: raise OSError("simulated interrupted compact copy")
        return actual(source,destination,digest)
    monkeypatch.setattr(full,"atomic_copy_verified",interrupt)
    with pytest.raises(OSError,match="interrupted"):
        cache.prepare_batch([row])
    new=cache.path(*row)
    assert native_path(new).exists() and not new.exists()
    native_bytes=native_path(new).read_bytes()
    result=cache.prepare_batch([row])[0]
    assert result["outcome"]=="recovered" and not predictor.calls
    assert native_path(new).read_bytes()==native_bytes==native_path(old).read_bytes()
    assert new.read_bytes()==old.read_bytes()
    assert cache.prepare_batch([row])[0]["outcome"]=="reused"


def test_corrupt_adopted_native_or_compact_fails_and_does_not_rerun(tmp_path):
    row=pair(tmp_path);old=stored(tmp_path/"pilot",row);_,plan=adoption(tmp_path,row,old)
    predictor=engine();cache=full.FullFlowCache(predictor,tmp_path/"full",plan);cache.prepare_batch([row])
    new=cache.path(*row);packet=torch.load(native_path(new),weights_only=True)
    packet["forward"][0,100,100]+=1;atomic_save(packet,native_path(new))
    with pytest.raises(ValueError,match="pinned original"):
        cache.prepare_batch([row])
    assert not predictor.calls


def test_atomic_copy_verifies_bytes_and_never_replaces_existing(tmp_path):
    source=tmp_path/"source";source.write_bytes(b"correct")
    dest=tmp_path/"destination"
    with pytest.raises(ValueError,match="changed during copy"):
        full.atomic_copy_verified(source,dest,"0"*64)
    assert not dest.exists() and not list(tmp_path.glob("*.partial"))
    full.atomic_copy_verified(source,dest,sha256(source))
    source.write_bytes(b"different")
    with pytest.raises(FileExistsError): full.atomic_copy_verified(source,dest,sha256(source))
    assert dest.read_bytes()==b"correct"


def test_progress_follows_index_order_for_mixed_new_and_reused_batch(tmp_path):
    rows=[pair(tmp_path,i) for i in range(3)]
    predictor=engine();cache=full.FullFlowCache(predictor,tmp_path/"full")
    for row in rows[1:]:
        path=cache.path(*row);save_pair(path,torch.zeros(2,576,1024),None,predictor.provenance(*row[:2]))
    receipts=[];results=cache.prepare_batch(rows,on_pair=receipts.append)
    assert [r["source"] for r in receipts]==[r[0]["token"] for r in rows]
    assert [r["outcome"] for r in results]==["created","reused","reused"]
    assert predictor.calls==[1]


def test_partial_prediction_failure_records_only_committed_prefix(tmp_path):
    rows=[pair(tmp_path,i) for i in range(3)]
    predictor=engine();cache=full.FullFlowCache(predictor,tmp_path/"full")
    def interrupted(jobs):
        source,target,path,_=jobs[0]
        save_pair(path,torch.zeros(2,576,1024),None,predictor.provenance(source,target))
        raise RuntimeError("simulated device interruption")
    predictor.pairs=interrupted;receipts=[]
    with pytest.raises(RuntimeError,match="device interruption"):
        cache.prepare_batch(rows,on_pair=receipts.append)
    assert len(receipts)==1 and receipts[0]["source"]==rows[0][0]["token"]
    assert receipts[0]["outcome"]=="created"


@pytest.mark.parametrize("limit,status,count",[(0,"complete",3),(2,"sample_complete",2)])
def test_cli_reports_exact_coverage_digest_and_bounded_receipts(tmp_path,monkeypatch,limit,status,count):
    import prepare_full_flow as cli
    rows=[pair(tmp_path,i) for i in range(3)]
    ipath=tmp_path/"index.json";ipath.write_text("fixed index test fixture")
    report=tmp_path/"progress.json"
    monkeypatch.setattr(cli,"code_version",lambda **kwargs:copy.deepcopy(CURRENT))
    monkeypatch.setattr(cli,"load_index",lambda path:dict(identity="c"*64,counts=dict(pairs=3)))
    monkeypatch.setattr(cli,"iter_groups",lambda *a,**kw:[dict(pair_count=3)])
    monkeypatch.setattr(cli,"iter_pairs",lambda *a,**kw:iter(rows))
    monkeypatch.setattr(cli,"FrozenRaft",lambda *a,**kw:engine(kw["generation_code"]))
    monkeypatch.setattr(sys,"argv",["prepare_full_flow.py","--index",str(ipath),"--output-root",str(tmp_path/"full"),
                                   "--report",str(report),"--device","cpu","--limit",str(limit)])
    cli.main()
    value=json.loads(report.read_text())
    expected=hashlib.sha256("".join(f"{r[0]['token']} {r[1]['token']}\n" for r in rows[:count]).encode()).hexdigest()
    assert value["status"]==status and value["processed_pair_sha256"]==expected
    assert value["pairs_processed"]==count and value["remaining_pairs"]==0
    assert value["total"]==value["expected_pairs"]==value["all_total"]==3
    assert value["counts"]==dict(created=count,reused=0,recovered=0,adopted=0)
    assert len(value["sample_receipts"])==len(value["last_receipts"])==count
    assert value["bytes"]==value["bytes_written"]>0
