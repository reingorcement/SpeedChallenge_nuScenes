"""Collector protocol tests use tiny files; native tensor checks have a CPU case."""
import copy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import collect_full_flow as collector
from purevision_v5.common import ROOT, object_hash, read_json, sha256, write_json
from purevision_v5.flow import CONFIGURABLE_SCHEMA, cache_path, derived_contract, native_contract, native_path, save_pair
from purevision_v5.full_flow_index import build_index, iter_pairs, load_index
from purevision_v5.packed_flow import ARRAY_SPECS, _manifest_pairs


CODE = dict(commit="a"*40, dirty=False, verified_remote_commit="a"*40)


@pytest.fixture
def completed_cache(tmp_path, monkeypatch):
    root, index_dir, cache_root = tmp_path/"raw", tmp_path/"index", tmp_path/"cache"
    source = tmp_path/"dataset.json"
    entries = []
    for version, names in (("v1.0-trainval", ["scene-a", "scene-b"]), ("v1.0-test", ["scene-c"])):
        rows, scenes, samples = [], [], []
        for scene in names:
            scenes.append(dict(token=scene, name=scene))
            samples.append(dict(token=scene+"-sample", scene_token=scene))
            frames = []
            for n in range(6):
                token=f"{scene}-{n}"
                filename=f"samples/CAM_FRONT/{token}.jpg"
                image=root/filename
                image.parent.mkdir(parents=True,exist_ok=True)
                Image.new("RGB",(16,9),(10,20,30)).save(image)
                rows.append(dict(token=token,sample_token=scene+"-sample",calibrated_sensor_token="cal",
                                 filename=filename,width=1600,height=900,timestamp=1_000_000+n*100_000,
                                 prev=f"{scene}-{n-1}" if n else "",next=f"{scene}-{n+1}" if n<5 else ""))
                frames.append(dict(token=token,time_s=1+n/10,filename=filename))
            if version=="v1.0-trainval":
                partition="train_core" if scene=="scene-a" else "dev"
                relative=f"sequences/{scene}.json"
                write_json(tmp_path/relative,dict(schema="v5_sequence_v1",scene=scene,partition=partition,
                                                log_token=scene+"-log",frames=frames,windows=[4]))
                entries.append(dict(scene=scene,partition=partition,log_token=scene+"-log",file=relative,sha256=sha256(tmp_path/relative)))
        for name,value in dict(scene=scenes,sample=samples,sample_data=rows,
                               sensor=[dict(token="sensor",channel="CAM_FRONT")],
                               calibrated_sensor=[dict(token="cal",sensor_token="sensor")]).items():
            write_json(root/version/(name+".json"),value)
    write_json(source,dict(schema="v5_dataset_v1",version="v1.0-trainval",seq_len=5,dataroot=str(root),split_sha256="b"*64,sequences=entries))
    monkeypatch.setattr("purevision_v5.full_flow_index.code_version",lambda **kw:CODE)
    index_path=build_index(root,source,index_dir,versions=["v1.0-trainval","v1.0-test"],cameras=["CAM_FRONT"])
    index=load_index(index_path)
    adoption=tmp_path/"adoption.json"
    write_json(adoption,dict(schema="v5_full_flow_adoption_v1",origins=[]))
    config=read_json(ROOT/"configs/next/full_nuscenes_flow.json")
    config["devices"]=["cuda:0","cuda:1","cuda:2"]
    config_path=tmp_path/"config.json";write_json(config_path,config)
    jobs=[];receipts={};all_rows={}
    for shard in range(3):
        rows=list(iter_pairs(index_path,shard_index=shard,shard_count=3)); samples=[];digest=hashlib.sha256()
        for src,tgt,g in rows:
            path=cache_path(cache_root/g['version']/g['camera'],src['token'],tgt['token'],schema=CONFIGURABLE_SCHEMA)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((src['token']+' compact').encode());native_path(path).write_bytes((src['token']+' native').encode())
            receipt=dict(source=src['token'],target=tgt['token'],version=g['version'],camera=g['camera'],scene=g['scene'],
                         outcome="created",compact_sha256=sha256(path),native_sha256=sha256(native_path(path)),
                         identity=object_hash([src['token'],tgt['token']]),generation_code=CODE,
                         bytes=path.stat().st_size+native_path(path).stat().st_size)
            receipt['bytes_written']=receipt['bytes'];samples.append(receipt);receipts[str(path)]=receipt
            all_rows[(src['token'],tgt['token'])]=(src,tgt,g)
            digest.update(f"{src['token']} {tgt['token']}\n".encode())
        report_path=tmp_path/f"flow{shard}.json";status_path=tmp_path/f"flow{shard}_status.json"
        count=len(rows)
        report=dict(status="complete",scope="full_dataset_generation",code=CODE,index=str(index_path),index_sha256=sha256(index_path),
                    index_identity=index['identity'],flow_py_sha256=sha256(ROOT/'purevision_v5/flow.py'),
                    arguments=dict(limit=0,shard_index=shard,shard_count=3,batch_size=8,device=f'cuda:{shard}',
                                   output_root=str(cache_root),index=str(index_path),index_sha256=sha256(index_path),adoption=str(adoption)),
                    adoption=dict(path=str(adoption),sha256=sha256(adoption)),native_contract=native_contract((576,1024)),
                    derived_contract=derived_contract((576,1024)),pairs_processed=count,expected_pairs=count,target_pairs=count,total=count,
                    remaining_pairs=0,all_total=index['counts']['pairs'],counts=dict(created=count,reused=0,recovered=0,adopted=0),
                    processed_pair_sha256=digest.hexdigest(),bytes=sum(x['bytes'] for x in samples),bytes_written=sum(x['bytes'] for x in samples),
                    sample_receipts=samples[:3],last_receipts=samples[-3:],elapsed_s=10.0,cuda_peak_allocated_bytes=0,
                    started_at_utc='2026-10-30T00:00:00+00:00',finished_at_utc='2026-10-30T00:00:10+00:00')
        write_json(report_path,report);write_json(status_path,dict(status='completed',exit_code=0,commit=CODE['commit']))
        jobs.append(dict(name=f'flow{shard}',kind='flow',device=f'cuda:{shard}',progress_path=str(report_path),status_path=str(status_path),expected_pairs=count))
    pack_root=tmp_path/'packed';pack_root.mkdir();expected=_manifest_pairs(source,('train_core','dev'));arrays={}
    for name,(dtype,shape) in ARRAY_SPECS.items():
        path=pack_root/(name+'.npy');np.save(path,np.zeros((len(expected),*shape),dtype=dtype))
        arrays[name]=dict(file=path.name,sha256=sha256(path),shape=[len(expected),*shape],dtype=dtype.name)
    from purevision_v5.flow import RAFT_SHA256
    contract=dict(flow_schema=CONFIGURABLE_SCHEMA,raft_sha256=RAFT_SHA256,native_contract=native_contract((576,1024)),
                  derived_contract=derived_contract((576,1024)),dataset_sha256=sha256(source),partitions=['train_core','dev'],
                  fb_available=False,source_root=str(cache_root/'v1.0-trainval/CAM_FRONT'))
    pack=dict(schema='v5_packed_flow_v1',code={**CODE,'verified_remote_commit':None},contract=contract,arrays=arrays,
              pairs=[dict(source=a,target=b,source_time=t0,target_time=t1,
                          source_sha256=receipts[str(cache_path(cache_root/'v1.0-trainval/CAM_FRONT',a,b,schema=CONFIGURABLE_SCHEMA))]['compact_sha256']) for a,b,t0,t1 in expected])
    pack['identity']=object_hash(pack);write_json(pack_root/'packed_index.json',pack)
    pack_status=tmp_path/'pack_status.json';write_json(pack_status,dict(status='completed',exit_code=0,commit=CODE['commit']))
    jobs += [dict(name='pack',kind='pack',status_path=str(pack_status)),dict(name='verify',kind='verify')]
    plan=dict(schema='v5_pipeline_v1',stage='full_nuscenes_flow',full_generation_authorized=True,training_authorized=False,
              code=CODE,code_root=str(ROOT),manifest=str(index_path),manifest_sha256=sha256(index_path),
              data_manifest=str(source),data_manifest_sha256=sha256(source),adoption=str(adoption),adoption_sha256=sha256(adoption),
              config_path=str(config_path),config_sha256=sha256(config_path),config=config,index_identity=index['identity'],counts=index['counts'],
              output_root=str(cache_root),packed_train_dev=str(pack_root),jobs=jobs)
    plan['identity']=object_hash(plan);plan_path=tmp_path/'plan.json';write_json(plan_path,plan)
    class SyntheticPacketVerifier:
        def path(self,source,target,group):
            return cache_path(cache_root/group['version']/group['camera'],source['token'],target['token'],schema=CONFIGURABLE_SCHEMA)
        def _verify(self,path,source,target,**kwargs):
            return copy.deepcopy(receipts[str(path)])
    monkeypatch.setattr(collector,'packet_verifier',lambda *args:SyntheticPacketVerifier())
    return plan_path,tmp_path/'report',plan


def test_complete_checks_cover_index_shards_packed_arrays_and_nullable_pack_remote(completed_cache):
    plan,output,_=completed_cache
    result=collector.collect(plan,output)
    assert result['status']=='complete',result.get('error')
    assert result['counts']==dict(groups=3,frames=18,pairs=15)
    assert result['packed_train_dev']['pairs']==8
    assert result['packed_train_dev']['hashed_array_bytes']>0
    assert result['validation_scope']['native_and_compact_checksum_tensor_samples']==15
    assert result['model_training_started'] is False and result['model_metrics'] is None
    assert result['outcomes']==dict(created=15,reused=0,recovered=0,adopted=0)
    assert 'no model-training or accuracy result' in (output/'summary.md').read_text()
    with pytest.raises(FileExistsError):collector.collect(plan,output)


@pytest.mark.parametrize('corruption',['digest','limit','worker','receipt','pack_partition','pack_times','array','missing_native'])
def test_corruption_fails_honestly_and_running_receipt_is_written_first(completed_cache,monkeypatch,corruption):
    plan_path,output,plan=completed_cache
    flow_path=Path(plan['jobs'][0]['progress_path']);flow=read_json(flow_path)
    if corruption=='digest':flow['processed_pair_sha256']='0'*64
    elif corruption=='limit':flow['arguments']['limit']=1
    elif corruption=='receipt':flow['sample_receipts'][0]['native_sha256']='0'*64
    elif corruption=='worker':write_json(plan['jobs'][0]['status_path'],dict(status='failed',exit_code=1,commit=CODE['commit']))
    elif corruption=='missing_native':
        r=flow['sample_receipts'][0];native_path(cache_path(Path(plan['output_root'])/r['version']/r['camera'],r['source'],r['target'],schema=CONFIGURABLE_SCHEMA)).unlink()
    else:
        ip=Path(plan['packed_train_dev'])/'packed_index.json';pack=read_json(ip)
        if corruption=='pack_partition':pack['contract']['partitions']=['train_core','dev','final_system_test']
        elif corruption=='pack_times':pack['pairs'][0]['source_time']-=.01
        elif corruption=='array':
            path=Path(plan['packed_train_dev'])/pack['arrays']['flow']['file'];data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        pack['identity']=object_hash({k:v for k,v in pack.items() if k!='identity'});write_json(ip,pack)
    write_json(flow_path,flow)
    statuses=[];actual=collector.write_json
    def capture(path,value):
        statuses.append(value['status']);return actual(path,value)
    monkeypatch.setattr(collector,'write_json',capture)
    result=collector.collect(plan_path,output)
    assert statuses[0]=='running' and statuses[-1]=='failed'
    assert result['status']=='failed' and result['error']
    assert read_json(output/'summary.json')['status']=='failed'
    assert 'Failure:' in (output/'summary.md').read_text()


def test_real_cpu_receipt_load_and_native_tamper_without_raft(tmp_path):
    plan=dict(code=CODE,config=dict(batch_size=8),output_root=str(tmp_path/'cache'))
    verifier=collector.packet_verifier(plan,dict(schema='v5_full_flow_adoption_v1',origins=[]))
    group=dict(version='v1.0-trainval',camera='CAM_FRONT',scene='scene-a')
    frames=[]
    for name,t in (('source',1.),('target',1.1)):
        image=tmp_path/(name+'.jpg');Image.new('RGB',(16,9),(10,20,30)).save(image)
        frames.append(dict(token=name,time_s=t,filename=str(image),dataset='nuScenes',**group))
    source,target=frames;path=verifier.path(source,target,group)
    previous_threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        save_pair(path,torch.zeros(2,576,1024),None,verifier.engine.provenance(source,target))
        checked=verifier._verify(path,source,target)
        receipt=dict(checked,source=source['token'],target=target['token'],**group,outcome='created')
        result=collector.verify_receipt(verifier,(source,target,group),receipt)
        assert result['generation_code']==CODE
        with native_path(path).open('ab') as stream:stream.write(b'tampered')
        with pytest.raises(ValueError,match='checksum'):
            collector.verify_receipt(verifier,(source,target,group),receipt)
    finally:
        torch.set_num_threads(previous_threads)
