"""Three fresh runs: frozen RGB, real RGB fine-tuning, same-start flow continuation."""
import argparse
import re
import sys
import time
from pathlib import Path
from collect_results import require
from encoder_comparison import comparison_training_job
from pipeline import initialize, lock, tick
from purevision_v5.common import ROOT, code_version, object_hash, read_json, sha256
from purevision_v5.initialization import FLOW_SOURCE_SHA, RGB_SOURCE_SHA, resolve_source

NAMES=("R03_Frozen","R03_FT","R02_FLEX_CONT")
DIFFERENCES={"experiment","run_variant","rgb_policy","rgb_initialization","rgb_learning_rate","rgb_input"}


def validate_inputs(plan):
    require(sha256(plan['manifest'])==plan['manifest_sha256'],'Manifest changed')
    require(sha256(plan['source_summary'])==plan['source_summary_sha256'],'Flow completion evidence changed')
    source=read_json(plan['source_summary'])
    require(source['status']=='complete' and source['generation_code']['commit']=='87d7468f911da5579bbbc39f79ec6bc85fa16dd6','Wrong flow source')
    pack=source['packed_train_dev']
    require(plan['flow_root']==pack['root'] and sha256(Path(pack['root'])/'packed_index.json')==pack['index_sha256'],'Packed flow changed')
    require(sha256(Path(plan['features'])/'index.json')==plan['rgb_features_index_sha256'],'Frozen feature index changed')
    require(sha256(Path(plan['rgb_frames'])/'index.json')==plan['rgb_frames_index_sha256'],'Image view index changed')
    require({j['name'] for j in plan['jobs']}==set(NAMES)|{'compare'} and len(plan['jobs'])==4,'Unexpected jobs')
    configs=[]
    for name in NAMES:
        job=next(j for j in plan['jobs'] if j['name']==name);config=read_json(job['config_path'])
        require(Path(job['config_path']).resolve().is_relative_to(Path(plan['code_root']).resolve()),'Config outside fixed checkout')
        require(sha256(job['config_path'])==plan['config_hashes'][name],'Config changed')
        require(config['encoder_variant']=='flex' and config['seed']==42 and config['epochs']==40
                and config['batch_size']==64 and config['micro_batch_size']==64 and config['early_stopping'] is False,
                'Changed authorized protocol')
        require(config['learning_rate']==1e-3 and config['weight_decay']==1e-4 and config['grad_clip']==2
                and config['scheduler_policy']=='cosine_each_group_to_0.1' and config['amp'] is True,'Changed optimization')
        init=config['flow_initialization']
        require(init['sha256']==FLOW_SOURCE_SHA and sha256(resolve_source(init['path']))==FLOW_SOURCE_SHA,'Seed42 source changed')
        policy={'R03_Frozen':'frozen','R03_FT':'finetune','R02_FLEX_CONT':'not_used_flow_only'}[name]
        require(config['run_variant']==name and config['rgb_policy']==policy
                and config['experiment']==('R02' if name=='R02_FLEX_CONT' else 'R03'),'Wrong model construction')
        if name!='R02_FLEX_CONT':
            rgb=config['rgb_initialization']
            require(config['rgb_learning_rate']==1e-4 and rgb['sha256']==RGB_SOURCE_SHA
                    and sha256(resolve_source(rgb['path']))==RGB_SOURCE_SHA,'RGB source/policy changed')
        configs.append({k:v for k,v in config.items() if k not in DIFFERENCES})
    require(configs[0]==configs[1]==configs[2],'Shared policies differ')
    preflight=read_json(plan['verification'])
    require(preflight['status']=='passed' and sha256(plan['verification'])==plan['verification_sha256'],'Real preflight not verified')


def build_plan(root,manifest,prefix,version,*,features,rgb_frames,verification,code_root=ROOT,python=sys.executable):
    root,manifest,code_root=map(lambda p:Path(p).resolve(),(root,manifest,code_root))
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',prefix)),'Invalid prefix')
    source=root/'reports/full_flow_20261030_full_nuscenes_front1024/summary.json'
    flow=read_json(source)['packed_train_dev']['root'];path=root/'pipeline'/prefix/'plan.json'
    jobs=[]
    for name,device in (('R03_FT','cuda:0'),('R03_Frozen','cuda:2'),('R02_FLEX_CONT','cuda:2')):
        jobs.append(comparison_training_job(root,code_root,python,prefix,manifest,flow,name,device,
            code_root/'configs/next'/f'{name}.json',experiment='R02' if name=='R02_FLEX_CONT' else 'R03',
            after=['R03_Frozen'] if name=='R02_FLEX_CONT' else [],
            features=features if name=='R03_Frozen' else None,rgb_frames=rgb_frames if name=='R03_FT' else None))
    report=root/'reports'/f'fusion_comparison_{prefix}'
    jobs.append(dict(name='compare',job_id=f'{prefix}_compare',kind='comparison',device=None,requires=list(NAMES),after=[],
        commands=[[str(python),'-B',str(code_root/'collect_fusion.py'),'--plan',str(path),'--output',str(report)]],
        session=f'v5next_{prefix}_compare',log_path=str(root/'logs'/f'{prefix}_compare.log'),
        status_path=str(path.parent/'jobs/compare.json'),progress_path=None,run_dir=None,fresh_output=str(report),
        checks=[dict(path=str(report/'summary.json'),field='status',equals='completed')]))
    index=read_json(Path(features)/'index.json');frames=read_json(Path(rgb_frames)/'index.json')
    plan=dict(schema='v5_pipeline_v1',stage='r03_flex_continuation',scope='full_train_dev',root=str(root),
        code_root=str(code_root),python=str(python),prefix=prefix,code=version,manifest=str(manifest),
        manifest_sha256=sha256(manifest),flow_root=flow,source_summary=str(source),source_summary_sha256=sha256(source),
        features=str(features),features_status=None,rgb_features_contract=index['contract'],
        rgb_features_index_sha256=sha256(Path(features)/'index.json'),rgb_frames=str(rgb_frames),
        rgb_frames_identity=frames['identity'],rgb_frames_index_sha256=sha256(Path(rgb_frames)/'index.json'),
        verification=str(verification),verification_sha256=sha256(verification),
        config_hashes={j['name']:sha256(j['config_path']) for j in jobs if j.get('config_path')},jobs=jobs,
        worker_network_file=str(root/'monitoring/network.json'),startup_gate_jobs=['R03_FT','R03_Frozen'],
        expected_counts=dict(train_core=dict(scenes=582,windows=132978),dev=dict(scenes=103,windows=23480)))
    validate_inputs(plan);return dict(plan,identity=object_hash(plan))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','manifest','run-prefix','features','rgb-frames','verification'):p.add_argument('--'+key,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    plan=build_plan(a.root,a.manifest,a.run_prefix,code_version(require_pushed=not a.dry_run),
        features=a.features,rgb_frames=a.rgb_frames,verification=a.verification)
    if a.dry_run:
        import json
        print(json.dumps(plan,indent=2));return
    with lock(Path(a.root)/'pipeline/coordinator.lock'):
        path=initialize(plan);print(dict(plan=str(path),identity=plan['identity']),flush=True)
        while True:
            done,launched=tick(plan,path)
            if launched:print(dict(launched=launched),flush=True)
            if done:break
            time.sleep(10)


if __name__=='__main__':main()
