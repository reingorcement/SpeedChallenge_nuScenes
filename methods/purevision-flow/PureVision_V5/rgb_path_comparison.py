"""One appearance+diff training run and read-only evaluation of paused diff FT."""
import argparse
import json
import time
from pathlib import Path
from collect_results import require
from encoder_comparison import comparison_training_job
from pipeline import initialize, lock, tick, validate_plan
from purevision_v5.common import ROOT, code_version, object_hash, read_json, sha256
from purevision_v5.initialization import FLOW_SOURCE_SHA, RGB_SOURCE_SHA, resolve_source

CANDIDATE='R03_APPEAR_FT_s42_18ep'
OLD_RUN='20261031_r03_flex_s42_R03_FT'
OLD_COMMIT='2922f8c4527b6c46eb664d02a4a041cd2c45f588'


def old_history(path):
    rows=[json.loads(s) for s in (Path(path)/'metrics.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['epoch']<18]
    require(len(rows)==18 and [r['epoch'] for r in rows]==list(range(18)),'Old complete-epoch history changed')
    require(all(r['global_step']==2078*(r['epoch']+1) for r in rows),'Old complete-epoch batch budget differs')
    return rows


def validate_inputs(plan):
    root=Path(plan['root']);old=Path(plan['old_run']);run=read_json(old/'run.json')
    require(run['code']['commit']==OLD_COMMIT and run['run_id']==OLD_RUN,'Old FT identity changed')
    require(read_json(old/'status.json')['status']=='interrupted','Old FT must remain paused')
    for key,path in [('manifest_sha256',plan['manifest']),('old_best_sha256',old/'best.pt'),
                     ('old_last_sha256',old/'last.pt'),('old_history_sha256',old/'metrics.jsonl'),
                     ('pause_sha256',root/'evidence/r03_ft_user_pause_20261101.json'),
                     ('flow_index_sha256',Path(plan['flow_root'])/'packed_index.json'),
                     ('rgb_frames_index_sha256',Path(plan['rgb_frames'])/'index.json'),
                     ('verification_sha256',plan['verification']),('recovery_sha256',plan['recovery'])]:
        require(sha256(path)==plan[key],f'Changed locked input: {key}')
    require(read_json(plan['verification'])['status']=='passed' and read_json(plan['recovery'])['status']=='passed','Verification incomplete')
    history=old_history(old)
    require(min(history,key=lambda r:r['validation']['overall']['epe'])['epoch']==0,'Old18 best changed')
    conf=read_json(Path(plan['code_root'])/'configs/next'/f'{CANDIDATE}.json')
    require(sha256(Path(plan['code_root'])/'configs/next'/f'{CANDIDATE}.json')==plan['config_hashes'][CANDIDATE],'Config changed')
    differences={'epochs','scheduler_epochs','rgb_variant','run_variant'}
    require({k:v for k,v in conf.items() if k not in differences}=={k:v for k,v in run['config'].items() if k not in differences},'Common old/new protocol differs')
    require(conf['epochs']==18 and conf['scheduler_epochs']==40 and conf['rgb_variant']=='appearance_difference','Wrong authorized variant/budget')
    for name,expected in [('flow_initialization',FLOW_SOURCE_SHA),('rgb_initialization',RGB_SOURCE_SHA)]:
        require(conf[name]['sha256']==expected and sha256(resolve_source(conf[name]['path']))==expected,'Initialization changed')
    require([j['name'] for j in plan['jobs'] if j['kind']=='training']==[CANDIDATE],'Only one new training run is authorized')


def build_plan(root,manifest,prefix,version,*,rgb_frames,verification,recovery,python,code_root=ROOT):
    root,manifest,code_root=map(lambda p:Path(p).resolve(),(root,manifest,code_root))
    flow=read_json(root/'reports/full_flow_20261030_full_nuscenes_front1024/summary.json')['packed_train_dev']['root']
    old=root/'runs'/OLD_RUN;path=root/'pipeline'/prefix/'plan.json'
    candidate=comparison_training_job(root,code_root,python,prefix,manifest,flow,CANDIDATE,'cuda:0',code_root/'configs/next'/f'{CANDIDATE}.json',experiment='R03',rgb_frames=rgb_frames)
    devout=Path(candidate['commands'][1][-1]) # helper's final argument is --output path
    require(candidate['commands'][1][-2]=='--output','Unexpected evaluation command')
    trainout=devout.with_name(devout.name+'_train_core')
    traincommand=candidate['commands'][1].copy();traincommand[-1]=str(trainout);traincommand+=['--partition','train_core']
    candidate['commands'].append(traincommand)
    candidate['checks'].append(dict(path=str(trainout/'evaluation.json'),field='status',equals='completed'))
    oldout=root/'evaluations'/f'{prefix}_old_diff18'
    def evalcommand(partition):
        return [str(python),'-B',str(code_root/'evaluate.py'),'--checkpoint',str(old/'best.pt'),'--manifest',str(manifest),
                '--flow',str(flow),'--rgb-frames',str(rgb_frames),'--device','cuda:2','--workers','4','--partition',partition,'--output',str(oldout/partition)]
    oldjob=dict(name='old_diff18_eval',job_id=f'{prefix}_old_diff18_eval',kind='evaluation',device='cuda:2',requires=[],after=[],source_training_commit=OLD_COMMIT,
        commands=[evalcommand(p) for p in ['dev','train_core']],session=f'v5next_{prefix}_old_diff18_eval',
        log_path=str(root/'logs'/f'{prefix}_old_diff18_eval.log'),status_path=str(path.parent/'jobs/old_diff18_eval.json'),
        progress_path=None,run_dir=None,fresh_output=str(oldout),checks=[dict(path=str(oldout/p/'evaluation.json'),field='status',equals='completed') for p in ['dev','train_core']])
    report=root/'reports'/f'rgb_path_{prefix}'
    compare=dict(name='compare',job_id=f'{prefix}_compare',kind='comparison',device=None,requires=[CANDIDATE,'old_diff18_eval'],after=[],
        commands=[[str(python),'-B',str(code_root/'collect_rgb_path.py'),'--plan',str(path),'--output',str(report)]],
        session=f'v5next_{prefix}_compare',log_path=str(root/'logs'/f'{prefix}_compare.log'),status_path=str(path.parent/'jobs/compare.json'),
        progress_path=None,run_dir=None,fresh_output=str(report),checks=[dict(path=str(report/'summary.json'),field='status',equals='completed')])
    plan=dict(schema='v5_pipeline_v1',stage='r03_rgb_path_18epoch',scope='full_train_dev',root=str(root),code_root=str(code_root),python=str(python),
        prefix=prefix,code=version,manifest=str(manifest),manifest_sha256=sha256(manifest),flow_root=flow,flow_index_sha256=sha256(Path(flow)/'packed_index.json'),
        rgb_frames=str(rgb_frames),rgb_frames_index_sha256=sha256(Path(rgb_frames)/'index.json'),features=None,features_status=None,
        old_run=str(old),old_best_sha256=sha256(old/'best.pt'),old_last_sha256=sha256(old/'last.pt'),old_history_sha256=sha256(old/'metrics.jsonl'),
        pause_sha256=sha256(root/'evidence/r03_ft_user_pause_20261101.json'),verification=str(verification),verification_sha256=sha256(verification),recovery=str(recovery),recovery_sha256=sha256(recovery),
        config_hashes={CANDIDATE:sha256(candidate['config_path'])},jobs=[candidate,oldjob,compare],startup_gate_jobs=[CANDIDATE],
        worker_network_file=str(root/'monitoring/network.json'),completion_report=str(report/'summary.json'),
        expected_counts=dict(train_core=dict(scenes=582,windows=132978),dev=dict(scenes=103,windows=23480)),
        evaluation_paths={'difference':{'dev':str(oldout/'dev'),'train_core':str(oldout/'train_core')},
                          'appearance_difference':{'dev':str(devout),'train_core':str(trainout)}})
    validate_inputs(plan);return dict(plan,identity=object_hash(plan))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=read_json(a.plan)
    validate_plan(plan);validate_inputs(plan);version=code_version(require_pushed=True)
    require(version['commit']==plan['code']['commit'],'Wrong fixed training commit')
    with lock(Path(plan['root'])/'pipeline/coordinator.lock'):
        initialize(plan)
        while True:
            done,launched=tick(plan,a.plan)
            if launched:print(dict(launched=launched),flush=True)
            if done:break
            time.sleep(10)

if __name__=='__main__':main()
