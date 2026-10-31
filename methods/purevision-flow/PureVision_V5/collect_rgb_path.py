"""Fair 18-epoch RGB-path comparison; never reinterpret the paused epoch19 last."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from collect_results import require, target_identity, check_summary
from pipeline import validate_plan
from rgb_path_comparison import CANDIDATE, OLD_COMMIT, old_history, validate_inputs
from purevision_v5.common import now, read_json, write_json, sha256, object_hash
from purevision_v5.evaluation import summarize, lateral_acceptance
from purevision_v5.selection import label_rows


def collect(plan_path):
    plan=read_json(plan_path);validate_plan(plan);validate_inputs(plan)
    candidate=next(j for j in plan['jobs'] if j['name']==CANDIDATE);old=Path(plan['old_run']);new=Path(candidate['run_dir'])
    require(read_json(candidate['status_path'])['status']=='completed','Candidate training/evaluation is incomplete')
    require(read_json(new/'status.json')['status']=='completed','Candidate has not finished')
    paths={'difference':old,'appearance_difference':new};labels={p:label_rows(plan['manifest'],p)[0] for p in ['train_core','dev']}
    runs={name:read_json(path/'run.json') for name,path in paths.items()};histories={name:old_history(path) for name,path in paths.items()}
    oldrun,newrun=runs['difference'],runs['appearance_difference']
    require(newrun['code']['commit']==plan['code']['commit'] and newrun['code']['verified_remote_commit']==plan['code']['commit'] and not newrun['code']['dirty'],'New training version not fixed/pushed')
    require(oldrun['code']['commit']==OLD_COMMIT and oldrun['status']=='interrupted','Old source identity changed')
    require(newrun['contract']['execution']=={'smoke':False,'max_batches':0},'Subset training is not comparable')
    for key in ['data','train_selection','dev_selection']:
        require(oldrun['contract'][key]==newrun['contract'][key],f'Different common contract: {key}')
    require(oldrun['sample_order']==newrun['sample_order'],'Different sample-order policy')
    for a,b in zip(histories['difference'],histories['appearance_difference']):
        require(a['learning_rates']==b['learning_rates'],'Learning rate differed within first18 epochs')
        require(a['sample_order_sha256']==b['sample_order_sha256'],'Sample order differed')
    for key in ['head_state_sha256','encoder_state_sha256','flow_checkpoint_sha256','rgb_checkpoint_sha256']:
        require(oldrun['initialization'][key]==newrun['initialization'][key],f'Common initialization differs: {key}')
    require(oldrun['initialization']['rgb_state_sha256']==newrun['initialization']['common_rgb_state_sha256'],'Shared RGB state differs')
    last=torch.load(new/'last.pt',map_location='cpu',weights_only=True)
    require(last['epoch']==18 and last['next_batch']==0 and last['scheduler']['T_max']==40,'Wrong ending/scheduler budget')
    require(last['global_step']==last['processed_batches']==last['processed_microbatches']==37404,'Incomplete processed batch budget')
    require(last['optimizer_updates']+last['amp_skips']==37404,'Wrong update counters')
    rows={};rule=newrun['config']['lateral_rule']
    for name,directory in paths.items():
        run=runs[name];history=histories[name];best=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        index=int(np.argmin([v['validation']['overall']['epe'] for v in history]))
        require(best['best_epoch']==index and best['best_epe']==history[index]['validation']['overall']['epe'],'Best is not first18 dev selection')
        require(best['run_id']==run['run_id'] and best['contract']==run['contract'] and best['code']==run['code'],'Best checkpoint contract/run mismatch')
        row=dict(run_id=run['run_id'],training_commit=run['code']['commit'],budget_complete_epochs=18,
            original_run_status=run['status'],best_epoch_one_based=index+1,best_checkpoint=str(directory/'best.pt'),best_sha256=sha256(directory/'best.pt'),
            trainable_parameters=run['trainable_parameters'],total_parameters=run['total_parameters'],
            processed_batches=history[-1]['processed_batches'],optimizer_updates=history[-1]['optimizer_updates'],amp_skips=history[-1]['amp_skips'],
            first18_wall_seconds=(datetime.fromisoformat(history[-1]['time'])-datetime.fromisoformat(run['started_at'])).total_seconds(),
            cost_scope='run startup through epoch18 validation; includes loading/validation/checkpoint IO; excludes old partial epoch19',
            peak_cuda_allocated_bytes=read_json(directory/'status.json')['cuda_peak_allocated_bytes'],
            peak_scope='whole observed run; old peak may include partial epoch19',evaluations={})
        require(row['processed_batches']==37404 and row['optimizer_updates']+row['amp_skips']==37404,'Epoch18 counters differ')
        row['throughput_windows_s']=132978*18/row['first18_wall_seconds']
        for part,out in plan['evaluation_paths'][name].items():
            out=Path(out);receipt=read_json(out/'evaluation.json');metrics=read_json(out/'metrics.json')
            require(receipt['status']=='completed' and receipt['partition']==part,'Independent evaluation incomplete')
            require(receipt['checkpoint_sha256']==row['best_sha256'] and receipt['training_run']==run['run_id'],'Evaluation checkpoint differs')
            require(receipt['training_code']==run['code'] and receipt['evaluation_code']['commit']==plan['code']['commit'] and not receipt['evaluation_code']['dirty'],'Wrong evaluation code')
            require(receipt['contract_hash']==run['contract_hash'] and receipt['data']==run['contract']['data'],'Evaluation data contract differs')
            selection_key='train_selection' if part=='train_core' else 'dev_selection'
            require(receipt['selection_hash']==run['contract'][selection_key],'Wrong evaluation selection')
            with (out/'predictions.csv').open(newline='') as f:predictions=list(csv.DictReader(f))
            require(target_identity(predictions)==target_identity(labels[part]),'Prediction labels/windows differ')
            actual=summarize([[float(r['vx']),float(r['vy'])] for r in predictions],[[float(r['target_vx']),float(r['target_vy'])] for r in predictions],[r['scene'] for r in predictions],rule)
            check_summary(metrics,actual)
            if part=='dev':check_summary(read_json(directory/'best_metrics.json'),actual)
            row['evaluations'][part]=dict(metrics=actual,predictions=str(out/'predictions.csv'),predictions_sha256=sha256(out/'predictions.csv'),count=len(predictions))
        row['generalization_gap_dev_minus_train_epe']=row['evaluations']['dev']['metrics']['overall']['epe']-row['evaluations']['train_core']['metrics']['overall']['epe']
        rows[name]=row
    a,b=[rows[n]['evaluations']['dev']['metrics'] for n in ['appearance_difference','difference']]
    delta={k:a['overall'][k]-b['overall'][k] for k in ['epe','mae_vx','mae_vy','rmse_vx','rmse_vy','bias_vx','bias_vy']}
    scenes={s:{k:a['per_scene'][s][k]-v[k] for k in delta} for s,v in b['per_scene'].items()}
    curves={name:[dict(epoch=r['epoch']+1,train_scaled_smoothl1=r['train_loss'],dev=r['validation']['overall'],learning_rates=r['learning_rates'],processed_batches=r['processed_batches'],optimizer_updates=r['optimizer_updates'],amp_skips=r['amp_skips']) for r in h] for name,h in histories.items()}
    baseline={}
    median=np.median([[r['target_vx'],r['target_vy']] for r in labels['train_core']],axis=0)
    targets=np.asarray([[r['target_vx'],r['target_vy']] for r in labels['dev']]);sc=[r['scene'] for r in labels['dev']]
    baseline['train_median_xy']=dict(constant_m_s=median.tolist(),fit_partition='train_core',metrics=summarize(np.broadcast_to(median,targets.shape),targets,sc,rule))
    baseline['zero_vy_mae']=float(np.abs(targets[:,1]).mean())
    verdict='lower' if delta['epe']<0 else 'higher' if delta['epe']>0 else 'equal'
    return dict(schema='v5_rgb_path_18epoch_comparison_v1',status='completed',finished_at=now(),plan_identity=plan['identity'],rows=rows,
        comparison=dict(delta_appearance_minus_difference=delta,per_scene_deltas=scenes,improved_scenes=sum(v['epe']<0 for v in scenes.values()),degraded_scenes=sum(v['epe']>0 for v in scenes.values()),lateral=lateral_acceptance(a,b,rule)),
        curves=curves,baselines=baseline,recommendation=f'Appearance bypass has {verdict} dev EPE within this seed42 common18-epoch trajectory; inspect lateral and train/dev gaps before a route decision.',
        limits=['18 complete epochs with original40-epoch cosine; old partial epoch19 excluded. Old last is not epoch18 endpoint.',
                'Single candidate/seed; no40-epoch claim or isolated proof that appearance is inherently useful.',
                'Dev used for selection; original scene split not log-disjoint; PLANAR_PROXY rear-axle supervision.',
                '31 independent lateral windows/5scenes: unchanged100-window/5-scene rule, insufficient_data.',
                'Frozen/CONT/R02 40-epoch results are historical context, excluded from this primary comparison.'])


def markdown(r):
    lines=['# RGB appearance bypass vs difference:18-epoch comparison','',r['recommendation'],'',
        '| Path | Dev EPE | Train EPE | vx MAE | vy MAE | Updates | AMP skips |','|---|---:|---:|---:|---:|---:|---:|']
    for name,row in r['rows'].items():
        d=row['evaluations']['dev']['metrics']['overall'];t=row['evaluations']['train_core']['metrics']['overall']
        lines.append(f"| {name} | {d['epe']:.6f} | {t['epe']:.6f} | {d['mae_vx']:.6f} | {d['mae_vy']:.6f} | {row['optimizer_updates']} | {row['amp_skips']} |")
    lines+=['','Full component errors, scene deltas, curves, lateral rule, costs and hashes are in summary.json.','']+['- '+s for s in r['limits']]
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    report=collect(a.plan);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'summary.json',report);(out/'summary.md').write_text(markdown(report));print(report['recommendation'],flush=True)
